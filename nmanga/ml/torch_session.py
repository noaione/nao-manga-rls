"""
MIT License

Copyright (c) 2022-present noaione

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# The PyTorch execution backend: device, layout, precision and compilation.

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .contracts import MLDataType, ModelInfo, TorchPreset, TorchRuntimeConfig
from .model import load_checkpoint
from .opt_arch import build_fast_model

if TYPE_CHECKING:
    import torch

__all__ = ("RUNTIME_DTYPE", "TorchRuntime", "configure_torch_runtime", "create_runtime", "resolve_device")

RUNTIME_DTYPE = MLDataType.FP32
"""The runtime precision: FP32 is the contract, never FP16/BF16 from a name or metadata."""

_COMPILE_MODE = "reduce-overhead"
"""The measured `torch.compile` mode for the fast CUDA recipe."""


def configure_torch_runtime() -> None:
    """
    Configure FP32 CUDA math with TF32 disabled, matching the validated recipe.

    `NVIDIA_TF32_OVERRIDE` and the `torch.backends` flags have to be in place before any CUDA
    kernel runs, so `create_runtime` calls this before it resolves a device. `cudnn.benchmark`
    matches the benchmark setup; both tile shapes are fixed, so the autotune happens once during
    the warm-up pass.
    """
    from ..lazy import get_torch

    torch = get_torch()
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True


def resolve_device(spec: str) -> torch.device:
    """
    Resolve a device specification into a torch device.

    Parameters
    ----------
    spec : str
        `auto` (CUDA when available, otherwise CPU), `cpu`, `cuda` or `cuda:N`.

    Returns
    -------
    torch.device
        The resolved device, always with an explicit CUDA index.

    Raises
    ------
    RuntimeError
        If CUDA is requested explicitly and this PyTorch build cannot use CUDA.
    ValueError
        If the specification is unknown or names a CUDA device that does not exist.

    """
    from ..lazy import get_torch

    torch = get_torch()
    if not isinstance(spec, str):
        raise TypeError(f"Device specification must be a string, got {type(spec).__name__}.")
    text = spec.strip().lower()
    if text in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    if text == "cpu":
        return torch.device("cpu")
    if text == "cuda" or text.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"The device {spec!r} was requested, but CUDA is not available in this PyTorch "
                "build (torch.cuda.is_available() is False). Re-run with --device cpu."
            )
        try:
            device = torch.device(text)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid device specification {spec!r}: {exc}") from exc
        index = device.index if device.index is not None else torch.cuda.current_device()
        available = torch.cuda.device_count()
        if index < 0 or index >= available:
            raise ValueError(
                f"The device {spec!r} does not exist: this machine exposes {available} CUDA "
                f"device(s), numbered 0-{available - 1}."
            )
        return torch.device("cuda", index)
    raise ValueError(f"Unsupported device specification {spec!r}: use 'auto', 'cpu' or 'cuda:N'.")


class TorchRuntime:
    """
    One loaded model, executed as an FP32 CPU-in/CPU-out inference callable.

    The runtime owns exactly one model instance for its whole lifetime. It never runs under
    `torch.autocast`, always uses `torch.inference_mode()` with gradients disabled, and keeps
    TF32 disabled through `configure_torch_runtime`.
    """

    _model: Any

    def __init__(
        self,
        model: torch.nn.Module,
        info: ModelInfo,
        *,
        config: TorchRuntimeConfig | None = None,
        device: torch.device | str | None = None,
        load_seconds: float | None = None,
    ) -> None:
        """
        Prepare a runtime around an already loaded model.

        Parameters
        ----------
        model : torch.nn.Module
            The model returned by `load_checkpoint`.
        info : ModelInfo
            Its normalized contract.
        config : TorchRuntimeConfig | None
            Runtime settings; the defaults are the validated fast recipe.
        device : torch.device | str | None
            An already resolved device; when omitted, `config.device` is resolved instead.
        load_seconds : float | None
            How long loading the checkpoint took, recorded for the CLI's timing report.

        Raises
        ------
        ValueError
            If the preset is unknown, has no adapter for this architecture, or if compilation was
            requested for a combination the measured recipe does not cover.

        """
        from ..lazy import get_torch

        torch = get_torch()
        settings = config or TorchRuntimeConfig()
        if settings.preset not in {"fast", "original"}:
            raise ValueError(f"Unknown preset {settings.preset!r}: use 'fast' or 'original'.")
        configure_torch_runtime()
        resolved = torch.device(device) if device is not None else resolve_device(settings.device)
        self.info = info
        self.preset: TorchPreset = settings.preset
        self.device = resolved
        self.dtype: torch.dtype = torch.float32
        self.load_seconds = load_seconds
        self.prepare_seconds: float | None = None
        self.compile_seconds: float | None = None
        self.warmup_seconds: float | None = None
        self._closed = False
        if settings.enable_compile:
            # Validate the requested recipe before building the model: an unsupported
            # combination must not cost a load or a rewrite first.
            self._validate_compile(settings)
        prepared_start = time.perf_counter()
        prepared = build_fast_model(model, info) if self.preset == "fast" else model
        prepared = prepared.eval().requires_grad_(False).to(device=resolved)
        self._model = prepared
        self.channels_last = bool(settings.use_channels_last) and resolved.type == "cuda" and self.preset == "fast"
        self._memory_format = torch.channels_last if self.channels_last else torch.contiguous_format
        if self.channels_last:
            self._model = self._model.to(memory_format=torch.channels_last)
        self.prepare_seconds = time.perf_counter() - prepared_start
        self.compiled = False
        if settings.enable_compile:
            compile_start = time.perf_counter()
            try:
                self._model = torch.compile(self._model, mode=_COMPILE_MODE, dynamic=False)
                self.compiled = True
                self._warm(settings.tiling.tile_size)
            except Exception as exc:
                # Inductor needs a working Triton and a C toolchain; without them the failure only
                # surfaces on the first tile. Release the model and report the compiler error.
                self.close()
                raise RuntimeError(
                    f"torch.compile failed while compiling and warming the first tile: "
                    f"{type(exc).__name__}: {exc}. torch.compile with mode={_COMPILE_MODE!r} needs "
                    "a working Triton installation and C toolchain on this platform; fix that, or "
                    "re-run without --compile to use the validated eager recipe."
                ) from exc
            self.compile_seconds = time.perf_counter() - compile_start
        elif resolved.type == "cuda":
            self._warm(settings.tiling.tile_size)

    def _validate_compile(self, settings: TorchRuntimeConfig) -> None:
        """Reject compile requests outside the measured fast CUDA recipe."""
        if settings.preset != "fast":
            raise ValueError(
                "torch.compile is only wired up for the `fast` preset; drop --compile or run with --preset fast."
            )
        if self.device.type != "cuda":
            raise ValueError(
                f"torch.compile with mode={_COMPILE_MODE!r} needs a CUDA device, but {self.device} was "
                "selected; re-run with --device cuda:0 or drop --compile."
            )

    def _warm(self, tile_size: int) -> None:
        """Run one fixed-shape tile so compiled graphs, cuDNN heuristics and buffers are ready."""
        from ..lazy import get_torch

        torch = get_torch()
        size = max(1, int(tile_size))
        sample = torch.zeros(1, self.info.in_channels, size, size, dtype=torch.float32)
        started = time.perf_counter()
        self.predict(sample)
        self.warmup_seconds = time.perf_counter() - started

    def _to_host(self, output: torch.Tensor) -> torch.Tensor:
        """Move a prediction to an ordinary FP32 CPU tensor outside `inference_mode`."""
        from ..lazy import get_torch

        torch = get_torch()
        result = output.detach().to(device="cpu", dtype=torch.float32)
        if result.is_inference():
            # A tensor created inside inference_mode stays an inference tensor; in-place updates on
            # it are rejected outside the context and the tiler accumulates into its buffers. The
            # clone runs in normal mode and hands callers an ordinary tensor.
            result = result.clone()
        return result

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Run one FP32 CPU tile through the model.

        Parameters
        ----------
        tensor : Tensor
            One FP32 CPU NCHW image tile with the model's external input channels.

        Returns
        -------
        Tensor
            One FP32 CPU NCHW result whose spatial dimensions are scaled by `info.scale`. The
            returned tensor is an ordinary tensor, safe to accumulate into.

        Raises
        ------
        RuntimeError
            If the runtime was closed, or the model returned an unexpected shape.
        TypeError
            If the argument is not a torch tensor.
        ValueError
            If the tensor is not an FP32 CPU single-image tile in the expected layout.

        """
        from ..lazy import get_torch

        torch = get_torch()
        if self._closed:
            raise RuntimeError("This TorchRuntime has been closed; create a new one with create_runtime().")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"TorchRuntime.predict expects a torch.Tensor, got {type(tensor).__name__}.")
        if tensor.device.type != "cpu":
            raise ValueError(f"TorchRuntime.predict expects a CPU tensor, got one on {tensor.device}.")
        if tensor.dtype != torch.float32:
            raise ValueError(f"TorchRuntime.predict expects FP32, got {tensor.dtype}.")
        if tensor.ndim != 4:
            raise ValueError(f"TorchRuntime.predict expects an NCHW tile, got shape {tuple(tensor.shape)}.")
        if tensor.shape[0] != 1:
            raise ValueError(
                f"TorchRuntime.predict expects a single image (batch size 1), got batch {tensor.shape[0]}."
            )
        if tensor.shape[1] != self.info.in_channels:
            raise ValueError(f"TorchRuntime.predict expects {self.info.in_channels} channels, got {tensor.shape[1]}.")
        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
        if height < 1 or width < 1:
            raise ValueError(f"TorchRuntime.predict expects a non-empty tile, got {height}x{width}.")
        staged = tensor.to(device=self.device, dtype=torch.float32, memory_format=self._memory_format)
        with torch.inference_mode():
            output = self._model(staged)
        result = self._to_host(output)
        expected = (1, self.info.out_channels, height * self.info.scale, width * self.info.scale)
        if tuple(result.shape) != expected:
            raise RuntimeError(
                f"The model returned {tuple(result.shape)} for the {tuple(tensor.shape)} tile; expected "
                f"{expected} for a {self.info.scale}x model."
            )
        return result

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Alias for `predict`, so the tiler can take the runtime as a plain callable."""
        return self.predict(tensor)

    def close(self) -> None:
        """Release the model and the cached CUDA memory. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._model = None
        gc.collect()
        from ..lazy import get_torch

        torch = get_torch()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> TorchRuntime:
        """Enter the runtime as a context manager."""
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Release the runtime when leaving the context manager."""
        self.close()

    def __repr__(self) -> str:
        """Describe the runtime without materializing the model."""
        state = "closed" if self._closed else f"{self.info.architecture} {self.info.scale}x {RUNTIME_DTYPE.value}"
        return (
            f"TorchRuntime({state}, device={self.device}, preset={self.preset!r}, "
            f"channels_last={self.channels_last}, compiled={self.compiled})"
        )


def create_runtime(model_path: str | Path, config: TorchRuntimeConfig | None = None) -> TorchRuntime:
    """
    Load a checkpoint and prepare one runtime for it.

    Parameters
    ----------
    model_path : str | Path
        The `.safetensors` checkpoint to run.
    config : TorchRuntimeConfig | None
        Device, preset, layout, compilation and warm-up tile settings.

    Returns
    -------
    TorchRuntime
        A runtime owning one model instance, ready for `predict`.

    Raises
    ------
    ValueError
        If the checkpoint cannot be described, the `fast` preset has no adapter for its
        architecture or structure, or the requested execution combination is unsupported.

    """
    settings = config or TorchRuntimeConfig()
    configure_torch_runtime()
    started = time.perf_counter()
    model, info = load_checkpoint(model_path)
    load_seconds = time.perf_counter() - started
    return TorchRuntime(model, info, config=settings, load_seconds=load_seconds)

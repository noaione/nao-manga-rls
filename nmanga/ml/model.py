"""
MIT License

Copyright (c) 2022-present noaione
Copyright (c) 2025-present anon

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

# resselt checkpoint loading and `ModelInfo` normalization.

from __future__ import annotations

import hashlib
import operator
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import MLDataType, ModelInfo
from .opt_arch import adapter_for

if TYPE_CHECKING:
    from torch import nn

__all__ = ("CHECKPOINT_SUFFIX", "LOADER_NAME", "load_checkpoint", "read_model_info")

CHECKPOINT_SUFFIX = ".safetensors"
"""The checkpoint format `nmanga torch infer` accepts; resselt loads it natively."""

LOADER_NAME = "resselt"
"""The library that detects, constructs and loads the architecture."""

_FLOAT_DTYPES = {
    "F32": MLDataType.FP32.value,
    "F16": MLDataType.FP16.value,
    "BF16": MLDataType.BF16.value,
    "F64": "fp64",
}
"""safetensors dtype tag -> the short name recorded in `ModelInfo.stored_dtypes`."""

_SHA256_CHUNK = 1024 * 1024


def _loader_version() -> str | None:
    """Return the installed resselt version, or ``None`` when it is not importable metadata."""
    try:
        return importlib_metadata.version(LOADER_NAME)
    except importlib_metadata.PackageNotFoundError:
        return None


def _validate_checkpoint_path(model_path: str | Path) -> Path:
    """Resolve a checkpoint path and reject formats this backend cannot inspect."""
    path = Path(model_path).expanduser()
    if not path.exists():
        raise ValueError(f"Checkpoint not found: {path}")
    if not path.is_file():
        raise ValueError(f"Checkpoint is not a file: {path}")
    if path.suffix.lower() != CHECKPOINT_SUFFIX:
        raise ValueError(
            f"Unsupported checkpoint format {path.suffix or '(none)'!r}: {path.name} is not a "
            f"{CHECKPOINT_SUFFIX} file. PyTorch inference loads resselt checkpoints; an ONNX or "
            f"TensorRT model needs the existing `upscale-trt` command instead."
        )
    return path


def _read_header(path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """
    Read every tensor name in a safetensors checkpoint with its stored dtype and shape.

    Only the header is read; no tensor data and no model graph is created.
    """
    from safetensors import safe_open

    header: dict[str, tuple[str, tuple[int, ...]]] = {}
    with safe_open(str(path), framework="numpy") as handle:
        for key in handle.keys():
            slice_ = handle.get_slice(key)
            header[key] = (str(slice_.get_dtype()), tuple(int(size) for size in slice_.get_shape()))
    return header


def _stored_float_dtypes(header: dict[str, tuple[str, tuple[int, ...]]]) -> tuple[str, ...]:
    """
    Collect the dtypes actually stored in the checkpoint's floating-point tensors.

    The filenames of the supplied checkpoints claim ``fp16``/``bf16`` while their tensors store
    FP32, so the reported precision comes from the tensors and never from the file name or the
    safetensors metadata comment.
    """
    found: set[str] = set()
    for dtype, _shape in header.values():
        if dtype == "BF16" or dtype.startswith("F"):
            found.add(_FLOAT_DTYPES.get(dtype, dtype.lower()))
    return tuple(sorted(found))


def _sha256(path: Path) -> str:
    """Hash the checkpoint contents so a report can name the exact weights."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_SHA256_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_channels(value: object, label: str, path: Path) -> int:
    """Validate one external channel count reported by resselt."""
    if isinstance(value, bool):
        raise ValueError(f"resselt reported a boolean {label} ({value!r}) for {path.name}.")
    try:
        channels = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError(
            f"resselt reported a non-integer {label} ({value!r}) for {path.name}; "
            "the normalized tensor contract needs a positive integer channel count."
        ) from None
    if channels <= 0:
        raise ValueError(f"resselt reported {label}={channels} for {path.name}; channel counts must be positive.")
    return channels


def _normalize_scale(value: object, path: Path) -> int:
    """
    Validate the effective upscale factor, accepting resselt's ``int`` or sequence type.

    resselt's metadata type allows a sequence of per-axis factors. Only an isotropic integer
    scale fits the NCHW tensor contract, and a sequence is never cast to ``int`` blindly.
    """
    scale: object = value
    if not isinstance(value, (str, bytes)):
        try:
            scale = operator.index(value)  # type: ignore[arg-type]
        except TypeError:
            scale = value
    if isinstance(scale, int) and not isinstance(scale, bool):
        if scale < 1:
            raise ValueError(f"resselt reported upscale={scale} for {path.name}; the scale must be at least 1.")
        return scale
    try:
        factors = tuple(operator.index(item) for item in scale)  # type: ignore[union-attr]
    except TypeError:
        raise ValueError(
            f"resselt reported upscale={value!r} for {path.name}; expected a positive integer "
            "or a sequence of positive integers."
        ) from None
    if not factors:
        raise ValueError(f"resselt reported an empty upscale sequence for {path.name}.")
    if len(set(factors)) != 1:
        raise ValueError(
            f"resselt reported the anisotropic upscale {factors} for {path.name}; this backend "
            "only supports a single isotropic integer scale."
        )
    if factors[0] < 1:
        raise ValueError(f"resselt reported upscale={factors[0]} for {path.name}; the scale must be at least 1.")
    return factors[0]


def _normalize_metadata(
    model: nn.Module,
    *,
    checkpoint_sha256: str | None = None,
    stored_dtypes: tuple[str, ...] = (),
    loader_version: str | None = None,
    path: str | Path | None = None,
) -> ModelInfo:
    """Build the normalized contract from resselt's `parameters_info` and the constructed model."""
    from resselt.factory.arch import ModelMetadata

    checkpoint = Path(path) if path is not None else Path("<memory>")
    metadata = getattr(model, "parameters_info", None)
    if not isinstance(metadata, ModelMetadata):
        raise ValueError(
            f"resselt did not attach a parameters_info metadata record to the "
            f"{type(model).__name__} model for {checkpoint.name}; the checkpoint cannot be described "
            "by the normalized tensor contract."
        )
    architecture = metadata.name
    if not isinstance(architecture, str) or not architecture.strip():
        raise ValueError(f"resselt reported the architecture name {architecture!r} for {checkpoint.name}.")
    in_channels = _normalize_channels(metadata.in_channels, "in_channels", checkpoint)
    out_channels = _normalize_channels(metadata.out_channels, "out_channels", checkpoint)
    scale = _normalize_scale(metadata.upscale, checkpoint)
    adapter = adapter_for(architecture)
    unshuffle = adapter.unshuffle_of(model, checkpoint) if adapter is not None else 1
    info = ModelInfo(
        architecture=architecture.strip(),
        scale=scale,
        in_channels=in_channels,
        out_channels=out_channels,
        unshuffle=unshuffle,
        stored_dtypes=stored_dtypes,
        checkpoint_sha256=checkpoint_sha256,
        loader=LOADER_NAME,
        loader_version=loader_version,
    )
    if adapter is not None:
        adapter.cross_check(model, info, checkpoint)
    return info


def _detect_architecture(header: dict[str, tuple[str, tuple[int, ...]]]) -> str | None:
    """Ask resselt which architecture claims these tensors, without constructing the model."""
    from resselt.archs import internal_registry

    # resselt's detectors only test key membership; the values are never inspected.
    keys = dict.fromkeys(header)
    for architecture in internal_registry:
        try:
            detected = architecture.detect(keys)
        except Exception:
            detected = False
        if detected:
            return architecture.id
    return None


def read_model_info(model_path: str | Path) -> ModelInfo:
    """
    Describe a checkpoint without running it.

    The stored dtypes, tensor names and shapes come from the safetensors header, and an
    architecture with a header adapter is described from that header as well, so the model is not
    constructed for it. Any other architecture is described by loading it through resselt.

    Parameters
    ----------
    model_path : str | Path
        The checkpoint to inspect.

    Returns
    -------
    ModelInfo
        The normalized contract for the checkpoint.

    Raises
    ------
    ValueError
        If the checkpoint is missing, is not a `.safetensors` file, or resselt cannot describe it.
    """
    path = _validate_checkpoint_path(model_path)
    header = _read_header(path)
    stored_dtypes = _stored_float_dtypes(header)
    checkpoint_sha256 = _sha256(path)
    loader_version = _loader_version()
    architecture = _detect_architecture(header)
    if architecture is not None:
        adapter = adapter_for(architecture)
        if adapter is not None:
            info = adapter.info_from_header(
                path,
                header,
                architecture=architecture,
                stored_dtypes=stored_dtypes,
                checkpoint_sha256=checkpoint_sha256,
                loader_version=loader_version,
            )
            if info is not None:
                return info
    _model, info = load_checkpoint(path)
    return info


def load_checkpoint(model_path: str | Path) -> tuple[nn.Module, ModelInfo]:
    """
    Load a checkpoint through resselt as an untouched FP32 evaluation model.

    The loader runs on the CPU and leaves architecture ownership with resselt: it detects the
    architecture, constructs it and loads the state dict. The returned model is promoted to FP32,
    switched to evaluation mode and has gradients disabled, which is the reference an optimized
    copy is compared against.

    Parameters
    ----------
    model_path : str | Path
        The `.safetensors` checkpoint to load.

    Returns
    -------
    tuple[nn.Module, ModelInfo]
        The resselt model (a `torch.nn.Module`) and its normalized contract.

    Raises
    ------
    ImportError
        If torch or resselt is not installed.
    ValueError
        If the checkpoint is missing, is not a safetensors file, or its metadata cannot be
        normalized into the tensor contract.
    """
    path = _validate_checkpoint_path(model_path)
    from ..lazy import get_torch

    get_torch()
    try:
        from resselt import load_from_file
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise ImportError(
            "resselt is required to load PyTorch checkpoints. Install the PyTorch extras with "
            "`pip install nmanga[denoiser]`."
        ) from exc

    header = _read_header(path)
    stored_dtypes = _stored_float_dtypes(header)
    checkpoint_sha256 = _sha256(path)
    loader_version = _loader_version()
    try:
        model = load_from_file(str(path))
    except Exception as exc:
        raise ValueError(
            f"resselt {loader_version or ''} could not load {path.name} as a supported checkpoint: {exc}"
        ) from exc
    # FP32 is the precision contract: the stored precision and the file name never select a
    # half-precision model, and the promoted weights are the reference for the fast preset.
    model = model.float().eval().requires_grad_(False)
    info = _normalize_metadata(
        model,
        checkpoint_sha256=checkpoint_sha256,
        stored_dtypes=stored_dtypes,
        loader_version=loader_version,
        path=path,
    )
    return model, info

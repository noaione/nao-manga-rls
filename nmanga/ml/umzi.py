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

# Umzi native runtime: package inspection, session options and checkpoint export.
#
# `umzi_native` ships CUDA wheels only, so it stays an optional extra and nothing here imports
# it at module level.

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol, cast

__all__ = (
    "UMZI_CHECKPOINT_SUFFIXES",
    "UMZI_FAMILIES",
    "UMZI_PRECISIONS",
    "UmziPackageInfo",
    "UmziPrecision",
    "UmziSession",
    "build_kernel_options",
    "create_umzi_session",
    "describe_package",
    "export_ummodel",
    "get_umzi_native",
    "parse_kernel_options",
    "read_ummodel",
    "recommended_kernel_options",
    "resolve_precision",
)

UmziPrecision = Literal["mixed", "fp16", "fp32"]
"""The precision modes `umzi_native.SessionOptions` accepts."""

UMZI_PRECISIONS: tuple[UmziPrecision, ...] = ("mixed", "fp16", "fp32")
"""Every precision mode, in the order help output lists them."""

UMZI_FAMILIES: tuple[str, ...] = ("gaterv3", "gaterv2", "moesr", "gfisrv2")
"""The kernel families a `.ummodel` package can declare."""

UMZI_CHECKPOINT_SUFFIXES: tuple[str, ...] = (".safetensors", ".pth", ".pt", ".ckpt")
"""Every checkpoint format the exporter reads."""

_RECOMMENDED_KERNEL_OPTIONS: dict[str, dict[str, int]] = {
    # Every family spells its switches out instead of relying on the runtime defaults: the report
    # line then shows what the session really runs, and an umzi upgrade cannot change it silently.
    # GateRV3 fuses the gate/pool and SiLU kernels and times the extra cuBLASLt projections; the
    # other families only pin the fused kernel that umzi already defaults on.
    "gaterv3": {"fuse_gate_pool": 1, "tune_projections": 1, "fuse_silu": 1},
    "gaterv2": {"tune_plans": 1},
    "moesr": {"fused_gate": 1},
    "gfisrv2": {"fused_conv": 1},
}

_FP32_FAMILIES = frozenset({"gfisrv2"})
"""The only family with an FP32 path; every other one runs mixed precision or FP16."""


class UmziSession(Protocol):
    """The part of an open `umzi_native.InferenceSession` the shared tiler uses."""

    scale_factor: int
    context: int

    def get_inputs(self) -> list[Any]: ...

    def get_outputs(self) -> list[Any]: ...

    def run(self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any]) -> list[Any]: ...

    def close(self) -> None: ...


@functools.lru_cache(maxsize=1)
def get_umzi_native() -> ModuleType:
    """
    Import the `umzi_native` runtime once.

    :return: The imported module
    :raises ImportError: If the runtime is not installed, with the install command in the message
    """
    try:
        import umzi_native  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "umzi-native is required for the `nmanga umzi` commands. Install it with "
            "`pip install nmanga[denoiser]`; it only publishes CUDA wheels for Linux x86_64 and "
            "Windows AMD64."
        ) from exc
    return umzi_native


@dataclass(frozen=True)
class UmziPackageInfo:
    """What a `.ummodel` package declares, normalized away from the runtime's own dataclasses."""

    path: Path
    """The package that was read."""

    family: str | None
    """The kernel family, `None` when the package declares one this build does not know."""

    scale: int
    """The upscale factor the package declares, `1` for a denoiser."""

    in_channels: int
    """The number of channels the package reads."""

    out_channels: int
    """The number of channels the package returns."""

    alignment: int | None
    """The tile alignment the package pads to, `None` for a family that does not declare one."""

    size: int
    """The package size in bytes."""

    digest: str
    """The SHA-256 digest recorded in the package header."""

    checkpoint: str | None
    """The checkpoint the package was exported from, when the payload records it."""

    problems: tuple[str, ...] = ()
    """Why the installed runtime refuses the package; empty when it can open it."""

    @property
    def supported(self) -> bool:
        """Whether the installed runtime can open this package."""
        return not self.problems


def _normalize_package(info: Any) -> UmziPackageInfo:
    """Flatten a `umzi_native.ModelInfo` and its family payload into :class:`UmziPackageInfo`."""
    payload = info.payload
    alignment = getattr(payload, "alignment", None)
    return UmziPackageInfo(
        path=Path(info.path),
        family=info.family_name,
        scale=int(getattr(payload, "scale", 0) or 0),
        in_channels=int(getattr(payload, "input_channels", 0) or 0),
        out_channels=int(getattr(payload, "output_channels", 0) or 0),
        alignment=None if alignment is None else int(alignment),
        size=int(info.size),
        digest=str(info.digest),
        checkpoint=getattr(payload, "checkpoint", None),
        problems=tuple(info.problems),
    )


def read_ummodel(model_path: str | Path) -> UmziPackageInfo:
    """
    Read a `.ummodel` header without initializing CUDA.

    :param model_path: The package to inspect
    :return: The normalized description of the package
    :raises ImportError: If `umzi_native` is not installed
    :raises ValueError: If the file is not a well-formed Umzi package
    """
    umzi = get_umzi_native()
    return _normalize_package(umzi.peek_model(Path(model_path).expanduser()))


def _format_size(size: int) -> str:
    """Render a byte count for the report lines."""
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if size >= scale:
            return f"{size / scale:.2f} {unit}"
    return f"{size} bytes"


def describe_package(info: UmziPackageInfo) -> list[str]:
    """
    Render one package as report lines.

    :param info: The package to describe
    :return: One line per fact, with the runtime's problems last
    """
    family = info.family or "unknown family"
    summary = f"{info.path}: {family}"
    if info.scale:
        summary += f", x{info.scale} scale"
    summary += ", supported" if info.supported else ", unsupported"
    lines = [summary, f"  {_format_size(info.size)}, sha256 {info.digest}"]
    if info.in_channels and info.out_channels:
        channels = f"  {info.in_channels} -> {info.out_channels} channels"
        if info.alignment is not None:
            channels += f", {info.alignment}-pixel alignment"
        lines.append(channels)
    if info.checkpoint:
        lines.append(f"  checkpoint: {info.checkpoint}")
    lines.extend(f"  problem: {problem}" for problem in info.problems)
    return lines


def recommended_kernel_options(family: str | None) -> dict[str, int]:
    """
    The kernel switches the umzi quickstart recommends for a family.

    :param family: The `family_name` a package declares
    :return: A fresh mapping; empty for a family that needs none, or is unknown
    """
    if family is None:
        return {}
    return dict(_RECOMMENDED_KERNEL_OPTIONS.get(family, {}))


def build_kernel_options(family: str | None, overrides: Mapping[str, int] | None = None) -> dict[str, int]:
    """
    Start from the family's recommended switches and let explicit ones win.

    Unknown names and out-of-range values are the runtime's call: it rejects them when the
    session opens, so the table here cannot drift away from the native one.

    :param family: The `family_name` a package declares
    :param overrides: The switches the user asked for
    :return: The kernel options to open the session with
    """
    options = recommended_kernel_options(family)
    if overrides:
        options.update(overrides)
    return options


def parse_kernel_options(values: Iterable[str]) -> dict[str, int]:
    """
    Parse `name=value` kernel switches from the command line.

    :param values: The raw `--kernel-option` values
    :return: The switches, keyed by name
    :raises ValueError: If a value is not `name=<integer>`
    """
    options: dict[str, int] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        name = name.strip()
        if not separator or not name or not raw.strip():
            raise ValueError(f"Kernel option {value!r} is not `name=value`.")
        try:
            options[name] = int(raw.strip())
        except ValueError:
            raise ValueError(f"Kernel option {name!r} needs an integer value, got {raw.strip()!r}.") from None
    return options


def resolve_precision(family: str | None, precision: str) -> UmziPrecision:
    """
    Check a requested precision against the family that has to run it.

    :param family: The `family_name` a package declares
    :param precision: The requested precision, case insensitive
    :return: The normalized precision
    :raises ValueError: If the precision is unknown, or only a GFISRv2 model has that path
    """
    normalized = precision.strip().lower()
    if normalized not in UMZI_PRECISIONS:
        raise ValueError(f"Unknown precision {precision!r}, expected one of: {', '.join(UMZI_PRECISIONS)}.")
    if normalized == "fp32" and family not in _FP32_FAMILIES:
        raise ValueError(
            f"The fp32 path is only available for GFISRv2 models, but this package is {family or 'unknown'}. "
            "Use `--precision mixed` or `--precision fp16`."
        )
    return normalized


def _load_torch_cuda_libraries() -> None:
    """Import torch first, so the cuBLASLt and cuDNN its CUDA build loaded are in the process."""
    try:
        from ..lazy import get_torch
    except ImportError:
        return
    try:
        get_torch()
    except ImportError:
        # Without torch the native loader has to find the CUDA libraries on its own.
        pass


def create_umzi_session(
    model_path: str | Path,
    *,
    family: str | None,
    device_id: int = 0,
    precision: str = "fp16",
    cache_dir: str | Path | None = None,
    enable_cuda_graph: bool = True,
    kernel_options: Mapping[str, int] | None = None,
) -> UmziSession:
    """
    Open a native session for a package, with the kernel switches its family recommends.

    :param model_path: The `.ummodel` package to open
    :param family: The `family_name` :func:`read_ummodel` reported
    :param device_id: The CUDA device index to run on
    :param precision: `mixed`, `fp16` or `fp32`, as :func:`resolve_precision` accepts it
    :param cache_dir: Where to keep the tuned plan files, `None` disables the disk cache
    :param enable_cuda_graph: Capture the forward pass into a CUDA graph
    :param kernel_options: Explicit switches that override the recommended ones
    :return: The open `umzi_native.InferenceSession`
    :raises ImportError: If `umzi_native` is not installed
    :raises ValueError: If the precision cannot be used, or the runtime rejects an option
    """
    _load_torch_cuda_libraries()
    umzi = get_umzi_native()
    options = umzi.SessionOptions(
        precision=resolve_precision(family, precision),
        device_id=device_id,
        cache_dir=None if cache_dir is None else Path(cache_dir),
        enable_cuda_graph=enable_cuda_graph,
        kernel_options=build_kernel_options(family, kernel_options),
    )
    return cast("UmziSession", umzi.InferenceSession(Path(model_path).expanduser(), options))


def export_ummodel(
    checkpoint: str | Path,
    destination: str | Path,
    *,
    kernel: str | None = None,
) -> UmziPackageInfo:
    """
    Pack a checkpoint into a `.ummodel` package.

    :param checkpoint: The `.safetensors`, `.pth`, `.pt` or `.ckpt` state dictionary
    :param destination: Where to write the package, the name must end in `.ummodel`
    :param kernel: When given, fail unless the checkpoint detects as that family
    :return: The normalized description of the written package
    :raises ImportError: If the exporter extra is not installed
    :raises ValueError: If the checkpoint has no native packer, or detects as another family
    """
    # Import the runtime first, so an unsupported platform reports that instead of a missing extra.
    get_umzi_native()
    try:
        from umzi_native.exporter import export_model  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "Exporting a checkpoint needs the umzi-native exporter extra, which adds resselt and "
            'torch: `pip install "umzi-native[exporter]"`.'
        ) from exc
    info = export_model(Path(checkpoint).expanduser(), Path(destination).expanduser(), kernel=kernel)
    return _normalize_package(info)

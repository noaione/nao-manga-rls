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
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol, cast

__all__ = (
    "UMZI_CHECKPOINT_SUFFIXES",
    "UMZI_FAMILIES",
    "UMZI_PRECISIONS",
    "FiGSRDetails",
    "GFISRv2Details",
    "GateRV2Details",
    "GateRV3Details",
    "MoESRDetails",
    "SMoSRDetails",
    "UmziPackageDetails",
    "UmziPackageInfo",
    "UmziPrecision",
    "UmziSection",
    "UmziSession",
    "UmziSessionPool",
    "build_kernel_options",
    "create_umzi_session",
    "describe_package",
    "export_ummodel",
    "get_umzi_native",
    "parse_kernel_options",
    "read_ummodel",
    "recommended_kernel_options",
    "resolve_precision",
    "run_umzi_tiled_image",
    "umzi_tiling_constraints",
)

UmziPrecision = Literal["mixed", "fp16", "fp32"]
"""The precision modes `umzi_native.SessionOptions` accepts."""

UMZI_PRECISIONS: tuple[UmziPrecision, ...] = ("mixed", "fp16", "fp32")
"""Every precision mode, in the order help output lists them."""

UMZI_FAMILIES: tuple[str, ...] = ("gaterv3", "gaterv2", "moesr", "gfisrv2", "smosr", "figsr")
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

_PRECISION_KERNEL_OPTIONS: dict[tuple[str, str], dict[str, int]] = {
    # SMoSR and FiGSR reject switches outside their precision scope at session open; SMoSR's FP16
    # path already gets its fastest plans from `fuse_silu` without re-timing unfused body convs.
    ("smosr", "fp16"): {"fuse_silu": 1},
    ("smosr", "mixed"): {"tune_convolutions": 1, "fuse_silu": 1},
    ("smosr", "fp32"): {},
    ("figsr", "fp16"): {"tune_convolutions": 1, "fuse_expand_gate": 1},
    ("figsr", "mixed"): {"tune_convolutions": 1, "fuse_row_gate": 1, "fuse_expand_gate": 1},
    ("figsr", "fp32"): {"tune_convolutions": 1, "fp32_nchw": 1},
}

_FP32_FAMILIES = frozenset({"gfisrv2", "smosr", "figsr"})
"""Every family with an FP32 path; the others only run mixed precision or FP16."""

_MIN_TILE_SIZES: dict[str, int] = {
    "moesr": 2,
    "smosr": 3,
    "figsr": 6,
}
"""Minimum spatial input dimension required by families with internal reflection padding."""


class UmziSession(Protocol):
    """The part of an open `umzi_native.InferenceSession` the shared tiler uses."""

    scale_factor: int
    context: int

    def get_inputs(self) -> list[Any]: ...

    def get_outputs(self) -> list[Any]: ...

    def run(self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any]) -> list[Any]: ...

    def run_tiled(
        self,
        image: Any,
        *,
        tile_size: int,
        batch_size: int,
        overlap: int = 0,
        tiling: Literal["fixed", "variable"] = "fixed",
        background: Literal["white", "black"] = "white",
    ) -> Any: ...

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
class UmziSection:
    """One entry of the package's section table."""

    id: int
    """The section kind, `1` architecture, `2` schedule, `3` weights and `4` checkpoint."""

    name: str
    """Its name as the runtime spells it."""

    length: int
    """Its size in bytes."""


@dataclass(frozen=True)
class GateRV3Details:
    """The GateRV3 payload: a fixed 1x topology with a declared operation schedule."""

    tensors: int
    operations: int
    weight_bytes: int
    op_kinds: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class GateRV2Details:
    """The GateRV2 payload: encoder and decoder depths around latent attention blocks."""

    encoder_depths: tuple[int, ...]
    decoder_depths: tuple[int, ...]
    latent_blocks: int
    attention_eps: float
    parameters: int
    weight_bytes: int


@dataclass(frozen=True)
class MoESRDetails:
    """The MoESR payload: groups of full- and reduced-resolution blocks."""

    groups: int
    full_blocks: int
    reduced_blocks: int
    full_hidden: int
    reduced_hidden: int
    weight_bytes: int


@dataclass(frozen=True)
class GFISRv2Details:
    """The GFISRv2 payload: a 4x Fourier block stack."""

    blocks: int
    hidden_channels: int
    fourier_channels: int
    weight_bytes: int


@dataclass(frozen=True)
class SMoSRDetails:
    """The SMoSR payload: a 2x direct pixel-shuffle stack with three block stages."""

    reflection_padding: int
    first_blocks: int
    middle_blocks: int
    end_blocks: int
    weight_bytes: int


@dataclass(frozen=True)
class FiGSRDetails:
    """The FiGSR payload: a 2x Fourier and dense spatial block stack."""

    reflection_padding: int
    blocks_per_group: int
    hidden_channels: int
    fourier_channels: int
    square_kernel: int
    band_kernel: int
    weight_bytes: int


UmziPackageDetails = GateRV3Details | GateRV2Details | MoESRDetails | GFISRv2Details | SMoSRDetails | FiGSRDetails
"""The family-specific half of a package header."""


@dataclass(frozen=True)
class UmziPackageInfo:
    """What a `.ummodel` package declares, normalized away from the runtime's own dataclasses."""

    path: Path
    """The package that was read."""

    version: int
    """The container format version."""

    family_id: int
    """The numeric kernel family, reported when this build does not know its name."""

    family: str | None
    """The kernel family, `None` when the package declares one this build does not know."""

    schema: int
    """The payload schema version."""

    size: int
    """The package size in bytes."""

    digest: str
    """The SHA-256 digest recorded in the package header."""

    sections: tuple[UmziSection, ...] = ()
    """The section table, in file order."""

    scale: int | None = None
    """The upscale factor the package declares, `1` for a denoiser."""

    in_channels: int | None = None
    """The number of channels the package reads."""

    out_channels: int | None = None
    """The number of channels the package returns."""

    alignment: int | None = None
    """The tile alignment the package pads to, `None` when it does not declare one."""

    base_channels: int | None = None
    """The width of the first convolution."""

    checkpoint: str | None = None
    """The checkpoint the package was exported from, when the payload records it."""

    details: UmziPackageDetails | None = None
    """The family-specific payload, `None` when this build cannot read it."""

    problems: tuple[str, ...] = ()
    """Why the installed runtime refuses the package; empty when it can open it."""

    @property
    def supported(self) -> bool:
        """Whether the installed runtime can open this package."""
        return not self.problems


def _payload_details(family: str | None, payload: Any) -> UmziPackageDetails | None:
    """Read the family-specific half of a payload, `None` when the runtime could not parse it."""
    if payload is None:
        return None
    if family == "gaterv3":
        return GateRV3Details(
            tensors=int(payload.tensors),
            operations=int(payload.operations),
            weight_bytes=int(payload.weight_bytes),
            op_kinds=tuple((str(name), int(count)) for name, count in payload.op_kinds.items()),
        )
    if family == "gaterv2":
        return GateRV2Details(
            encoder_depths=tuple(int(depth) for depth in payload.encoder_depths),
            decoder_depths=tuple(int(depth) for depth in payload.decoder_depths),
            latent_blocks=int(payload.latent_blocks),
            attention_eps=float(payload.attention_eps),
            parameters=int(payload.parameters),
            weight_bytes=int(payload.weight_bytes),
        )
    if family == "moesr":
        return MoESRDetails(
            groups=int(payload.groups),
            full_blocks=int(payload.full_blocks),
            reduced_blocks=int(payload.reduced_blocks),
            full_hidden=int(payload.full_hidden),
            reduced_hidden=int(payload.reduced_hidden),
            weight_bytes=int(payload.weight_bytes),
        )
    if family == "gfisrv2":
        return GFISRv2Details(
            blocks=int(payload.blocks),
            hidden_channels=int(payload.hidden_channels),
            fourier_channels=int(payload.fourier_channels),
            weight_bytes=int(payload.weight_bytes),
        )
    if family == "smosr":
        return SMoSRDetails(
            reflection_padding=int(payload.reflection_padding),
            first_blocks=int(payload.first_blocks),
            middle_blocks=int(payload.middle_blocks),
            end_blocks=int(payload.end_blocks),
            weight_bytes=int(payload.weight_bytes),
        )
    if family == "figsr":
        return FiGSRDetails(
            reflection_padding=int(payload.reflection_padding),
            blocks_per_group=int(payload.blocks_per_group),
            hidden_channels=int(payload.hidden_channels),
            fourier_channels=int(payload.fourier_channels),
            square_kernel=int(payload.square_kernel),
            band_kernel=int(payload.band_kernel),
            weight_bytes=int(payload.weight_bytes),
        )
    return None


def _normalize_package(info: Any) -> UmziPackageInfo:
    """Flatten a `umzi_native.ModelInfo` and its family payload into :class:`UmziPackageInfo`."""
    payload = info.payload
    return UmziPackageInfo(
        path=Path(info.path),
        version=int(info.version),
        family_id=int(info.family),
        family=info.family_name,
        schema=int(info.schema),
        size=int(info.size),
        digest=str(info.digest),
        sections=tuple(
            UmziSection(id=int(section.id), name=str(section.name), length=int(section.length))
            for section in info.sections
        ),
        scale=getattr(payload, "scale", None),
        in_channels=getattr(payload, "input_channels", None),
        out_channels=getattr(payload, "output_channels", None),
        alignment=getattr(payload, "alignment", None),
        base_channels=getattr(payload, "base_channels", None),
        checkpoint=getattr(payload, "checkpoint", None),
        details=_payload_details(info.family_name, payload),
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
    """Render a byte count the way the umzi CLI does, with the exact byte count."""
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if size >= scale:
            return f"{size / scale:.2f} {unit} ({size:,} bytes)"
    return f"{size:,} bytes"


def describe_package(info: UmziPackageInfo) -> list[str]:
    """
    Render one package as report lines.

    :param info: The package to describe
    :return: One line per fact, with the runtime's problems last
    """
    family = info.family or f"kernel family {info.family_id}"
    state = "supported" if info.supported else "unsupported"
    lines = [
        f"{info.path}: {family} schema {info.schema}, {state}",
        f"  {_format_size(info.size)}, sha256 {info.digest}",
    ]
    lines.extend(f"  section {section.id} {section.name}: {_format_size(section.length)}" for section in info.sections)

    if info.in_channels is not None and info.out_channels is not None and info.scale is not None:
        geometry = f"  {info.in_channels} -> {info.out_channels} channels, x{info.scale} scale"
        if info.alignment is not None:
            geometry += f", {info.alignment}-pixel padding"
        if info.base_channels is not None:
            geometry += f", {info.base_channels} base channels"
        lines.append(geometry)

    details = info.details
    if isinstance(details, GateRV3Details):
        lines.append(
            f"  {details.tensors} tensors, {details.operations} operations, "
            f"{_format_size(details.weight_bytes)} of weights"
        )
        if details.op_kinds:
            lines.append("  operations: " + ", ".join(f"{name} {count}" for name, count in details.op_kinds))
    elif isinstance(details, GateRV2Details):
        lines.append(
            f"  encoder {details.encoder_depths}, decoder {details.decoder_depths}, "
            f"{details.latent_blocks} latent attention blocks (eps {details.attention_eps:g})"
        )
        lines.append(f"  {details.parameters} FP32 parameters, {_format_size(details.weight_bytes)} of weights")
    elif isinstance(details, MoESRDetails):
        lines.append(
            f"  {details.groups} groups of {details.full_blocks} full-resolution and "
            f"{details.reduced_blocks} reduced-resolution blocks"
        )
        lines.append(
            f"  full {details.full_hidden} and reduced {details.reduced_hidden} hidden channels, "
            f"{_format_size(details.weight_bytes)} of weights"
        )
    elif isinstance(details, GFISRv2Details):
        lines.append(
            f"  {details.blocks} blocks, {details.hidden_channels} hidden and "
            f"{details.fourier_channels} Fourier channels"
        )
        lines.append(f"  {_format_size(details.weight_bytes)} of weights")
    elif isinstance(details, SMoSRDetails):
        lines.append(
            f"  {details.first_blocks} + {details.middle_blocks} + {details.end_blocks} blocks, "
            f"{details.reflection_padding}-pixel reflection padding"
        )
        lines.append(f"  {_format_size(details.weight_bytes)} of weights")
    elif isinstance(details, FiGSRDetails):
        lines.append(
            f"  2 groups of {details.blocks_per_group} blocks, {details.hidden_channels} hidden and "
            f"{details.fourier_channels} Fourier channels"
        )
        lines.append(
            f"  {details.square_kernel}x{details.square_kernel} and 1x{details.band_kernel} spatial kernels, "
            f"{details.reflection_padding}-pixel reflection padding"
        )
        lines.append(f"  {_format_size(details.weight_bytes)} of weights")

    if info.checkpoint:
        lines.append(f"  checkpoint: {info.checkpoint}")
    lines.extend(f"  problem: {problem}" for problem in info.problems)
    return lines


def recommended_kernel_options(family: str | None, precision: str = "fp16") -> dict[str, int]:
    """
    The kernel switches the umzi quickstart recommends for a family and precision.

    :param family: The `family_name` a package declares
    :param precision: The session precision (`mixed`, `fp16`, or `fp32`)
    :return: A fresh mapping; empty for a family that needs none, or is unknown
    """
    if family is None:
        return {}
    normalized = precision.strip().lower()
    if (family, normalized) in _PRECISION_KERNEL_OPTIONS:
        return dict(_PRECISION_KERNEL_OPTIONS[family, normalized])
    return dict(_RECOMMENDED_KERNEL_OPTIONS.get(family, {}))


def build_kernel_options(
    family: str | None,
    overrides: Mapping[str, int] | None = None,
    *,
    precision: str = "fp16",
) -> dict[str, int]:
    """
    Start from the family's recommended switches and let explicit ones win.

    Unknown names and out-of-range values are the runtime's call: it rejects them when the
    session opens, so the table here cannot drift away from the native one.

    :param family: The `family_name` a package declares
    :param overrides: The switches the user asked for
    :param precision: The session precision (`mixed`, `fp16`, or `fp32`)
    :return: The kernel options to open the session with
    """
    options = recommended_kernel_options(family, precision)
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
    :raises ValueError: If the precision is unknown, or the family has no FP32 path
    """
    normalized = precision.strip().lower()
    if normalized not in UMZI_PRECISIONS:
        raise ValueError(f"Unknown precision {precision!r}, expected one of: {', '.join(UMZI_PRECISIONS)}.")
    if normalized == "fp32" and family not in _FP32_FAMILIES:
        raise ValueError(
            f"The fp32 path is only available for GFISRv2, SMoSR, and FiGSR models, "
            f"but this package is {family or 'unknown'}. Use `--precision mixed` or `--precision fp16`."
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
    normalized_precision = resolve_precision(family, precision)
    options = umzi.SessionOptions(
        precision=normalized_precision,
        device_id=device_id,
        cache_dir=None if cache_dir is None else Path(cache_dir),
        enable_cuda_graph=enable_cuda_graph,
        kernel_options=build_kernel_options(family, kernel_options, precision=normalized_precision),
    )
    return cast("UmziSession", umzi.InferenceSession(Path(model_path).expanduser(), options))


def umzi_tiling_constraints(package: UmziPackageInfo) -> tuple[int, int]:
    """
    Return the ``(alignment, min_tile_size)`` spatial constraints a package requires.

    :param package: The inspected `.ummodel` metadata
    :return: ``(alignment, min_tile_size)`` in input pixels
    """
    alignment = package.alignment if package.alignment and package.alignment > 0 else 1
    min_tile = _MIN_TILE_SIZES.get(package.family or "", 1)
    return alignment, min_tile


def run_umzi_tiled_image(
    image: Any,
    session: UmziSession,
    *,
    batch_size: int,
    tile_size: int,
    tile_overlap: int,
    tiling: Literal["fixed", "variable"],
    background: Literal["white", "black"],
    contrast_stretch: bool,
) -> Any:
    """Run native GPU tiling and retain the input image's alpha and output mode."""
    from PIL import Image

    from ..lazy import get_numpy

    np = get_numpy()
    has_alpha = "A" in image.getbands() or (image.mode == "P" and image.info.get("transparency") is not None)
    alpha = image.getchannel("A") if "A" in image.getbands() else None
    if has_alpha and alpha is None:
        with image.convert("RGBA") as rgba:
            alpha = rgba.getchannel("A")

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if contrast_stretch:
        low = int(rgb.min())
        high = int(rgb.max())
        if high > low:
            rgb = np.rint((rgb.astype(np.float32) - low) * (255.0 / (high - low))).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)
    output = session.run_tiled(
        rgb,
        tile_size=tile_size,
        batch_size=batch_size,
        overlap=tile_overlap,
        tiling=tiling,
        background=background,
    )
    result = Image.fromarray(output, mode="RGB")

    if alpha is not None:
        result = result.convert("L" if image.mode == "LA" else "RGB")
        result = result.convert("LA" if image.mode == "LA" else "RGBA")
        resized_alpha = alpha.resize(result.size, Image.Resampling.BICUBIC)
        result.putalpha(resized_alpha)
        resized_alpha.close()
        alpha.close()
    elif image.palette:
        palette_image = Image.new("P", (1, 1))
        palette_image.putpalette(image.palette)
        result = result.quantize(palette=palette_image, dither=Image.Dither.NONE)
    elif image.mode not in ("RGB", "RGBA"):
        result = result.convert(image.mode)
    return result


def _is_oom_error(exc: BaseException) -> bool:
    return isinstance(exc, MemoryError) or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


class UmziSessionPool:
    """
    Keep up to `max_sessions` prepared `InferenceSession` instances keyed by batch shape.

    Each native session holds one prepared ``(N, C, H, W)`` shape at a time, so variable edge
    tiling across a page would otherwise re-prepare and re-capture CUDA Graphs repeatedly.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        family: str | None,
        device_id: int = 0,
        precision: str = "fp16",
        cache_dir: str | Path | None = None,
        enable_cuda_graph: bool = True,
        kernel_options: Mapping[str, int] | None = None,
        max_sessions: int = 4,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1.")
        self._model_path = model_path
        self._family = family
        self._device_id = device_id
        self._precision = precision
        self._cache_dir = cache_dir
        self._enable_cuda_graph = enable_cuda_graph
        self._kernel_options = dict(kernel_options) if kernel_options is not None else None
        self._max_sessions = max_sessions
        self._closed = False

        primary = self._open_session()
        self.scale_factor: int = primary.scale_factor
        self.context: int = primary.context
        self._inputs = primary.get_inputs()
        self._outputs = primary.get_outputs()
        self._sessions: OrderedDict[tuple[int, int, int] | None, UmziSession] = OrderedDict([(None, primary)])

    def _open_session(self) -> UmziSession:
        return create_umzi_session(
            self._model_path,
            family=self._family,
            device_id=self._device_id,
            precision=self._precision,
            cache_dir=self._cache_dir,
            enable_cuda_graph=self._enable_cuda_graph,
            kernel_options=self._kernel_options,
        )

    @property
    def session_count(self) -> int:
        """Number of open native sessions currently held in the pool."""
        return len(self._sessions)

    def get_inputs(self) -> list[Any]:
        return list(self._inputs)

    def get_outputs(self) -> list[Any]:
        return list(self._outputs)

    def _evict_oldest(self, *, keep: UmziSession | None = None) -> bool:
        for key in list(self._sessions):
            candidate = self._sessions[key]
            if candidate is keep:
                continue
            self._sessions.pop(key)
            candidate.close()
            return True
        return False

    def run(self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any]) -> list[Any]:
        if self._closed:
            raise RuntimeError("The Umzi session pool is closed.")
        input_name = self._inputs[0].name if self._inputs else next(iter(input_feed))
        sample = input_feed[input_name]
        shape_key = (int(sample.shape[0]), int(sample.shape[2]), int(sample.shape[3]))

        if shape_key in self._sessions:
            session = self._sessions[shape_key]
            self._sessions.move_to_end(shape_key)
            return session.run(output_names, input_feed)

        if None in self._sessions:
            session = self._sessions.pop(None)
            self._sessions[shape_key] = session
            return session.run(output_names, input_feed)

        if len(self._sessions) < self._max_sessions:
            try:
                session = self._open_session()
            except Exception as exc:
                if not _is_oom_error(exc) or not self._evict_oldest():
                    raise
                # Opening an extra session hit VRAM limits; fall back to reusing the LRU slot.
                _, session = self._sessions.popitem(last=False)
            try:
                result = session.run(output_names, input_feed)
            except Exception as exc:
                if not _is_oom_error(exc):
                    session.close()
                    raise
                # Free cached sessions and retry once on this session before giving up.
                while self._evict_oldest(keep=session):
                    pass
                self._sessions[shape_key] = session
                return session.run(output_names, input_feed)
            self._sessions[shape_key] = session
            return result

        _, session = self._sessions.popitem(last=False)
        self._sessions[shape_key] = session
        try:
            return session.run(output_names, input_feed)
        except Exception as exc:
            if not _is_oom_error(exc) or len(self._sessions) <= 1:
                raise
            while self._evict_oldest(keep=session):
                pass
            return session.run(output_names, input_feed)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while self._sessions:
            _, session = self._sessions.popitem(last=False)
            session.close()

    def __enter__(self) -> UmziSessionPool:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


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

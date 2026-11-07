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

# resselt checkpoint loading, metadata normalization and provenance.
#
# resselt owns architecture detection, construction and the state dict load. This module only
# turns what resselt reports into the normalized `ModelInfo` contract, cross-checks the parts of
# an FDAT checkpoint the normalized contract depends on, and records where the numbers came from.
#
# Nothing here may import torch or resselt at module import time: `nmanga.ml.model` is imported
# by code paths (`nmanga --help`, the CLI option parsing) that must stay lightweight.

from __future__ import annotations

import hashlib
import math
import operator
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import MLDataType, ModelInfo

if TYPE_CHECKING:
    from torch import nn

__all__ = ("CHECKPOINT_SUFFIX", "LOADER_NAME", "load_checkpoint", "read_model_info")

CHECKPOINT_SUFFIX = ".safetensors"
"""The checkpoint format `nmanga torch-infer` accepts; resselt loads it natively."""

LOADER_NAME = "resselt"
"""The library that detects, constructs and loads the architecture."""

_FLOAT_DTYPES = {
    "F32": MLDataType.FP32.value,
    "F16": MLDataType.FP16.value,
    "BF16": MLDataType.BF16.value,
    "F64": "fp64",
}
"""safetensors dtype tag -> the short name recorded in `ModelInfo.stored_dtypes`."""

_FDAT_METADATA_KEY = "upsampler.MetaUpsample"
"""The FDAT upsampler buffer resselt reads to rebuild the architecture arguments."""

_FDAT_UNSHUFFLE_CONV = "conv_first.1.weight"
"""The FDAT pixel-unshuffle convolution: present only when the model unshuffles pixels."""

_SHA256_CHUNK = 1024 * 1024


def _ensure_typing_self() -> None:
    """Provide `typing.Self` for resselt on interpreters that predate it (3.10 and older)."""
    import typing

    if not hasattr(typing, "Self"):
        from typing_extensions import Self

        typing.Self = Self  # type: ignore[attr-defined]


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
    """Read every tensor name in a safetensors checkpoint with its stored dtype and shape.

    Only the header is read; no tensor data and no model graph is created.
    """
    from safetensors import safe_open

    header: dict[str, tuple[str, tuple[int, ...]]] = {}
    with safe_open(str(path), framework="numpy") as handle:
        for key in handle.keys():
            slice_ = handle.get_slice(key)
            header[key] = (str(slice_.get_dtype()), tuple(int(size) for size in slice_.get_shape()))
    return header


def _read_integer_tensor(path: Path, key: str) -> tuple[int, ...]:
    """Read one small integer tensor out of a checkpoint without constructing the model."""
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as handle:
        return tuple(int(value) for value in handle.get_tensor(key).tolist())


def _stored_float_dtypes(header: dict[str, tuple[str, tuple[int, ...]]]) -> tuple[str, ...]:
    """Collect the dtypes actually stored in the checkpoint's floating-point tensors.

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
    """Validate the effective upscale factor, accepting resselt's ``int`` or sequence type.

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


def _fdat_unshuffle(model: nn.Module, path: Path) -> int:
    """Read the pixel-unshuffle factor out of resselt's constructed FDAT model."""
    import torch

    conv_first = getattr(model, "conv_first", None)
    if isinstance(conv_first, torch.nn.Sequential) and len(conv_first) > 0:
        first = conv_first[0]
        if not isinstance(first, torch.nn.PixelUnshuffle):
            raise ValueError(
                f"resselt built {type(model).__name__} with conv_first={type(conv_first).__name__} whose "
                f"first layer is {type(first).__name__} for {path.name}; the fast-preset adapter expects "
                "a pixel-unshuffle first layer, so this resselt structure is unsupported."
            )
        return int(first.downscale_factor)
    if isinstance(conv_first, torch.nn.Conv2d):
        return 1
    raise ValueError(
        f"resselt built {type(model).__name__} without a recognisable conv_first "
        f"({type(conv_first).__name__}) for {path.name}; the normalized FDAT contract cannot be "
        "cross-checked against this resselt structure."
    )


def _fdat_conv_first_in_channels(model: nn.Module, unshuffle: int, path: Path) -> int:
    """Return the channel count entering the first convolution of the unshuffle path."""
    import torch

    conv_first = model.conv_first
    if unshuffle > 1:
        conv = conv_first[1]
    else:
        conv = conv_first
    if not isinstance(conv, torch.nn.Conv2d):
        raise ValueError(
            f"resselt built {type(model).__name__} with a {type(conv).__name__} after its "
            f"pixel-unshuffle layer for {path.name}; the normalized FDAT contract cannot be checked."
        )
    return int(conv.in_channels)


def _cross_check_fdat(model: nn.Module, info: ModelInfo, path: Path) -> None:
    """Verify the normalized FDAT contract against the model resselt actually constructed."""
    internal_scale = int(model.upsampler.MetaUpsample[2].item())
    pads = int(getattr(model, "pad", 0))
    expected_pad = info.unshuffle if info.unshuffle > 1 else 0
    conv_in = _fdat_conv_first_in_channels(model, info.unshuffle, path)
    expected_conv_in = info.in_channels * info.unshuffle**2
    problems: list[str] = []
    if conv_in != expected_conv_in:
        problems.append(f"conv_first takes {conv_in} channels, expected {expected_conv_in}")
    if pads != expected_pad:
        problems.append(f"check_img_size pads by {pads}, expected {expected_pad}")
    if int(model.upscale) != info.scale:
        problems.append(f"the model upscales by {int(model.upscale)}, including cropping, expected {info.scale}")
    if internal_scale != info.scale * info.unshuffle:
        problems.append(
            f"the upsampler is built for {internal_scale}x, expected {info.scale * info.unshuffle}x "
            f"({info.scale}x effective over a {info.unshuffle}x unshuffle)"
        )
    if problems:
        raise ValueError(
            f"resselt reported FDAT metadata for {path.name} that does not match the constructed model: "
            + "; ".join(problems)
            + ". The installed resselt structure is not the one this backend was validated against."
        )


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
    is_fdat = architecture.strip().upper() == "FDAT"
    unshuffle = _fdat_unshuffle(model, checkpoint) if is_fdat else 1
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
    if is_fdat:
        _cross_check_fdat(model, info, checkpoint)
    return info


def _detect_architecture(header: dict[str, tuple[str, tuple[int, ...]]]) -> str | None:
    """Ask resselt which architecture claims these tensors, without constructing the model."""
    _ensure_typing_self()
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


def _info_from_fdat_header(
    path: Path,
    header: dict[str, tuple[str, tuple[int, ...]]],
    *,
    architecture: str,
    stored_dtypes: tuple[str, ...],
    checkpoint_sha256: str,
    loader_version: str | None,
) -> ModelInfo | None:
    """Normalize FDAT metadata straight from the safetensors header, without loading the model.

    The rules mirror resselt's FDAT loader: the upsampler buffer carries the internal scale and
    the external channels, while the first convolution says whether pixels are unshuffled. A
    structure that does not fit these rules returns ``None`` so the caller can fall back to a real
    load and resselt's own error.
    """
    if _FDAT_METADATA_KEY not in header:
        return None
    meta = _read_integer_tensor(path, _FDAT_METADATA_KEY)
    if len(meta) != 7:
        return None
    _version, _upsampler_index, internal_scale, _embed_dim, out_channels, _mid_dim, _group = meta
    if internal_scale < 1 or out_channels < 1:
        return None
    if _FDAT_UNSHUFFLE_CONV in header:
        conv_in = header[_FDAT_UNSHUFFLE_CONV][1][1]
        if conv_in <= 0 or conv_in % out_channels:
            return None
        unshuffle = math.isqrt(conv_in // out_channels)
    else:
        conv_first = header.get("conv_first.weight")
        if conv_first is None or conv_first[1][1] <= 0:
            return None
        unshuffle = 1
        conv_in = conv_first[1][1]
    if unshuffle < 1 or conv_in != out_channels * unshuffle**2:
        return None
    in_channels = conv_in // unshuffle**2
    if internal_scale % unshuffle:
        return None
    scale = internal_scale // unshuffle
    if scale < 1:
        return None
    return ModelInfo(
        architecture=architecture,
        scale=scale,
        in_channels=in_channels,
        out_channels=out_channels,
        unshuffle=unshuffle,
        stored_dtypes=stored_dtypes,
        checkpoint_sha256=checkpoint_sha256,
        loader=LOADER_NAME,
        loader_version=loader_version,
    )


def read_model_info(model_path: str | Path) -> ModelInfo:
    """
    Describe a checkpoint without running it.

    The stored dtypes, tensor names and shapes come from the safetensors header, and the FDAT
    metadata is read from that header as well, so the model is not constructed for the supported
    architecture family. Any other architecture is described by loading it through resselt.

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
    if architecture is not None and architecture.strip().upper() == "FDAT":
        info = _info_from_fdat_header(
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
    _ensure_typing_self()
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

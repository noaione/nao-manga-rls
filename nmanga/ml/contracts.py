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

# Runtime contracts shared by the ONNX and PyTorch backends.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

__all__ = (
    "DEFAULT_TILE_OVERLAP",
    "DEFAULT_TILE_SIZE",
    "MLDataType",
    "ModelInfo",
    "TilingConfig",
    "TorchPreset",
    "TorchRuntimeConfig",
)

DEFAULT_TILE_SIZE = 256
"""The tile size the validated PyTorch recipes were measured with."""

DEFAULT_TILE_OVERLAP = 32
"""The tile overlap the validated PyTorch recipes were measured with."""

TorchPreset = Literal["fast", "original"]
"""The PyTorch inference recipe: the adapted lean model, or the untouched resselt one."""


class MLDataType(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    @classmethod
    def from_str(cls, type_str: str) -> MLDataType:
        type_str = type_str.lower()
        if type_str == "fp32":
            return cls.FP32
        elif type_str == "fp16":
            return cls.FP16
        elif type_str == "bf16":
            return cls.BF16
        else:
            raise ValueError(f"Unknown MLDataType string: {type_str}")

    def to_onv(self) -> str:
        if self == MLDataType.FP32:
            return "f32"
        elif self == MLDataType.FP16:
            return "f16"
        elif self == MLDataType.BF16:
            return "bf16"
        else:
            raise ValueError(f"Unsupported MLDataType for OpenVINO: {self}")


@dataclass(frozen=True)
class ModelInfo:
    """
    Normalized description of a loaded checkpoint.

    The channel counts are the *external* contract: the RGB channels a caller feeds the
    model and receives back. An architecture that unshuffles pixels internally (FDAT-M's
    12 channel representation) does not change them.
    """

    architecture: str
    """The architecture the checkpoint was detected as, e.g. `FDAT`."""

    scale: int
    """The effective isotropic upscale factor, after any internal unshuffle."""

    in_channels: int
    """The number of RGB channels the model accepts."""

    out_channels: int
    """The number of RGB channels the model returns."""

    unshuffle: int = 1
    """The internal pixel-unshuffle factor, `1` when the architecture does not unshuffle."""

    stored_dtypes: tuple[str, ...] = ()
    """Every floating dtype actually stored in the checkpoint, inspected from its tensors."""

    checkpoint_sha256: str | None = None
    """The SHA-256 digest of the checkpoint file, when one was computed."""

    loader: str = "resselt"
    """The library that detected and constructed the architecture."""

    loader_version: str | None = None
    """The version of the loading library, when it could be determined."""

    def to_dict(self) -> dict[str, object]:
        """Return the contract as a JSON-serializable dictionary."""
        return {
            "architecture": self.architecture,
            "scale": self.scale,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "unshuffle": self.unshuffle,
            "stored_dtypes": list(self.stored_dtypes),
            "checkpoint_sha256": self.checkpoint_sha256,
            "loader": self.loader,
            "loader_version": self.loader_version,
        }


@dataclass(frozen=True)
class TilingConfig:
    """
    The validated JaNai tile grid.

    `tile_overlap` is the *total* overlap between two neighbouring tiles, so the stride is
    `tile_size - tile_overlap`. The ONNX overlap tiler in `nmanga.ml.onnx_backend` uses the same
    stride, but the two are still not interchangeable because the attention and pooling of a torch
    model depend on the whole tile.
    """

    tile_size: int = DEFAULT_TILE_SIZE
    tile_overlap: int = DEFAULT_TILE_OVERLAP

    def __post_init__(self):
        if self.tile_size < 1:
            raise ValueError("Tile size must be at least 1.")
        if self.tile_overlap < 0:
            raise ValueError("Tile overlap must be non-negative.")
        if self.tile_size and self.tile_overlap >= self.tile_size:
            raise ValueError("Tile overlap must be smaller than the tile size.")

    @property
    def stride(self) -> int:
        """The distance between the top-left corner of two neighbouring tiles."""
        return self.tile_size - self.tile_overlap


@dataclass(frozen=True)
class TorchRuntimeConfig:
    """Everything that decides how the PyTorch backend executes a model."""

    device: str = "auto"
    """`auto`, `cpu` or `cuda:N`; `auto` picks CUDA when it is available."""

    preset: TorchPreset = "fast"
    """The rewriting recipe applied to the loaded model."""

    use_channels_last: bool | None = None
    """An explicit memory-format override: `None` lets the architecture's adapter choose."""

    enable_compile: bool = False
    """Whether to wrap the prepared model with `torch.compile`."""

    tiling: TilingConfig = field(default_factory=TilingConfig)
    """The tile grid used to run one page through the model."""

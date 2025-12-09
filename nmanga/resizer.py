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

from __future__ import annotations

from enum import Enum
from typing import Literal, overload

from cykooz.resizer import FilterType, ResizeAlg, ResizeOptions, Resizer
from PIL import Image

__all__ = (
    "ResizeKernel",
    "ResizeMode",
    "ResizeTarget",
    "rescale_image",
)


class ResizeKernel(str, Enum):
    """
    Resampling kernels for image rescaling.
    """

    Nearest = "nearest"
    """Nearest neighbor resampling"""
    Box = "box"
    """Box resampling"""
    Bicubic = "bicubic"
    """Bicubic resampling"""
    Bilinear = "bilinear"
    """Bilinear resampling"""
    CatmullRom = "catmull_rom"
    """Catmull-Rom resampling"""
    Mitchell = "mitchell"
    """Mitchell resampling"""
    Gaussian = "gaussian"
    """Gaussian resampling"""
    Lanczos = "lanczos"
    """Lanczos 3-taps resampling"""
    Hamming = "hamming"
    """Hamming resampling"""

    def to_resample(self) -> Image.Resampling | ResizeOptions:
        match self:
            case ResizeKernel.Nearest:
                return ResizeOptions(
                    resize_alg=ResizeAlg.nearest(),
                    use_alpha=False,  # We don't care about alpha channel for PIL resampling
                )
            case ResizeKernel.Box:
                return ResizeOptions(
                    resize_alg=ResizeAlg.convolution(FilterType.box),
                    use_alpha=False,  # We don't care about alpha channel for PIL resampling
                )
            case ResizeKernel.Bicubic:
                return Image.Resampling.BICUBIC
            case ResizeKernel.Bilinear:
                return ResizeOptions(
                    resize_alg=ResizeAlg.convolution(FilterType.bilinear),
                    use_alpha=False,  # We don't care about alpha channel for PIL resampling
                )
            case ResizeKernel.CatmullRom:
                return ResizeOptions(
                    resize_alg=ResizeAlg.convolution(FilterType.catmull_rom),
                    use_alpha=False,  # We don't care about alpha channel for PIL resampling
                )
            case ResizeKernel.Mitchell:
                return ResizeOptions(
                    resize_alg=ResizeAlg.convolution(FilterType.mitchell),
                    use_alpha=False,  # We don't care about alpha channel for PIL resampling
                )
            case ResizeKernel.Gaussian:
                return ResizeOptions(
                    resize_alg=ResizeAlg.convolution(FilterType.gaussian),
                    use_alpha=False,  # We don't care about alpha channel for PIL resampling
                )
            case ResizeKernel.Lanczos:
                return ResizeOptions(
                    resize_alg=ResizeAlg.convolution(FilterType.lanczos3),
                    use_alpha=False,  # We don't care about alpha channel for PIL resampling
                )
            case ResizeKernel.Hamming:
                return Image.Resampling.HAMMING


class ResizeMode(str, Enum):
    """
    Resize modes for image rescaling.
    """

    Exact = "exact"
    """Resize to exact dimensions, possibly changing aspect ratio."""
    Height = "height"
    """Resize to target height, adjusting width to maintain aspect ratio."""
    Width = "width"
    """Resize to target width, adjusting height to maintain aspect ratio."""
    Fit = "fit"
    """Resize to fit to largest within target dimensions, maintaining aspect ratio."""
    Multiply = "multiply"
    """Resize by multiplying both dimensions by a factor."""


class ResizeTarget:
    mode: ResizeMode
    """The resize mode."""
    target_width: int | float | None
    """The target width and height, or None if not applicable."""
    target_height: int | float | None
    """The target width and height, or None if not applicable."""

    @overload
    def __init__(self, mode: Literal[ResizeMode.Exact], *, width: int, height: int) -> None: ...
    @overload
    def __init__(self, mode: Literal[ResizeMode.Height], *, height: int) -> None: ...
    @overload
    def __init__(self, mode: Literal[ResizeMode.Width], *, width: int) -> None: ...
    @overload
    def __init__(self, mode: Literal[ResizeMode.Fit], *, width: int, height: int) -> None: ...
    @overload
    def __init__(self, mode: Literal[ResizeMode.Multiply], *, factor: float) -> None: ...

    def __init__(self, mode: ResizeMode, **kwargs: int | float) -> None:
        # Unpack args based on mode, also calculate the width/height if needed
        self.mode = mode

        match mode:
            case ResizeMode.Exact:
                try:
                    self.target_width = kwargs["width"]
                    self.target_height = kwargs["height"]
                except KeyError as e:
                    raise ValueError("Exact mode requires both `width` and `height`") from e
            case ResizeMode.Height:
                try:
                    self.target_height = kwargs["height"]
                except KeyError as e:
                    raise ValueError("Height mode requires `height`") from e
            case ResizeMode.Width:
                try:
                    self.target_width = kwargs["width"]
                    self.target_height = None
                except KeyError as e:
                    raise ValueError("Width mode requires `width`") from e
            case ResizeMode.Fit:
                try:
                    self.target_width = kwargs["width"]
                    self.target_height = kwargs["height"]
                except KeyError as e:
                    raise ValueError("Fit mode requires both `width` and `height`") from e
            case ResizeMode.Multiply:
                try:
                    factor = kwargs["factor"]
                    self.target_width = factor
                    self.target_height = factor
                except KeyError as e:
                    raise ValueError("Multiply mode requires `factor`") from e

        is_factor = mode in {ResizeMode.Multiply}

        # Check if target_width and target_height are positive integers or None
        if self.target_width is not None and (not isinstance(self.target_width, int) or self.target_width <= 0):
            raise ValueError("Target width must be a positive integer or None")
        if self.target_height is not None and (not isinstance(self.target_height, int) or self.target_height <= 0):
            raise ValueError("Target height must be a positive integer or None")
        if is_factor and (
            not isinstance(self.target_width, (int, float)) or not isinstance(self.target_height, (int, float))
        ):
            raise ValueError("Multiply mode requires numeric factor for width and height")
        if is_factor and (
            (isinstance(self.target_width, (int, float)) and self.target_width <= 0)
            or (isinstance(self.target_height, (int, float)) and self.target_height <= 0)
        ):
            raise ValueError("Multiply mode factor must be positive")

    def __str__(self) -> str:
        match self.mode:
            case ResizeMode.Exact:
                return f"Exact({self.target_width}x{self.target_height})"
            case ResizeMode.Height:
                return f"Height({self.target_height})"
            case ResizeMode.Width:
                return f"Width({self.target_width})"
            case ResizeMode.Fit:
                return f"Fit({self.target_width}x{self.target_height})"
            case ResizeMode.Multiply:
                return f"Multiply(factor={self.target_width})"

    def get_scale(self, img: Image.Image) -> tuple[int, int]:
        original_width, original_height = img.size

        match self.mode:
            case ResizeMode.Exact:
                if self.target_width is None or self.target_height is None:
                    raise ValueError("Exact mode requires both width and height")
                return (int(self.target_width), int(self.target_height))
            case ResizeMode.Height:
                if self.target_height is None:
                    raise ValueError("Height mode requires height")
                new_height = int(self.target_height)
                new_width = int((new_height / original_height) * original_width)
                return (new_width, new_height)
            case ResizeMode.Width:
                if self.target_width is None:
                    raise ValueError("Width mode requires width")
                new_width = int(self.target_width)
                new_height = int((new_width / original_width) * original_height)
                return (new_width, new_height)
            case ResizeMode.Fit:
                if self.target_width is None or self.target_height is None:
                    raise ValueError("Fit mode requires both width and height")
                ratio = min(
                    self.target_width / original_width,
                    self.target_height / original_height,
                )
                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)
                return (new_width, new_height)
            case ResizeMode.Multiply:
                factor = self.target_width  # Both width and height use the same factor
                if factor is None:
                    raise ValueError("Multiply mode requires factor")
                new_width = int(original_width * factor)
                new_height = int(original_height * factor)
                return (new_width, new_height)


def rescale_image(
    img: Image.Image,
    target: ResizeTarget,
    kernel: ResizeKernel,
) -> Image.Image:
    """
    Rescale an image using the specified target and kernel.

    :param img: The input image to rescale.
    :param target: The resize target specifying mode and dimensions.
    :param kernel: The resampling kernel to use.
    :return: The rescaled image.
    :rtype: Image.Image
    """

    new_width, new_height = target.get_scale(img)
    resample = kernel.to_resample()

    if isinstance(resample, ResizeOptions):
        resizer = Resizer()
        dst_image = Image.new(img.mode, (new_width, new_height))
        resizer.resize_pil(img, dst_image, resample)
        return dst_image
    elif isinstance(resample, Image.Resampling):
        return img.resize((new_width, new_height), resample=resample)
    else:
        raise ValueError("Invalid resample type")

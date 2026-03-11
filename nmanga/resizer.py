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

from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Literal, cast, overload

from cykooz.resizer import FilterType, ImageData, PixelType, ResizeAlg, ResizeOptions, Resizer
from PIL import Image
from wand.image import Image as WandImage

__all__ = (
    "ResizeKernel",
    "ResizeMode",
    "ResizeTarget",
    "rescale_image",
)


@dataclass
class WandParamMap:
    a: str | None = None
    b: str | None = None


class WandResizeKernel(str, Enum):
    Hermite = "hermite"
    Hanning = "hanning"
    Hamming = "hamming"
    Blackman = "blackman"
    Quadratic = "quadratic"
    Cubic = "cubic"
    Jinc = "jinc"
    Sinc = "sinc"
    SincFast = "sincfast"
    Kaiser = "kaiser"
    Welsh = "welsh"
    Parzen = "parzen"
    Bohman = "bohman"
    Bartlett = "bartlett"
    Lagrange = "lagrange"
    Lanczos = "lanczos"
    LanczosSharp = "lanczossharp"
    Robidoux = "robidoux"
    RobidouxSharp = "robidouxsharp"
    Cosine = "cosine"
    Spline = "spline"


NumpyLib = None
# Basically since our CLI would have --param-a and --param-b for extra parameters for certain kernels
WAND_PARAM_MAPPINGS = {
    "lanczos": WandParamMap(a="support", b="scale-blur"),  # Blur for "sharp" variant
    "lanczossharp": WandParamMap(a="support"),
    "cubic": WandParamMap(a="B", b="C"),
    "spline": WandParamMap(a="lobes"),
    "sinc": WandParamMap(a="support"),
    "sincfast": WandParamMap(a="support"),
    "kaiser": WandParamMap(a="lobes", b="kaiser-beta"),
    "hanning": WandParamMap(a="window-support"),
    "hamming": WandParamMap(a="window-support"),
    "lagrange": WandParamMap(a="support"),
    "robidoux": WandParamMap(a="lobes"),
    "robidouxsharp": WandParamMap(a="lobes"),
}


def try_imports():
    global NumpyLib

    try:
        import numpy as np  # type: ignore

        NumpyLib = np
    except ImportError as exc:
        raise ImportError("numpy is required to use some specific Wand kernel. Please install numpy.") from exc


class WandResizeOptions:
    def __init__(self, kernel: WandResizeKernel, filters: dict[str, float | int] | None = None):
        self.kernel = kernel
        self.filters = filters or {}

    def apply_filters(self, param_a: float | int | None, param_b: float | int | None) -> dict[str, float | int]:
        mapping = WAND_PARAM_MAPPINGS.get(self.kernel.value)
        if not mapping:
            raise ValueError(f"No parameter mapping defined for kernel {self.kernel}")
        filters = {}
        if mapping.a and param_a is not None:
            filters["filter:" + mapping.a] = param_a
        if mapping.b and param_b is not None:
            filters["filter:" + mapping.b] = param_b
        return filters

    def resize_image(
        self,
        img: Image.Image,
        width: int,
        height: int,
        *,
        param_a: float | int | None = None,
        param_b: float | int | None = None,
    ) -> Image.Image:
        try_imports()

        filters = self.apply_filters(param_a, param_b)
        img_arr = NumpyLib.array(img)  # type: ignore
        with WandImage.from_array(img_arr, channel_map=img.mode) as wand_img:
            # Apply the filters as needed based on the kernel
            for filter_name, filter_value in filters.items():
                wand_img.artifacts[filter_name] = filter_value  # type: ignore
            wand_img.resize(width, height, filter=self.kernel.value)
            bytes_data = BytesIO(wand_img.make_blob("png"))  # type: ignore
            return Image.open(bytes_data).convert(img.mode)


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
    """Lanczos resampling"""
    LanczosSharp = "lanczossharp"
    """Lanczos resampling with sharper results, typically by adjusting the blurring factor"""
    Hamming = "hamming"
    """Hamming resampling"""
    Hanning = "hanning"
    """Hanning resampling"""
    Hermite = "hermite"
    """Hermite resampling"""
    Blackman = "blackman"
    """Blackman resampling"""
    Quadratic = "quadratic"
    """Quadratic resampling"""
    Jinc = "jinc"
    """Jinc resampling, which is a windowed sinc function that can provide good results for downscaling"""
    Sinc = "sinc"
    """Sinc resampling"""
    SincFast = "sincfast"
    """SincFast is a faster, less accurate version of the Sinc kernel"""
    Kaiser = "kaiser"
    """Kaiser resampling"""
    Welsh = "welsh"
    """Welsh resampling"""
    Parzen = "parzen"
    """Parzen resampling"""
    Bohman = "bohman"
    """Bohman resampling"""
    Bartlett = "bartlett"
    """Bartlett resampling"""
    Lagrange = "lagrange"
    """Lagrange resampling"""
    Robidoux = "robidoux"
    """Robidoux resampling, similar to Lanczos"""
    RobidouxSharp = "robidouxsharp"
    """Robidoux resampling with sharper results, typically by adjusting the blurring factor"""
    Cosine = "cosine"
    """Cosine resampling"""
    Spline = "spline"
    """Spline resampling"""

    def to_resample(self) -> Image.Resampling | ResizeOptions | WandResizeOptions:
        match self:
            case ResizeKernel.Nearest:
                return ResizeOptions(
                    resize_alg=ResizeAlg.nearest(),
                )
            case ResizeKernel.Box:
                return ResizeOptions(
                    resize_alg=ResizeAlg.convolution(FilterType.box),
                )
            case ResizeKernel.Bicubic:
                return WandResizeOptions(kernel=WandResizeKernel.Cubic)
            case ResizeKernel.Bilinear:
                return ResizeOptions(
                    resize_alg=ResizeAlg.convolution(FilterType.bilinear),
                )
            case ResizeKernel.CatmullRom:
                return ResizeOptions(
                    resize_alg=ResizeAlg.convolution(FilterType.catmull_rom),
                )
            case ResizeKernel.Mitchell:
                return ResizeOptions(
                    resize_alg=ResizeAlg.convolution(FilterType.mitchell),
                )
            case ResizeKernel.Gaussian:
                return ResizeOptions(
                    resize_alg=ResizeAlg.convolution(FilterType.gaussian),
                )
            case ResizeKernel.Lanczos:
                return WandResizeOptions(kernel=WandResizeKernel.Lanczos)
            case ResizeKernel.LanczosSharp:
                return WandResizeOptions(kernel=WandResizeKernel.LanczosSharp)
            case ResizeKernel.Hamming:
                return WandResizeOptions(kernel=WandResizeKernel.Hamming)
            case ResizeKernel.Hanning:
                return WandResizeOptions(kernel=WandResizeKernel.Hanning)
            case ResizeKernel.Hermite:
                return WandResizeOptions(kernel=WandResizeKernel.Hermite)
            case ResizeKernel.Blackman:
                return WandResizeOptions(kernel=WandResizeKernel.Blackman)
            case ResizeKernel.Quadratic:
                return WandResizeOptions(kernel=WandResizeKernel.Quadratic)
            case ResizeKernel.Jinc:
                return WandResizeOptions(kernel=WandResizeKernel.Jinc)
            case ResizeKernel.Sinc:
                return WandResizeOptions(kernel=WandResizeKernel.Sinc)
            case ResizeKernel.SincFast:
                return WandResizeOptions(kernel=WandResizeKernel.SincFast)
            case ResizeKernel.Kaiser:
                return WandResizeOptions(kernel=WandResizeKernel.Kaiser)
            case ResizeKernel.Welsh:
                return WandResizeOptions(kernel=WandResizeKernel.Welsh)
            case ResizeKernel.Parzen:
                return WandResizeOptions(kernel=WandResizeKernel.Parzen)
            case ResizeKernel.Bohman:
                return WandResizeOptions(kernel=WandResizeKernel.Bohman)
            case ResizeKernel.Bartlett:
                return WandResizeOptions(kernel=WandResizeKernel.Bartlett)
            case ResizeKernel.Lagrange:
                return WandResizeOptions(kernel=WandResizeKernel.Lagrange)
            case ResizeKernel.Robidoux:
                return WandResizeOptions(kernel=WandResizeKernel.Robidoux)
            case ResizeKernel.RobidouxSharp:
                return WandResizeOptions(kernel=WandResizeKernel.RobidouxSharp)
            case ResizeKernel.Cosine:
                return WandResizeOptions(kernel=WandResizeKernel.Cosine)
            case ResizeKernel.Spline:
                return WandResizeOptions(kernel=WandResizeKernel.Spline)
            case _:
                raise ValueError(f"Unsupported resize kernel: {self.value}")


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
    def __init__(self, mode: Literal[ResizeMode.Multiply], *, factor: int | float) -> None: ...
    @overload
    def __init__(
        self, mode: ResizeMode, *, factor: int | float | None, width: int | None, height: int | None
    ) -> None: ...

    def __init__(self, mode: ResizeMode, **kwargs: int | float | None) -> None:
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
        target_width = getattr(self, "target_width", None)
        target_height = getattr(self, "target_height", None)
        if target_width and (not isinstance(target_width, int) or target_width <= 0):
            raise ValueError("Target width must be a positive integer or None")
        if target_height is not None and (not isinstance(target_height, int) or target_height <= 0):
            raise ValueError("Target height must be a positive integer or None")
        if is_factor and (not isinstance(target_width, (int, float)) or not isinstance(target_height, (int, float))):
            raise ValueError("Multiply mode requires numeric factor for width and height")
        if is_factor and (
            (isinstance(target_width, (int, float)) and target_width <= 0)
            or (isinstance(target_height, (int, float)) and target_height <= 0)
        ):
            raise ValueError("Multiply mode factor must be positive")

        if hasattr(self, "target_width") is True:
            self.target_width = cast(int | float, self.target_width)
        if hasattr(self, "target_height") is True:
            self.target_height = cast(int | float, self.target_height)

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


def get_image_mode(img: Image.Image) -> PixelType:
    match img.mode:
        case "L":
            return PixelType.U8
        case "LA":
            return PixelType.U8x2
        case "RGB":
            return PixelType.U8x3
        case "RGBA" | "RGBa" | "CMYK":
            return PixelType.U8x4
        case "I":
            return PixelType.I32
        case "F":
            return PixelType.F32
        case _:
            raise ValueError(f"Unsupported image mode: {img.mode}")


def make_image_pair(in_img: Image.Image, out_size: tuple[int, int]) -> tuple[ImageData, ImageData]:
    pixels = in_img.tobytes()
    img_mode = get_image_mode(in_img)
    src_img = ImageData(
        width=in_img.width,
        height=in_img.height,
        pixel_type=img_mode,
        pixels=pixels,
    )
    dest_img = ImageData(
        width=out_size[0],
        height=out_size[1],
        pixel_type=img_mode,
    )
    return src_img, dest_img


def _do_cykooz_resize(img: Image.Image, resample: ResizeOptions, new_width: int, new_height: int) -> Image.Image:
    resizer = Resizer()
    src_image, dst_image = make_image_pair(img, (new_width, new_height))
    resizer.resize(src_image, dst_image, resample)
    out_img = Image.frombytes(img.mode, (dst_image.width, dst_image.height), dst_image.get_buffer())
    return out_img


def _do_wand_resize(
    img: Image.Image,
    resample: WandResizeOptions,
    new_width: int,
    new_height: int,
    *,
    param_a: float | int | None = None,
    param_b: float | int | None = None,
) -> Image.Image:
    return resample.resize_image(img, new_width, new_height, param_a=param_a, param_b=param_b)


def rescale_image(
    img: Image.Image,
    target: ResizeTarget,
    kernel: ResizeKernel,
    *,
    param_a: float | int | None = None,
    param_b: float | int | None = None,
) -> Image.Image:
    """
    Rescale an image using the specified target and kernel.

    :param img: The input image to rescale.
    :param target: The resize target specifying mode and dimensions.
    :param kernel: The resampling kernel to use.
    :param param_a: Optional parameter A for certain kernels (e.g., Lanczos support).
    :param param_b: Optional parameter B for certain kernels (e.g., Lanczos blur factor).
    :return: The rescaled image.
    :rtype: Image.Image
    """

    new_width, new_height = target.get_scale(img)
    resample = kernel.to_resample()

    if isinstance(resample, ResizeOptions):
        return _do_cykooz_resize(img, resample, new_width, new_height)
    elif isinstance(resample, WandResizeOptions):
        return _do_wand_resize(img, resample, new_width, new_height, param_a=param_a, param_b=param_b)
    elif isinstance(resample, Image.Resampling):
        return img.resize((new_width, new_height), resample=resample)
    else:
        raise ValueError("Invalid resample type")

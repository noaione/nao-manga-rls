"""
Original code by anon

Adapted for nmanga
"""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, TypedDict, Union

from PIL import Image

__all__ = (
    "analyze_gray_shades",
    "apply_levels",
    "apply_levels",
    "create_magick_params",
    "find_local_peak",
    "gamma_correction",
    "pad_shades_to_bpc",
    "posterize_image_by_bits",
    "posterize_image_by_shades",
    "try_imports",
)

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)

NumpyLib = None
ScipySignalLib = None


class ShadeAnalysis(TypedDict):
    shade: int
    """The gray shade value in integer (0-255)"""
    percentage: float
    """The percentage of pixels in the image that have this shade."""


def try_imports():
    global NumpyLib, ScipySignalLib

    try:
        import numpy as np  # type: ignore

        NumpyLib = np
    except ImportError:
        raise ImportError("numpy is required to use autolevel. Please install numpy.")

    try:
        from scipy import signal  # type: ignore

        ScipySignalLib = signal
    except ImportError:
        raise ImportError("scipy is required to use autolevel. Please install scipy.")


def find_local_peak(img_path: Union[Path, BytesIO, Image.Image], upper_limit: int = 60) -> Tuple[int, Path, bool]:
    """
    Automatically determine the optimal black level for an image by finding local peaks in its histogram.

    Returns a tuple of (black_level, img_path, force_gray).
    """
    global NumpyLib, ScipySignalLib

    if NumpyLib is None or ScipySignalLib is None:
        try_imports()

    image = img_path if isinstance(img_path, Image.Image) else Image.open(img_path)
    force_gray = image.mode != "L"
    if force_gray:
        image = image.convert("L")  # force grayscale

    img_array = NumpyLib.array(image)
    hist, binedges = NumpyLib.histogram(img_array, bins=256, range=(0, 255))

    width = NumpyLib.arange(start=1, stop=upper_limit, step=1)
    result = ScipySignalLib.find_peaks_cwt(hist, widths=width)

    if not isinstance(img_path, Image.Image):
        # temp image, close
        image.close()

    if result.any():
        if hist[result[0] - 1] > hist[result[0]]:
            result[0] = result[0] - 1
        elif hist[result[0] + 1] > hist[result[0]]:
            result[0] = result[0] + 1

        black_level = math.ceil(binedges[result[0]])
        return black_level, img_path, force_gray
    return 0, img_path, force_gray


def gamma_correction(black_level: int) -> int:
    black_point = black_level / 255
    white_point = 1
    midpoint = 0.5
    gamma = math.log(midpoint) / math.log((midpoint - black_point) / (white_point - black_point))
    return round(1 / gamma, 2)


def create_magick_params(black_level: int, peak_offset: int = 0) -> str:
    gamma = gamma_correction(black_level)
    black_point_pct = round(black_level / 255 * 100, 2) + peak_offset
    return f"{black_point_pct},100%,{gamma}"


def apply_levels(image: Image.Image, black_point: float, white_point: float, gamma: float):
    """
    Apply level adjustments similar to ImageMagick's -level command.
    """
    delta = white_point - black_point
    inv_gamma = 1.0 / gamma

    def lookup_table(value: int):
        if value < black_point:
            return 0
        elif value > white_point:
            return 255
        else:
            normalized = (value - black_point) / delta
            gamma_corrected = math.pow(normalized, inv_gamma)
            return round(gamma_corrected * 255)

    # Create lookup table
    lut_adjustments = [lookup_table(i) for i in range(256)]

    # Apply the lookup table
    return image.point(lut_adjustments * len(image.getbands()))


def analyze_gray_shades(image: Image.Image, threshold: float = 0.01) -> List[ShadeAnalysis]:
    """
    Analyze the amount of gray shades in a "grayscale" image.
    """

    global NumpyLib

    if NumpyLib is None:
        try_imports()

    if image.mode != "L":
        image = image.convert("L")  # force grayscale

    img_array = NumpyLib.array(image)

    total_pixels = img_array.size
    unique_shades, counter = NumpyLib.unique(img_array, return_counts=True)

    filtered_shades = []
    for shade, count in zip(unique_shades, counter):
        percentage = (count / total_pixels) * 100
        if percentage >= threshold:
            # Get the RGB value of the shade
            filtered_shades.append({"shade": int(shade), "percentage": percentage})

    # Sort by highest percentage first
    filtered_shades.sort(key=lambda x: x["percentage"], reverse=True)
    return filtered_shades


def posterize_image_by_bits(image: Image.Image, num_bits: int) -> Image.Image:
    """
    Posterize an image to the specified gray shades.

    This would return 2^num_bits shades.
    """

    if image.mode != "L":
        image = image.convert("L")  # force grayscale

    quantized = image.quantize(colors=2**num_bits, method=Image.Dither.FLOYDSTEINBERG)
    return quantized


def pad_shades_to_bpc(shades: List[int]) -> List[int]:
    """
    Pad the shades list to the closest bpc value.

    Shades is a list of gray shade values (0-255).
    """

    num_shades = len(shades)
    if num_shades <= 1:
        # Uhhhh, nothing to posterize
        return shades

    # Known shade values:
    # 1bpc -> 2 shades
    # 2bpc -> 4 shades
    # 3bpc -> 8 shades
    # 4bpc -> 16 shades
    # 5bpc -> 32 shades
    # 6bpc -> 64 shades
    # 7bpc -> 128 shades
    # 8bpc -> 256 shades
    num_bits = math.ceil(math.log2(num_shades))
    num_palette_shades = int(math.pow(2, num_bits))
    diffs_count = num_palette_shades - num_shades

    corrected = sorted(shades.copy())  # sort ascending
    corrected.extend([255] * diffs_count)  # Pad with white
    return corrected


def posterize_image_by_shades(image: Image.Image, shades: List[int]) -> Image.Image:
    """
    Posterize an image to the specified gray shades.

    Shades is a list of gray shade values (0-255).
    """

    if image.mode != "L":
        image = image.convert("L")  # force grayscale

    palette: List[Tuple[int, int, int]] = []
    for shade in shades:
        palette.extend([shade] * 3)  # R, G, B

    palette_img = Image.new("P", (1, 1))
    palette_img.putpalette(palette)

    dithered_img = image.convert("P", palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
    return dithered_img

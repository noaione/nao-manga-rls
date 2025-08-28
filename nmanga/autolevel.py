"""
Original code by anon

Adapted for nmanga
"""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Tuple, Union

from PIL import Image

__all__ = (
    "apply_levels",
    "create_magick_params",
    "find_local_peak",
    "gamma_correction",
    "try_imports",
)

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)

NumpyLib = None
ScipySignalLib = None


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

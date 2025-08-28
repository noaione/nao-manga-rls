"""
Original code by anon

Adapted for nmanga
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

from PIL import Image

__all__ = (
    "create_magick_params",
    "find_local_peak",
    "try_imports",
)

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024 ** 3) // 3)

NumpyLib = None
ScipySignalLib = None


def try_imports():
    global NumpyLib, ScipySignalLib

    try:
        import numpy as np

        NumpyLib = np
    except ImportError:
        raise ImportError("numpy is required to use autolevel. Please install numpy.")

    try:
        from scipy import signal

        ScipySignalLib = signal
    except ImportError:
        raise ImportError("scipy is required to use autolevel. Please install scipy.")


def find_local_peak(img_path: Path, upper_limit: int = 60) -> Tuple[int, Path, bool]:
    """
    Automatically determine the optimal black level for an image by finding local peaks in its histogram.

    Returns a tuple of (black_level, img_path, force_gray).
    """
    global NumpyLib, ScipySignalLib

    if NumpyLib is None or ScipySignalLib is None:
        try_imports()

    with Image.open(img_path) as image:
        force_gray = image.mode != "L"
        if force_gray:
            image = image.convert("L")  # force grayscale

        img_array = NumpyLib.array(image)
        hist, binedges = NumpyLib.histogram(img_array, bins=256, range=(0, 255))

        width = NumpyLib.arange(start=1, stop=upper_limit, step=1)
        result = ScipySignalLib.find_peaks_cwt(hist, widths=width)
        image.close()

        if result.any():
            if hist[result[0] - 1] > hist[result[0]]:
                result[0] = result[0] - 1
            elif hist[result[0] + 1] > hist[result[0]]:
                result[0] = result[0] + 1

            black_level = math.ceil(binedges[result[0]])
            return black_level, img_path, force_gray
        return 0, img_path, force_gray


def create_magick_params(black_level: int, peak_offset: int = 0) -> str:
    black_point = black_level / 255
    white_point = 1
    midpoint = 0.5
    gamma = math.log(midpoint) / math.log((midpoint - black_point) / (white_point - black_point))
    gamma = round(1 / gamma, 2)
    black_point_pct = round(black_level / 255 * 100, 2) + peak_offset
    return f"{black_point_pct},100%,{gamma}"

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

import math
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple

__all__ = (
    "create_magick_params",
    "find_local_peak",
    "find_local_peaks",
    "try_imports",
)
ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp", ".jxl"]

ImageLib = None
NumpyLib = None
ScipySignalLib = None


def try_imports():
    global ImageLib, NumpyLib, ScipySignalLib
    try:
        from PIL import Image as PilImage

        ImageLib = PilImage
    except ImportError:
        raise ImportError("Pillow is required to use autolevel. Please install Pillow.")

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
    global ImageLib, NumpyLib, ScipySignalLib

    if ImageLib is None or NumpyLib is None or ScipySignalLib is None:
        try_imports()

    image = ImageLib.open(img_path)
    force_gray = image.mode != "L"
    if force_gray:
        image = image.convert("L")  # force grayscale

    img_array = NumpyLib.array(image)
    hist, binedges = NumpyLib.histogram(img_array, bins=256, range=(0, 255))

    width = NumpyLib.arange(start=1, stop=upper_limit, step=1)
    result = ScipySignalLib.find_peaks_cwt(hist, widths=width)

    if result.any():
        if hist[result[0] - 1] > hist[result[0]]:
            result[0] = result[0] - 1
        elif hist[result[0] + 1] > hist[result[0]]:
            result[0] = result[0] + 1

        black_level = math.ceil(binedges[result[0]])
        return black_level, img_path, force_gray
    return 0, img_path, force_gray


def find_local_peaks(source_dir: Path, upper_limit: int = 60) -> List[Tuple[int, Path]]:
    """
    Find local peaks for all images in a directory.
    """
    if not source_dir.is_dir():
        raise NotADirectoryError(f"{source_dir} is not a valid directory.")

    files = {file for file in source_dir.glob("*") if file.suffix.lower() in ALLOWED_EXTENSIONS}
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(find_local_peak, [(file, upper_limit) for file in files])
    return results


def create_magick_params(black_level: int, peak_offset: int = 0) -> str:
    black_point = black_level / 255
    white_point = 1
    midpoint = 0.5
    gamma = math.log(midpoint) / math.log(
        (midpoint - black_point) / (white_point - black_point)
    )
    gamma = round(1 / gamma, 2)
    black_point_pct = round(black_level / 255 * 100, 2) + peak_offset
    return f"{black_point_pct},100%,{gamma}"

"""
Original code by anon

Adapted for nmanga
"""

from __future__ import annotations

import math
import subprocess as sp
from io import BytesIO
from pathlib import Path
from typing import TypedDict

from PIL import Image

__all__ = (
    "analyze_gray_shades",
    "apply_levels",
    "apply_levels",
    "create_magick_params",
    "detect_nearest_bpc",
    "find_local_peak",
    "gamma_correction",
    "pad_shades_to_bpc",
    "posterize_image_by_bits",
    "posterize_image_by_shades",
    "posterize_image_with_imagemagick",
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


def is_grayscale_palette(palette: list[int]) -> bool:
    for i in range(0, len(palette), 3):
        r, g, b = palette[i : i + 3]
        if (r != g) or (g != b):
            return False

    return True


def find_local_peak(
    img_path: Path | BytesIO | Image.Image,
    *,
    upper_limit: int = 60,
    peak_percentage: float = 0.25,
    skip_white_check: bool = False,
) -> tuple[int, int, bool]:
    """
    Automatically determine the optimal black level for an image by finding local peaks in its histogram.

    Courtesy of anon.
    """

    global NumpyLib, ScipySignalLib

    if NumpyLib is None or ScipySignalLib is None:
        try_imports()

    image = img_path if isinstance(img_path, Image.Image) else Image.open(img_path)
    force_gray = image.mode != "L"

    if image.mode == "P":
        palette = image.getpalette()
        if (palette is not None) and is_grayscale_palette(palette):
            force_gray = True

    if force_gray:
        image = image.convert("L")  # force grayscale

    if peak_percentage <= 0 and peak_percentage >= 100:
        peak_percentage = 0.25  # default value

    img_width, img_height = image.size

    min_px_count = math.ceil((img_width * img_height) * (peak_percentage / 100.0))
    min_prominence = int(min_px_count / 2)

    img_array = NumpyLib.array(image)
    hist, binedges = NumpyLib.histogram(img_array, bins=256, range=(0, 256))

    # Region of interest: only consider peaks in the lower range for black level
    roi_min, roi_max = 0, upper_limit
    roi_mask = (binedges[:-1] >= roi_min) & (binedges[:-1] < roi_max)
    hist_roi = hist[roi_mask]
    binedges_roi = binedges[:-1][roi_mask]

    # Pad left and right to avoid edge peaks
    hist_padded = NumpyLib.pad(hist_roi, (1, 1), mode="constant", constant_values=0)

    # Find peaks on padded array
    peaks_padded, _ = ScipySignalLib.find_peaks(hist_padded, height=min_px_count, prominence=min_prominence)

    peaks = peaks_padded - 1  # Adjust indices back to original hist_roi
    valid = (peaks >= 0) & (peaks < len(hist_roi))
    peaks = peaks[valid]

    black_level = 0
    if len(peaks) > 0:
        peak_val = math.ceil(binedges_roi[peaks[0]])
        black_level = peak_val
    else:
        # Find without max height constraint
        peaks_padded, _ = ScipySignalLib.find_peaks(hist_padded, height=0)

        peaks = peaks_padded - 1  # Adjust indices back to original hist_roi
        valid = (peaks >= 0) & (peaks < len(hist_roi))
        peaks = peaks[valid]
        if len(peaks) > 0:
            peak_val = math.ceil(binedges_roi[peaks[0]])
            black_level = peak_val

    if not isinstance(img_path, Image.Image):
        # temp image, close
        image.close()

    white_level = 255
    if skip_white_check:
        return black_level, white_level, force_gray

    # White level detection: find the highest peak near white
    roi_min_w, roi_max_w = 195, 256
    roi_mask_w = (binedges[:-1] >= roi_min_w) & (binedges[:-1] < roi_max_w)
    hist_roi_w = hist[roi_mask_w]
    binedges_roi_w = binedges[:-1][roi_mask_w]
    if hist_roi_w.size == 0:
        return black_level, white_level, force_gray

    hist_padded_w = NumpyLib.pad(hist_roi_w, (1, 1), mode="constant", constant_values=0)
    peaks_padded_w, _ = ScipySignalLib.find_peaks(hist_padded_w, height=min_px_count, prominence=min_prominence)

    peaks_w = peaks_padded_w - 1
    valid_w = (peaks_w >= 0) & (peaks_w < len(hist_roi_w))
    peaks_w = peaks_w[valid_w]

    if len(peaks_w) > 0:
        peak_val_w = math.ceil(binedges_roi_w[peaks_w[-1]])
        white_level = peak_val_w
    else:
        # Find without max height constraint
        peaks_padded_w, _ = ScipySignalLib.find_peaks(hist_padded_w, height=0)

        peaks_w = peaks_padded_w - 1
        valid_w = (peaks_w >= 0) & (peaks_w < len(hist_roi_w))
        peaks_w = peaks_w[valid_w]
        if len(peaks_w) > 0:
            peak_val_w = math.ceil(binedges_roi_w[peaks_w[-1]])
            white_level = peak_val_w

    return black_level, white_level, force_gray


def gamma_correction(black_level: int) -> int:
    black_point = black_level / 255
    white_point = 1
    midpoint = 0.5
    gamma = math.log(midpoint) / math.log((midpoint - black_point) / (white_point - black_point))
    return round(1 / gamma, 2)


def create_magick_params(black_level: int, white_point: int, peak_offset: int = 0) -> str:
    gamma = gamma_correction(black_level)
    black_point_pct = round(black_level / 255 * 100, 2) + peak_offset
    white_point_pct = round(white_point / 255 * 100, 2)
    return f"{black_point_pct},{white_point_pct}%,{gamma}"


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


def analyze_gray_shades(image: Image.Image, threshold: float = 0.01) -> list[ShadeAnalysis]:
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
    hist, _ = NumpyLib.histogram(img_array, bins=256)

    pixel_thresh = math.ceil(total_pixels * (threshold / 100.0))

    # Find the shades (indices) that have more pixels than the threshold
    significant_indices = NumpyLib.where(hist > pixel_thresh)[0]

    # If no colors are significant, return an empty list
    if significant_indices.size == 0:
        return []

    # Get the counts for just the significant shades
    significant_counts = hist[significant_indices]

    filtered_shades = []
    for i in range(len(significant_indices)):
        count = significant_counts[i]
        pct = (count / total_pixels) * 100
        filtered_shades.append(
            {
                "shade": significant_indices[i],
                "percentage": pct,
            }
        )

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

    palette = image.quantize(colors=2**num_bits)
    quantized = image.quantize(colors=2**num_bits, palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
    return quantized


def posterize_image_with_imagemagick(
    img_path: Path,
    output_dir: Path,
    num_bits: int,
    magick_path: str = "magick",
) -> None:
    # use mogrify
    magick_cmd = [
        magick_path,
        "mogrify",
        "-format",
        "png",
        "+dither",
        "-posterize",
        str(2**num_bits),
        "-depth",
        str(num_bits),
        "-path",
        str(output_dir),
        str(img_path),
    ]

    sp.run(magick_cmd, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)


def npow2(num: int) -> int:
    return 1 if num == 0 else 2 ** (num - 1).bit_length()


def detect_nearest_bpc(shades: list[ShadeAnalysis]) -> int:
    num_shades = len(shades)
    if num_shades <= 1:
        return 1  # 1 bpc
    bpc = math.ceil(math.log2(num_shades))
    bitdepth = npow2(bpc)
    return bitdepth


def pad_shades_to_bpc(shades: list[ShadeAnalysis]) -> list[int]:
    """
    Pad the shades list to the closest bpc value.

    Accepts the output of analyze_gray_shades.
    Returns a list of gray shade values (0-255).
    """

    num_shades = len(shades)
    if num_shades <= 1:
        # Uhhhh, nothing to posterize
        shade_values = [shade_info["shade"] for shade_info in shades]
        return shade_values

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

    # Now calculate to the correct nearest bpc
    # PNG should support:
    # - 1bpc
    # - 2bpc
    # - 4bpc
    # - 8bpc
    nearest_bpc = min([1, 2, 4, 8], key=lambda x: abs(x - num_bits))
    target_shades: int = 2**nearest_bpc

    diffs = target_shades - num_shades
    if diffs < 0:
        # We cut down the top shades already
        cut_down = sorted(shades.copy(), key=lambda x: x["shade"])[:target_shades]
        return [shade_info["shade"] for shade_info in cut_down]

    # We need to pad the shades
    corrected = shades.copy()
    shade_values = [shade_info["shade"] for shade_info in corrected]
    shade_values.extend([255] * diffs)  # pad with white
    return shade_values


def posterize_image_by_shades(image: Image.Image, shades: list[int]) -> Image.Image:
    """
    Posterize an image to the specified gray shades.

    Shades is a list of gray shade values (0-255).
    """

    if image.mode != "L":
        image = image.convert("L")  # force grayscale

    palette: list[tuple[int, int, int]] = []
    for shade in shades:
        palette.extend([shade] * 3)  # R, G, B

    palette_img = Image.new("P", (1, 1))
    palette_img.putpalette(palette)

    dithered_img = image.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
    return dithered_img

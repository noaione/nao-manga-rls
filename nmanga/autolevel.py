"""
MIT License

Copyright (c) 2022-present noaione, anon

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

# Automatic level detection and grayscale posterization.

from __future__ import annotations

import math
import subprocess as sp
from contextlib import nullcontext
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from PIL import Image

from .lazy import get_numpy, get_scipy_signal
from .ogsov import is_grayscale_palette

if TYPE_CHECKING:
    from numpy import ndarray

__all__ = (
    "PosterizePointMethod",
    "analyze_gray_shades",
    "apply_levels",
    "create_magick_params",
    "detect_nearest_bpc",
    "find_local_peak",
    "find_local_peak_legacy",
    "gamma_correction",
    "pad_shades_to_bpc",
    "posterize_image_by_bits",
    "posterize_image_by_shades",
    "posterize_image_with_imagemagick",
)

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)
LLOYD_ITERATIONS = 40


class ShadeAnalysis(TypedDict):
    shade: int
    """The gray shade value in integer (0-255)"""
    percentage: float
    """The percentage of pixels in the image that have this shade."""


class PosterizePointMethod(str, Enum):
    EVEN = "even"
    """Even distribution of shades."""
    LLOYD = "lloyd"
    """Lloyd-Max quantization algorithm."""

    @classmethod
    def from_str(cls, type_str: str) -> PosterizePointMethod:
        type_str = type_str.lower()
        if type_str == "even":
            return cls.EVEN
        elif type_str == "lloyd":
            return cls.LLOYD
        else:
            raise ValueError(f"Unknown PosterizePointMethod string: {type_str}")

    def to_nimages(self):
        if self == PosterizePointMethod.EVEN:
            return 0
        elif self == PosterizePointMethod.LLOYD:
            return 1
        else:
            raise ValueError(f"Unknown PosterizePointMethod: {self}")


def _find_histogram_peak(hist: list[int], min_pix_count: int, min_prominence: int) -> int | None:
    np = get_numpy()
    scisignal = get_scipy_signal()

    # Zero padding makes the first and last shades eligible peaks.
    padded = np.zeros(len(hist) + 2, dtype=np.float64)
    padded[1:-1] = hist
    peaks, properties = scisignal.find_peaks(
        padded, height=min_pix_count, prominence=min_prominence if min_prominence > 0 else None
    )
    if peaks.size == 0:
        peaks, properties = scisignal.find_peaks(padded, height=0)
    if peaks.size == 0:
        return None

    # argmax keeps the lower shade when peak heights tie.
    return int(peaks[properties["peak_heights"].argmax()]) - 1


def find_local_peak(
    img_path: Path | BytesIO | Image.Image,
    *,
    upper_limit: int = 60,
    peak_percentage: float | None = 0.25,
    peak_prominence: float | None = None,
    skip_white_check: bool = False,
) -> tuple[int, int, bool]:
    """
    Determine black and white levels from the tallest local peaks in the grayscale histogram.

    If no peak passes the percentage and prominence thresholds, use the tallest peak anyway.
    An empty region defaults to black 0 or white 255. The final flag indicates whether the
    input needed grayscale conversion, not whether the thresholds passed.

    Courtesy of anon.

    :return: Black level, white level, and whether the input needed grayscale conversion.
    """

    image_context = nullcontext(img_path) if isinstance(img_path, Image.Image) else Image.open(img_path)
    with image_context as image:
        force_gray = image.mode != "L"
        if force_gray:
            gray_image = image.convert("L")
            try:
                hist = gray_image.histogram()
            finally:
                gray_image.close()
        else:
            hist = image.histogram()
        total_pixels = image.width * image.height

    if upper_limit < 0:
        return 0, 255, force_gray

    min_pix_count = math.ceil(total_pixels * (peak_percentage / 100.0)) if peak_percentage is not None else 0
    min_prominence = math.ceil(total_pixels * (peak_prominence / 100.0)) if peak_prominence is not None else 0

    black_peak = _find_histogram_peak(hist[: min(256, upper_limit + 1)], min_pix_count, min_prominence)
    black_peak_level = 0 if black_peak is None else black_peak
    if skip_white_check:
        return black_peak_level, 255, force_gray

    white_start = max(0, 255 - upper_limit)
    white_peak = _find_histogram_peak(hist[white_start:], min_pix_count, min_prominence)
    white_peak_level = 255 if white_peak is None else white_start + white_peak
    return black_peak_level, white_peak_level, force_gray


def find_local_peak_legacy(
    img_path: Path | BytesIO | Image.Image, *, upper_limit: int = 60, skip_white_peaks: bool = False
) -> tuple[int, int, bool]:
    """
    Automatically determine the optimal black level for an image by finding local peaks in its histogram.

    Returns a tuple of (black_level, img_path, force_gray).
    """

    np = get_numpy()
    scisignal = get_scipy_signal()

    image = img_path if isinstance(img_path, Image.Image) else Image.open(img_path)
    force_gray = image.mode != "L"

    if image.mode == "P":
        palette = image.getpalette()
        if (palette is not None) and is_grayscale_palette(palette):
            force_gray = True

    gray_img = image.convert("L")  # force grayscale

    img_array = np.array(gray_img)  # pyright: ignore[reportOptionalMemberAccess]
    hist, binedges = np.histogram(img_array, bins=256, range=(0, 255))  # pyright: ignore[reportOptionalMemberAccess]

    width = np.arange(1, stop=upper_limit, step=1)  # pyright: ignore[reportOptionalMemberAccess]
    result = scisignal.find_peaks_cwt(hist, widths=width)  # pyright: ignore[reportOptionalMemberAccess]

    if not isinstance(img_path, Image.Image):
        # temp image, close
        image.close()

    black_level = 0
    if result.any():
        if hist[result[0] - 1] > hist[result[0]]:
            result[0] = result[0] - 1
        elif hist[result[0] + 1] > hist[result[0]]:
            result[0] = result[0] + 1

        black_level = math.ceil(binedges[result[0]])

    if skip_white_peaks:
        return black_level, 255, force_gray

    # TODO: White level detection
    return black_level, 255, force_gray


def gamma_correction(black_level: int) -> int | float:
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

    np = get_numpy()

    if image.mode != "L":
        image = image.convert("L")  # force grayscale

    img_array = np.array(image)

    total_pixels = img_array.size
    hist, _ = np.histogram(img_array, bins=256)

    pixel_thresh = math.ceil(total_pixels * (threshold / 100.0))

    # Find the shades (indices) that have more pixels than the threshold
    significant_indices = np.where(hist > pixel_thresh)[0]

    # If no colors are significant, return an empty list
    if significant_indices.size == 0:
        return []

    # Get the counts for just the significant shades
    significant_counts = hist[significant_indices]

    filtered_shades = []
    for i in range(len(significant_indices)):
        count = significant_counts[i]
        pct = (count / total_pixels) * 100
        filtered_shades.append({
            "shade": significant_indices[i],
            "percentage": pct,
        })

    # Sort by highest percentage first
    filtered_shades.sort(key=lambda x: x["percentage"], reverse=True)
    return filtered_shades


def _lloyd_max_levels(img: Image.Image, colors: int) -> "ndarray":
    """Returns a 256-entry LUT mapping each gray value to one of `colors` levels, with 0 and 255 pinned."""
    np = get_numpy()
    hist = np.bincount(np.asarray(img).ravel(), minlength=256).astype(float)
    x = np.arange(256)
    levels = np.linspace(0, 255, colors)
    for _ in range(LLOYD_ITERATIONS):
        edges = np.concatenate([[-1], (levels[:-1] + levels[1:]) / 2, [256]])
        bucket = np.digitize(x, edges) - 1
        for j in range(1, colors - 1):
            mask = bucket == j
            if hist[mask].sum() > 0:
                levels[j] = (hist[mask] * x[mask]).sum() / hist[mask].sum()
    edges = np.concatenate([[-1], (levels[:-1] + levels[1:]) / 2, [256]])
    return np.round(levels[np.digitize(x, edges) - 1]).astype(np.uint8)


def posterize_image_by_bits(
    image: Image.Image, num_bits: int, *, method: PosterizePointMethod = PosterizePointMethod.EVEN
) -> Image.Image:
    """
    Posterize an image to the specified gray shades.

    This would return 2^num_bits shades.
    """

    if image.mode != "L":
        image = image.convert("L")  # force grayscale

    colors: int = 2**num_bits

    def _method_even(x: float) -> float:
        return round(x * (colors - 1) / 255) * 255 / (colors - 1)

    # Posterize without fucking up luminance
    if method == PosterizePointMethod.LLOYD:
        lut = _lloyd_max_levels(image, colors).tolist()
    else:
        lut = _method_even
    posterized = image.point(lut)
    # Maybe can be commented out? optimizing with pingo give similar ssimulacra2 scores
    quantized = posterized.quantize(colors=colors, dither=Image.Dither.NONE)
    return quantized.convert("L")


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

    palette: list[int] = []
    for shade in shades:
        palette.extend([shade] * 3)  # R, G, B

    palette_img = Image.new("P", (1, 1))
    palette_img.putpalette(palette)

    dithered_img = image.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
    return dithered_img

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

import subprocess as sp
from enum import Enum
from pathlib import Path
from typing import Sequence

from PIL import Image

from .file_handler import random_name

__all__ = (
    "SpreadDirection",
    "join_spreads",
    "join_spreads_imagemagick",
    "select_exts",
    "split_spreads",
)

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)


class SpreadDirection(str, Enum):
    """
    The direction of the spread to join images.
    """

    LTR = "ltr"
    """Spread direction from left to right."""
    RTL = "rtl"
    """Spread direction from right to left, or reversed."""


def select_exts(files: list[Path]) -> str:
    extensions = [x.suffix for x in files]
    select_ext = ".jpg"
    if ".png" in extensions:
        select_ext = ".png"
    # Check if only webp
    if all(".webp" in x for x in extensions):
        select_ext = ".webp"
    # Check if has webp and mix with other formats
    if ".webp" in extensions and select_ext != ".webp":
        select_ext = ".png"
    return select_ext


def join_spreads(
    images: Sequence[Image.Image], direction: SpreadDirection = SpreadDirection.LTR, gap: int = 0
) -> Image.Image:
    """Join images into spreads.

    Parameters
    ----------
    images: :class:`list` of :class:`PIL.Image.Image`
        The list of images to join.
    direction: :class:`SpreadDirection`
        The direction of the spread. Defaults to `SpreadDirection.LTR`.
    gap: :class:`int`
        The gap (in pixels) to add on each side of an image that touches another image.
        Between any two adjacent images the visible gap will be ``gap * 2``. Defaults to ``0``.

    Returns
    -------
    :class:`PIL.Image.Image`
        The joined image.
    """

    widths, heights = zip(*(i.size for i in images), strict=True)

    total_width = sum(widths) + (gap * 2 * (len(images) - 1))
    max_height = max(heights)

    # Get all the current image modes, force RGB if one of them is not "L" (also check if there's an alpha channel)
    modes = {im.mode for im in images}
    if len(modes) > 1 or (len(modes) == 1 and "L" not in modes):
        images = [im.convert("RGB") for im in images]
        mode = "RGB"
    else:
        mode = "L"

    bg_color = 255 if mode == "L" else (255, 255, 255)
    new_im = Image.new(mode, (total_width, max_height), color=bg_color)
    x_offset = 0
    ordered_images = images if direction == SpreadDirection.LTR else list(reversed(images))
    for idx, im in enumerate(ordered_images):
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
        if gap > 0 and idx != len(ordered_images) - 1:
            x_offset += gap * 2
    return new_im


def split_spreads(
    image: Image.Image, direction: SpreadDirection = SpreadDirection.LTR, gap: int = 0
) -> tuple[Image.Image, Image.Image]:
    """Split a spread image into two pages.

    Parameters
    ----------
    image: :class:`PIL.Image.Image`
        The spread image to split.
    direction: :class:`SpreadDirection`
        The order of the returned pages. Defaults to `SpreadDirection.LTR`.
    gap: :class:`int`
        The gap (in pixels) that was added on each side of the two joined images, i.e. the
        reverse of the ``gap`` used in :func:`join_spreads`. The ``gap * 2`` pixels in the
        middle of the image will be discarded. Defaults to ``0``.

    Returns
    -------
    tuple[:class:`PIL.Image.Image`, :class:`PIL.Image.Image`]
        The left and right pages, ordered according to ``direction``.
    """

    content_width = image.width - (gap * 2)
    split_at = content_width // 2
    left = image.crop((0, 0, split_at, image.height))
    right = image.crop((split_at + (gap * 2), 0, image.width, image.height))
    if direction == SpreadDirection.RTL:
        return right, left
    return left, right


def join_spreads_imagemagick(
    images: list[Path],
    output_directory: Path,
    quality: float = 100.0,
    direction: SpreadDirection = SpreadDirection.LTR,
    output_format: str = "auto",
    magick_path: str = "magick",
    gap: int = 0,
) -> str:
    """Join images into spreads using ImageMagick.

    Parameters
    ----------
    images: :class:`list` of :class:`pathlib.Path`
        The list of image paths to join.
    output_directory: :class:`pathlib.Path`
        The output directory to save the joined image.
    quality: :class:`float`
        The quality of the output image. Defaults to 100.0.
    direction: :class:`SpreadDirection`
        The direction of the spread. Defaults to `SpreadDirection.LTR`.
    output_format: :class:`str`
        The output format of the image. Defaults to "auto", which will select the best format based on input images.
        Supported formats are "jpg", "png", and "webp".
    magick_path: :class:`str`
        The path to the ImageMagick `magick` executable. Defaults to "magick".
    gap: :class:`int`
        The gap (in pixels) to add on each side of an image that touches another image.
        Between any two adjacent images the visible gap will be ``gap * 2``. Defaults to ``0``.

    Returns
    -------
    :class:`str`
        The output image path.
    """

    selected_extensions = select_exts(images) if output_format == "auto" else f".{output_format.lower()}"
    output_name = random_name(length=12) + selected_extensions

    images.sort(key=lambda x: x.name, reverse=direction == SpreadDirection.RTL)

    commands = [magick_path]
    commands.extend(str(x) for x in images)
    commands.extend(["-quality", f"{quality:.2f}%"])
    if gap > 0:
        commands.extend(["-background", "white", "+smush", str(gap * 2)])
    else:
        commands.append("+append")
    commands.append(f"{output_directory / output_name}")

    try:
        sp.run(commands, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    except sp.CalledProcessError as e:
        raise RuntimeError("Failed to join spreads using ImageMagick.") from e
    return output_name

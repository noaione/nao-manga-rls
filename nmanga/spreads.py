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

import subprocess as sp
from enum import Enum
from pathlib import Path

from PIL import Image

from .file_handler import random_name

__all__ = (
    "SpreadDirection",
    "join_spreads",
    "join_spreads_imagemagick",
    "select_exts",
)

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)


class SpreadDirection(str, Enum):
    RTL = "rtl"
    LTR = "ltr"


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


def join_spreads(images: list[Image.Image], direction: SpreadDirection = SpreadDirection.LTR) -> Image.Image:
    """Join images into spreads.

    Parameters
    ----------
    images: :class:`list` of :class:`PIL.Image.Image`
        The list of images to join.
    direction: :class:`SpreadDirection`
        The direction of the spread. Defaults to `SpreadDirection.LTR`.

    Returns
    -------
    :class:`PIL.Image.Image`
        The joined image.
    """

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    # Get all the current image modes, force RGB if one of them is not "L" (also check if there's an alpha channel)
    modes = {im.mode for im in images}
    if len(modes) > 1 or (len(modes) == 1 and "L" not in modes):
        images = [im.convert("RGB") for im in images]
        mode = "RGB"
    else:
        mode = "L"

    new_im = Image.new(mode, (total_width, max_height))
    x_offset = 0
    for im in images if direction == SpreadDirection.LTR else reversed(images):
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def join_spreads_imagemagick(
    images: list[Path],
    output_directory: Path,
    quality: float = 100.0,
    direction: SpreadDirection = SpreadDirection.LTR,
    output_format: str = "auto",
    magick_path: str = "magick",
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

    Returns
    -------
    :class:`str`
        The output image path.
    """

    selected_extensions = select_exts(images) if output_format == "auto" else f".{output_format.lower()}"
    output_name = random_name(length=12) + selected_extensions

    images.sort(key=lambda x: x.name, reverse=direction == SpreadDirection.RTL)

    commands = [magick_path]
    commands.extend(x.name for x in images)
    commands.extend("-quality", f"{quality:.2f}%", "+append", f"{output_directory / output_name}")

    try:
        sp.run(commands, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    except sp.CalledProcessError as e:
        raise RuntimeError("Failed to join spreads using ImageMagick.") from e
    return output_name

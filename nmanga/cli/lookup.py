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

# Collection of lookup commands

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import rich_click as click
from PIL import Image

from .. import file_handler, term
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)
console = term.get_console()


@click.group(
    name="lookup",
    help="Lookup information from files or directories",
)
def lookup_group():
    """Lookup information from files or directories"""
    pass


@lookup_group.command(
    name="imagesize",
    help="Lookup image sizes inside a manga archive or folder",
    cls=NMangaCommandHandler,
)
@options.path_or_archive()
@options.recursive
@time_program
def lookup_imagesize(path_or_archive: Path, recursive: bool):
    """
    Lookup image sizes inside a manga archive or folder
    """

    if recursive and not path_or_archive.is_dir():
        console.warning("The --recursive option is only applicable when a folder is given.")
        raise click.UsageError("The --recursive option is only applicable when a folder is given.")

    candidates: list[Path] = []
    if not recursive:
        console.info(f"Looking up image sizes in {path_or_archive}...")
        candidates.append(path_or_archive)
    else:
        console.info(f"Recursively looking up image sizes in {path_or_archive}...")
        for comic in file_handler.collect_all_comics(path_or_archive):
            candidates.append(comic)
        console.info(f"Found {len(candidates)} archives/folders to lookup.")

    if not candidates:
        console.warning("No valid archives or folders found to lookup.")
        return

    for candidate in candidates:
        if recursive:
            console.info(f"Processing: {candidate}")
        grouped_size: dict[str, list[str]] = {}
        with file_handler.MangaArchive(candidate) as archive:
            for image, _ in archive:
                img_read = archive.read(image)
                with Image.open(BytesIO(img_read)) as img:
                    img_size = f"{img.width}x{img.height}"
                    if img_size not in grouped_size:
                        grouped_size[img_size] = []
                    grouped_size[img_size].append(image.filename)

        console.info("Found the following image sizes:")
        for img_size, image_list in grouped_size.items():
            console.info(f" - {img_size}: {len(image_list)} images")
            if console.debugged:
                for img_name in image_list:
                    console.log(f"    - {img_name}")

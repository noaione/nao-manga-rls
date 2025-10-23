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

# Quickly do zero-index renaming of all images in a folder
from __future__ import annotations

from pathlib import Path

import rich_click as click

from .. import file_handler, term
from ..common import format_volume_text
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler

console = term.get_console()


@click.command(
    "shiftname",
    help="Quickly do renaming using padded start number",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-s",
    "--start",
    "start_index",
    type=options.ZERO_POSITIVE_INT,
    default=0,
    show_default=True,
    help="The starting index to rename the files to",
)
@click.option(
    "-r",
    "--reverse",
    "reverse",
    is_flag=True,
    help="Reverse the direction of renaming",
)
@options.manga_title_optional
@options.manga_volume
@time_program
def shift_renamer(
    path_or_archive: Path,
    start_index: int,
    reverse: bool,
    manga_title: str | None,
    manga_volume: int | float | None,
):
    """
    Quickly rename all images in a folder to a padded number starting from START_INDEX.
    """

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    volume_text = format_volume_text(manga_volume=manga_volume, manga_chapter=None)
    console.info(f"Bulk renaming images in {path_or_archive} with starting index {start_index}...")

    all_images: list[Path] = []
    for image_file, _, _, _ in file_handler.collect_image_from_folder(path_or_archive):
        all_images.append(image_file.resolve())

    all_images.sort(key=lambda x: x.stem, reverse=reverse)

    # we do the padding at minimum 3 digits, if page number has more than 3 digits we add additional padding
    total_files = len(all_images) + start_index
    padding = max(3, len(str(total_files + start_index - 1)))

    console.status(f"Renaming {len(all_images)} images...")
    remapped_names = set()
    should_revert = False
    total_rename = 0
    for idx, image_path in enumerate(all_images):
        console.status(f"Renaming images [{idx + 1}/{total_files}]...")
        new_name = f"p{str(start_index + idx).zfill(padding)}"

        manga_title_join = manga_title
        if volume_text is not None and manga_title_join:
            manga_title_join += f" - {volume_text}"

        if manga_title_join:
            # final: Manga Title - v01 - 001.ext
            new_name = f"{manga_title_join} - {new_name}"

        img_suffix = image_path.suffix.lower()
        new_path = image_path.with_name(new_name).with_suffix(img_suffix)
        remapped_names.add((image_path, new_path))
        if new_path.exists():
            console.warning("Conflict detected, reverting all changes...")
            console.log(f"Conflicting file: {new_path}")
            should_revert = True
            break
        image_path.rename(new_path)
        total_rename += 1

    if should_revert:
        console.stop_status(f"Renamed {total_rename} images, reverting...")
        console.log()
        console.status("Reverting all changes...")
        for old_path, new_path in remapped_names:
            if new_path.exists():
                new_path.rename(old_path)
        console.stop_status("Reverted all changes.")
        return 1

    console.stop_status(f"Renamed {total_rename} images successfully.")
    return 0

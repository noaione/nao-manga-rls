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

# Split volumes into chapters
# This utilize the filename inside the chapter
# to split everything apart

from __future__ import annotations

from os import path
from pathlib import Path
from typing import Dict, List, Optional

import click

from .. import exporter, file_handler, term, utils
from . import options
from ._deco import time_program
from .base import CatchAllExceptionsCommand, RegexCollection
from .common import check_cbz_exist, create_chapter

console = term.get_console()


@click.command(
    name="autosplit",
    help="Automatically split volumes into chapters using regex",
    cls=CatchAllExceptionsCommand,
)
@options.path_or_archive()
@click.option(
    "-t",
    "--title",
    "title",
    required=True,
    help="The title of the series (used on volume filename, and inner filename)",
)
@click.option(
    "-pub",
    "--publisher",
    "publisher",
    required=False,
    default=None,
    help="The publisher of the series (used on inner filename)",
)
@click.option(
    "-it",
    "--inner-title",
    "inner_title",
    required=False,
    default=None,
    help="The title of the series (used on inner filename, will override --title)",
)
@click.option(
    "-lt",
    "--limit-to",
    "limit_to_credit",
    required=False,
    default=None,
    help="Limit the volume regex to certain ripper only.",
)
@click.option(
    "-oshot",
    "--is-oneshot",
    "is_oneshot",
    is_flag=True,
    default=False,
    help="Mark the series as oneshot",
)
@time_program
def auto_split(
    path_or_archive: Path,
    title: str,
    publisher: Optional[str] = None,
    inner_title: Optional[str] = None,
    limit_to_credit: Optional[str] = None,
    is_oneshot: bool = False,
):
    """
    Automatically split volumes into chapters using regex
    """
    if inner_title is None:
        inner_title = title

    all_comic_files: Dict[str, Path] = {}
    parent_dir = path_or_archive
    volume_re = RegexCollection.volume_re(title, limit_to_credit)
    if file_handler.is_archive(path_or_archive):
        match_re = volume_re.match(path_or_archive.name)
        if not match_re:
            console.warning("Unable to match volume regex, falling back to v00...")
            volume_num = "00"
        else:
            volume_num = match_re.group(1)
        all_comic_files[volume_num] = path_or_archive
        parent_dir = path_or_archive.parent
    else:
        # Valid cbz/cbr/cb7 files
        for comic_file in file_handler.collect_all_comics(path_or_archive):
            if is_oneshot:
                if not all_comic_files:
                    next_key_data = "00"
                else:
                    next_keys = int(list(all_comic_files.keys())[-1]) + 1
                    next_key_data = f"{next_keys:02d}"
                console.info(f"Marking as oneshot (v{next_key_data}): {comic_file}")
                all_comic_files[next_key_data] = comic_file
                continue

            match_re = volume_re.match(comic_file.name)
            if not match_re:
                continue
            volume_num = match_re.group(1)
            if not volume_num:
                continue
            all_comic_files[volume_num] = comic_file
        if not all_comic_files:
            console.error("No valid comic files found with title provided!")
            return 1

    chapter_re = RegexCollection.chapter_re(inner_title, publisher)
    for volume, file_path in all_comic_files.items():
        console.info(f"[?] Processing: {file_path}")
        target_path = parent_dir / f"v{volume}"

        # Just need to mirror automatically.
        # Read to memory, then dump to disk
        collected_chapters: Dict[str, exporter.CBZMangaExporter] = {}
        skipped_chapters: List[str] = []
        with file_handler.MangaArchive(file_path) as archive:
            for image, _ in archive:
                filename = image.filename
                match_re = chapter_re.match(path.basename(filename))
                if not match_re:
                    console.error(f"[{volume}][!] Unable to match chapter: {filename}")
                    console.error(f"[{volume}][!] Exiting...")
                    return 1
                chapter_data = create_chapter(match_re, publisher is not None)
                if chapter_data in skipped_chapters:
                    continue

                if chapter_data not in collected_chapters:
                    if check_cbz_exist(target_path, utils.secure_filename(chapter_data)):
                        console.warning(f"[{volume}][?] Skipping chapter: {chapter_data}")
                        skipped_chapters.append(chapter_data)
                        continue
                    console.info(f"[{volume}][+] Creating chapter: {chapter_data}")
                    collected_chapters[chapter_data] = exporter.CBZMangaExporter(
                        utils.secure_filename(chapter_data), target_path
                    )

                image_bita = archive.read(image)
                collected_chapters[chapter_data].add_image(path.basename(filename), image_bita)

        for chapter, cbz_export in collected_chapters.items():
            console.info(f"[{volume}][+] Finishing chapter: {chapter}")
            cbz_export.close()
        console.enter()
    return 0

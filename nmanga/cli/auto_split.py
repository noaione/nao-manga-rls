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
from typing import Dict, Match, Optional

import click
from py7zr import FileInfo, SevenZipFile

from .. import exporter, file_handler, term, utils
from . import options
from .base import CatchAllExceptionsCommand, RegexCollection

console = term.get_console()


def create_chapter(match: Match[str], has_publisher: bool = False):
    chapter_num = int(match.group("ch"))
    chapter_extra = match.group("ex")
    chapter_vol = match.group("vol")
    if utils.is_oneshot(chapter_vol):
        chapter_vol = 0
    else:
        chapter_vol = int(chapter_vol[1:])

    chapter_title: Optional[str] = None
    try:
        chapter_title = match.group("title")
        if chapter_title is not None:
            chapter_title = utils.clean_title(chapter_title)
    except IndexError:
        pass

    chapter_data = f"{chapter_vol:02d}.{chapter_num:03d}"
    if chapter_extra is not None:
        chapter_data += f".{int(chapter_extra[1:]) + 4}"
    if chapter_title is not None:
        chapter_data += f" - {chapter_title}"
    if chapter_title is None and has_publisher and chapter_extra is not None:
        ch_ex = int(chapter_extra[1:])
        chapter_data += f" - Extra {ch_ex}"

    return chapter_data


@click.command(
    name="autosplit",
    help="Automatically split volumes into chapters using regex",
    cls=CatchAllExceptionsCommand,
)
@options.path_or_archive
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
    "-oshot",
    "--is-oneshot",
    "is_oneshot",
    is_flag=True,
    default=False,
    help="Mark the series as oneshot",
)
def auto_split(
    path_or_archive: Path,
    title: str,
    publisher: Optional[str] = None,
    inner_title: Optional[str] = None,
    is_oneshot: bool = False,
):
    """
    Automatically split volumes into chapters using regex
    """
    if inner_title is None:
        inner_title = title

    all_comic_files: Dict[str, Path] = {}
    parent_dir = path_or_archive
    volume_re = RegexCollection.volume_re(title)
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
        for image, handler, _, _ in file_handler.collect_image_archive(file_path):
            filename = image.filename
            match_re = chapter_re.match(path.basename(filename))
            if not match_re:
                console.error(f"[{volume}][!] Unable to match chapter: {filename}")
                console.error(f"[{volume}][!] Exiting...")
                return 1
            chapter_data = create_chapter(match_re, publisher is not None)
            if chapter_data not in collected_chapters:
                console.info(f"[{volume}][+] Creating chapter: {chapter_data}")
                collected_chapters[chapter_data] = exporter.CBZMangaExporter(
                    utils.secure_filename(chapter_data), target_path
                )

            in_img = image
            if isinstance(image, FileInfo):
                in_img = [image.filename]
            image_bita = handler.read(in_img)
            if isinstance(handler, SevenZipFile):
                handler.reset()
                image_bita = list(image_bita.values())[0].read()
            collected_chapters[chapter_data].add_image(path.basename(filename), image_bita)

        for chapter, cbz_export in collected_chapters.items():
            console.info(f"[{volume}][+] Finishing chapter: {chapter}")
            cbz_export.close()
        print()
    console.info("Done!")
    return 0

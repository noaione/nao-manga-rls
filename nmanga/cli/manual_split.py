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

from __future__ import annotations

import re
from os import path
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Union

import click

from .. import exporter, file_handler, term, utils
from . import options
from ._deco import time_program
from .base import CatchAllExceptionsCommand
from .common import (
    ChapterRange,
    PseudoChapterMatch,
    check_cbz_exist,
    create_chapter,
    inquire_chapter_ranges,
    safe_int,
)

console = term.get_console()


def extract_page_num(
    filename: str, custom_data: Dict[str, int] = {}, regex_data: Optional[Pattern[str]] = None
) -> List[int]:
    # Remove until pXXX
    if regex_data is not None:
        filename = re.sub(regex_data, r"\1-\2", filename)
    for pg_match, pg_num in custom_data.items():
        if pg_match in filename:
            return [pg_num]
    if filename.endswith("-"):
        filename = filename[:-1]
    name, _ = path.splitext(filename)
    try:
        first, second = name.split("-")
        return [int(first), int(second)]
    except ValueError:
        return [int(name)]


def coerce_number_page(number: Union[int, float]) -> str:
    if isinstance(number, int):
        return f"{number:03d}"
    base, floating = str(number).split(".")
    return f"{int(base):03d}.{floating}"


def _collect_custom_page():
    custom_data: Dict[str, int] = {}
    console.enter()
    console.info("Please input the custom page mapping:")
    while True:
        page = console.inquire("Page number", lambda y: safe_int(y) is not None)
        page_number = int(page)

        file_name = console.inquire("Page filename match", lambda y: len(y.strip()) > 0)
        custom_data[file_name] = page_number

        do_more = console.confirm("Do you want to add another naming?")
        if not do_more:
            break
    console.enter()
    return custom_data


def _collect_archive_to_chapters(
    target_path: Path,
    archive_file: Path,
    chapters_mapping: List[ChapterRange],
    volume_num: Optional[int] = None,
    custom_data: Dict[str, int] = {},
    regex_data: Optional[Pattern[str]] = None,
):
    console.info(f"Collecting chapters from {archive_file.name}")

    collected_chapters: Dict[str, exporter.CBZMangaExporter] = {}
    skipped_chapters: List[str] = []
    with file_handler.MangaArchive(archive_file) as archive:
        for image, _ in archive:
            filename = image.filename
            page_numbers = extract_page_num(path.basename(filename), custom_data, regex_data)

            first_page = page_numbers[0]
            selected_chapter: ChapterRange = None
            for chapter in chapters_mapping:
                if chapter.is_single:
                    if first_page >= chapter.range[0]:
                        selected_chapter = chapter
                        break
                else:
                    if first_page in chapter.range:
                        selected_chapter = chapter
                        break

            if selected_chapter is None:
                console.warning(f"Page {first_page} is not in any chapter ranges, skipping!")
                continue

            chapter_info = PseudoChapterMatch()
            as_bnum = selected_chapter.bnum.split("x", 1)
            chapter_info.set("ch", as_bnum[0])
            if len(as_bnum) > 1:
                chapter_info.set("ex", "x" + as_bnum[1])
            if volume_num is not None:
                chapter_info.set("vol", f"v{volume_num:02d}")
            if selected_chapter.name is not None:
                chapter_info.set("title", selected_chapter.name)

            chapter_data = create_chapter(chapter_info)
            if chapter_data in skipped_chapters:
                continue

            if chapter_data not in collected_chapters:
                if check_cbz_exist(target_path, utils.secure_filename(chapter_data)):
                    console.warning(f"[?] Skipping chapter: {chapter_data}")
                    skipped_chapters.append(chapter_data)
                    continue
                console.info(f"[+] Creating chapter: {chapter_data}")
                collected_chapters[chapter_data] = exporter.CBZMangaExporter(
                    utils.secure_filename(chapter_data), target_path
                )

            collected_chapters[chapter_data].add_image(path.basename(filename), archive.read(image))

    for chapter, cbz_export in collected_chapters.items():
        console.info(f"[+] Finishing chapter: {chapter}")
        cbz_export.close()
    console.enter()


def _handle_page_number_mode(archive_file: Path, volume_num: Optional[int], custom_mode_enabled: bool = False):
    console.info(f"Handling in page number mode (custom enabled? {custom_mode_enabled!r})")

    custom_data: Dict[str, int] = {}
    if custom_mode_enabled:
        custom_data = _collect_custom_page()

    has_ch_title = console.confirm("Does this volume have chapter titles?")
    split_chapter_ranges = inquire_chapter_ranges(
        "Please input information for each chapter",
        "Do you want to add another chapter?",
        has_ch_title,
    )

    parent_dir = archive_file.parent
    if volume_num is not None:
        TARGET_DIR = parent_dir / f"v{volume_num:02d}"
    else:
        TARGET_DIR = parent_dir / "v00"

    _collect_archive_to_chapters(TARGET_DIR, archive_file, split_chapter_ranges, volume_num, custom_data)


def _handle_regex_mode(archive_file: Path, volume_num: Optional[int], custom_mode_enabled: bool = False):
    console.info(f"Handling in regex mode (custom enabled? {custom_mode_enabled!r})")

    default_regex = r"p(?:([\d]{1,4})(?:-)?([\d]{1,4})?).*"
    regex_data = console.inquire("Enter regex", default=default_regex)

    custom_data: Dict[str, int] = {}
    if custom_mode_enabled:
        custom_data = _collect_custom_page()

    regex_compiled = re.compile(regex_data)
    has_ch_title = console.confirm("Does this volume have chapter titles?")
    split_chapter_ranges = inquire_chapter_ranges(
        "Please input information for each chapter",
        "Do you want to add another chapter?",
        has_ch_title,
    )

    parent_dir = archive_file.parent
    if volume_num is not None:
        TARGET_DIR = parent_dir / f"v{volume_num:02d}"
    else:
        TARGET_DIR = parent_dir / "v00"

    _collect_archive_to_chapters(
        TARGET_DIR, archive_file, split_chapter_ranges, volume_num, custom_data, regex_compiled
    )


@click.command(
    name="manualsplit",
    help="Manually split volumes into chapters using multiple modes",
    cls=CatchAllExceptionsCommand,
)
@options.path_or_archive(disable_folder=True)
@click.option(
    "-vol",
    "--volume",
    "volume_num",
    type=int,
    required=False,
    help="The volume number for the archive",
    default=None,
)
@time_program
def manual_split(path_or_archive: Path, volume_num: Optional[int] = None):
    """
    Manually split volumes into chapters using multiple modes
    """

    if path_or_archive.is_dir():
        console.warning("Directory split is not supported yet")
        return 1

    if not file_handler.is_archive(path_or_archive):
        console.warning("Provided path is not a valid archive!")
        return 1

    select_option = console.choice(
        "Select mode",
        choices=[
            term.ConsoleChoice("page_number", "Page number mode (all filename must be page number)"),
            term.ConsoleChoice("regex", "Regex mode (Enter regex that should atleast match the page number!)"),
            term.ConsoleChoice("page_number_and_custom", "Page number mode with custom page number mapping"),
            term.ConsoleChoice("regex_and_custom", "Regex mode with custom page number mapping"),
        ],
    )

    select_name = select_option.name
    if select_name.startswith("page_number"):
        _handle_page_number_mode(path_or_archive, volume_num, "_and_custom" in select_name)
    elif select_name.startswith("regex"):
        _handle_regex_mode(path_or_archive, volume_num, "_and_custom" in select_name)
    else:
        console.error("Unknown mode selected!")
        return 1

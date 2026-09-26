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
from pathlib import Path
from typing import Pattern

import rich_click as click

from .. import chapter_split, file_handler, term
from .._ntypes import VolumeNumberT
from ..chapter_split import ChapterSplitObserver, SplitReport
from ..common import (
    ChapterRange,
    format_volume_text,
    inquire_chapter_ranges,
    safe_int,
)
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler

console = term.get_console()


class _ConsoleSplitObserver(ChapterSplitObserver):
    """Forward chapter split progress events to the terminal console."""

    def on_chapter_start(self, chapter: str) -> None:
        console.info(f"[+] Creating chapter: {chapter}")

    def on_chapter_finish(self, chapter: str, target: Path) -> None:
        console.info(f"[+] Finishing chapter: {chapter}")

    def on_chapter_skip(self, chapter: str) -> None:
        console.warning(f"[?] Skipping chapter: {chapter}")

    def on_page_unassigned(self, page: int, filename: str) -> None:
        console.warning(f"Page {page} ({filename}) is not in any chapter ranges, skipping!")


def _collect_custom_page():  # pragma: no cover
    custom_data: dict[str, int] = {}
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


def _default_output_dir(source: Path, volume_num: VolumeNumberT | None) -> Path:
    """Derive the default output folder from the source path and its volume."""
    parent_dir = source.parent
    if volume_num is not None:
        volume_text = format_volume_text(manga_volume=volume_num)
        if volume_text is not None:
            return parent_dir / volume_text
    return parent_dir / "v00"


def _run_split(
    source: Path,
    target_dir: Path,
    chapters: list[ChapterRange],
    volume_num: VolumeNumberT | None,
    custom_data: dict[str, int] | None = None,
    regex_data: Pattern[str] | None = None,
    overwrite: bool = False,
) -> SplitReport:  # pragma: no cover
    console.info(f"Splitting {source.name} into chapter archives...")
    report = chapter_split.split_chapters(
        source,
        target_dir,
        chapters,
        volume=volume_num,
        custom_data=custom_data,
        regex_data=regex_data,
        overwrite=overwrite,
        observer=_ConsoleSplitObserver(),
    )

    console.enter()
    if report.pages_total < 1:
        console.warning(f"No image found in {source}!")
    elif report.created_chapters > 0:
        console.success(f"[+] Created {report.created_chapters} chapter archive(s) in {target_dir}")
    if report.skipped:
        console.warning(f"[?] Skipped {len(report.skipped)} chapter(s) that already exist")
    if report.has_unassigned:
        console.warning(f"[!] {len(report.unassigned_pages)} page(s) were not assigned to any chapter")
    return report


def _handle_page_number_mode(
    source: Path,
    target_dir: Path,
    volume_num: VolumeNumberT | None,
    custom_mode_enabled: bool = False,
    ask_volume: bool = False,
    overwrite: bool = False,
):  # pragma: no cover
    console.info(f"Handling in page number mode (custom enabled? {custom_mode_enabled!r})")

    custom_data: dict[str, int] = {}
    if custom_mode_enabled:
        custom_data = _collect_custom_page()

    has_ch_title = console.confirm("Does this volume have chapter titles?")
    split_chapter_ranges = inquire_chapter_ranges(
        "Please input information for each chapter",
        "Do you want to add another chapter?",
        has_ch_title,
        ask_volume=ask_volume,
        default_volume=volume_num,
    )

    _run_split(source, target_dir, split_chapter_ranges, volume_num, custom_data, None, overwrite)


def _handle_regex_mode(
    source: Path,
    target_dir: Path,
    volume_num: VolumeNumberT | None,
    custom_mode_enabled: bool = False,
    ask_volume: bool = False,
    overwrite: bool = False,
):  # pragma: no cover
    console.info(f"Handling in regex mode (custom enabled? {custom_mode_enabled!r})")

    default_regex = r"p(?:([\d]{1,4})(?:-)?([\d]{1,4})?).*"
    console.info("Only needed when the page number is neither a bare number nor the standard `p001`/`p001-002` naming.")
    regex_data = console.inquire("Enter regex", default=default_regex)

    custom_data: dict[str, int] = {}
    if custom_mode_enabled:
        custom_data = _collect_custom_page()

    regex_compiled = re.compile(regex_data)
    has_ch_title = console.confirm("Does this volume have chapter titles?")
    split_chapter_ranges = inquire_chapter_ranges(
        "Please input information for each chapter",
        "Do you want to add another chapter?",
        has_ch_title,
        ask_volume=ask_volume,
        default_volume=volume_num,
    )

    _run_split(source, target_dir, split_chapter_ranges, volume_num, custom_data, regex_compiled, overwrite)


@click.command(
    name="manualsplit",
    help="Manually split volumes into chapters using multiple modes",
    cls=NMangaCommandHandler,
)
@options.path_or_archive()
@click.option(
    "-vol",
    "--volume",
    "volume_num",
    type=options.VOLUME_NUMBER,
    required=False,
    help="The volume number for the source, use `1-2` for an omnibus range",
    default=None,
)
@click.option(
    "-pcv",
    "--per-chapter-volume",
    "per_chapter_volume",
    is_flag=True,
    default=False,
    help="Ask for a volume for each chapter range (useful for omnibus volumes)",
)
@options.dest_output(optional=True)
@options.force
@time_program
def manual_split(
    path_or_archive: Path,
    volume_num: VolumeNumberT | None = None,
    per_chapter_volume: bool = False,
    dest_output: Path | None = None,
    force: bool = False,
):  # pragma: no cover
    """
    Manually split volumes into chapters using multiple modes
    """

    if path_or_archive.is_file() and not file_handler.is_archive(path_or_archive):
        console.warning("Provided path is not a valid archive!")
        return 1

    if path_or_archive.is_dir() and not any(path_or_archive.glob("*")):
        console.warning("Provided folder is empty!")
        return 1

    target_dir = dest_output if dest_output is not None else _default_output_dir(path_or_archive, volume_num)

    select_option = console.choice(
        "Select mode",
        choices=[
            term.ConsoleChoice(
                "page_number",
                "Page number mode (bare `001`, or the standard `p001`/`p001-002` naming)",
            ),
            term.ConsoleChoice(
                "regex",
                "Regex mode (custom regex with two capture groups for the page number)",
            ),
            term.ConsoleChoice("page_number_and_custom", "Page number mode with custom page number mapping"),
            term.ConsoleChoice("regex_and_custom", "Regex mode with custom page number mapping"),
        ],
    )

    select_name = select_option.name
    if select_name.startswith("page_number"):
        _handle_page_number_mode(
            path_or_archive,
            target_dir,
            volume_num,
            "_and_custom" in select_name,
            per_chapter_volume,
            force,
        )
    elif select_name.startswith("regex"):
        _handle_regex_mode(
            path_or_archive,
            target_dir,
            volume_num,
            "_and_custom" in select_name,
            per_chapter_volume,
            force,
        )
    else:
        console.error("Unknown mode selected!")
        return 1

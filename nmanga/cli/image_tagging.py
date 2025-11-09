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

# Image tagging command
# This file is part of nmanga.
from __future__ import annotations

import subprocess as sp
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import rich_click as click
from click.exceptions import Exit

from nmanga import file_handler

from .. import term
from ..common import (
    ALLOWED_TAG_EXTENSIONS,
    format_archive_filename,
    format_volume_text,
    make_metadata_command,
    threaded_worker,
)
from ..constants import MangaPublication
from . import options
from ._deco import check_config_first, time_program
from .base import NMangaCommandHandler, is_executeable_global_path, test_or_find_exiftool

console = term.get_console()


def _threaded_tagging(
    log_q: term.MessageOrInterface,
    exiftool_exe: str,
    image_path: Path,
    archive_filename: str,
    rls_email: str,
) -> None:
    """Threaded helper for tagging images"""
    ext = image_path.suffix.lower().lstrip(".")
    if ext not in ALLOWED_TAG_EXTENSIONS:
        cnsl = term.with_thread_queue(log_q)
        cnsl.warning(f"Skipping unsupported image format for tagging: {image_path.name}")
        return
    base_cmd = make_metadata_command(exiftool_exe, archive_filename, rls_email)
    proc = sp.Popen(
        [*base_cmd, str(image_path)],
        stdout=sp.PIPE,
        stderr=sp.PIPE,
    )
    proc.wait()


def _threaded_tagging_star(args: tuple[term.MessageQueue, str, Path, str, str]) -> None:
    """Star wrapper for threaded tagging"""
    return _threaded_tagging(*args)


@click.command(
    name="tag",
    help="Tag images with metadata",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-t",
    "--title",
    "manga_title",
    required=True,
    help="The title of the series",
)
@options.manga_volume
@options.manga_chapter
@options.manga_year
@options.manga_publication_type()
@options.rls_credit
@options.rls_email
@options.rls_revision
@options.rls_extra_metadata
@options.use_bracket_type
@options.exiftool_path
@options.threads_alt
@check_config_first
@time_program
def image_tagging(
    path_or_archive: Path,
    manga_title: str,
    manga_volume: int | float | None,
    manga_chapter: int | float | None,
    manga_year: int,
    manga_publication_type: MangaPublication,
    rls_credit: str,
    rls_email: str,
    rls_revision: int,
    rls_extra_metadata: str | None,
    bracket_type: Literal["square", "round", "curly"],
    exiftool_path: str,
    threads: int,
):  # pragma: no cover
    """
    Tag images with metadata
    """

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    force_search = not is_executeable_global_path(exiftool_path, "exiftool")
    exiftool_exe = test_or_find_exiftool(exiftool_path, force_search)
    if exiftool_exe is None:
        console.error("Exiftool not found, unable to tag image with exif metadata!")
        raise Exit(1)

    current_pst = datetime.now(timezone(timedelta(hours=-8)))
    current_year = manga_year or current_pst.year

    volume_text = format_volume_text(manga_volume=manga_volume, manga_chapter=manga_chapter)
    if volume_text is None:
        raise click.BadParameter(
            "Please provide either a chapter or a volume number.",
            param_hint="manga_volume",
        )

    archive_filename = format_archive_filename(
        manga_title=manga_title,
        manga_year=current_year,
        publication_type=manga_publication_type,
        ripper_credit=rls_credit,
        bracket_type=bracket_type,
        manga_volume_text=volume_text,
        rls_revision=rls_revision,
        extra_metadata=rls_extra_metadata,
    )

    console.info("Tagging images with exif metadata...")
    progress = console.make_progress()

    precollect_images = [file_path for file_path, _, _, _ in file_handler.collect_image_from_folder(path_or_archive)]
    precollect_images.sort(key=lambda p: p.name)

    task = progress.add_task("Tagging images...", finished_text="Tagged images", total=len(precollect_images))

    if threads > 1:
        console.info(f"Using {threads} CPU threads for processing.")
        with threaded_worker(console, threads) as (pool, log_q):
            for _ in pool.imap_unordered(
                _threaded_tagging_star,
                ((log_q, exiftool_exe, image, archive_filename, rls_email) for image in precollect_images),
            ):
                progress.update(task, advance=1)
    else:
        for image_path in precollect_images:
            _threaded_tagging(console, exiftool_exe, image_path, archive_filename, rls_email)
            progress.update(task, advance=1)
    console.stop_progress(progress, "Tagged all possible images with exif metadata.")


@click.command(
    name="rawtag",
    help="Tag images with provided metadata",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-t",
    "--title",
    "manga_title",
    required=True,
    help="The title of the series",
)
@options.rls_email
@options.exiftool_path
@options.threads_alt
@check_config_first
@time_program
def image_tagging_raw(
    path_or_archive: Path,
    manga_title: str,
    rls_email: str,
    exiftool_path: str,
    threads: int,
):  # pragma: no cover
    """
    Tag images with anything provided by user.
    """

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    force_search = not is_executeable_global_path(exiftool_path, "exiftool")
    exiftool_exe = test_or_find_exiftool(exiftool_path, force_search)
    if exiftool_exe is None:
        console.error("Exiftool not found, unable to tag image with exif metadata!")
        raise Exit(1)

    console.info("Tagging images with exif metadata...")
    progress = console.make_progress()

    precollect_images = [file_path for file_path, _, _, _ in file_handler.collect_image_from_folder(path_or_archive)]
    precollect_images.sort(key=lambda p: p.name)

    task = progress.add_task("Tagging images...", finished_text="Tagged images", total=len(precollect_images))

    if threads > 1:
        console.info(f"Using {threads} CPU threads for processing.")
        with threaded_worker(console, threads) as (pool, log_q):
            for _ in pool.imap_unordered(
                _threaded_tagging_star,
                ((log_q, exiftool_exe, image, manga_title, rls_email) for image in precollect_images),
            ):
                progress.update(task, advance=1)
    else:
        for image_path in precollect_images:
            _threaded_tagging(console, exiftool_exe, image_path, manga_title, rls_email)
            progress.update(task, advance=1)
    console.stop_progress(progress, "Tagged all possible images with exif metadata.")

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

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional, Union

import click

from .. import config, term
from . import options
from ._deco import check_config_first, time_program
from .base import NMangaCommandHandler, is_executeable_global_path, test_or_find_exiftool
from .common import format_archive_filename, inject_metadata
from .constants import MangaPublication

console = term.get_console()
conf = config.get_config()
TARGET_TITLE = "{mt} {vol} ({year}) ({pt}) {cpa}{c}{cpb}"


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
@options.use_bracket_type
@options.exiftool_path
@check_config_first
@time_program
def image_tagging(
    path_or_archive: Path,
    manga_title: str,
    manga_volume: Optional[int],
    manga_chapter: Optional[Union[int, float]],
    manga_year: int,
    manga_publication_type: MangaPublication,
    rls_credit: str,
    rls_email: str,
    rls_revision: int,
    bracket_type: Literal["square", "round", "curly"],
    exiftool_path: str,
):
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
        raise click.exceptions.Exit(1)

    current_pst = datetime.now(timezone(timedelta(hours=-8)))
    current_year = manga_year or current_pst.year

    tag_sep = conf.defaults.ch_special_tag

    volume_text: Optional[str] = None
    if manga_chapter is not None:
        if isinstance(manga_chapter, float):
            float_string = str(manga_chapter)
            base_float, decimal_float = float_string.split(".")
            dec_float = int(decimal_float)
            if dec_float - 4 > 0:
                dec_float -= 4
            volume_text = f"{int(base_float):03d}{tag_sep}{dec_float}"
        else:
            volume_text = f"{manga_chapter:03d}"
        if conf.defaults.ch_add_c_prefix:
            volume_text = f"c{volume_text}"
    if manga_volume is not None:
        volume_text = f"v{manga_volume:02d}"
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
    )

    console.info("Tagging images with exif metadata...")
    inject_metadata(exiftool_exe, path_or_archive, archive_filename, rls_email)

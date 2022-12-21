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
from typing import Literal

import click

from .. import term
from . import options
from .base import CatchAllExceptionsCommand, test_or_find_exiftool
from .common import BRACKET_MAPPINGS, inject_metadata, time_program

console = term.get_console()
TARGET_TITLE = "{mt} {vol} ({year}) (Digital) {cpa}{c}{cpb}"


def _is_default_path(path: str) -> bool:
    path = path.lower()
    if path == "exiftool":
        return True
    if path == "./exiftool":
        return True
    if path == ".\\exiftool":
        return True
    return False


@click.command(
    name="tag",
    help="Tag images with metadata",
    cls=CatchAllExceptionsCommand,
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-t",
    "--title",
    "manga_title",
    required=True,
    help="The title of the series",
)
@click.option(
    "-vol",
    "--volume",
    "manga_volume",
    required=True,
    type=int,
    help="The volume of the series release",
)
@click.option(
    "-y",
    "--year",
    "manga_year",
    default=None,
    type=int,
    help="The year of the series release",
)
@click.option(
    "-c",
    "--credit",
    "rls_credit",
    help="The ripper credit for this series",
    show_default=True,
    default="nao",
)
@click.option(
    "-e",
    "--email",
    "rls_email",
    help="The ripper email for this series",
    show_default=True,
    default="noaione@protonmail.com",
)
@options.use_bracket_type
@options.exiftool_path
@time_program
def image_tagging(
    path_or_archive: str,
    manga_title: str,
    manga_volume: int,
    manga_year: int,
    rls_credit: str,
    rls_email: str,
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

    force_search = not _is_default_path(exiftool_path)
    exiftool_exe = test_or_find_exiftool(exiftool_path, force_search)
    if exiftool_exe is None:
        console.error("Exiftool not found, unable to tag image with exif metadata!")
        raise click.exceptions.Exit(1)

    current_pst = datetime.now(timezone(timedelta(hours=-8)))
    current_year = manga_year or current_pst.year

    pair_left, pair_right = BRACKET_MAPPINGS.get(bracket_type.lower(), BRACKET_MAPPINGS["square"])
    image_title = TARGET_TITLE.format(
        mt=manga_title,
        vol=f"{manga_volume:02d}",
        year=current_year,
        c=rls_credit,
        cpa=pair_left,
        cpb=pair_right,
    )
    console.info("Tagging images with exif metadata...")
    inject_metadata(exiftool_exe, path_or_archive, image_title, rls_email)

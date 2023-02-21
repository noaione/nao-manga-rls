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

import os
from pathlib import Path

import click


def path_or_archive(disable_archive: bool = False, disable_folder: bool = False):
    if disable_archive and disable_folder:
        raise click.UsageError("You can't disable both archive and folder")

    metavar = "path_or_archive_file"
    if disable_archive:
        metavar = "folder_path"
    elif disable_folder:
        metavar = "archive_file"

    return click.argument(
        "path_or_archive",
        metavar=metavar.upper(),
        required=True,
        type=click.Path(
            exists=True,
            resolve_path=True,
            file_okay=not disable_archive,
            dir_okay=not disable_folder,
            path_type=Path,
        ),
    )


class FloatIntParamType(click.ParamType):
    name = "int_or_float"

    def convert(self, value, param, ctx):
        if isinstance(value, int):
            return value

        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            self.fail(f"{value!r} is not a valid integer or floating type", param, ctx)


FLOAT_INT = FloatIntParamType()


archive_file = click.argument(
    "archive_file",
    metavar="ARCHIVE_FILE",
    required=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False, path_type=Path),
)
output_dir = click.option(
    "-o",
    "--output",
    "output_dirpath",
    type=click.Path(
        exists=True, resolve_path=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    default=os.getcwd(),
    help="Existing folder to write the output [default: The current directory]",
)
magick_path = click.option(
    "-me",
    "--magick-exec",
    "magick_path",
    default="magick",
    help="Path to the magick executable",
    show_default=True,
)
exiftool_path = click.option(
    "-ee",
    "--exiftool-exec",
    "exiftool_path",
    default="exiftool",
    help="Path to the exiftool executable",
    show_default=True,
)
pingo_path = click.option(
    "-pe",
    "--pingo-exec",
    "pingo_path",
    default="pingo",
    help="Path to the pingo executable",
    show_default=True,
)
debug_mode = click.option(
    "-v",
    "--verbose",
    "debug_mode",
    is_flag=True,
    default=False,
    help="Enable debug mode",
)
use_bracket_type = click.option(
    "-br",
    "--bracket-type",
    "bracket_type",
    default="square",
    help="Bracket to use to surround the ripper name",
    show_default=True,
    type=click.Choice(["square", "round", "curly"]),
)
manga_volume = click.option(
    "-vol",
    "--volume",
    "manga_volume",
    type=int,
    help="The volume of the series release",
    default=None,
)
manga_chapter = click.option(
    "-ch",
    "--chapter",
    "manga_chapter",
    type=FLOAT_INT,
    help="The chapter of the series release",
    default=None,
)
manga_ripper = click.option(
    "-ch",
    "--chapter",
    "manga_chapter",
    type=FLOAT_INT,
    help="The chapter of the series release",
    default=None,
)
manga_year = click.option(
    "-y",
    "--year",
    "manga_year",
    default=None,
    type=int,
    help="The year of the series release",
)
rls_credit = click.option(
    "-c",
    "--credit",
    "rls_credit",
    help="The ripper credit for this series",
    show_default=True,
    default="nao",
)
rls_email = click.option(
    "-e",
    "--email",
    "rls_email",
    help="The ripper email for this series",
    show_default=True,
    default="noaione@protonmail.com",
)
rls_revision = click.option(
    "-r",
    "--revision",
    "rls_revision",
    help="The revision of the release, if the number 1 provided it will not put in the filename",
    type=click.IntRange(min=1, max_open=True),
    default=1,
    show_default=True,
)

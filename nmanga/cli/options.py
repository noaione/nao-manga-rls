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

from pathlib import Path
from typing import Optional

import click

from ..config import get_config
from ..exporter import ExporterType
from .constants import MANGA_PUBLICATION_TYPES

config = get_config()


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


class MangaPublicationParamType(click.ParamType):
    name = "publication_type"

    def convert(self, value, param, ctx):
        if not isinstance(value, str):
            self.fail(f"{value!r} is not a valid string", param, ctx)

        pub_type = MANGA_PUBLICATION_TYPES.get(value)
        pub_keys_list = list(MANGA_PUBLICATION_TYPES.keys())
        pub_keys = "`" + "`, `".join(pub_keys_list) + "`"

        if pub_type is None:
            self.fail(f"{value!r} is not a valid publication type (must be either: {pub_keys})")
        return pub_type

    def get_metavar(self, param: "click.core.Parameter") -> Optional[str]:
        choices_str = "|".join(list(MANGA_PUBLICATION_TYPES.keys()))

        if param.required and param.param_type_name == "argument":
            return f"{{{choices_str}}}"
        return f"[{choices_str}]"


FLOAT_INT = FloatIntParamType()
PUBLICATION_TYPE = MangaPublicationParamType()


def manga_publication_type(chapter_mode: bool = False):
    default_arg = config.defaults.rls_pub_type
    if chapter_mode:
        default_arg = config.defaults.rls_ch_pub_type

    return click.option(
        "-pt",
        "--publication-type",
        "manga_publication_type",
        type=PUBLICATION_TYPE,
        help="The publication type for this series, use none to remove it from image filename",
        default=default_arg,
        show_default=True,
    )


archive_file = click.argument(
    "archive_file",
    metavar="ARCHIVE_FILE",
    required=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False, path_type=Path),
)
output_mode = click.option(
    "-m",
    "--mode",
    "output_mode",
    type=click.Choice(ExporterType),
    help="The output mode for the archive packing",
    default=ExporterType.cbz,
    show_default=True,
)
magick_path = click.option(
    "-me",
    "--magick-exec",
    "magick_path",
    default=config.executables.magick_path,
    help="Path to the magick executable",
    show_default=True,
)
exiftool_path = click.option(
    "-ee",
    "--exiftool-exec",
    "exiftool_path",
    default=config.executables.exiftool_path,
    help="Path to the exiftool executable",
    show_default=True,
)
pingo_path = click.option(
    "-pe",
    "--pingo-exec",
    "pingo_path",
    default=config.executables.pingo_path,
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
    default=config.defaults.bracket_type,
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
manga_title = click.option(
    "-t",
    "--title",
    "manga_title",
    required=True,
    help="The title of the series",
)
manga_publisher = click.option(
    "-pub",
    "--publisher",
    "manga_publisher",
    help="The publisher of the series",
    required=True,
)

rls_credit = click.option(
    "-c",
    "--credit",
    "rls_credit",
    help="The ripper credit for this series",
    show_default=True,
    default=config.defaults.ripper_credit,
)
rls_email = click.option(
    "-e",
    "--email",
    "rls_email",
    help="The ripper email for this series",
    show_default=True,
    default=config.defaults.ripper_email,
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

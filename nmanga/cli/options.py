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

from __future__ import annotations

import re
from multiprocessing import cpu_count
from pathlib import Path
from typing import Generator, Literal

import rich_click as click
from click.shell_completion import CompletionItem

from ..config import get_config
from ..constants import MANGA_PUBLICATION_TYPES
from ..exporter import ExporterType
from ..pdfs import PdfBoxExpansion

config = get_config()


class MarkerRange:
    def __init__(self, kind: Literal["odd", "even"]):
        self.kind = kind

    def with_total(self, total: int) -> Generator[int, None, None]:
        start_pg = 0 if self.kind == "odd" else 1
        for pg in range(start_pg, total, 2):
            yield pg

    def __str__(self):
        return self.kind

    def __repr__(self):
        return f"MarkerRange({self.kind})"

    def __eq__(self, other):
        return self.kind == other.kind

    def __hash__(self):
        return hash(("MarkerRange", self.kind))


class AnyPageRange:
    __slots__ = ("__markers", "__ranges")

    def __init__(self):
        self.__ranges: set[int] = set()
        self.__markers: set[MarkerRange] = set()  # so no duplicates

    def __repr__(self) -> str:
        return f"AnyPageRange({len(self.__ranges)!r} total, {self.__markers!r})"

    def add(self, thing: int | range | MarkerRange):
        if isinstance(thing, range):
            self.__ranges.update(thing)
        elif isinstance(thing, int):
            self.__ranges.add(thing)
        elif isinstance(thing, MarkerRange):
            self.__markers.add(thing)
        else:
            raise TypeError(f"Expected range or MarkerRange, got {type(thing)}")

    def iterate(self, total: int) -> Generator[int, None, None]:
        # sort everything
        pages: set[int] = set(self.__ranges)

        for marker in self.__markers:
            pages.update(marker.with_total(total))

        yield from sorted(pages)


def path_or_archive(
    disable_archive: bool = False, disable_folder: bool = False, *, param_name: str = "path_or_archive"
):
    if disable_archive and disable_folder:
        raise click.UsageError("You can't disable both archive and folder")

    metavar = "path_or_archive_file"
    if disable_archive:
        metavar = "folder_path"
    elif disable_folder:
        metavar = "archive_file"

    return click.argument(
        param_name,
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


class PositiveIntParamType(click.ParamType):
    name = "positive_int"

    def __init__(self, start_from_zero: bool = False):
        super().__init__()
        self.start_from_zero = start_from_zero

    def get_metavar(self, param: click.Parameter, ctx: click.Context) -> str | None:
        marker = "[x>=0]" if self.start_from_zero else "[x>0]"
        return f"INTEGER RANGE {marker}"

    def convert(self, value, param, ctx):
        try:
            value = int(value)
        except ValueError:
            self.fail(f"{value!r} is not a valid integer", param, ctx)

        min_val = 0 if self.start_from_zero else 1
        if value < min_val:
            self.fail(f"{value!r} is not a positive integer", param, ctx)
        return value


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

    def get_metavar(self, param: click.Parameter, ctx: click.Context) -> str | None:
        choices_str = "|".join(list(MANGA_PUBLICATION_TYPES.keys()))

        if param.required and param.param_type_name == "argument":
            return f"{{{choices_str}}}"
        return f"[{choices_str}]"

    def shell_complete(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> list[CompletionItem]:
        return [CompletionItem(key) for key in MANGA_PUBLICATION_TYPES.keys()]


class PdfBoxShorthand(click.ParamType):
    name = "pdf_box"

    def get_metavar(self, param: click.Parameter, ctx: click.Context) -> str | None:
        return "[x0 / y0 / x1 / y1]"

    def convert(self, value, param, ctx) -> PdfBoxExpansion:
        if isinstance(value, PdfBoxExpansion):
            return value

        raw = str(value).strip()

        if not raw:
            self.fail("Expected 1 to 4 numbers.", param, ctx)

        # Supports:
        #   "10"
        #   "10 20"
        #   "10 20 30"
        #   "10 20 30 40"
        #   "10/20"
        #   "10/20/30/40"
        #   "10 / 20 / 30 / 40"
        parts = re.split(r"(?:\s+|/)+", raw)

        try:
            values = [float(part) for part in parts if part]
        except ValueError:
            self.fail(
                f"Invalid box shorthand {value!r}. Expected 1 to 4 float numbers.",
                param,
                ctx,
            )

        if not 1 <= len(values) <= 4:
            self.fail(
                f"Invalid box shorthand {value!r}. Expected 1 to 4 numbers.",
                param,
                ctx,
            )

        top = values[0]

        if len(values) == 1:
            right = bottom = left = top
        elif len(values) == 2:
            right = left = values[1]
            bottom = top
        elif len(values) == 3:
            right = left = values[1]
            bottom = values[2]
        else:
            right = values[1]
            bottom = values[2]
            left = values[3]

        return PdfBoxExpansion(
            top=top,
            right=right,
            bottom=bottom,
            left=left,
        )


class PagesRange(click.ParamType):
    name = "pages_range"

    def get_metavar(self, param: click.Parameter, ctx: click.Context) -> str | None:
        return "1-5,10,odd,even,..."

    def convert(self, value, param, ctx) -> AnyPageRange:
        if not isinstance(value, str):
            self.fail(f"{value!r} is not a valid string", param, ctx)

        page_range = AnyPageRange()
        for part in value.split(","):
            part = part.strip().lower()
            if part == "odd":
                page_range.add(MarkerRange("odd"))
            elif part == "even":
                page_range.add(MarkerRange("even"))
            elif "-" in part:
                start, end = part.split("-", 1)
                start, end = int(start.strip()), int(end.strip())
                if start < 1:
                    self.fail(f"Invalid page range {part!r}. Start must be greater than 0", param, ctx)
                page_range.add(range(start - 1, end))
            else:
                start = int(part)
                if start < 1:
                    self.fail(f"Invalid page index {part!r}. Must be greater than 0", param, ctx)
                page_range.add(start - 1)
        return page_range


FLOAT_INT = FloatIntParamType()
POSITIVE_INT = PositiveIntParamType()
ZERO_POSITIVE_INT = PositiveIntParamType(start_from_zero=True)
PUBLICATION_TYPE = MangaPublicationParamType()
PDF_BOX = PdfBoxShorthand()
PAGES_RANGE = PagesRange()


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
        panel="Release Options",
    )


def half_cpu_threads():
    return max(cpu_count() // 2, 1)


CPU_THREADS = half_cpu_threads()


archive_file = click.argument(
    "archive_file",
    metavar="ARCHIVE_FILE",
    required=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False, path_type=Path),
)
recursive = click.option(
    "-r",
    "--recursive",
    "recursive",
    is_flag=True,
    default=False,
    help="Recursively lookup in subdirectories (only when a folder is given)",
    panel="Input Options",
)
force = click.option(
    "-f",
    "--force",
    "force",
    is_flag=True,
    default=False,
    help="Force overwrite or move existing files",
    panel="Input Options",
)
output_mode = click.option(
    "-m",
    "--mode",
    "output_mode",
    type=click.Choice(ExporterType),
    help="The output mode for the archive packing",
    default=ExporterType.cbz,
    panel="Archive Options",
    show_default=True,
)
compression_level = click.option(
    "-cl",
    "--compression-level",
    "compression_level",
    type=click.IntRange(min=0, max=9),
    help="The compression level to use for packing the archive (0-9)",
    default=7,
    show_default=True,
    panel="Archive Options",
)
magick_path = click.option(
    "-me",
    "--magick-exec",
    "magick_path",
    default=config.executables.magick_path,
    help="Path to the magick executable",
    panel="Executable Path",
    show_default=True,
)
exiftool_path = click.option(
    "-ee",
    "--exiftool-exec",
    "exiftool_path",
    default=config.executables.exiftool_path,
    help="Path to the exiftool executable",
    panel="Executable Path",
    show_default=True,
)
pingo_path = click.option(
    "-pe",
    "--pingo-exec",
    "pingo_path",
    default=config.executables.pingo_path,
    help="Path to the pingo executable",
    panel="Executable Path",
    show_default=True,
)
w2x_trt_path = click.option(
    "-w2exe",
    "--w2x-trt-exec",
    "w2x_trt_path",
    default=config.executables.w2x_trt_path,
    help="Path to the waifu2x-tensorrt executable",
    panel="Executable Path",
    show_default=True,
)
cjpegli_path = click.option(
    "-cje",
    "--cjpegli-exec",
    "cjpegli_path",
    default=config.executables.cjpegli_path,
    help="Path to the cjpegli executable",
    panel="Executable Path",
    show_default=True,
)
use_bracket_type = click.option(
    "-br",
    "--bracket-type",
    "bracket_type",
    default=config.defaults.bracket_type,
    help="Bracket to use to surround the ripper name",
    show_default=True,
    type=click.Choice(["square", "round", "curly"]),
    panel="Release Options",
)
manga_volume = click.option(
    "-vol",
    "--volume",
    "manga_volume",
    type=FLOAT_INT,
    help="The volume of the series release",
    default=None,
    panel="Release Options",
)
manga_chapter = click.option(
    "-ch",
    "--chapter",
    "manga_chapter",
    type=FLOAT_INT,
    help="The chapter of the series release",
    default=None,
    panel="Release Options",
)
manga_ripper = click.option(
    "-ch",
    "--chapter",
    "manga_chapter",
    type=FLOAT_INT,
    help="The chapter of the series release",
    default=None,
    panel="Release Options",
)
manga_year = click.option(
    "-y",
    "--year",
    "manga_year",
    default=None,
    type=int,
    help="The year of the series release",
    panel="Release Options",
)
manga_title = click.option(
    "-t",
    "--title",
    "manga_title",
    required=True,
    help="The title of the series",
    panel="Release Options",
)
manga_title_optional = click.option(
    "-t",
    "--title",
    "manga_title",
    required=False,
    help="The title of the series",
    panel="Release Options",
)
manga_publisher = click.option(
    "-pub",
    "--publisher",
    "manga_publisher",
    help="The publisher of the series",
    panel="Release Options",
    required=True,
)

rls_credit = click.option(
    "-c",
    "--credit",
    "rls_credit",
    help="The ripper credit for this series",
    show_default=True,
    default=config.defaults.ripper_credit,
    panel="Release Options",
)
rls_email = click.option(
    "-e",
    "--email",
    "rls_email",
    help="The ripper email for this series",
    show_default=True,
    default=config.defaults.ripper_email,
    panel="Release Options",
)
rls_revision = click.option(
    "-r",
    "--revision",
    "rls_revision",
    help="The revision of the release, if the number 1 provided it will not put in the filename",
    type=click.IntRange(min=1, max_open=True),
    default=1,
    show_default=True,
    panel="Release Options",
)
rls_extra_metadata = click.option(
    "-ex",
    "--extra-meta",
    "rls_extra_metadata",
    help="Extra metadata to add to the pack filename",
    default=None,
    required=False,
    panel="Release Options",
)

is_oneshot = click.option(
    "-oshot",
    "--is-oneshot",
    "is_oneshot",
    is_flag=True,
    default=False,
    help="Mark the series as oneshot, this will not add volume to the filename",
    panel="Release Options",
)

threads = click.option(
    "-t",
    "--threads",
    "threads",
    type=POSITIVE_INT,
    default=CPU_THREADS,
    show_default=True,
    help="The number of threads to use for processing",
    panel="Performance Options",
)
threads_alt = click.option(
    "-th",
    "--threads",
    "threads",
    type=POSITIVE_INT,
    default=CPU_THREADS,
    show_default=True,
    help="The number of threads to use for processing",
    panel="Performance Options",
)


def dest_output(file_okay: bool = False, dir_okay: bool = True, optional: bool = False):
    if not file_okay and not dir_okay:
        raise click.UsageError("You can't disable both file and folder")

    return click.option(
        "-o",
        "--output",
        "dest_output",
        type=click.Path(file_okay=file_okay, dir_okay=dir_okay, path_type=Path),
        required=not optional,
        help="The output directory/file to save the results",
        panel="Output Options",
    )

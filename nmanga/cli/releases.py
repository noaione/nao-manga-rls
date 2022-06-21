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

# Prepare manga for releasing to the public

import subprocess as sp
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import click

from .. import file_handler, term
from . import options
from .base import CatchAllExceptionsCommand, RegexCollection, test_or_find_exiftool

console = term.get_console()
TARGET_FORMAT = "{mt} - c{ch} ({vol}) - p{pg}{ex}[dig] [{t}] [{pb}] [{c}]"  # noqa
TARGET_FORMAT_ALT = "{mt} - c{ch} ({vol}) - p{pg}{ex}[dig] [{pb}] [{c}]"  # noqa
TARGET_TITLE = "{mt} {vol} ({year}) (Digital) [{c}]"


def _is_default_path(path: str) -> bool:
    path = path.lower()
    if path == "exiftool":
        return True
    if path == "./exiftool":
        return True
    if path == ".\\exiftool":
        return True
    return False


def safe_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        return None


def int_or_float(value: str) -> Optional[Union[int, float]]:
    if "." in value:
        try:
            return float(value)
        except ValueError:
            pass
    return safe_int(value)


def parse_ch_ranges(data: str) -> Tuple[List[int], bool]:
    split_range = data.split("-")
    if len(split_range) < 2:
        return [int(data)], True

    first, second = split_range
    return list(range(int(first), int(second) + 1)), False


def validate_ch_ranges(_, current: str):
    split_range = current.strip().split("-")
    if len(split_range) < 2:
        return safe_int(current) is not None

    first, second = split_range
    if not isinstance(second, str):
        return False

    first = safe_int(first)
    second = safe_int(second)
    if first is None or second is None:
        return False
    return True


class SimpleRange:
    def __init__(self, number: int, name: Optional[str], range: List[int], is_single: bool = False):
        self.number = number
        self.name = name
        self.range = range
        self.is_single = is_single

    def __repr__(self):
        if isinstance(self.number, float):
            return f"<SimpleRange c{self.number} - {self.name}>"
        return f"<SimpleRange c{self.number:03d} - {self.name}>"

    @property
    def bnum(self):
        if isinstance(self.number, int):
            return f"{self.number:03d}"
        base, floating = str(self.number).split(".")
        floating = int(floating)
        if floating - 4 >= 1:
            # Handle split chapter (.1, .2, etc)
            floating -= 4
        return f"{int(base):03d}x{floating}"


class SpecialNaming:
    def __init__(self, page: int, data: str):
        self.page = page
        self.data = data


def inject_metadata(exiftool_dir: str, current_directory: Path, image_title: str, image_email: str):
    base_cmd = [exiftool_dir]
    update_tags = {
        "XPComment": image_email,
        "Artist": image_email,
        "XPAuthor": image_email,
        "XPTitle": image_title,
        "ImageDescription": image_title,
        "Title": image_title,
        "Description": image_title,
    }
    for tag, value in update_tags.items():
        base_cmd.append(f"-{tag}={value}")
    base_cmd.append("-overwrite_original_in_place")
    full_dir = current_directory.resolve() / "*"
    base_cmd.append(str(full_dir))
    proc = sp.Popen(base_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    proc.wait()


@click.command(
    name="releases",
    help="Prepare a release of a manga series.",
    cls=CatchAllExceptionsCommand,
)
@options.path_or_archive
@click.option(
    "-t",
    "--title",
    "manga_title",
    required=True,
    help="The title of the series",
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
    "-pub",
    "--publisher",
    "publisher",
    help="The publisher of the series",
    required=True,
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
@click.option(
    "-hq",
    "--is-high-quality",
    "is_high_quality",
    is_flag=True,
    help="Whether this is a high quality release",
    default=False,
)
@options.exiftool_path
def prepare_releases(
    path_or_archive: Path,
    manga_title: str,
    manga_year: Optional[int],
    publisher: str,
    rls_credit: str,
    rls_email: str,
    is_high_quality: bool,
    exiftool_path: str,
):
    """
    Prepare a release of a manga series.
    """

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    current_pst = datetime.now(timezone(timedelta(hours=-8)))
    current_year = manga_year or current_pst.year

    force_search = not _is_default_path(exiftool_path)
    exiftool_exe = test_or_find_exiftool(exiftool_path, force_search)
    if exiftool_exe is None:
        console.warning("Exiftool not found, will skip tagging image with exif metadata!")

    has_ch_title = console.confirm("Does this release have chapter titles?")
    rls_information: List[SimpleRange] = []
    while True:
        console.info("Please input information regarding this release...")
        ch_title: Optional[str] = None
        if has_ch_title:
            ch_title = console.inquire("Chapter title", lambda _, y: len(y.strip()) > 0)

        ch_number = console.inquire("Chapter number", lambda _, y: int_or_float(y) is not None)
        ch_number = int_or_float(ch_number)

        ch_ranges = console.inquire("Chapter ranges (x-y or x)", validate_ch_ranges)
        actual_ranges, is_single = parse_ch_ranges(ch_ranges)
        simple_range = SimpleRange(ch_number, ch_title, actual_ranges, is_single)
        rls_information.append(simple_range)

        do_more = console.confirm("Do you want to add another release?")
        if not do_more:
            break

    special_naming: Dict[int, SpecialNaming] = {}
    do_special_get = console.confirm("Do you want to add some special naming?")
    if do_special_get:
        while True:
            page = console.inquire("Page number", lambda _, y: safe_int(y) is not None)
            page_number = int(page)

            data = console.inquire("Page type", lambda _, y: len(y.strip()) > 0)
            special_naming[page_number] = SpecialNaming(page_number, data)

            do_more = console.confirm("Do you want to add another naming?")
            if not do_more:
                break

    console.info("Preparing release...")
    console.info(f"Has {len(rls_information)} chapters")
    cmx_re = RegexCollection.cmx_re()
    current = 1
    console.info("Processing: 1/???")
    image_titling: Optional[str] = None
    for image, _, total_img, _ in file_handler.collect_image_from_folder(path_or_archive):
        console.status(f"Processing: {current}/{total_img}")
        title_match = cmx_re.match(image.name)
        if title_match is None:
            console.error("Unmatching file name: {}".format(image.name))
            return 1

        p01 = title_match.group("a")
        p01_copy = int(title_match.group("a"))
        p02 = title_match.group("b")
        vol = title_match.group("vol")
        if p02 is not None:
            p01 = f"{p01}-{p02}"

        # print(p01_copy)
        selected_range: SimpleRange = None
        for rls_info in rls_information:
            if rls_info.is_single:
                if p01_copy >= rls_info.range[0]:
                    selected_range = rls_info
                    break
            else:
                if p01_copy in rls_info.range:
                    selected_range = rls_info
                    break
        if selected_range is None:
            console.warning(f"Page {p01} are not included in any range, skipping!")
            current += 1
            continue
        extra_name = " "
        if p01_copy in special_naming:
            extra_name = f" [{special_naming[p01_copy].data}] "

        chapter_num = selected_range.bnum
        ch_title_name = selected_range.name
        extension = image.suffix

        final_filename = TARGET_FORMAT_ALT.format(
            mt=manga_title,
            ch=chapter_num,
            vol=vol,
            pg=p01,
            ex=extra_name,
            pb=publisher,
            c=rls_credit,
        )
        if has_ch_title:
            final_filename = TARGET_FORMAT.format(
                mt=manga_title,
                ch=chapter_num,
                vol=vol,
                pg=p01,
                ex=extra_name,
                t=ch_title_name,
                pb=publisher,
                c=rls_credit,
            )
        if is_high_quality:
            final_filename += r" {HQ}"

        if not image_titling:
            image_titling = TARGET_TITLE.format(
                mt=manga_title,
                vol=vol,
                year=current_year,
                c=rls_credit,
            )

        final_filename += extension
        new_name = image.parent / final_filename
        image.rename(new_name)
        current += 1
    console.stop_status()

    if exiftool_exe is not None:
        console.info("Tagging images with exif metadata...")
        inject_metadata(exiftool_exe, path_or_archive, image_titling, rls_email)
    console.info("Done!")

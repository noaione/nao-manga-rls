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

import subprocess as sp
import time
from pathlib import Path
from typing import Dict, List, Match, Optional, Tuple, Union

from .. import term, utils

__all__ = (
    "BRACKET_MAPPINGS",
    "PseudoChapterMatch",
    "ChapterRange",
    "check_cbz_exist",
    "actual_or_fallback",
    "create_chapter",
    "inquire_chapter_ranges",
    "safe_int",
    "time_program",
    "inject_metadata",
)


console = term.get_console()

BRACKET_MAPPINGS = {
    "square": ["[", "]"],
    "round": ["(", ")"],
    "curly": ["{", "}"],
}


class PseudoChapterMatch:
    def __init__(self):
        self._contents: Dict[str, str] = {}

    def set(self, key: str, value: str):
        self._contents[key] = value

    def get(self, key: str):
        return self._contents.get(key)

    def group(self, key: Union[str, int]):
        if isinstance(key, int):
            try:
                actual = list(self._contents.keys())[key]
            except IndexError:
                return None
            return self.get(actual)
        return self.get(key)


class ChapterRange:
    def __init__(self, number: int, name: Optional[str], range: List[int], is_single: bool = False):
        self.number = number
        self.name = name
        self.range = range
        self.is_single = is_single

    def __repr__(self):
        if isinstance(self.number, float):
            return f"<ChapterRange c{self.number} - {self.name}>"
        return f"<ChapterRange c{self.number:03d} - {self.name}>"

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


def check_cbz_exist(base_path: Path, filename: str):
    full_path = base_path / f"{filename}.cbz"
    return full_path.exists() and full_path.is_file()


def actual_or_fallback(actual_ch: Optional[str], chapter_num: int) -> str:
    proper_ch = f"{chapter_num:03d}"
    if not actual_ch:
        return proper_ch
    if "." in actual_ch:
        try:
            base, floating = str(float(actual_ch)).split(".", 1)
        except ValueError:
            return proper_ch
        try:
            return f"{int(base):03d}.{floating}"
        except ValueError:
            return proper_ch
    try:
        return f"{int(actual_ch):03d}"
    except ValueError:
        return proper_ch


def create_chapter(match: Union[Match[str], PseudoChapterMatch], has_publisher: bool = False):
    chapter_num = int(match.group("ch"))
    chapter_extra = match.group("ex")
    chapter_vol = match.group("vol")
    chapter_actual = match.group("actual")
    if chapter_vol is not None:
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

    act_ch_num = actual_or_fallback(chapter_actual, chapter_num)

    if chapter_vol is not None:
        chapter_data = f"{chapter_vol:02d}.{act_ch_num}"
    else:
        chapter_data = act_ch_num
    if chapter_extra is not None:
        add_num = int(chapter_extra[1:])
        if "." not in chapter_extra:
            add_num += 4
        chapter_data += f".{add_num}"
    if chapter_title is not None:
        chapter_data += f" - {chapter_title}"
    if chapter_title is None and has_publisher and chapter_extra is not None:
        ch_ex = int(chapter_extra[1:])
        if "." in chapter_extra:
            ch_ex -= 4
        chapter_data += f" - Extra {ch_ex}"

    return chapter_data


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


def validate_ch_ranges(current: str):
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


def inquire_chapter_ranges(
    initial_prompt: str, continue_prompt: str, ask_title: bool = False
) -> List[ChapterRange]:
    chapter_ranges: List[ChapterRange] = []
    while True:
        console.info(initial_prompt)

        ch_number = console.inquire("Chapter number", lambda y: int_or_float(y) is not None)
        ch_number = int_or_float(ch_number)

        ch_ranges = console.inquire("Chapter ranges (x-y or x)", validate_ch_ranges)
        actual_ranges, is_single = parse_ch_ranges(ch_ranges)

        ch_title: Optional[str] = None
        if ask_title:
            ch_title = console.inquire("Chapter title", lambda y: len(y.strip()) > 0)
        simple_range = ChapterRange(ch_number, ch_title, actual_ranges, is_single)
        chapter_ranges.append(simple_range)

        do_more = console.confirm(continue_prompt)
        if not do_more:
            break

    return chapter_ranges


def time_program(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        if isinstance(result, int) and result > 0:
            console.error(f"Failure! (Took {delta:.2f}s) [exit code {result}]")
        else:
            console.info(f"Done! (Took {delta:.2f}s)")
        return result

    return wrapper


def inject_metadata(exiftool_dir: str, current_directory: Path, image_title: str, image_email: str):
    resolve_dir = current_directory.resolve()
    any_jpg = len(list(resolve_dir.glob("*.jp[e]?g"))) > 0
    any_tiff = len(list(resolve_dir.glob("*.tiff"))) > 0
    if not any_jpg and not any_tiff:
        console.warning("No valid images found in directory, skipping metadata injection")
        return
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
    if any_jpg:
        full_dir = current_directory.resolve() / "*.jpg"
        base_cmd.append(str(full_dir))
        console.info("Injecting metadata into JP(e)G files...")
        proc = sp.Popen(base_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        proc.wait()
        base_cmd.pop()
    if any_tiff:
        full_dir = current_directory.resolve() / "*.tiff"
        base_cmd.append(str(full_dir))
        console.info("Injecting metadata into TIFF files...")
        proc = sp.Popen(base_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        proc.wait()

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

# Join spreads from a directory of images.

from dataclasses import dataclass
import re
import subprocess as sp
from os import path
from pathlib import Path
from shutil import move as mv
from typing import Dict, List, Optional, TypedDict

import click

from .. import file_handler, term
from . import options
from .base import CatchAllExceptionsCommand, RegexCollection, test_or_find_magick
from .common import time_program

console = term.get_console()
_SpreadsRe = re.compile(r"[\d]{1,3}(-[\d]{1,3}){1,}")


def _is_default_path(path: str) -> bool:
    path = path.lower()
    if path == "magick":
        return True
    if path == "./magick":
        return True
    if path == ".\\magick":
        return True
    return False


def make_prefix_convert(magick_exe: str):
    name = path.splitext(path.basename(magick_exe))[0]
    if name.lower() == "convert":
        return ["convert"]
    return ["magick", "convert"]


@dataclass
class _ExportedImage:
    path: Path
    prefix: Optional[str] = None
    postfix: Optional[str] = None


def execute_spreads_join(
    magick_dir: str, quality: float, input_imgs: List[_ExportedImage], out_dir: Path, reverse_mode: bool
):
    extensions = [x.path.suffix for x in input_imgs]
    select_ext = ".jpg"
    if ".png" in extensions:
        select_ext = ".png"
    output_name = file_handler.random_name() + select_ext
    execute_this = make_prefix_convert(magick_dir)
    input_imgs.sort(key=lambda x: x.path.name)
    if reverse_mode:
        input_imgs.reverse()
    execute_this += list(map(str, input_imgs))
    execute_this += ["-quality", f"{quality:.2f}%", "+append", f"{out_dir / output_name}"]
    try:
        sp.run(execute_this, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    except sp.CalledProcessError as e:
        console.error(f"Error: {e.output.decode('utf-8')}")
        raise e
    return output_name


class _ExportedImages(TypedDict):
    imgs: List[_ExportedImage]
    pattern: List[int]


@click.command(
    name="spreads", help="Join multiple spreads into a single image", cls=CatchAllExceptionsCommand
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-q",
    "--quality",
    "quality",
    default=100.0,
    show_default=True,
    type=click.FloatRange(1.0, 100.0),
    help="The quality of the output image",
)
@click.option(
    "-s",
    "--spreads",
    "spreads_data",
    required=True,
    multiple=True,
    help="The spread information, can be repeated and must contain something like: 1-2",
    metavar="A-B",
)
@click.option(
    "-r",
    "--reverse",
    "reverse",
    is_flag=True,
    default=False,
    help="Reverse the order of the spreads (manga mode)",
)
@options.magick_path
@time_program
def spreads_join(
    path_or_archive: Path,
    quality: float,
    spreads_data: List[str],
    reverse: bool,
    magick_path: str,
):
    """
    Join multiple spreads into a single image
    """
    force_search = not _is_default_path(magick_path)
    magick_exe = test_or_find_magick(magick_path, force_search)
    if magick_exe is None:
        console.error("Could not find the magick executable")
        return 1
    console.info("Using magick executable: {}".format(magick_exe))

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    # Validate spreads data
    spreads_data = [x.strip() for x in spreads_data]
    valid_spreads_data: Dict[str, List[int]] = {}
    for idx, spread in enumerate(spreads_data):
        matched_data = _SpreadsRe.match(spread)
        if not matched_data:
            console.error(f"Invalid spread data: {spread}")
            return 1
        matched_data = matched_data.group(0)
        matched_data = matched_data.split("-")
        matched_data = [int(x) for x in matched_data]
        valid_spreads_data[f"spread_{idx}"] = matched_data

    page_re = RegexCollection.page_re()

    exported_imgs: Dict[str, _ExportedImages] = {
        x: {"imgs": [], "pattern": y} for x, y in valid_spreads_data.items()
    }
    console.info("Collecting image for spreads...")
    with file_handler.MangaArchive(path_or_archive) as archive:
        for image, _ in archive:
            title_match = page_re.match(image.stem)

            if title_match is None:
                console.error("Unmatching file name: {}".format(image.filename))
                return 1

            a_part = title_match.group("a")
            b_part = title_match.group("b")
            prefix_text = title_match.group("any")
            postfix_text = title_match.group("anyback")
            if b_part:
                continue
            a_part = int(a_part)
            for spd, spreads in valid_spreads_data.items():
                if a_part in spreads:
                    im_data = _ExportedImage(image.access(), prefix_text, postfix_text)
                    exported_imgs[spd]["imgs"].append(im_data)

    total_match_spread = len(list(exported_imgs.keys()))
    current = 1
    for spread, imgs in exported_imgs.items():
        console.status(f"Joining spreads: {current}/{total_match_spread}")
        temp_output = execute_spreads_join(magick_exe, quality, imgs["imgs"], path_or_archive, reverse)
        # Rename back
        pattern = imgs["pattern"]
        pattern.sort()
        first_val = pattern[0]
        last_val = pattern[-1]
        first_img = imgs["imgs"][0]
        pre_t = first_img.prefix or ""
        post_t = first_img.postfix or ""

        extension = path.splitext(temp_output)[1]
        final_filename = f"{pre_t}p{first_val:03d}-{last_val:03d}{post_t}"
        final_filename += extension
        final_path = path_or_archive / final_filename
        temp_output_path = path_or_archive / temp_output
        temp_output_path.rename(final_path)
        current += 1
    console.stop_status(f"Joined {total_match_spread} spreads")

    BACKUP_DIR = path_or_archive / "backup"
    BACKUP_DIR.mkdir(exist_ok=True)
    console.info("Backing up old files to: {}".format(BACKUP_DIR))
    for img_data in exported_imgs.values():
        for image in img_data["imgs"]:
            try:
                mv(image, BACKUP_DIR / path.basename(image.name))
            except FileNotFoundError:
                pass

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

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from shutil import move as mv
from typing import TypedDict, cast

import rich_click as click
from PIL import Image

from .. import file_handler, term
from ..common import RegexCollection, lowest_or, threaded_worker
from ..spreads import SpreadDirection, join_spreads, join_spreads_imagemagick, select_exts, split_spreads
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler, test_or_find_magick

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


@dataclass
class _ExportedImage:
    path: Path
    prefix: str | None = None
    postfix: str | None = None


class _ExportedImages(TypedDict):
    imgs: list[_ExportedImage]
    pattern: list[int]


@dataclass
class _SplitSpreads:
    img: _ExportedImage
    a_part: int
    b_part: int


def _split_output_extension(image_path: Path, image_fmt: str) -> str:
    if image_fmt != "auto":
        return f".{image_fmt.lower()}"
    if image_path.suffix.lower() in {".png", ".webp"}:
        return ".png"
    return ".jpg"


def _save_split_image(image: Image.Image, output_path: Path, quality: float) -> None:
    save_image = image
    if output_path.suffix.lower() in {".jpg", ".jpeg"} and image.mode not in {"1", "L", "RGB", "CMYK"}:
        save_image = image.convert("RGB")
    try:
        save_kwargs = {"quality": int(quality)} if output_path.suffix.lower() in {".jpg", ".jpeg"} else {}
        save_image.save(output_path, format=output_path.suffix[1:].upper(), **save_kwargs)
    finally:
        if save_image is not image:
            save_image.close()


def _runner_spreads_split(
    split_spread: _SplitSpreads,
    output_dir: Path,
    quality: float,
    image_fmt: str,
    direction: SpreadDirection,
    gap: int,
) -> None:
    image_data = split_spread.img
    output_extension = _split_output_extension(image_data.path, image_fmt)
    first_val = split_spread.a_part
    second_val = split_spread.b_part
    prefix = image_data.prefix or ""
    postfix = image_data.postfix or ""
    first_path = output_dir / f"{prefix}p{first_val:03d}{postfix}{output_extension}"
    second_path = output_dir / f"{prefix}p{second_val:03d}{postfix}{output_extension}"

    with Image.open(image_data.path) as image:
        first_image, second_image = split_spreads(image, direction, gap=gap)
    try:
        _save_split_image(first_image, first_path, quality)
        _save_split_image(second_image, second_path, quality)
    finally:
        first_image.close()
        second_image.close()


def _runner_spreads_split_star(
    args: tuple[_SplitSpreads, Path, float, str, SpreadDirection, int],
) -> None:
    _runner_spreads_split(*args)


quality_option = click.option(
    "-q",
    "--quality",
    "quality",
    default=100.0,
    show_default=True,
    type=click.FloatRange(1.0, 100.0),
    help="The quality of the output image",
)
reverse_direction = click.option(
    "-r",
    "--reverse",
    "reverse",
    is_flag=True,
    default=False,
    help="Reverse the order of the spreads (manga mode)",
)
format_output = click.option(
    "-f",
    "--format",
    "image_fmt",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "png", "jpg"]),
    help="The format of the output image, auto will detect the format from the input images",
)
gap_option = click.option(
    "-g",
    "--gap",
    "gap",
    default=0,
    show_default=True,
    type=click.IntRange(0),
    help=(
        "The gap (in pixels) to add on each side of an image that touches another image. "
        "The visible gap between two adjacent images will be twice this value."
    ),
)


@click.group(name="spreads", help="Manage spreads from a directory of images")
def spreads():
    pass


@spreads.command(name="join", help="Join multiple spreads into a single image", cls=NMangaCommandHandler)
@options.path_or_archive(disable_archive=True)
@quality_option
@click.option(
    "-s",
    "--spreads",
    "spreads_data",
    required=True,
    multiple=True,
    help="The spread information, can be repeated and must contain something like: 1-2",
    metavar="A-B",
)
@reverse_direction
@format_output
@gap_option
@click.option(
    "--use-pil",
    "use_pil",
    is_flag=True,
    default=False,
    help="Use PIL to join images instead of ImageMagick (may use more memory)",
)
@options.magick_path
@time_program
def spreads_join(
    path_or_archive: Path,
    quality: float,
    spreads_data: list[str],
    reverse: bool,
    image_fmt: str,
    gap: int,
    use_pil: bool,
    magick_path: str,
):
    """
    Join multiple spreads into a single image
    """
    force_search = not _is_default_path(magick_path)
    magick_exe = test_or_find_magick(magick_path, force_search)
    if magick_exe is None and not use_pil:
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
    valid_spreads_data: dict[str, list[int]] = {}
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

    exported_imgs: dict[str, _ExportedImages] = {x: {"imgs": [], "pattern": y} for x, y in valid_spreads_data.items()}
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
                    im_data = _ExportedImage(cast(Path, image.access()), prefix_text, postfix_text)
                    exported_imgs[spd]["imgs"].append(im_data)

    total_match_spread = len(list(exported_imgs.keys()))
    current = 1

    direction = SpreadDirection.RTL if reverse else SpreadDirection.LTR

    for imgs in exported_imgs.values():
        console.status(f"Joining spreads: {current}/{total_match_spread}")

        # Rename back
        pattern = imgs["pattern"]
        pattern.sort()
        first_val = pattern[0]
        last_val = pattern[-1]
        first_img = imgs["imgs"][0]
        pre_t = first_img.prefix or ""
        post_t = first_img.postfix or ""

        final_filename = f"{pre_t}p{first_val:03d}-{last_val:03d}{post_t}"

        all_img_paths = [x.path for x in imgs["imgs"]]
        if use_pil:
            # Load all images
            loaded_images = [Image.open(p) for p in all_img_paths]

            joined_image = join_spreads(loaded_images, direction, gap=gap)
            extension = select_exts(all_img_paths)
            if image_fmt != "auto":
                extension = f".{image_fmt}"
            final_filename += extension

            joined_image.save(path_or_archive / final_filename, quality=int(quality))
        else:
            temp_output = join_spreads_imagemagick(
                all_img_paths,
                output_directory=path_or_archive,
                quality=quality,
                direction=direction,
                output_format=image_fmt,
                magick_path=cast(str, magick_exe),
                gap=gap,
            )
            extension = Path(temp_output).suffix

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
                mv(image.path, BACKUP_DIR / Path(image.path.name).name)
            except FileNotFoundError:
                pass


@spreads.command(name="split", help="Split a joined spreads into two images", cls=NMangaCommandHandler)
@options.path_or_archive(disable_archive=True)
@quality_option
@reverse_direction
@format_output
@gap_option
@options.threads
@time_program
def spreads_split(
    path_or_archive: Path,
    quality: float,
    reverse: bool,
    image_fmt: str,
    gap: int,
    threads: int,
):
    """
    Split a joined spreads into two images
    """
    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    image_list: list[_SplitSpreads] = []
    page_re = RegexCollection.page_re()
    console.info("Collecting image for spreads...")
    with file_handler.MangaArchive(path_or_archive) as archive:
        for image, _ in archive:
            title_match = page_re.match(image.stem)

            if title_match is None:
                console.warning("Unmatching file name: {}".format(image.filename))
                continue

            a_part = title_match.group("a")
            b_part = title_match.group("b")
            prefix_text = title_match.group("any")
            postfix_text = title_match.group("anyback")
            if not b_part:
                continue
            a_part = int(a_part)
            b_part = int(b_part)
            im_data = _ExportedImage(cast(Path, image.access()), prefix_text, postfix_text)
            split_spread = _SplitSpreads(img=im_data, a_part=a_part, b_part=b_part)
            image_list.append(split_spread)
    console.info(f"Found {len(image_list)} spreads to split")

    if image_list:
        direction = SpreadDirection.RTL if reverse else SpreadDirection.LTR
        progress = console.make_progress()
        task = progress.add_task("Splitting spreads...", finished_text="Split spreads", total=len(image_list))
        console.info(f"Using {threads} CPU threads for processing.")
        with threaded_worker(console, lowest_or(threads, image_list)) as (pool, _):
            for _ in pool.imap_unordered(
                _runner_spreads_split_star,
                (
                    (split_spread, path_or_archive, quality, image_fmt, direction, gap)
                    for split_spread in image_list
                ),
            ):
                progress.update(task, advance=1)
        console.stop_progress(progress, f"Split {len(image_list)} spreads")

    BACKUP_DIR = path_or_archive / "backup"
    BACKUP_DIR.mkdir(exist_ok=True)
    for image in image_list:
        try:
            mv(image.img.path, BACKUP_DIR / image.img.path.name)
        except FileNotFoundError:
            pass

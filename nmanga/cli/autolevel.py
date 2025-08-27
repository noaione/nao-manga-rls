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

# Automatically adjust the levels of images in a directory using ImageMagick
# Recommended only for images that is grayscaled

from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import subprocess
from pathlib import Path
from typing import List

import click

from nmanga import file_handler

from .. import term
from ..autolevel import create_magick_params, find_local_peak
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler, test_or_find_magick

console = term.get_console()


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
    name = Path(magick_exe).name
    if name.lower() == "convert":
        return ["convert"]
    return ["magick", "convert"]


def _autolevel_exec(command: List[str]) -> None:
    output = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # check exit code
    if output.returncode != 0:
        console.error(f"Command {' '.join(command)} failed with exit code {output.returncode}")


def determine_image_format(img_path: Path, prefer: str) -> str:
    if prefer != "auto":
        return f".{prefer}"

    ext = img_path.suffix.lower()
    if not ext.startswith("."):
        return f".{ext}"
    return ext


@click.command(
    name="autolevel",
    help="Automatically adjust the levels of images in a directory using ImageMagick",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-o",
    "--output",
    "dest_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The output directory to save the processed images",
)
@click.option(
    "-ul",
    "--upper-limit",
    "upper_limit",
    type=click.IntRange(1, 255),
    default=60,
    show_default=True,
    help="The upper limit for finding local peaks in the histogram",
)
@click.option(
    "-po",
    "--peak-offset",
    "peak_offset",
    type=click.IntRange(-100, 100),
    default=0,
    show_default=True,
    help="The offset to add to the detected black level percentage",
)
@click.option(
    "-f",
    "--format",
    "image_fmt",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "png", "jpg"]),
    help="The format of the output image, auto will detect the format from the input images",
)
@options.threads
@options.magick_path
@time_program
def autolevel(
    path_or_archive: Path,
    dest_dir: Path,
    upper_limit: int,
    peak_offset: int,
    image_fmt: str,
    threads: int,
    magick_path: str,
):  # pragma: no cover
    """
    Automatically adjust the levels of all images in a directory based on local peaks in their histograms.
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

    with file_handler.MangaArchive(path_or_archive) as archive:
        all_files: list[Path] = archive.contents()

    console.info(f"Found {len(all_files)} files in the directory.")

    is_continue = console.confirm("Proceed with autoleveling calculation")
    if not is_continue:
        console.info("Aborting autolevel.")
        return 0

    console.status("Calculating black levels for images...")
    with mp.Pool(threads) as pool:
        results = pool.starmap(find_local_peak, [(file, upper_limit) for file in all_files])
    console.stop_status("Calculated black levels for images.")

    commands: List[str] = []
    to_be_copied: List[Path] = []
    magick_cmd: List[str] = make_prefix_convert(magick_exe)

    console.info(f"Saving processed images to: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    dumped_data_temp = []
    for black_level, img_path, force_gray in results:
        dumped_data_temp.append(
            {
                "image": str(img_path),
                "black_level": black_level,
                "force_gray": force_gray,
            }
        )

    # Dump the results for debugging
    dumped_data = json.dumps(dumped_data_temp, indent=4)
    current_cwd = Path.cwd()
    dump_path = current_cwd / "autolevel_debug.json"
    dump_path.write_text(dumped_data, encoding="utf-8")
    console.info(f"Dumped autolevel debug data to: {dump_path}")

    # Pre-compute all the image magick commands
    total_results = len(results)
    for idx, (black_level, img_path, force_gray) in enumerate(results):
        console.status(f"Preparing autolevel commands [{idx + 1}/{total_results}]...")
        if black_level == 0:
            # Skip images that don't need autolevel
            to_be_copied.append(img_path)
            continue
        if black_level > upper_limit:
            # Skip images that are too bright
            to_be_copied.append(img_path)
            continue

        params = create_magick_params(black_level, peak_offset)
        image_name = img_path.stem + determine_image_format(img_path, image_fmt)
        dest_path = dest_dir / image_name

        cmd = [
            *magick_cmd,
            str(img_path),
            "-alpha",
            "off",
        ]
        if force_gray:
            cmd.extend(["-colorspace", "Gray"])
        cmd.extend(["-level", params, str(dest_path)])
        commands.append(cmd)

    console.stop_status("Prepared autolevel commands.")

    console.info(f"Images to be autoleveled: {len(commands)}")
    console.info(f"Images to be copied without autolevel: {len(to_be_copied)}")
    console.info(f"Using {threads} CPU threads for processing.")
    is_continue = console.confirm("Proceed with autoleveling?")
    if not is_continue:
        console.info("Aborting autolevel.")
        return 0

    console.status(f"Copying {len(to_be_copied)} images without autolevel...")
    # Do copying first
    for img_path in to_be_copied:
        dest_path = dest_dir / img_path.name
        if dest_path.exists():
            console.warning(f"Skipping existing file: {dest_path}")
            continue
        shutil.copy2(img_path, dest_path)

    console.stop_status(f"Copied {len(to_be_copied)} images without autolevel.")

    console.status(f"Processing {len(commands)} images with autolevel...")
    with mp.Pool(threads) as pool:
        pool.map(_autolevel_exec, commands)

    console.stop_status(f"Processed {len(commands)} images with autolevel.")

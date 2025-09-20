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
import signal
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import click
from PIL import Image

from .. import file_handler, term
from ..autolevel import (
    analyze_gray_shades,
    apply_levels,
    create_magick_params,
    find_local_peak,
    gamma_correction,
    pad_shades_to_bpc,
    posterize_image_by_bits,
    posterize_image_by_shades,
)
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler, test_or_find_magick

console = term.get_console()


class AutoLevelResult(int, Enum):
    PROCESSED = 1
    GRAYSCALED = 2
    COPIED = 3


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
    if name.lower() == "magick":
        return ["magick"]
    raise ValueError("Invalid magick executable name, must be 'magick' or 'convert'")


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


def _init_worker():
    """Initialize worker processes to handle keyboard interrupts properly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@click.command(
    name="autolevel",
    help="Automatically adjust the levels of images in a directory using ImageMagick",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.dest_output()
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
    dest_output: Path,
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

    all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(path_or_archive)]
    console.info(f"Found {len(all_files)} files in the directory.")

    is_continue = console.confirm("Proceed with autoleveling calculation")
    if not is_continue:
        console.info("Aborting autolevel.")
        return 0

    console.status("Calculating black levels for images...")
    if threads <= 1:
        results = [find_local_peak(file, upper_limit) for file in all_files]
    else:
        try:
            with mp.Pool(threads, initializer=_init_worker) as pool:
                results = pool.starmap(find_local_peak, [(file, upper_limit) for file in all_files])
        except KeyboardInterrupt:
            console.warning("Autoleveling interrupted by user.")
            pool.terminate()
            pool.join()
            return 1
    console.stop_status("Calculated black levels for images.")

    commands: List[str] = []
    to_be_copied: List[Path] = []
    magick_cmd: List[str] = make_prefix_convert(magick_exe)

    console.info(f"Saving processed images to: {dest_output}")
    dest_output.mkdir(parents=True, exist_ok=True)

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
        dest_path = dest_output / image_name

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
        dest_path = dest_output / img_path.name
        if dest_path.exists():
            console.warning(f"Skipping existing file: {dest_path}")
            continue
        shutil.copy2(img_path, dest_path)

    console.stop_status(f"Copied {len(to_be_copied)} images without autolevel.")

    console.status(f"Processing {len(commands)} images with autolevel...")
    if threads <= 1:
        for command in commands:
            _autolevel_exec(command)
    else:
        try:
            with mp.Pool(threads, initializer=_init_worker) as pool:
                pool.map(_autolevel_exec, commands)
        except KeyboardInterrupt:
            console.warning("Autoleveling interrupted by user.")
            pool.terminate()
            pool.join()
            return 1

    console.stop_status(f"Processed {len(commands)} images with autolevel.")


@dataclass
class Autolevel2Config:
    upper_limit: int
    peak_offset: int
    force_gray: bool
    keep_colorspace: bool
    image_fmt: str


def _autolevel2_wrapper(img_path: Path, dest_output: Path, config: Autolevel2Config) -> AutoLevelResult:
    img = Image.open(img_path)
    black_level, _, _ = find_local_peak(img, upper_limit=60)

    if black_level <= 0 or black_level > config.upper_limit:
        dest_path = dest_output / img_path.name
        if config.force_gray:
            img = img.convert("L")
            img.save(dest_path.with_suffix(".png"), format="PNG")
            img.close()
            return AutoLevelResult.GRAYSCALED

        img.close()

        if dest_path.exists():
            console.warning(f"Skipping existing file: {dest_path}")
            return
        shutil.copy2(img_path, dest_path)
        return AutoLevelResult.COPIED

    dest_path = dest_output / img_path.with_suffix(f".{config.image_fmt}").name
    if dest_path.exists():
        console.warning(f"Skipping existing file: {dest_path}")
        return

    # Apply the black level with Pillow
    if not config.keep_colorspace:
        # Force convert to grayscale
        img = img.convert("L")
    gamma_correct = gamma_correction(black_level)

    adjusted_img = apply_levels(img, black_point=black_level + config.peak_offset, white_point=255, gamma=gamma_correct)

    # if jpeg, set quality to 98
    params = {}
    if config.image_fmt == "jpg":
        params["quality"] = 98
    adjusted_img.save(dest_path, format=config.image_fmt.upper(), **params)
    adjusted_img.close()
    img.close()
    return AutoLevelResult.PROCESSED


@click.command(
    name="autolevel2",
    help="Automatically adjust the levels of images in a directory using Pillow (experimental)",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.dest_output()
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
    "-gr",
    "--force-gray",
    "force_gray",
    is_flag=True,
    default=False,
    help="Force convert all images to grayscale for image that is not autoleveled",
)
@click.option(
    "-kc",
    "--keep-colorspace",
    "keep_colorspace",
    is_flag=True,
    default=False,
    help="Keep the original colorspace of the image instead of converting to grayscale",
)
@click.option(
    "-f",
    "--format",
    "image_fmt",
    default="png",
    show_default=True,
    type=click.Choice(["png", "jpg"]),
    help="The format of the output image",
)
@options.threads
@time_program
def autolevel2(
    path_or_archive: Path,
    dest_output: Path,
    upper_limit: int,
    peak_offset: int,
    force_gray: bool,
    keep_colorspace: bool,
    image_fmt: str,
    threads: int,
):  # pragma: no cover
    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(path_or_archive)]
    total_files = len(all_files)
    console.info(f"Found {total_files} files in the directory.")

    full_config = Autolevel2Config(
        upper_limit=upper_limit,
        peak_offset=peak_offset,
        force_gray=force_gray,
        keep_colorspace=keep_colorspace,
        image_fmt=image_fmt,
    )

    console.status("Processing images with autolevel...")
    dest_output.mkdir(parents=True, exist_ok=True)
    results: List[AutoLevelResult] = []
    if threads <= 1:
        for idx, img_path in enumerate(all_files):
            console.status(f"Processing image with autolevel... [{idx + 1}/{total_files}]")
            results.append(_autolevel2_wrapper(img_path, dest_output, full_config))
    else:
        console.info(f"Using {threads} CPU threads for processing.")
        try:
            with mp.Pool(threads, initializer=_init_worker) as pool:
                results = pool.starmap(
                    _autolevel2_wrapper,
                    [(img_path, dest_output, full_config) for img_path in all_files],
                )
        except KeyboardInterrupt:
            console.warning("Autoleveling interrupted by user.")
            pool.terminate()
            pool.join()
            return 1

    autolevel_count = sum(1 for result in results if result == AutoLevelResult.PROCESSED)
    copied_count = sum(1 for result in results if result == AutoLevelResult.COPIED)
    grayscaled_count = sum(1 for result in results if result == AutoLevelResult.GRAYSCALED)
    console.stop_status(f"Processed {total_files} images with autolevel2.")

    if copied_count > 0:
        console.info(f"Copied {copied_count} images without autolevel.")
    if autolevel_count > 0:
        console.info(f"Autoleveled {autolevel_count} images.")
    if grayscaled_count > 0:
        console.info(f"Grayscaled {grayscaled_count} images.")


def _forcegray_exec(command: List[str]) -> None:
    output = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # check exit code
    if output.returncode != 0:
        console.error(f"Command {' '.join(command)} failed with exit code {output.returncode}")


@click.command(
    name="forcegray",
    help="Force convert all images in a directory to grayscale using ImageMagick",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.dest_output(optional=True)
@options.magick_path
@options.threads
@time_program
def force_gray(
    path_or_archive: Path,
    dest_output: Optional[Path],
    magick_path: str,
    threads: int,
):
    """
    Force convert all images in a directory to grayscale using ImageMagick.

    This will always use PNG as the output format.
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

    all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(path_or_archive)]

    console.info(f"Found {len(all_files)} files in the directory.")
    commands: List[List[str]] = []
    magick_cmd: List[str] = make_prefix_convert(magick_exe)

    total_files = len(all_files)
    for idx, img_path in enumerate(all_files):
        console.status(f"Preparing force grayscale commands [{idx + 1}/{total_files}]...")
        if dest_output is not None and dest_output != path_or_archive:
            # Use the destination output directory
            dest_output.mkdir(parents=True, exist_ok=True)
            dest_path = dest_output / img_path.with_suffix(".png").name
        else:
            dest_path = img_path.with_suffix(".png")

        cmd = [
            *magick_cmd,
            str(img_path),
            "-alpha",
            "off",
            "-colorspace",
            "Gray",
            str(dest_path),
        ]
        commands.append(cmd)

    console.stop_status("Prepared force grayscale commands.")

    # Create a backup directory
    backup_dir = path_or_archive / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    console.info(f"Using {threads} CPU threads for processing.")
    console.status(f"Processing {len(commands)} images to grayscale...")
    if threads <= 1:
        for command in commands:
            _forcegray_exec(command)
    else:
        try:
            with mp.Pool(threads, initializer=_init_worker) as pool:
                pool.map(_forcegray_exec, commands)
        except KeyboardInterrupt:
            console.warning("Force grayscale interrupted by user.")
            pool.terminate()
            pool.join()
            return 1

    console.stop_status(f"Processed {len(commands)} images to grayscale.")

    # Move all original files to backup directory
    if dest_output is None or dest_output == path_or_archive:
        console.status(f"Backing up original files to {backup_dir}...")
        for img_path in all_files:
            dest_path = backup_dir / img_path.name
            img_path.rename(dest_path)
        console.stop_status(f"Backed up original files to {backup_dir}.")


def _posterize_simple_wrapper(img_path: Path, dest_output: Path, num_bits: int):
    img = Image.open(img_path)

    posterized = posterize_image_by_bits(img, num_bits)
    dest_path = dest_output / img_path.with_suffix(".png").name

    posterized.save(dest_path, format="PNG")
    posterized.close()
    img.close()


@click.command(
    name="posterize",
    help="Force posterize images to a specific bit depth using Pillow",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.dest_output(optional=True)
@click.option(
    "-b",
    "--bits",
    "num_bits",
    type=click.IntRange(1, 8),
    default=4,
    show_default=True,
    help="The number of bits to posterize the image to (1-8)",
)
@options.threads
@time_program
def posterize_simple(
    path_or_archive: Path,
    dest_output: Optional[Path],
    num_bits: int,
    threads: int,
):
    """
    Posterize images in a directory to a specific bit depth using Pillow.

    This will always use PNG as the output format.
    """
    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(path_or_archive)]
    total_files = len(all_files)
    console.info(f"Found {total_files} files in the directory.")

    console.status("Processing images with posterize...")
    dest_output.mkdir(parents=True, exist_ok=True)
    if threads <= 1:
        for idx, img_path in enumerate(all_files):
            console.status(f"Processing image with posterize... [{idx + 1}/{total_files}]")
            _posterize_simple_wrapper(img_path, dest_output, num_bits)
    else:
        console.info(f"Using {threads} CPU threads for processing.")
        try:
            with mp.Pool(threads, initializer=_init_worker) as pool:
                pool.starmap(
                    _posterize_simple_wrapper,
                    [(img_path, dest_output, num_bits) for img_path in all_files],
                )
        except KeyboardInterrupt:
            console.warning("Posterize interrupted by user.")
            pool.terminate()
            pool.join()
            return 1

    console.stop_status(f"Processed {total_files} images with posterize.")


class AutoPosterizeResult:
    PROCESSED = 1
    COPIED = 2


def _autoposterize_wrapper(img_path: Path, dest_output: Path, threshold: float) -> AutoLevelResult:
    img = Image.open(img_path)
    dest_path = dest_output / img_path.with_suffix(".png").name

    shades = analyze_gray_shades(img, threshold)

    # Pad the shades to the nearest bpc value
    shades_nums = pad_shades_to_bpc([shade_info["shade"] for shade_info in shades])
    if len(shades_nums) == 0:
        # No significant shades found, just copy the image
        img.close()
        if dest_path.exists():
            console.warning(f"Skipping existing file: {dest_path}")
            return AutoLevelResult.COPIED
        shutil.copy2(img_path, dest_path)
        return AutoLevelResult.COPIED
    if len(shades_nums) > 128:
        # same 8bpc, just copy the image
        img.close()
        if dest_path.exists():
            console.warning(f"Skipping existing file: {dest_path}")
            return AutoLevelResult.COPIED
        shutil.copy2(img_path, dest_path)
        return AutoLevelResult.COPIED

    posterized = posterize_image_by_shades(img, shades_nums)

    posterized.save(dest_path, format="PNG")
    posterized.close()
    img.close()
    return AutoLevelResult.PROCESSED


@click.command(
    name="autoposterize",
    help="(Experimental) Analyze and posterize images to optimal bit depth using Pillow",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.dest_output(optional=True)
@click.option(
    "-th",
    "--threshold",
    "threshold_pct",
    type=click.FloatRange(0.0, 100.0),
    default=0.01,
    show_default=True,
    help="The threshold percentage to consider a shade as significant (0-100%)",
)
@options.threads
@time_program
def auto_posterize(
    path_or_archive: Path,
    dest_output: Optional[Path],
    threshold_pct: float,
    threads: int,
):
    """
    Automatically analyze and posterize images in a directory to an optimal bit depth using Pillow.

    This will always use PNG as the output format.
    """
    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    if threshold_pct == 0.0:
        raise click.BadParameter(
            "Threshold percentage cannot be 0.0, as this will consider all shades as significant.",
            param_hint="threshold_pct",
        )

    all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(path_or_archive)]
    total_files = len(all_files)
    console.info(f"Found {total_files} files in the directory.")

    console.status("Processing images with autoposterize...")
    dest_output.mkdir(parents=True, exist_ok=True)
    results: List[AutoPosterizeResult] = []
    if threads <= 1:
        for idx, img_path in enumerate(all_files):
            console.status(f"Processing image with autoposterize... [{idx + 1}/{total_files}]")
            results.append(_autoposterize_wrapper(img_path, dest_output, threshold_pct))
    else:
        console.info(f"Using {threads} CPU threads for processing.")
        try:
            with mp.Pool(threads, initializer=_init_worker) as pool:
                results = pool.starmap(
                    _autoposterize_wrapper,
                    [(img_path, dest_output, threshold_pct) for img_path in all_files],
                )
        except KeyboardInterrupt:
            console.warning("Auto-posterizing interrupted by user.")
            pool.terminate()
            pool.join()
            return 1

    posterized_count = sum(1 for result in results if result == AutoLevelResult.PROCESSED)
    copied_count = sum(1 for result in results if result == AutoLevelResult.COPIED)
    grayscaled_count = sum(1 for result in results if result == AutoLevelResult.GRAYSCALED)
    console.stop_status(f"Processed {total_files} images with autoposterize.")

    if copied_count > 0:
        console.info(f"Copied {copied_count} images without autoposterize.")
    if posterized_count > 0:
        console.info(f"Posterized {posterized_count} images.")
    if grayscaled_count > 0:
        console.info(f"Grayscaled {grayscaled_count} images.")


@click.command(
    name="analyze-shades",
    help="(Experimental) Analyze and show the gray shades in images in a directory using Pillow",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-th",
    "--threshold",
    "threshold_pct",
    type=click.FloatRange(0.0, 100.0),
    default=0.01,
    show_default=True,
    help="The threshold percentage to consider a shade as significant (0-100%)",
)
@time_program
def analyze_shades(
    path_or_archive: Path,
    threshold_pct: float,
):
    """
    Analyze and show the gray shades in images in a directory using Pillow.
    """
    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    if threshold_pct == 0.0:
        raise click.BadParameter(
            "Threshold percentage cannot be 0.0, as this will consider all shades as significant.",
            param_hint="threshold_pct",
        )

    all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(path_or_archive)]
    total_files = len(all_files)
    console.info(f"Found {total_files} files in the directory.")

    console.status("Processing images with analyzer...")
    for image_path in all_files:
        console.status(f"Analyzing image... {image_path}")
        shades = analyze_gray_shades(Image.open(image_path), threshold_pct)
        if len(shades) == 0:
            console.info(f"No significant shades found in {image_path}")
            continue

        closest_bpc = pad_shades_to_bpc([shade_info["shade"] for shade_info in shades])
        total_shades = len(shades)
        console.info(
            f"Shades found in {image_path}: (Total: {total_shades}, Closest bpc: {len(closest_bpc).bit_length() - 1})"
        )

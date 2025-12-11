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
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypedDict

import rich_click as click
from PIL import Image

from .. import file_handler, term
from ..autolevel import (
    apply_levels,
    create_magick_params,
    find_local_peak,
    find_local_peak_legacy,
    gamma_correction,
)
from ..common import lowest_or, threaded_worker
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


def _autolevel_exec(term_q: term.MessageOrInterface, command: list[str]) -> None:
    output = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # check exit code
    if output.returncode != 0:
        cnsl = term.with_thread_queue(term_q)
        cnsl.error(f"Command {' '.join(command)} failed with exit code {output.returncode}")


def _autolevel_exec_star(args: tuple[term.MessageQueue, list[str]]) -> None:
    return _autolevel_exec(*args)


def determine_image_format(img_path: Path, prefer: str) -> str:
    if prefer != "auto":
        return f".{prefer}"

    ext = img_path.suffix.lower()
    if not ext.startswith("."):
        return f".{ext}"
    return ext


def _find_local_peak_magick_wrapper(
    img_path: Path, upper_limit: int, peak_min_pct: float, skip_white: bool, is_legacy: bool
) -> tuple[int, int, Path, bool]:
    if not is_legacy:
        black_level, white_level, force_gray = find_local_peak(
            img_path, upper_limit=upper_limit, peak_percentage=peak_min_pct, skip_white_check=skip_white
        )
    else:
        black_level, white_level, force_gray = find_local_peak_legacy(
            img_path, upper_limit=upper_limit, skip_white_peaks=skip_white
        )
    return black_level, white_level, img_path, force_gray


def _find_local_peak_magick_wrapper_star(args: tuple[Path, int, float, bool, bool]) -> tuple[int, int, Path, bool]:
    return _find_local_peak_magick_wrapper(*args)


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
    "-pmp",
    "--peak-min-pct",
    "peak_min_pct",
    type=click.FloatRange(0.0, 100.0),
    default=0.25,
    show_default=True,
    help="The minimum percentage of pixels for a peak to be considered valid",
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
@click.option(
    "--no-white",
    "no_white",
    is_flag=True,
    default=False,
    help="Do not adjust white level, only adjust black level",
)
@click.option(
    "--legacy",
    is_flag=True,
    default=False,
    help="Use legacy autolevel analysis",
)
@options.threads
@options.magick_path
@time_program
def autolevel(
    path_or_archive: Path,
    dest_output: Path,
    upper_limit: int,
    peak_min_pct: float,
    peak_offset: int,
    image_fmt: str,
    no_white: bool,
    legacy: bool,
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

    progress = console.make_progress()
    task_calculate = progress.add_task("Analzying images...", finished_text="Analzyed images", total=len(all_files))

    results: list[tuple[int, int, Path, bool]] = []
    console.info(f"Using {threads} CPU threads for processing.")
    with threaded_worker(console, lowest_or(threads, all_files)) as (pool, _):
        for result in pool.imap_unordered(
            _find_local_peak_magick_wrapper_star,
            ((file, upper_limit, peak_min_pct, no_white, legacy) for file in all_files),
        ):
            results.append(result)
            progress.update(task_calculate, advance=1)
    progress.update(task_calculate, completed=len(all_files))

    commands: list[list[str]] = []
    to_be_copied: list[Path] = []
    magick_cmd: list[str] = make_prefix_convert(magick_exe)

    console.info(f"Saving processed images to: {dest_output}")
    dest_output.mkdir(parents=True, exist_ok=True)

    dumped_data_temp = []
    for black_level, white_level, img_path, force_gray in results:
        dumped_data_temp.append({
            "image": str(img_path),
            "black_level": black_level,
            "white_level": white_level,
            "force_gray": force_gray,
        })

    # Dump the results for debugging
    dumped_data = json.dumps(dumped_data_temp, indent=4)
    current_cwd = Path.cwd()
    dump_path = current_cwd / "autolevel_debug.json"
    dump_path.write_text(dumped_data, encoding="utf-8")
    console.info(f"Dumped autolevel debug data to: {dump_path}")

    # Pre-compute all the image magick commands
    task_cmd = progress.add_task(
        "Preparing autolevel commands...", finished_text="Prepared autolevel commands", total=len(results)
    )
    total_results = len(results)
    for black_level, white_level, img_path, force_gray in results:
        if black_level == 0:
            # Skip images that don't need autolevel
            to_be_copied.append(img_path)
            continue
        if black_level > upper_limit:
            # Skip images that are too bright
            to_be_copied.append(img_path)
            continue

        params = create_magick_params(black_level, white_level, peak_offset)
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
        progress.update(task_cmd, advance=1)

    progress.update(task_cmd, completed=total_results)

    console.info(f"Images to be autoleveled: {len(commands)}")
    console.info(f"Images to be copied without autolevel: {len(to_be_copied)}")
    console.info(f"Using {threads} CPU threads for processing.")
    is_continue = console.confirm("Proceed with autoleveling?")
    if not is_continue:
        console.info("Aborting autolevel.")
        return 0

    task_copy = progress.add_task("Copying images...", finished_text="Copied images", total=len(to_be_copied))
    # Do copying first
    for img_path in to_be_copied:
        dest_path = dest_output / img_path.name
        if dest_path.exists():
            console.warning(f"Skipping existing file: {dest_path}")
            progress.update(task_copy, advance=1)
            continue
        shutil.copy2(img_path, dest_path)
        progress.update(task_copy, advance=1)

    progress.update(task_copy, completed=len(to_be_copied))

    task_proc = progress.add_task("Auto-leveling images...", finished_text="Auto-leveled images", total=len(commands))
    console.info(f"Using {threads} CPU threads for processing.")
    with threaded_worker(console, lowest_or(threads, commands)) as (pool, log_q):
        for _ in pool.imap_unordered(_autolevel_exec_star, [(log_q, cmd) for cmd in commands]):
            progress.update(task_proc, advance=1)

    console.stop_progress(progress, f"Auto-leveled {len(commands)} images.")


@dataclass
class Autolevel2Config:
    upper_limit: int
    peak_offset: int
    peak_min_pct: float
    force_gray: bool
    keep_colorspace: bool
    image_fmt: str
    no_white: bool


def _autolevel2_wrapper(
    log_q: term.MessageOrInterface, img_path: Path, dest_output: Path, config: Autolevel2Config, is_legacy: bool
) -> AutoLevelResult:
    img = Image.open(img_path)
    if not is_legacy:
        black_level, white_level, _ = find_local_peak(
            img, upper_limit=60, peak_percentage=config.peak_min_pct, skip_white_check=config.no_white
        )
    else:
        black_level, white_level, _ = find_local_peak_legacy(img, upper_limit=60, skip_white_peaks=config.no_white)

    cnsl = term.with_thread_queue(log_q)

    is_black_bad = black_level <= 0
    is_white_bad = white_level >= 255 if not config.no_white else False

    if (
        (is_black_bad and is_white_bad and not config.no_white)  # both levels are bad
        or (is_black_bad and config.no_white)
        or black_level > config.upper_limit
    ):
        dest_path = dest_output / img_path.name
        if config.force_gray:
            img = img.convert("L")
            img.save(dest_path.with_suffix(".png"), format="PNG")
            img.close()
            return AutoLevelResult.GRAYSCALED

        img.close()

        if dest_path.exists():
            cnsl.warning(f"Skipping existing file: {dest_path}")
            return AutoLevelResult.COPIED
        shutil.copy2(img_path, dest_path)
        return AutoLevelResult.COPIED

    dest_path = dest_output / img_path.with_suffix(f".{config.image_fmt}").name
    if dest_path.exists():
        cnsl.warning(f"Skipping existing file: {dest_path}")
        return AutoLevelResult.COPIED

    # Apply the black level with Pillow
    if not config.keep_colorspace:
        # Force convert to grayscale
        img = img.convert("L")
    gamma_correct = gamma_correction(black_level)

    adjusted_img = apply_levels(
        img,
        black_point=black_level + config.peak_offset,
        white_point=255 if config.no_white else white_level,
        gamma=gamma_correct,
    )

    # if jpeg, set quality to 98
    params = {}
    if config.image_fmt == "jpg":
        params["quality"] = 98
    adjusted_img.save(dest_path, format=config.image_fmt.upper(), **params)
    adjusted_img.close()
    img.close()
    return AutoLevelResult.PROCESSED


def _autolevel2_wrapper_star(args: tuple[term.MessageQueue, Path, Path, Autolevel2Config, bool]) -> AutoLevelResult:
    return _autolevel2_wrapper(*args)


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
    "-pmp",
    "--peak-min-pct",
    "peak_min_pct",
    type=click.FloatRange(0.0, 100.0),
    default=0.25,
    show_default=True,
    help="The minimum percentage of pixels for a peak to be considered valid",
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
@click.option(
    "--no-white",
    "no_white",
    is_flag=True,
    default=False,
    help="Do not adjust white level, only adjust black level",
)
@click.option(
    "--legacy",
    is_flag=True,
    default=False,
    help="Use legacy autolevel analysis",
)
@options.threads
@options.recursive
@time_program
def autolevel2(
    path_or_archive: Path,
    dest_output: Path,
    upper_limit: int,
    peak_min_pct: float,
    peak_offset: int,
    force_gray: bool,
    keep_colorspace: bool,
    image_fmt: str,
    no_white: bool,
    legacy: bool,
    threads: int,
    recursive: bool,
):  # pragma: no cover
    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    candidates: list[Path] = []
    if not recursive:
        candidates.append(path_or_archive)
    else:
        console.info(f"Recursively collecting folder in {path_or_archive}...")
        for comic in file_handler.collect_all_comics(path_or_archive, dir_only=True):
            candidates.append(comic)
        console.info(f"Found {len(candidates)} archives/folders to autolevel.")

    if not candidates and recursive:
        console.warning("No valid folders found to autolevel.")
        return 1

    for path_real in candidates:
        if recursive:
            console.info(f"Processing: {path_real}")
        all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(path_real)]
        total_files = len(all_files)
        console.info(f"Found {total_files} files in the directory.")

        full_config = Autolevel2Config(
            upper_limit=upper_limit,
            peak_offset=peak_offset,
            peak_min_pct=peak_min_pct,
            force_gray=force_gray,
            keep_colorspace=keep_colorspace,
            image_fmt=image_fmt,
            no_white=no_white,
        )

        real_output = dest_output
        if recursive:
            real_output = dest_output / path_real.name
        real_output.mkdir(parents=True, exist_ok=True)

        results: list[AutoLevelResult] = []
        progress = console.make_progress()
        task = progress.add_task("Processing images...", finished_text="Processed images", total=total_files)

        console.info(f"Using {threads} CPU threads for processing.")
        with threaded_worker(console, lowest_or(threads, all_files)) as (pool, log_q):
            for result in pool.imap_unordered(
                _autolevel2_wrapper_star,
                ((log_q, img_path, real_output, full_config, legacy) for img_path in all_files),
            ):
                results.append(result)
                progress.update(task, advance=1)

        console.stop_progress(progress, f"Processed {total_files} images.")
        autolevel_count = sum(1 for result in results if result == AutoLevelResult.PROCESSED)
        copied_count = sum(1 for result in results if result == AutoLevelResult.COPIED)
        grayscaled_count = sum(1 for result in results if result == AutoLevelResult.GRAYSCALED)

        if copied_count > 0:
            console.info(f"Copied {copied_count} images without autolevel.")
        if autolevel_count > 0:
            console.info(f"Autoleveled {autolevel_count} images.")
        if grayscaled_count > 0:
            console.info(f"Grayscaled {grayscaled_count} images.")
    if recursive:
        console.info(f"Finished processing {len(candidates)} folders.")


def _forcegray_exec(log_q: term.MessageOrInterface, command: list[str]) -> None:
    output = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # check exit code
    if output.returncode != 0:
        cnsl = term.with_thread_queue(log_q)
        cnsl.error(f"Command {' '.join(command)} failed with exit code {output.returncode}")


def _forcegray_exec_star(args: tuple[term.MessageQueue, list[str]]) -> None:
    return _forcegray_exec(*args)


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
    dest_output: Path | None,
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
    commands: list[list[str]] = []
    magick_cmd: list[str] = make_prefix_convert(magick_exe)

    total_files = len(all_files)
    progress = console.make_progress()
    task_prep = progress.add_task("Preparing commands...", finished_text="Prepared images", total=total_files)

    for img_path in all_files:
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
        progress.update(task_prep, advance=1)

    progress.update(task_prep, completed=total_files)

    # Create a backup directory
    backup_dir = path_or_archive / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    task_proc = progress.add_task("Processing images...", finished_text="Processed images", total=len(commands))
    console.info(f"Using {threads} CPU threads for processing.")
    with threaded_worker(console, lowest_or(threads, all_files)) as (pool, log_q):
        for _ in pool.imap_unordered(_forcegray_exec_star, [(log_q, cmd) for cmd in commands]):
            progress.update(task_proc, advance=1)

    console.stop_progress(progress, f"Processed {len(commands)} images to grayscale.")

    # Move all original files to backup directory
    if dest_output is None or dest_output == path_or_archive:
        console.info(f"Backing up original files to {backup_dir}...")
        for img_path in all_files:
            dest_path = backup_dir / img_path.name
            img_path.rename(dest_path)
        console.info(f"Backed up original files to {backup_dir}.")


class AutolevelAnalyzeResult(TypedDict):
    image: str
    black: int
    white: int
    gamma: float
    force_gray: bool


def _analyze_levels_wrapper(img_path: Path, config: Autolevel2Config, is_legacy: bool) -> AutolevelAnalyzeResult:
    img = Image.open(img_path)
    if not is_legacy:
        black_level, white_level, force_gray = find_local_peak(
            img, upper_limit=60, peak_percentage=config.peak_min_pct, skip_white_check=config.no_white
        )
    else:
        black_level, white_level, force_gray = find_local_peak_legacy(
            img, upper_limit=60, skip_white_peaks=config.no_white
        )
    img.close()
    gamma_correct = gamma_correction(black_level)
    return {
        "image": str(img_path),
        "black": black_level,
        "white": white_level,
        "gamma": gamma_correct,
        "force_gray": force_gray,
    }


def _analyze_level_wrapper_star(args: tuple[Path, Autolevel2Config, bool]) -> AutolevelAnalyzeResult:
    return _analyze_levels_wrapper(*args)


@click.command(
    name="analyze-peaks",
    help="Do a peak level analysis on images in a directory and output the results to a JSON file",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
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
    "-pmp",
    "--peak-min-pct",
    "peak_min_pct",
    type=click.FloatRange(0.0, 100.0),
    default=0.25,
    show_default=True,
    help="The minimum percentage of pixels for a peak to be considered valid",
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
    "--no-white",
    "no_white",
    is_flag=True,
    default=False,
    help="Do not adjust white level, only adjust black level",
)
@click.option(
    "--legacy",
    is_flag=True,
    default=False,
    help="Use legacy autolevel analysis",
)
@options.threads
@time_program
def analyze_level(
    path_or_archive: Path,
    upper_limit: int,
    peak_min_pct: float,
    peak_offset: int,
    no_white: bool,
    legacy: bool,
    threads: int,
):  # pragma: no cover
    """
    Analyze the peak levels of all images in a directory and output the results to a JSON file.
    """

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
        peak_min_pct=peak_min_pct,
        no_white=no_white,
        # Mocked values
        force_gray=False,
        keep_colorspace=False,
        image_fmt="png",
    )

    results: list[AutolevelAnalyzeResult] = []
    progress = console.make_progress()
    task = progress.add_task("Analzying images...", finished_text="Analyzed images", total=total_files)

    console.info(f"Using {threads} CPU threads for processing.")
    with threaded_worker(console, lowest_or(threads, all_files)) as (pool, _):
        for result in pool.imap_unordered(
            _analyze_level_wrapper_star,
            ((img_path, full_config, legacy) for img_path in all_files),
        ):
            results.append(result)
            progress.update(task, advance=1)

    console.stop_progress(progress, f"Analyzed {total_files} images peak levels.")

    results.sort(key=lambda x: x["image"])
    complete_name = f"{path_or_archive.name}_autolevel.json"
    dump_path = Path.cwd() / complete_name
    dumped_data = json.dumps(results, indent=4)
    dump_path.write_text(dumped_data, encoding="utf-8")
    console.info(f"Dumped autolevel analysis data to: {dump_path}")

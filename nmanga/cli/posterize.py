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

# Automatically posterize images based on shade analysis

from __future__ import annotations

import shutil
from enum import Enum
from pathlib import Path

import rich_click as click
from PIL import Image

from .. import file_handler, term
from ..autolevel import (
    analyze_gray_shades,
    detect_nearest_bpc,
    pad_shades_to_bpc,
    posterize_image_by_bits,
    posterize_image_by_shades,
)
from ..common import lowest_or, threaded_worker
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler

console = term.get_console()


class PosterizedResult(int, Enum):
    PROCESSED = 1
    COPIED = 2


def _posterize_simple_wrapper(img_path: Path, dest_output: Path, num_bits: int):
    img = Image.open(img_path)

    posterized = posterize_image_by_bits(img, num_bits)
    dest_path = dest_output / img_path.with_suffix(".png").name

    posterized.save(dest_path, format="PNG")
    posterized.close()
    img.close()


def _posterize_simple_wrapper_star(args: tuple[Path, Path, int]):
    return _posterize_simple_wrapper(*args)


@click.command(
    name="posterize",
    help="Force posterize images to a specific bit depth using Pillow",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.dest_output(optional=False)
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
    dest_output: Path,
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

    progress = console.make_progress()
    task = progress.add_task("Posterizing images...", finished_text="Posterized images", total=total_files)

    dest_output.mkdir(parents=True, exist_ok=True)
    console.info(f"Using {threads} CPU threads for processing.")
    with threaded_worker(console, lowest_or(threads, all_files)) as (pool, _):
        for _ in pool.imap_unordered(
            _posterize_simple_wrapper_star,
            [(img_path, dest_output, num_bits) for img_path in all_files],
        ):
            progress.update(task, advance=1)

    console.stop_progress(progress, f"Posterized {total_files} images to {num_bits} bits.")


def _autoposterize_wrapper(
    log_q: term.MessageOrInterface, img_path: Path, dest_output: Path, threshold: float, use_palette_mode: bool
) -> PosterizedResult:
    img = Image.open(img_path)
    dest_path = dest_output / img_path.with_suffix(".png").name

    shades = analyze_gray_shades(img, threshold)

    # Pad the shades to the nearest bpc value
    shades_nums = pad_shades_to_bpc(shades)
    bpc_count = max(len(shades_nums).bit_length() - 1, 1)
    if bpc_count >= 8:
        # same 8bpc, just copy the image
        img.close()
        if dest_path.exists():
            cnsl = term.with_thread_queue(log_q)
            cnsl.warning(f"Skipping existing file: {dest_path}")
            return PosterizedResult.COPIED
        shutil.copy2(img_path, dest_path)
        return PosterizedResult.COPIED

    if use_palette_mode:
        posterized = posterize_image_by_bits(img, bpc_count)
    else:
        posterized = posterize_image_by_shades(img, shades_nums)

    posterized.save(dest_path, format="PNG")
    posterized.close()
    img.close()
    return PosterizedResult.PROCESSED


def _autoposterize_wrapper_star(args: tuple[term.MessageQueue, Path, Path, float, bool]) -> PosterizedResult:
    return _autoposterize_wrapper(*args)


@click.command(
    name="autoposterize",
    help="(Experimental) Analyze and posterize images to optimal bit depth using Pillow",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.dest_output(optional=False)
@click.option(
    "-th",
    "--threshold",
    "threshold_pct",
    type=click.FloatRange(0.0, 100.0),
    default=0.01,
    show_default=True,
    help="The threshold percentage to consider a shade as significant (0-100%)",
)
@click.option(
    "-pm",
    "--palette-mode",
    "use_palette_mode",
    is_flag=True,
    default=False,
    help="Use palette mode for posterization instead of direct color mapping (may produce better results)",
)
@options.threads
@time_program
def auto_posterize(
    path_or_archive: Path,
    dest_output: Path,
    threshold_pct: float,
    use_palette_mode: bool,
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

    progress = console.make_progress()
    taks = progress.add_task("Auto-posterizing images...", finished_text="Auto-posterized images", total=total_files)
    dest_output.mkdir(parents=True, exist_ok=True)

    results: list[PosterizedResult] = []
    console.info(f"Using {threads} CPU threads for processing.")
    with threaded_worker(console, lowest_or(threads, all_files)) as (pool, log_q):
        for result in pool.imap_unordered(
            _autoposterize_wrapper_star,
            [(log_q, img_path, dest_output, threshold_pct, use_palette_mode) for img_path in all_files],
        ):
            results.append(result)
            progress.update(taks, advance=1)

    console.stop_progress(progress, f"Auto-posterized {total_files} images.")
    posterized_count = sum(1 for result in results if result == PosterizedResult.PROCESSED)
    copied_count = sum(1 for result in results if result == PosterizedResult.COPIED)

    if copied_count > 0:
        console.info(f"Copied {copied_count} images without autoposterize.")
    if posterized_count > 0:
        console.info(f"Posterized {posterized_count} images.")


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

    progress = console.make_progress()
    task = progress.add_task("Analyzing images...", finished_text="Analyzed images", total=total_files)
    for image_path in all_files:
        shades = analyze_gray_shades(Image.open(image_path), threshold_pct)
        if len(shades) == 0:
            console.info(f"No significant shades found in {image_path}")
            continue

        closest_bpc = detect_nearest_bpc(shades)
        total_shades = len(shades)
        console.info(f"Shades found in {image_path}: (Total: {total_shades}, Closest bpc: {closest_bpc}bpp)")
        progress.update(task, advance=1)

    console.stop_progress(progress, f"Analyzed {total_files} images!")

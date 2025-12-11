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

# Upscale images using a tiled approach with overlapping seams to prevent artifacts.

from __future__ import annotations

from enum import Enum
from pathlib import Path

import rich_click as click
from PIL import Image

from .. import file_handler, term
from ..common import lowest_or, threaded_worker
from ..resizer import ResizeKernel, ResizeMode, ResizeTarget
from ..resizer import rescale_image as rescale_image_func
from . import options
from ._deco import check_config_first, time_program
from .base import NMangaCommandHandler

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)

console = term.get_console()


class RescaleResult(int, Enum):
    PROCESSED = 1
    SKIPPED = 2


def _runner_rescale_threaded(
    log_q: term.MessageOrInterface,
    img_path: Path,
    output_dir: Path,
    target: ResizeTarget,
    kernel: ResizeKernel,
) -> RescaleResult:
    cnsl = term.with_thread_queue(log_q)

    dest_path = output_dir / f"{img_path.stem}.png"
    if dest_path.exists():
        cnsl.warning(f"Skipping existing file: {dest_path}")
        return RescaleResult.SKIPPED

    img = Image.open(img_path)
    rescaled_img = rescale_image_func(
        img,
        target=target,
        kernel=kernel,
    )

    rescaled_img.save(dest_path, format="PNG")
    img.close()
    rescaled_img.close()
    return RescaleResult.PROCESSED


def _runner_rescale_threaded_star(
    args: tuple[term.MessageQueue, Path, Path, ResizeTarget, ResizeKernel],
) -> RescaleResult:
    return _runner_rescale_threaded(*args)


@click.command(
    "rescale",
    help="Rescale images in a directory using various algorithms.",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.dest_output()
@click.option(
    "-k",
    "--kernel",
    type=click.Choice(ResizeKernel),
    required=True,
    help="Rescaling kernel to use.",
)
@click.option(
    "-f",
    "--factor",
    type=options.FLOAT_INT,
    help="Scaling factor to rescale images by (e.g., 2 for 2x upscaling).",
    required=False,
)
@click.option(
    "-w",
    "--width",
    type=options.POSITIVE_INT,
    help="Target width to resize images to.",
    required=False,
)
@click.option(
    "-ht",
    "--height",
    type=options.POSITIVE_INT,
    help="Target height to resize images to.",
    required=False,
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(ResizeMode),
    show_default=True,
    default=ResizeMode.Fit,
    help="Resizing mode to use when both width and height are specified.",
)
@options.recursive
@options.threads_alt
@check_config_first
@time_program
def rescale_image(
    path_or_archive: Path,
    dest_output: Path,
    kernel: ResizeKernel,
    factor: float | int | None,
    width: int | None,
    height: int | None,
    mode: ResizeMode,
    recursive: bool,
    threads: int,
) -> None:
    """Rescale images in a directory using various algorithms."""
    console.info("Preparing rescaling task...")

    candidates: list[Path] = []
    if not recursive:
        candidates.append(path_or_archive)
    else:
        console.info(f"Recursively collecting folder in {path_or_archive}...")
        for comic in file_handler.collect_all_comics(path_or_archive, dir_only=True):
            candidates.append(comic)
        console.info(f"Found {len(candidates)} archives/folders to denoise.")

    real_target = ResizeTarget(mode=mode, factor=factor, width=width, height=height)
    for path_real in candidates:
        if recursive:
            console.info(f"Processing: {path_real}")

        all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(path_real)]
        total_files = len(all_files)
        console.info(f"Found {total_files} files in the directory.")

        real_output = dest_output
        if recursive:
            real_output = dest_output / path_real.name
        real_output.mkdir(parents=True, exist_ok=True)

        results: list[RescaleResult] = []
        progress = console.make_progress()
        task = progress.add_task("Processing images...", finished_text="Processed images", total=total_files)

        console.info(f"Using {threads} CPU threads for processing.")
        with threaded_worker(console, lowest_or(threads, all_files)) as (pool, log_q):
            for result in pool.imap_unordered(
                _runner_rescale_threaded_star,
                ((log_q, img_path, real_output, real_target, kernel) for img_path in all_files),
            ):
                results.append(result)
                progress.update(task, advance=1)

        console.stop_progress(progress, f"Processed {total_files} images.")
        processed_count = sum(1 for r in results if r == RescaleResult.PROCESSED)
        ignored_count = sum(1 for r in results if r == RescaleResult.SKIPPED)

        if processed_count > 0:
            console.info(f"Rescaled {processed_count} images.")
        if ignored_count > 0:
            console.info(f"Skipped {ignored_count} existing images.")
    if recursive:
        console.info(f"Finished processing {len(candidates)} folders.")

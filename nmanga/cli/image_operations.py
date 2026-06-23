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

# Collection of image operations

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import rich_click as click
from PIL import Image

from .. import file_handler, term
from ..common import lowest_or, threaded_worker
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)
console = term.get_console()


class ThreadingResult(int, Enum):
    PROCESSED = 1
    SKIPPED = 2


@click.group(
    name="imops",
    help="Image operations toolsets",
)
def imops_group():
    """Image operations toolsets"""
    pass


@dataclass
class ChoppingParams:
    top_px: int = 0
    bottom_px: int = 0
    left_px: int = 0
    right_px: int = 0


def _runner_imops_chopping(
    log_q: term.MessageOrInterface,
    img_path: Path,
    output_dir: Path,
    params: ChoppingParams,
    force: bool,
) -> ThreadingResult:
    cnsl = term.with_thread_queue(log_q)

    dest_path = output_dir / f"{img_path.stem}.png"
    if dest_path.exists() and not force:
        cnsl.warning(f"Skipping existing file: {dest_path}")
        return ThreadingResult.SKIPPED

    img = Image.open(img_path)

    # crop images
    #  The crop rectangle, as a (left, upper, right, lower)-tuple.
    cropped = img.crop((params.left_px, params.top_px, img.width - params.right_px, img.height - params.bottom_px))
    cropped.save(dest_path, format="png")
    return ThreadingResult.PROCESSED


def _runner_imops_chopping_star(args: tuple[term.MessageQueue, Path, Path, ChoppingParams, bool]) -> ThreadingResult:
    return _runner_imops_chopping(*args)


@imops_group.command(
    name="chop",
    help="Chop/crop images in a directory",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-tp",
    "--top",
    "top_px",
    default=0,
    show_default=True,
    type=options.ZERO_POSITIVE_INT,
)
@click.option(
    "-bm",
    "--bottom",
    "bottom_px",
    default=0,
    show_default=True,
    type=options.ZERO_POSITIVE_INT,
)
@click.option(
    "-lt",
    "--left",
    "left_px",
    default=0,
    show_default=True,
    type=options.ZERO_POSITIVE_INT,
)
@click.option(
    "-rt",
    "--right",
    "right_px",
    default=0,
    show_default=True,
    type=options.ZERO_POSITIVE_INT,
)
@options.dest_output(optional=False)
@options.recursive
@options.threads
@options.force
@time_program
def image_ops_chop(
    path_or_archive: Path,
    top_px: int,
    bottom_px: int,
    left_px: int,
    right_px: int,
    dest_output: Path,
    recursive: bool,
    threads: int,
    force: bool,
) -> None:
    """Chop/crop images in a directory by pixels"""

    if top_px == bottom_px == left_px == right_px == 0:
        console.warning("No chopping parameters provided, nothing to do.")
        return

    candidates: list[Path] = []
    if not recursive:
        candidates.append(path_or_archive)
    else:
        console.info(f"Recursively collecting folder in {path_or_archive}...")
        for comic in file_handler.collect_all_comics(path_or_archive, dir_only=True):
            candidates.append(comic)
        console.info(f"Found {len(candidates)} archives/folders to denoise.")

    chop_params = ChoppingParams(top_px, bottom_px, left_px, right_px)
    for path_real in candidates:
        if recursive:
            console.info(f"Processing: {path_real}")
        image_candidates: list[Path] = [
            img_path for img_path, _, _, _ in file_handler.collect_image_from_folder(path_real)
        ]

        real_output = dest_output
        if recursive:
            real_output = dest_output / path_real.name

        total_images = len(image_candidates)
        if total_images == 0:
            console.info(f"No images found in {path_real}, skipping.")
            continue

        real_output.mkdir(parents=True, exist_ok=True)

        progress = console.make_progress()
        task = progress.add_task("Chopping images...", finished_text="Chopped images", total=total_images)

        console.info(f"Using {threads} CPU threads for processing.")
        with threaded_worker(console, lowest_or(threads, image_candidates)) as (pool, log_q):
            for _ in pool.imap_unordered(
                _runner_imops_chopping_star,
                ((log_q, image, real_output, chop_params, force) for image in image_candidates),
            ):
                progress.update(task, advance=1)
        console.stop_progress(progress, f"Chopped {total_images} images.")
    if recursive:
        console.info(f"Finished processing {len(candidates)} folders.")

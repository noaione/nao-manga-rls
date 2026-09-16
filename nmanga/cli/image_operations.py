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
from ..deblur import deblur_deconv, deblur_edge_sharp
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
        total_images = len(image_candidates)
        if total_images <= 0:
            console.warning(f"No images found in {path_real}, skipping.")
            continue

        real_output = dest_output
        if recursive:
            real_output = dest_output / path_real.name

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


@dataclass
class DeblurParams:
    method: str = "deconv"
    radius: float = 0.8
    strength: float = 0.65
    iterations: int = 6
    threshold: float = 2.0
    overshoot: float = 0.0


def _runner_imops_deblur(
    log_q: term.MessageOrInterface,
    img_path: Path,
    output_dir: Path,
    params: DeblurParams,
    force: bool,
) -> ThreadingResult:
    cnsl = term.with_thread_queue(log_q)

    dest_path = output_dir / f"{img_path.stem}.png"
    if dest_path.exists() and not force:
        cnsl.warning(f"Skipping existing file: {dest_path}")
        return ThreadingResult.SKIPPED

    with Image.open(img_path) as img:
        if params.method == "edge":
            deblurred = deblur_edge_sharp(
                img,
                radius=params.radius,
                strength=params.strength,
                threshold=params.threshold,
                overshoot=params.overshoot,
            )
        else:
            deblurred = deblur_deconv(
                img,
                radius=params.radius,
                strength=params.strength,
                iterations=params.iterations,
                threshold=params.threshold,
                overshoot=params.overshoot,
            )
        deblurred.save(dest_path, format="png")

    return ThreadingResult.PROCESSED


def _runner_imops_deblur_star(args: tuple[term.MessageQueue, Path, Path, DeblurParams, bool]) -> ThreadingResult:
    return _runner_imops_deblur(*args)


@imops_group.command(
    name="deblur",
    help="Deblur/sharpen images in a directory",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-dm",
    "--method",
    "method",
    type=click.Choice(["deconv", "edge"], case_sensitive=False),
    default="deconv",
    show_default=True,
    help="The deblurring method to use, deconv is a deconvolution pass, edge is an edge-masked unsharp mask",
)
@click.option(
    "-ra",
    "--radius",
    "radius",
    type=click.FloatRange(0.1, 10.0),
    default=0.8,
    show_default=True,
    help="The Gaussian blur radius in pixels, higher values target coarser softness, lower values target fine lines",
)
@click.option(
    "-s",
    "--strength",
    "strength",
    type=click.FloatRange(0.0, 5.0),
    default=None,
    help="How much of the correction is applied, higher values sharpen harder, lower values stay closer to the "
    "original (default is 0.65 for deconv and 0.85 for edge)",
)
@click.option(
    "-it",
    "--iterations",
    "iterations",
    type=options.ZERO_POSITIVE_INT,
    default=6,
    show_default=True,
    help="The number of refinement passes for the deconv method, higher values recover more detail and take longer, "
    "this option is ignored by the edge method",
)
@click.option(
    "-et",
    "--threshold",
    "threshold",
    type=click.FloatRange(0.0, 255.0),
    default=2.0,
    show_default=True,
    help="The edge threshold in 8-bit gray levels, higher values only sharpen stronger edges, lower values also touch "
    "faint edges and fine texture",
)
@click.option(
    "-ov",
    "--overshoot",
    "overshoot",
    type=click.FloatRange(0.0, 255.0),
    default=0.0,
    show_default=True,
    help="How far the result may exceed the local extremes in 8-bit steps, higher values look crisper but ring more",
)
@options.dest_output(optional=False)
@options.recursive
@options.threads
@options.force
@time_program
def image_ops_deblur(
    path_or_archive: Path,
    method: str,
    radius: float,
    strength: float | None,
    iterations: int,
    threshold: float,
    overshoot: float,
    dest_output: Path,
    recursive: bool,
    threads: int,
    force: bool,
) -> int | None:
    """Deblur/sharpen images in a directory, output will always be PNG"""

    if strength is None:
        strength = 0.85 if method.lower() == "edge" else 0.65

    candidates: list[Path] = []
    if not recursive:
        candidates.append(path_or_archive)
    else:
        console.info(f"Recursively collecting folder in {path_or_archive}...")
        for comic in file_handler.collect_all_comics(path_or_archive, dir_only=True):
            candidates.append(comic)
        console.info(f"Found {len(candidates)} archives/folders to deblur.")

    if len(candidates) <= 0:
        console.warning(f"No folders found in {path_or_archive}, nothing to do.")
        return 1

    deblur_params = DeblurParams(method.lower(), radius, strength, iterations, threshold, overshoot)
    console.info(f"Using {threads} CPU threads for processing.")
    console.info(
        f"Using deblur method: {deblur_params.method}, radius: {radius}, strength: {strength}, "
        f"iterations: {iterations}, threshold: {threshold}, overshoot: {overshoot}"
    )

    for path_real in candidates:
        if recursive:
            console.info(f"Processing: {path_real}")
        image_candidates: list[Path] = [
            img_path for img_path, _, _, _ in file_handler.collect_image_from_folder(path_real)
        ]
        total_images = len(image_candidates)
        if total_images <= 0:
            console.warning(f"No images found in {path_real}, skipping.")
            continue

        real_output = dest_output
        if recursive:
            real_output = dest_output / path_real.name

        real_output.mkdir(parents=True, exist_ok=True)

        progress = console.make_progress()
        task = progress.add_task("Deblurring images...", finished_text="Deblurred images", total=total_images)

        with threaded_worker(console, lowest_or(threads, image_candidates)) as (pool, log_q):
            for _ in pool.imap_unordered(
                _runner_imops_deblur_star,
                ((log_q, image, real_output, deblur_params, force) for image in image_candidates),
            ):
                progress.update(task, advance=1)
        console.stop_progress(progress, f"Deblurred {total_images} images.")
    if recursive:
        console.info(f"Finished processing {len(candidates)} folders.")
    return None

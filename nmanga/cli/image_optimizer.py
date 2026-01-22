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

# Image optimizer command
# This file is part of nmanga.
from __future__ import annotations

import shutil
import subprocess as sp
from pathlib import Path

import rich_click as click
from click.exceptions import Exit

from .. import file_handler, term
from ..common import lowest_or, optimize_images, threaded_worker
from . import options
from ._deco import check_config_first, time_program
from .base import NMangaCommandHandler, is_executeable_global_path, test_or_find_cjpegli, test_or_find_pingo

console = term.get_console()


@click.command(
    name="optimize",
    help="Optimize images with pingo",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-ax",
    "--aggressive",
    "aggresive_mode",
    default=False,
    show_default=True,
)
@options.pingo_path
@check_config_first
@time_program
def image_optimizer(
    path_or_archive: Path,
    aggresive_mode: bool,
    pingo_path: str,
):  # pragma: no cover
    """
    Optimize images with pingo
    """

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    force_search = not is_executeable_global_path(pingo_path, "exiftool")
    pingo_exe = test_or_find_pingo(pingo_path, force_search)
    if pingo_exe is None:
        console.error("pingo not found, unable to optimize images!")
        raise Exit(1)

    console.info(f"Using pingo at {pingo_exe}")
    console.info("Optimizing images...")
    optimize_images(pingo_exe, path_or_archive, aggresive_mode)


def _wrapper_jpegify_threaded(
    log_q: term.MessageOrInterface,
    img_path: Path,
    output_dir: Path,
    cjpegli: str,
    quality: int,
    force: bool,
) -> None:
    dest_path = output_dir / f"{img_path.stem}.jpg"
    if dest_path.exists() and not force:
        cnsl = term.with_thread_queue(log_q)
        cnsl.warning(f"Skipping existing file: {dest_path}")
        return

    cmd = [cjpegli, "-q", str(quality), str(img_path), str(dest_path)]
    sp.run(cmd, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)


def _wrapper_jpegify_threaded_star(
    args: tuple[term.MessageQueue, Path, Path, str, int, bool],
) -> None:
    return _wrapper_jpegify_threaded(*args)


@click.command(
    name="jpegify",
    help="Convert images to JPEG to save space",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-q",
    "--quality",
    "jpeg_quality",
    default=98,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Quality of the output JPEG images (1-100)",
)
@options.dest_output(optional=False)
@options.cjpegli_path
@options.threads
@options.recursive
@options.force
@check_config_first
@time_program
def image_jpegify(
    path_or_archive: Path,
    jpeg_quality: int,
    dest_output: Path,
    cjpegli_path: str,
    threads: int,
    recursive: bool,
    force: bool,
):  # pragma: no cover
    """
    Convert images to JPEG to save space
    """

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    force_search = not is_executeable_global_path(cjpegli_path, "cjpegli")
    cjpegli_exe = test_or_find_cjpegli(cjpegli_path, force_search)
    if cjpegli_exe is None:
        console.error("cjpegli not found, unable to convert images to JPEG!")
        raise Exit(1)

    console.info(f"Using cjpegli at {cjpegli_exe}")

    candidates: list[Path] = []
    if not recursive:
        candidates.append(path_or_archive)
    else:
        console.info(f"Recursively collecting folder in {path_or_archive}...")
        for comic in file_handler.collect_all_comics(path_or_archive, dir_only=True):
            candidates.append(comic)
        console.info(f"Found {len(candidates)} archives/folders to jpegify.")

    for path_real in candidates:
        if recursive:
            console.info(f"Processing: {path_real}")
        image_candidates: list[Path] = [
            img_path for img_path, _, _, _ in file_handler.collect_image_from_folder(path_real)
        ]

        real_output = dest_output
        if recursive:
            real_output = dest_output / path_real.name
        real_output.mkdir(parents=True, exist_ok=True)

        total_images = len(image_candidates)
        if total_images == 0:
            console.info(f"No images found in {path_real}, skipping.")
            continue
        quality = max(0, min(100, jpeg_quality))

        progress = console.make_progress()
        task = progress.add_task("Converting images...", finished_text="Converted images", total=total_images)

        console.info(f"Using {threads} CPU threads for processing.")
        with threaded_worker(console, lowest_or(threads, image_candidates)) as (pool, log_q):
            for _ in pool.imap_unordered(
                _wrapper_jpegify_threaded_star,
                ((log_q, image, real_output, cjpegli_exe, quality, force) for image in image_candidates),
            ):
                progress.update(task, advance=1)
        console.stop_progress(progress, f"Converted {total_images} images to JPEG.")
    if recursive:
        console.info(f"Finished processing {len(candidates)} folders.")


@click.command(
    name="mixmatch",
    help="Mix and match images from two directories",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True, param_name="dir_a")
@options.path_or_archive(disable_archive=True, param_name="dir_b")
@options.dest_output(optional=False)
@check_config_first
@time_program
def image_mixmatch(
    dir_a: Path,
    dir_b: Path,
    dest_output: Path,
):
    """
    Mix and match images from two directories
    """

    if not dir_a.is_dir():
        raise click.BadParameter(
            f"{dir_a} is not a directory. Please provide a directory.",
            param_hint="dir_a",
        )

    if not dir_b.is_dir():
        raise click.BadParameter(
            f"{dir_b} is not a directory. Please provide a directory.",
            param_hint="dir_b",
        )

    console.info(f"Mixing and matching images from {dir_a} and {dir_b}...")

    images_a = sorted([img_path for img_path, _, _, _ in file_handler.collect_image_from_folder(dir_a)])
    images_b = sorted([img_path for img_path, _, _, _ in file_handler.collect_image_from_folder(dir_b)])

    total_images = min(len(images_a), len(images_b))
    console.info(f"Found {len(images_a)} images in dir_a and {len(images_b)} images in dir_b.")
    console.info(f"Mixing and matching {total_images} images...")

    dest_output.mkdir(parents=True, exist_ok=True)

    progress = console.make_progress()
    task = progress.add_task(
        "Mixing and matching images...", finished_text="Mixed and matched images", total=total_images
    )

    for file_a, file_b in zip(images_a, images_b, strict=True):
        console.log(f"Dir A Image: {file_a}, Dir B Image: {file_b}")
        a_stem = file_a.stem
        b_stem = file_b.stem
        if a_stem != b_stem:
            console.warning(f"Image name mismatch: {a_stem} vs {b_stem}")
            progress.update(task, advance=1)
            continue

        a_size = file_a.stat().st_size
        b_size = file_b.stat().st_size
        console.log(f"Sizes: Dir A: {a_size}, Dir B: {b_size}")
        # Take smaller
        chosen_file = file_a if a_size <= b_size else file_b
        dest_file = dest_output / chosen_file.name
        if dest_file.exists():
            console.warning(f"Skipping existing file: {dest_file}")
        else:
            # Copy
            shutil.copy2(chosen_file, dest_file)
        progress.update(task, advance=1)

    console.stop_progress(progress, f"Mixed and matched {total_images} images.")

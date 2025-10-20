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

import subprocess as sp
from pathlib import Path

import click

from .. import file_handler, term
from ..common import optimize_images, threaded_worker
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
        raise click.exceptions.Exit(1)

    console.info(f"Using pingo at {pingo_exe}")
    console.info("Optimizing images...")
    optimize_images(pingo_exe, path_or_archive, aggresive_mode)


def _wrapper_jpegify_threaded(
    img_path: Path,
    output_dir: Path,
    cjpegli: str,
    quality: int,
) -> None:
    dest_path = output_dir / f"{img_path.stem}.jpg"
    if dest_path.exists():
        console.warning(f"Skipping existing file: {dest_path}")
        return

    cmd = [cjpegli, "-q", str(quality), str(img_path), str(dest_path)]
    sp.run(cmd, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)


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
@check_config_first
@time_program
def image_jpegify(
    path_or_archive: Path,
    jpeg_quality: int,
    dest_output: Path,
    cjpegli_path: str,
    threads: int,
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
        raise click.exceptions.Exit(1)

    console.info(f"Using cjpegli at {cjpegli_exe}")

    image_candidates: list[Path] = [
        img_path for img_path, _, _, _ in file_handler.collect_image_from_folder(path_or_archive)
    ]

    dest_output.mkdir(parents=True, exist_ok=True)

    total_images = len(image_candidates)
    console.status(f"Converting {total_images} images to JPEG with cjpegli...")
    quality = max(0, min(100, jpeg_quality))
    if threads > 1:
        console.info(f"Using {threads} CPU threads for processing.")
        with threaded_worker(console, threads) as pool:
            pool.starmap(
                _wrapper_jpegify_threaded,
                [(image, dest_output, cjpegli_exe, quality) for image in image_candidates],
            )
    else:
        for idx, image in enumerate(image_candidates):
            console.status(f"Converting image to JPEG... [{idx + 1}/{total_images}]")
            _wrapper_jpegify_threaded(image, dest_output, cjpegli_exe, quality)
    console.stop_status(f"Converted {total_images} images to JPEG.")

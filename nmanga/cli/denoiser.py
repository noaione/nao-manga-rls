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

# Bulk denoise images in a directory using Waifu2x-TensorRT
# https://github.com/z3lx/waifu2x-tensorrt

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional
from uuid import uuid4

import click

from nmanga import file_handler

from .. import term
from . import options
from ._deco import check_config_first, time_program
from .base import NMangaCommandHandler, test_or_find_magick, test_or_find_w2x_trt

console = term.get_console()


def get_precision_enum(precision: str) -> str:
    if precision == "fp16":
        return 1
    elif precision == "tf32":
        return 0
    else:
        raise ValueError("Invalid precision value, must be 'fp16' or 'tf32'")


@click.command(
    name="denoise",
    help="Denoise all images in a directory using Waifu2x-TensorRT",
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
    "-dn",
    "--denoise-level",
    "denoise_level",
    type=click.IntRange(0, 3),
    required=True,
    help="The denoise level to use, based on waifu2x-tensorrt levels",
)
@click.option(
    "-b",
    "--batch-size",
    "batch_size",
    type=options.POSITIVE_INT,
    default=32,
    show_default=True,
    help="The batch size to use for processing images, higher values use more VRAM",
)
@click.option(
    "-t",
    "--tile-size",
    "tile_size",
    type=click.Choice([64, 128, 256, 400, 640], case_sensitive=False),
    default=256,
    show_default=True,
    help="The tile size to use for processing images, higher values use more VRAM",
)
@click.option(
    "-p",
    "--precision",
    "precision",
    type=click.Choice(["fp16", "tf32"], case_sensitive=True),
    default="fp16",
    show_default=True,
    help="The precision to use for processing images, tf32 requires a more powerful GPU",
)
@click.option(
    "-tta",
    "--tta",
    "tta",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use TTA (Test Time Augmentation) for better quality, but slower processing",
)
@options.w2x_trt_path
@check_config_first
@time_program
def denoiser(
    path_or_archive: Path,
    dest_dir: Path,
    denoise_level: int,
    batch_size: int,
    tile_size: int,
    precision: str,
    tta: bool,
    w2x_trt_path: Optional[str],
):  # pragma: no cover
    """
    Automatically adjust the levels of all images in a directory based on local peaks in their histograms.
    """

    w2x_trt_exe = test_or_find_w2x_trt(w2x_trt_path, False)
    if w2x_trt_exe is None:
        console.error("Could not find the waifu2x-tensorrt executable, please configure it first.")
        return 1
    console.info("Using waifu2x-tensorrt executable: {}".format(w2x_trt_exe))

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    with file_handler.MangaArchive(path_or_archive) as archive:
        all_files: list[Path] = archive.contents()

    total_files = len(all_files)
    console.info(f"Found {total_files} files in the directory.")
    console.info(f"Using denoise level: {denoise_level}")
    console.info(f"Using batch size: {batch_size}")
    console.info(f"Using tile size: {tile_size}")
    console.info(f"Using precision: {precision}")
    console.info(f"TTA enabled?: {tta}")

    console.info("Destination directory: {}".format(dest_dir))

    is_continue = console.confirm("Proceed with denoising?")
    if not is_continue:
        console.info("Aborting denoise.")
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)

    precision_enum = get_precision_enum(precision)

    base_params = [
        str(w2x_trt_exe),
        "--model", "cunet/art",
        "--scale", "1",
        "--noise", str(denoise_level),
        "--batchSize", str(batch_size),
        "--tileSize", str(tile_size),
        "--precision", str(precision_enum),
        "render",
    ]
    final_params = ["-o", str(dest_dir)]

    console.status("Denoising images...")
    errors = []
    for idx, image in enumerate(all_files):
        params = [
            *base_params,
            "-i", str(image),
        ]
        if tta:
            params.append("--tta")
        params.extend(final_params)

        console.debug("Running command: {}".format(" ".join(params)))
        console.status(f"Denoising images... [{idx + 1}/{total_files}]")
        # silent output
        result = subprocess.run(params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            console.error(f"Error denoising image: {image}, skipping...")
            # get stderr output
            console.debug("Error output:")
            console.debug(result.stderr.decode("utf-8"))
            console.debug("STDOUT output:")
            console.debug(result.stdout.decode("utf-8"))
            errors.append(image)
            continue
    correct_amount = total_files - len(errors)
    console.stop_status(f"Denoised all {correct_amount} images.")
    if len(errors) > 0:
        console.error(f"Failed to denoise {len(errors)} images:")
        for err in errors:
            console.error(f"- {err}")
        return 1
    return 0


def _is_default_path(path: str) -> bool:
    path = path.lower()
    if path == "magick":
        return True
    if path == "./magick":
        return True
    if path == ".\\magick":
        return True
    return False


def make_prefix_identify(magick_exe: str):
    name = Path(magick_exe).name
    if name.lower() == "identify":
        return ["identify"]
    return ["magick", "identify"]


def recommended_level(quality: int) -> int:
    if quality < 0 or quality > 100:
        return -1  # Invalid quality

    if quality > 95:
        return -1
    elif quality >= 92:
        return 0
    elif quality >= 81:
        return 1
    elif quality >= 69:
        return 2
    else:
        return 3


@click.command(
    name="identify-quality",
    help="Identify images that may benefit from denoising using Waifu2x-TensorRT",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.magick_path
@time_program
def identify_denoise_candidates(
    path_or_archive: Path,
    magick_path: str,
):
    """
    Identify images that may benefit from denoising using Waifu2x-TensorRT
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

    cwd = Path.cwd()
    dump_data_path = f"{uuid4().hex.replace('-', '')}_identified.json"
    dump_data = []

    common_counters = {}
    console.info("Scanning images for quality...")
    for idx, (image, _, total_img, _) in enumerate(file_handler.collect_image_from_folder(path_or_archive)):
        console.status(f"Scanning images for quality... [{idx + 1}/{total_img}]")
        path_obj = image.resolve()
        img_ext = path_obj.suffix.lower().strip(".")
        if img_ext not in ["jpg", "jpeg"]:
            console.debug(f"Skipping non-JPEG image: {path_obj}")
            continue

        params = [*make_prefix_identify(magick_exe), "-format", "%Q", str(path_obj)]
        console.debug(f"Running command: {' '.join(params)}")
        executed = subprocess.run(
            params,
            capture_output=True,
            text=True,
        )
        if executed.returncode != 0:
            console.error(f"Error identifying image: {path_obj}, skipping...")
            continue
        quality_str = executed.stdout.strip()
        try:
            quality = int(quality_str)
            dump_data.append({"img": str(image), "q": quality})
        except ValueError:
            console.error(f"Error parsing quality for image: {path_obj}, got '{quality_str}', skipping...")
            continue
        common_counters[quality_str] = common_counters.get(quality_str, 0) + 1
    console.stop_status("Scanned all images.")

    # Save results to JSON file
    dump_path = cwd / dump_data_path
    dump_path.write_text(json.dumps(dump_data, indent=4), encoding="utf-8")

    console.info("Image quality distribution:")
    for quality, count in sorted(common_counters.items(), key=lambda x: int(x[0]) if x[0].isdigit() else -1):
        console.info(f"- Quality {quality}: {count} images")
        quality_int = int(quality) if quality.isdigit() else -1
        den_level = recommended_level(quality_int)
        if den_level >= 0:
            console.info(f"  Recommended denoise level: {den_level}")
        else:
            console.info("  No denoising recommended")

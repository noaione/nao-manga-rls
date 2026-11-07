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

import importlib.util
from pathlib import Path
from typing import Literal

import rich_click as click
from PIL import Image

from .. import file_handler, term
from ..ml.contracts import MLDataType
from ..ml.onnx_backend import denoise_single_image_with_overlap
from ..ml.onnx_session import prepare_model_runtime_builders
from . import options
from ._deco import check_config_first, time_program
from .base import NMangaCommandHandler

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)
console = term.get_console()


@click.command("upscale-trt", help="(Experimental) Upscale images using TensorRT models.", cls=NMangaCommandHandler)
@options.path_or_archive(disable_archive=True)
@options.dest_output()
@click.option(
    "-m",
    "--model",
    "model_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="The ONNX model file to use for denoising",
)
@click.option(
    "-d",
    "--device",
    "device_id",
    type=options.ZERO_POSITIVE_INT,
    default=0,
    show_default=True,
    help="The GPU device ID to use for processing",
)
@click.option(
    "-b",
    "--batch-size",
    "batch_size",
    type=options.POSITIVE_INT,
    default=64,
    show_default=True,
    help="The batch size to use for processing images, higher values use more VRAM",
)
@click.option(
    "-t",
    "--tile-size",
    "tile_size",
    type=options.POSITIVE_INT,
    default=128,
    show_default=True,
    help="The tile size to use for processing images, higher values use more VRAM",
)
@click.option(
    "-to",
    "--tile-overlap",
    "tile_overlap",
    type=options.ZERO_POSITIVE_INT,
    default=0,
    show_default=True,
    help="The tile overlap to use for processing images",
)
@click.option(
    "-cs",
    "--contrast-stretch",
    "contrast_stretch",
    is_flag=True,
    default=False,
    show_default=True,
    help="Apply contrast stretching in pre-processing, in some cases this could made it worse",
)
@click.option(
    "-bg",
    "--background",
    "background",
    type=click.Choice(["black", "white"], case_sensitive=True),
    default="black",
    show_default=True,
    help="The background color to use for padding",
)
@click.option(
    "-qs",
    "--quant-size",
    "quant_size",
    type=click.Choice(["fp16", "fp32", "bf16"], case_sensitive=False),
    default="fp16",
    show_default=True,
    help="The quantization size to use for inference",
)
@click.option(
    "-rtx",
    "--with-nvrtx",
    "with_nvidia_rtx",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use RTX for inference",
)
@click.option(
    "-ws",
    "--workspace-size",
    "workspace_size",
    type=options.POSITIVE_INT,
    default=None,
    help="The TensorRT workspace limit in MiB, defaults to 75% of the total GPU memory",
)
@click.option(
    "-cg",
    "--cuda-graph/--no-cuda-graph",
    "enable_cuda_graph",
    default=True,
    show_default=True,
    help="Enable CUDA graph replay for the TensorRT RTX engine",
)
@click.option(
    "-sep",
    "--strict-ep",
    "strict_ep",
    is_flag=True,
    default=False,
    show_default=True,
    help="Fail instead of falling back when the requested execution provider is unavailable",
)
@click.option(
    "--no-cpu",
    "no_cpu_fallback",
    is_flag=True,
    default=False,
    show_default=True,
    help="Disable CPU fallback for execution provider",
)
@options.recursive
@time_program
@check_config_first
def upscale_trt(
    path_or_archive: Path,
    dest_output: Path,
    model_path: Path,
    device_id: int,
    batch_size: int,
    tile_size: int,
    tile_overlap: int,
    contrast_stretch: bool,
    background: Literal["black", "white"],
    quant_size: str,
    with_nvidia_rtx: bool,
    workspace_size: int | None,
    enable_cuda_graph: bool,
    strict_ep: bool,
    no_cpu_fallback: bool,
    recursive: bool,
):
    """Upscale images using TensorRT models."""

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    if tile_overlap < 0 or tile_overlap * 2 >= tile_size:
        raise click.BadParameter(
            "Tile overlap must be non-negative and less than half the tile size.", param_hint="tile_overlap"
        )

    candidates: list[Path] = []
    if not recursive:
        candidates.append(path_or_archive)
    else:
        console.info(f"Recursively collecting folder in {path_or_archive}...")
        for comic in file_handler.collect_all_comics(path_or_archive, dir_only=True):
            candidates.append(comic)
        console.info(f"Found {len(candidates)} archives/folders to upscale.")

    if not candidates and recursive:
        console.warning("No valid folders found to upscale.")
        return 1

    console.info(f"Output folder: {dest_output.resolve()} (recursive? {recursive!r})")

    is_tensorrt_available = importlib.util.find_spec("tensorrt")

    # Try importing stuff here
    ml_quant = MLDataType.from_str(quant_size)
    console.info(f"Loading model: {model_path.name}...")
    console.info(f"Using device ID: {device_id}")
    if is_tensorrt_available is not None:
        console.info(
            f"Building/loading TensorRT Engine for {model_path.name}. "
            + "If it is building, you may need to wait up to 20-25 minutes for the first time."
        )
    sess = prepare_model_runtime_builders(
        model_path,
        device_id=device_id,
        is_verbose=console.debugged,
        tile_size=tile_size,
        batch_size=batch_size,
        data_type=ml_quant,
        with_nvrtx=with_nvidia_rtx,
        workspace_size=None if workspace_size is None else workspace_size * 1024 * 1024,
        enable_cuda_graph=enable_cuda_graph,
        strict_ep=strict_ep,
        disable_cpu=no_cpu_fallback,
    )

    if scale_factor := getattr(sess, "scale_factor", None):
        console.info(f"Detected model scale factor: x{scale_factor}")

    console.info("Model loaded successfully. Starting upscaling process...")

    for candidate in candidates:
        if recursive:
            console.info(f"Processing: {candidate}")
        all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(candidate)]
        total_files = len(all_files)
        if total_files <= 0:
            console.warning(f"No images found in {candidate}, skipping.")
            continue

        real_output = dest_output
        if recursive:
            real_output = dest_output / candidate.name
        real_output.mkdir(parents=True, exist_ok=True)

        progress = console.make_progress()
        task = console.make_task(progress, "Upscaling images...", total=total_files)

        for image_file in all_files:
            output_path = real_output / f"{image_file.stem}.png"

            img_file = Image.open(image_file)
            output_image = denoise_single_image_with_overlap(
                img_file,
                sess,
                batch_size=batch_size,
                tile_size=tile_size,
                contrast_stretch=contrast_stretch,
                background=background,
                tile_overlap=tile_overlap,
                use_fp32=not sess.use_halfp,
            )

            output_image.save(output_path, format="PNG")
            img_file.close()
            output_image.close()
            progress.advance(task)
        console.stop_progress(progress, f"Upscaled all {total_files} images to {real_output}")
    if recursive:
        console.info(f"Finished processing {len(candidates)} folders.")
    return 0

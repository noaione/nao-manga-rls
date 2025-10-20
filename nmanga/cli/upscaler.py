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

import sys
import traceback
from pathlib import Path

import click
from PIL import Image, ImageDraw

from .. import file_handler, term
from . import options
from ._deco import check_config_first, time_program
from .base import NMangaCommandHandler

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)
console = term.get_console()


@click.command(
    name="upscale-tiled",
    help="(Experimental) Upscale images with tiling to reduce VRAM usage and seams",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.dest_output()
@click.option(
    "-m",
    "--model",
    "model_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="The ONNX model file to use for upscaling",
)
@click.option(
    "-s",
    "--overlap-seam",
    "seam_size",
    type=options.POSITIVE_INT,
    default=16,
    show_default=True,
    help="The overlap size between tiles to reduce seam artifacts",
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
    default=32,
    show_default=True,
    help="The batch size to use for processing tiles, higher values use more VRAM",
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
@check_config_first
@time_program
def upscale_tiled(
    path_or_archive: Path,
    dest_output: Path,
    model_path: Path,
    seam_size: int,
    device_id: int,
    batch_size: int,
    tile_size: int,
):
    """
    Upscale images using a tiled approach with overlapping seams to prevent artifacts.
    """
    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(path_or_archive)]

    console.info("Importing required packages...")
    try:
        import numpy as np
        import onnxruntime as ort
        from einops import rearrange
    except ImportError as e:
        console.error(f"Missing required package: {e.name}. Please install it first.")
        traceback.print_exc()
        return 1

    console.info(f"Loading model: {model_path.name}...")

    verbose_level = 3 if not console.debugged else 0
    ort.set_default_logger_severity(verbose_level)
    ort.set_default_logger_verbosity(verbose_level)
    console.info(f"ONNX Runtime version: {ort.__version__}")
    console.info(f"Using device ID: {device_id}")
    console.info(f"Available Execution Providers: {ort.get_available_providers()}")

    # If mac, use CoreML EP
    if sys.platform == "darwin":
        CACHE_DIR = Path.home() / ".cache" / "nmanga-denoiser"
        providers = [
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "1",
                    "EnableOnSubgraphs": "1",
                    "ModelCacheDirectory": str(CACHE_DIR),
                    "SpecializationStrategy": "FastPrediction",
                },
            )
        ]
    else:
        providers = [
            (
                # Force use of TensorRT if available
                "TensorrtExecutionProvider",
                {
                    "device_id": device_id,
                    "trt_fp16_enable": True,
                    "trt_sparsity_enable": True,
                },
            ),
        ]

    sess_opt = ort.SessionOptions()
    sess = ort.InferenceSession(
        str(model_path),
        sess_options=sess_opt,
        providers=providers,
    )

    first_input = sess.get_inputs()[0]
    first_output = sess.get_outputs()[0]
    upscale_factor = first_output.shape[2] // first_input.shape[2]
    if first_output.shape[3] // first_input.shape[3] != upscale_factor:
        console.error("The model does not have a consistent upscale factor!")
        return 1
    if first_input.shape[1] != 3:
        console.error("The model does not have 3 input channels (RGB)!")
        return 1

    input_name = first_input.name
    output_name = first_output.name

    total_files = len(all_files)
    dest_output.mkdir(parents=True, exist_ok=True)

    upscaled_tile_size = tile_size * upscale_factor
    upscaled_seam_size = seam_size * upscale_factor

    mask_left = Image.new("L", (upscaled_seam_size, upscaled_tile_size), 0)
    draw_left = ImageDraw.Draw(mask_left)
    for i in range(upscaled_seam_size):
        alpha = int(255 * (i / (upscaled_seam_size - 1)))
        draw_left.line([(i, 0), (i, upscaled_tile_size)], fill=alpha)

    mask_top = Image.new("L", (upscaled_tile_size, upscaled_seam_size), 0)
    draw_top = ImageDraw.Draw(mask_top)
    for i in range(upscaled_seam_size):
        alpha = int(255 * (i / (upscaled_seam_size - 1)))
        draw_top.line([(0, i), (upscaled_tile_size, i)], fill=alpha)

    mask_corner = Image.new("L", (upscaled_seam_size, upscaled_seam_size), 0)
    for x in range(upscaled_seam_size):
        for y in range(upscaled_seam_size):
            alpha_x = int(255 * (x / (upscaled_seam_size - 1)))
            alpha_y = int(255 * (y / (upscaled_seam_size - 1)))
            mask_corner.putpixel((x, y), min(alpha_x, alpha_y))

    console.status("Upscaling [???/???]...")
    for idx, image_file in enumerate(all_files):
        console.status(f"Upscaling [{idx + 1}/{total_files}]...")
        output_path = dest_output / f"{image_file.stem}.png"

        img = Image.open(image_file).convert("RGB")
        source_width, source_height = img.size

        upscaled_width = source_width * upscale_factor
        upscaled_height = source_height * upscale_factor
        merged_image = Image.new("RGB", (upscaled_width, upscaled_height))

        step_size = tile_size - seam_size

        tile_infos = []
        for y in range(0, source_height, step_size):
            for x in range(0, source_width, step_size):
                box = (x, y, x + tile_size, y + tile_size)
                tile = img.crop(box)

                if tile.size[0] < tile_size or tile.size[1] < tile_size:
                    padded_tile = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                    padded_tile.paste(tile, (0, 0))
                    tile = padded_tile

                tile_infos.append({"x": x, "y": y, "tile": tile})

        for i in range(0, len(tile_infos), batch_size):
            batch_data = tile_infos[i : i + batch_size]

            batch_tiles = [info["tile"] for info in batch_data]

            # Pre-process batch
            # The line below was changed from np.float32 to np.float16
            np_batch = np.array([np.array(t) for t in batch_tiles], dtype=np.float16) / 255.0
            np_batch = rearrange(np_batch, "b h w c -> b c h w")

            # Run inference
            model_output = sess.run([output_name], {input_name: np_batch})[0]

            # Post-process batch
            output_batch = rearrange(model_output, "b c h w -> b h w c")
            # Ensure output_batch is a concrete numpy array with a numeric dtype before clipping
            output_batch = np.asarray(output_batch, dtype=np.float32)
            output_batch = np.clip(output_batch, 0.0, 1.0) * 255.0
            output_batch = output_batch.astype(np.uint8)

            for j, info in enumerate(batch_data):
                upscaled_tile_arr = output_batch[j]
                upscaled_tile = Image.fromarray(upscaled_tile_arr)

                del info["tile"]  # free memory
                info["tile_up"] = upscaled_tile

        for tile_info in tile_infos:
            x, y = tile_info["x"], tile_info["y"]

            upscaled_tile: Image.Image = tile_info["tile_up"]

            paste_x = (x // step_size) * (upscaled_tile_size - upscaled_seam_size)
            paste_y = (y // step_size) * (upscaled_tile_size - upscaled_seam_size)

            current_mask = Image.new("L", (upscaled_tile_size, upscaled_tile_size), 255)
            is_left_edge = x == 0
            is_top_edge = y == 0

            if not is_left_edge:
                current_mask.paste(mask_left, (0, 0))
            if not is_top_edge:
                current_mask.paste(mask_top, (0, 0))
            if not is_left_edge and not is_top_edge:
                current_mask.paste(mask_corner, (0, 0))

            merged_image.paste(upscaled_tile, (paste_x, paste_y), mask=current_mask)

        final_image = merged_image.crop((0, 0, upscaled_width, upscaled_height))
        final_image.save(output_path, "PNG")

    console.stop_status(f"Finished upscaling all {total_files} images.")

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

# Bulk denoise images in a directory using Waifu2x-TensorRT or ONNX Runtime
# https://github.com/z3lx/waifu2x-tensorrt
# Part of the code is adapted from anon

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional
from uuid import uuid4

import click
from PIL import Image

from .. import file_handler, term
from . import options
from ._deco import check_config_first, time_program
from .base import NMangaCommandHandler, test_or_find_magick, test_or_find_w2x_trt

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)
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
@options.dest_dir
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

    all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(path_or_archive)]

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
        "--model",
        "cunet/art",
        "--scale",
        "1",
        "--noise",
        str(denoise_level),
        "--batchSize",
        str(batch_size),
        "--tileSize",
        str(tile_size),
        "--precision",
        str(precision_enum),
        "render",
    ]
    final_params = ["-o", str(dest_dir)]

    console.status("Denoising images...")
    errors = []
    for idx, image in enumerate(all_files):
        params = [
            *base_params,
            "-i",
            str(image),
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


@click.command(
    name="denoise-trt",
    help="(Experimental) Denoise all images using TensorRT/ONNX Runtime",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.dest_dir
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
@check_config_first
@time_program
def denoiser_trt(
    path_or_archive: Path,
    dest_dir: Path,
    model_path: Path,
    device_id: int,
    batch_size: int,
    tile_size: int,
    contrast_stretch: bool,
    background: str,
):
    """
    Denoise all images using TensorRT/ONNX Runtime (Experimental)

    All of this code is originally created by anon.
    Adapted to fit into nmanga by me
    """

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(path_or_archive)]

    # Try importing stuff here
    console.info("Importing required packages...")
    try:
        import numpy as np  # type: ignore
        import onnxruntime as ort  # type: ignore
        from einops import rearrange  # type: ignore
    except ImportError as e:
        console.error(f"Missing required package: {e.name}. Please install it first.")
        return 1

    console.info(f"Loading model: {model_path.name}...")

    verbose_level = 3 if not console.debugged else 0
    ort.set_default_logger_severity(verbose_level)
    ort.set_default_logger_verbosity(verbose_level)

    sess_opt = ort.SessionOptions()
    sess = ort.InferenceSession(
        model_path,
        sess_options=sess_opt,
        providers=[
            (
                # Force use of TensorRT if available
                "TensorrtExecutionProvider",
                {
                    "device_id": device_id,
                    "trt_fp16_enable": True,
                    "trt_sparsity_enable": True,
                },
            ),
        ],
    )

    total_files = len(all_files)
    dest_dir.mkdir(parents=True, exist_ok=True)

    console.status(f"Denoising images... [???/{total_files}]")
    for idx, image_file in enumerate(all_files):
        console.status(f"Denoising images... [{idx + 1}/{total_files}]")
        output_path = dest_dir / f"{image_file.stem}.png"

        img_file = Image.open(image_file)
        orig_img_mode = img_file.mode
        orig_palette = img_file.palette

        img = img_file.convert("RGB")
        source_width, source_height = img.size
        # compute the new canvas size
        tile_count_height = int(np.ceil((source_height / tile_size)))
        tile_count_width = int(np.ceil((source_width / tile_size)))

        padded_height = int(tile_count_height * tile_size)
        padded_width = int(tile_count_width * tile_size)
        background_tuple = (0, 0, 0) if background == "black" else (255, 255, 255)

        new_padded_image = Image.new("RGB", (padded_width, padded_height), background_tuple)
        new_padded_image.paste(img, (0, 0))

        # Pre-process
        image_array = np.array(new_padded_image)
        # contrast stretching: scaling the image based on its own darkest and brightest pixels.
        # For example, a very dark photo (e.g., pixel values from 10 to 50) and a very bright photo (e.g., values
        # from 200 to 250) will both be stretched to the full [0.0, 1.0] range.
        # The model loses all information about the image's absolute brightness. Can be desirable with a white
        # background to reverse CMYK shift in rare (color) images.
        # Not desirable in general so the alternative path should be taken in the vast majority of cases.
        # See also: https://en.wikipedia.org/wiki/Normalization_(image_processing)#Contrast_Stretching_for_Image_Enhancement
        if contrast_stretch:
            image_array = (image_array - np.min(image_array)) / np.ptp(image_array)
        else:
            image_array = image_array / 255.0
        # Casting to FP16 since we use FP16 quantized models wherever possible
        image_array = image_array.astype(np.float16)
        # Rearranging (Width, Height, Channel) -> (Channel, Width, Height) to match the expected input shape
        image_array = rearrange(image_array, "w h c -> c w h")

        padded_tiles = rearrange(
            image_array,
            "c (h th) (w tw) -> (h w) c th tw",
            th=tile_size,
            tw=tile_size,
        )

        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        all_output_chunks = []
        num_chunks = padded_tiles.shape[0]

        for i in range(0, num_chunks, batch_size):
            batch_of_chunks = padded_tiles[i : (i + batch_size if i + batch_size < num_chunks else num_chunks)]

            # infer
            model_output = sess.run([output_name], {input_name: batch_of_chunks})
            # append the output
            all_output_chunks.append(model_output[0])

        # concat all chunks
        tiled_output_image = np.concatenate(all_output_chunks, axis=0)

        reconstructed_image_with_pad = rearrange(
            tiled_output_image,
            "(h w) c th tw -> (h th) (w tw) c",
            h=tile_count_height,
            w=tile_count_width,
        )

        # post process
        if contrast_stretch:
            postprocessed_array = reconstructed_image_with_pad.astype(np.float16)
            postprocessed_array = (postprocessed_array - np.min(postprocessed_array)) / np.ptp(postprocessed_array)
        else:
            postprocessed_array = np.clip(reconstructed_image_with_pad, 0.0, 1.0)

        # scaling back to [0, 255] from [0.0, 1.0]
        postprocessed_array = postprocessed_array * 255.0

        # rounding and cast back to uint8
        postprocessed_array = np.round(postprocessed_array)
        postprocessed_array = postprocessed_array.astype(np.uint8)
        output_image = Image.fromarray(postprocessed_array)

        output_image = output_image.crop((0, 0, source_width, source_height))
        if orig_palette:
            palette_image = Image.new("P", (1, 1))
            palette_image.putpalette(orig_palette)
            output_image = output_image.convert("RGB").quantize(palette=palette_image, dither=Image.Dither.NONE)
        else:
            output_image = output_image.convert(orig_img_mode)

        output_image.save(output_path, format="PNG")
    console.stop_status(f"Denoised all {total_files} images.")

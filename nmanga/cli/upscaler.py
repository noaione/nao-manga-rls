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

import importlib
import importlib.util
from pathlib import Path
from typing import Literal

import rich_click as click
from PIL import Image

from .. import file_handler, term
from ..denoiser import MLDataType, denoise_single_image, prepare_model_runtime_builders
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
    contrast_stretch: bool,
    background: Literal["black", "white"],
    quant_size: str,
    recursive: bool,
):
    """Upscale images using TensorRT models."""

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    candidates: list[Path] = []
    if not recursive:
        candidates.append(path_or_archive)
    else:
        console.info(f"Recursively collecting folder in {path_or_archive}...")
        for comic in file_handler.collect_all_comics(path_or_archive, dir_only=True):
            candidates.append(comic)
        console.info(f"Found {len(candidates)} archives/folders to denoise.")

    if not candidates and recursive:
        console.warning("No valid folders found to denoise.")
        return 1

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
    )

    console.info("Model loaded successfully. Starting upscaling process...")

    for candidate in candidates:
        if recursive:
            console.info(f"Processing: {candidate}")
        all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(candidate)]
        total_files = len(all_files)

        real_output = dest_output
        if recursive:
            real_output = dest_output / candidate.name
        real_output.mkdir(parents=True, exist_ok=True)

        progress = console.make_progress()
        task = console.make_task(progress, "Upscaling images...", total=total_files)

        for image_file in all_files:
            output_path = real_output / f"{image_file.stem}.png"

            img_file = Image.open(image_file)
            output_image = denoise_single_image(
                img_file,
                sess,
                batch_size=1,
                tile_size=tile_size,
                contrast_stretch=contrast_stretch,
                background=background,
                use_fp32=True,
            )

            output_image.save(output_path, format="PNG")
            img_file.close()
            output_image.close()
            progress.advance(task)
        console.stop_progress(progress, f"Upscaled all {total_files} images to {real_output}")
    if recursive:
        console.info(f"Finished processing {len(candidates)} folders.")
    return 0


@click.command(
    "to-onnx",
    help="Export torch model or safetensors model to ONNX format",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=False, disable_folder=False)
@click.option(
    "-qs",
    "--quant-size",
    "quant_size",
    type=click.Choice(["fp32", "bf16"], case_sensitive=True),
    default="fp32",
)
@click.option(
    "--dynamo",
    "use_dynamo",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use torch dynamo to optimize the model before exporting to ONNX",
)
@click.option(
    "-op",
    "--opset",
    "opset_version",
    type=options.POSITIVE_INT,
    default=20,
    show_default=True,
    help="The ONNX opset version to use for exporting the model",
)
@time_program
def export_to_onnx(
    path_or_archive: Path,
    quant_size: Literal["fp32", "bf16"],
    use_dynamo: bool,
    opset_version: int,
):
    """Export torch model or safetensors model to ONNX format"""

    file_candidates: list[Path] = []
    if not path_or_archive.is_file():
        for file in path_or_archive.glob("*"):
            if file.suffix in [".pt", ".pth", ".safetensors"]:
                file_candidates.append(file)
    else:
        if path_or_archive.suffix in [".pt", ".pth", ".safetensors"]:
            file_candidates.append(path_or_archive)
        else:
            raise click.BadParameter(
                f"{path_or_archive} is not a valid model file. Please provide a .pt, .pth or .safetensors file.",
                param_hint="path_or_archive",
            )

    if not file_candidates:
        raise click.BadParameter(
            f"No valid model files found in {path_or_archive}. Please provide a .pt, .pth or .safetensors file.",
            param_hint="path_or_archive",
        )

    try:
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise click.ClickException("PyTorch is not installed. Please install it to use this command.") from exc

    dtypes_maps = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }

    torch_dtype = dtypes_maps.get(quant_size)
    if torch_dtype is None:
        raise click.ClickException(f"Unsupported quantization size: {quant_size}")

    safetensors_available = importlib.util.find_spec("safetensors")
    if safetensors_available is None:
        raise click.ClickException("safetensors is not installed. Please install it to use this command.")

    # Monkeypatch typing.Self for resselt
    if not hasattr(importlib.import_module("typing"), "Self"):
        import typing

        from typing_extensions import Self as TypingSelf

        typing.Self = TypingSelf  # type: ignore

    try:
        from resselt import load_from_file  # pyright: ignore[reportMissingImports]
        from resselt.factory.arch import ModelMetadata
    except ImportError as exc:
        raise click.ClickException("resselt is not installed. Please install it to use this command.") from exc

    model_sfx = f"{quant_size}_op20"
    if use_dynamo:
        model_sfx += "_dynamo"

    for model_file in file_candidates:
        # Export only in fp32 for now
        final_name = f"{model_file.stem}_{model_sfx}.onnx"
        model_dir = model_file.parent
        onnx_path = model_dir / final_name
        if onnx_path.exists():
            console.info(f"ONNX model {onnx_path.name} already exists, skipping export.")
            continue

        console.info(f"Loading model file: {model_file.name}...")
        model_torch: torch.nn.Module = load_from_file(str(model_file))
        param_info = getattr(model_torch, "parameters_info", None)

        input_channel = 3
        if isinstance(param_info, ModelMetadata):
            input_channel = param_info.in_channels

        model_torch = model_torch.to("cpu", dtype=torch_dtype)
        autocast_ctx = torch.autocast(device_type="cpu", dtype=torch_dtype, enabled=True)
        input_tensor = torch.rand((1, input_channel, 32, 32))

        with autocast_ctx:
            torch.onnx.export(
                model_torch,
                (input_tensor,),
                str(onnx_path),
                external_data=False,
                opset_version=opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamo=use_dynamo,
                dynamic_axes={
                    "input": {0: "batch_size", 2: "width", 3: "height"},
                    "output": {0: "batch_size", 2: "width", 3: "height"},
                },
            )

        console.info(f"Exported {model_file.name} to {onnx_path.name}")

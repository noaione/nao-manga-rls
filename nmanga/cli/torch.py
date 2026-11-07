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

# PyTorch model tooling: run a checkpoint model directly, or export one to ONNX.
#
# `torch infer` loads the model through `resselt`, which detects the architecture and
# constructs it, so it needs neither an ONNX Runtime session nor a TensorRT engine.
# See docs/torch-infer-implementation-plan.md for the validated recipes and their limits.

from __future__ import annotations

import importlib
import importlib.util
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import rich_click as click
from PIL import Image

from .. import file_handler, term
from ..common import format_elapsed_time
from ..ml.contracts import DEFAULT_TILE_OVERLAP, DEFAULT_TILE_SIZE, TilingConfig, TorchRuntimeConfig
from . import options
from ._deco import check_config_first, time_program
from .base import NMangaCommandHandler

if TYPE_CHECKING:
    from ..ml.contracts import TorchPreset
    from ..ml.torch_backend import TorchRuntime

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)
console = term.get_console()

__all__ = ("torch_group",)


class DeviceParamType(click.ParamType):
    """The torch device to run the model on, as `auto`, `cpu` or `cuda`/`cuda:N`."""

    name = "device"

    def get_metavar(self, param: click.Parameter, ctx: click.Context) -> str | None:
        return "[auto|cpu|cuda:N]"

    def convert(self, value, param, ctx) -> str:
        if not isinstance(value, str):
            self.fail(f"{value!r} is not a valid device, expected auto, cpu or cuda:N", param, ctx)

        normalized = value.strip().lower()
        if normalized in {"auto", "cpu"}:
            return normalized
        if normalized == "cuda":
            return "cuda:0"
        if normalized.startswith("cuda:") and normalized.partition(":")[2].isdigit():
            return f"cuda:{int(normalized.partition(':')[2])}"
        self.fail(f"{value!r} is not a valid device, expected auto, cpu or cuda:N", param, ctx)


DEVICE = DeviceParamType()


def _describe_run(config: TorchRuntimeConfig, runtime: TorchRuntime) -> list[str]:
    """
    Describe the exact settings a run uses, so a failure can be reproduced or tuned.

    :param config: The requested runtime configuration
    :param runtime: The prepared runtime, which knows what was actually selected
    :return: One line per setting
    """
    info = runtime.info
    tiling = config.tiling
    return [
        f"Architecture: {info.architecture} (via {info.loader} {info.loader_version or 'unknown'})",
        f"Scale factor: x{info.scale} (channels: {info.in_channels} -> {info.out_channels})",
        f"Stored checkpoint dtype: {', '.join(info.stored_dtypes) or 'unknown'}",
        f"Runtime dtype: {runtime.dtype}",
        f"Device: {runtime.device}",
        f"Recipe: {config.preset} (channels-last: {runtime.channels_last!r})",
        f"Tiling: {tiling.tile_size}px tile, {tiling.tile_overlap}px overlap (stride {tiling.stride}px)",
        f"Compilation: {'enabled' if runtime.compiled else 'disabled'}",
    ]


@click.group(
    name="torch",
    help="(Experimental) Run Machine Learning models directly with PyTorch, or export them to ONNX",
)
def torch_group():
    """Machine Learning commands that run on PyTorch."""
    pass


@torch_group.command(
    name="infer",
    help="(Experimental) Upscale images with a PyTorch checkpoint model, without ONNX Runtime.",
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
    help="The safetensors checkpoint to run with PyTorch",
)
@click.option(
    "-d",
    "--device",
    "device",
    type=DEVICE,
    default="auto",
    show_default=True,
    help="The device to run the model on: auto picks CUDA when it is available",
)
@click.option(
    "--preset",
    "preset",
    type=click.Choice(["fast", "original"], case_sensitive=False),
    default="fast",
    show_default=True,
    help=(
        "The inference recipe: `fast` applies the adapted lean rewrites, "
        "`original` runs the untouched model as loaded by resselt"
    ),
)
@click.option(
    "-c",
    "--compile/--no-compile",
    "enable_compile",
    default=False,
    show_default=True,
    help="Wrap the prepared model with torch.compile (CUDA `fast` recipe only)",
)
@click.option(
    "-t",
    "--tile-size",
    "tile_size",
    type=options.POSITIVE_INT,
    default=DEFAULT_TILE_SIZE,
    show_default=True,
    help="The fixed tile size used for inference, a different size needs its own validation",
)
@click.option(
    "-to",
    "--tile-overlap",
    "tile_overlap",
    type=options.ZERO_POSITIVE_INT,
    default=DEFAULT_TILE_OVERLAP,
    show_default=True,
    help="Total overlap between two neighbouring tiles (stride is tile-size minus this), must be below the tile size",
)
@options.recursive
@time_program
@check_config_first
def infer(
    path_or_archive: Path,
    dest_output: Path,
    model_path: Path,
    device: str,
    preset: str,
    enable_compile: bool,
    tile_size: int,
    tile_overlap: int,
    recursive: bool,
):
    """Upscale images with a PyTorch checkpoint model."""

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    if model_path.suffix.lower() != ".safetensors":
        raise click.BadParameter(
            f"{model_path.name} is not a safetensors checkpoint. "
            "`torch infer` runs checkpoint models with PyTorch, use `upscale-trt` for ONNX graphs.",
            param_hint="model_path",
        )

    try:
        tiling = TilingConfig(tile_size=tile_size, tile_overlap=tile_overlap)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="tile_overlap") from exc

    selected_preset = cast("TorchPreset", preset.lower())

    if enable_compile and selected_preset != "fast":
        raise click.BadParameter(
            "Compilation is only validated for the `fast` preset; drop --compile or use --preset fast.",
            param_hint="enable_compile",
        )

    # The heavy runtime is imported here so `nmanga --help`, `torch infer --help` and every
    # unrelated command keep working on an installation without torch or resselt.
    try:
        from ..ml.torch_backend import create_runtime, resolve_device
    except ImportError as exc:
        raise click.ClickException(
            "PyTorch is not installed. Install the `denoiser` extra to use this command."
        ) from exc

    try:
        selected_device = resolve_device(device)
    except (RuntimeError, ValueError) as exc:
        raise click.BadParameter(str(exc), param_hint="device") from exc

    if enable_compile and selected_device.type != "cuda":
        raise click.BadParameter(
            "Compilation is only validated for the CUDA `fast` recipe; "
            f"the resolved device is {selected_device}. Drop --compile or pass --device cuda.",
            param_hint="enable_compile",
        )

    candidates: list[Path] = []
    if not recursive:
        candidates.append(path_or_archive)
    else:
        console.info(f"Recursively collecting folder in {path_or_archive}...")
        for comic in file_handler.collect_all_comics(path_or_archive, dir_only=True):
            candidates.append(comic)
        console.info(f"Found {len(candidates)} folders to upscale.")

    if not candidates and recursive:
        console.warning("No valid folders found to upscale.")
        return 1

    console.info(f"Output folder: {dest_output.resolve()} (recursive? {recursive!r})")

    runtime_config = TorchRuntimeConfig(
        device=device,
        preset=selected_preset,
        # The validated `fast` recipe uses channels-last on CUDA; `original` stays contiguous.
        use_channels_last=selected_preset == "fast",
        enable_compile=enable_compile,
        tiling=tiling,
    )

    console.info(f"Loading model: {model_path.name}...")
    load_start = time.perf_counter()
    try:
        runtime = create_runtime(model_path, runtime_config)
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        raise click.ClickException(f"Could not prepare {model_path.name}: {exc}") from exc
    load_elapsed = format_elapsed_time(time.perf_counter() - load_start)

    for line in _describe_run(runtime_config, runtime):
        console.info(line)
    console.info(f"Model loaded and warmed up in {load_elapsed}.")

    from ..ml.images import image_to_tensor, tensor_to_image
    from ..ml.tiling import infer_tensor

    info = runtime.info
    try:
        for candidate in candidates:
            if recursive:
                console.info(f"Processing: {candidate}")

            all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(candidate)]
            total_files = len(all_files)
            if total_files <= 0:
                console.warning(f"No images found in {candidate}, skipping.")
                continue

            # Keep the recursive folder identity instead of only its leaf name, so two
            # same-named nested folders cannot write into the same output directory.
            real_output = dest_output / candidate.relative_to(path_or_archive) if recursive else dest_output
            real_output.mkdir(parents=True, exist_ok=True)

            progress = console.make_progress()
            task = console.make_task(progress, "Upscaling images...", total=total_files)

            for image_file in all_files:
                output_path = real_output / f"{image_file.stem}.png"

                with Image.open(image_file) as img_file:
                    img_file.load()
                    input_tensor = image_to_tensor(img_file)
                    prediction = infer_tensor(input_tensor, runtime.predict, scale=info.scale, config=tiling)
                    output_image = tensor_to_image(prediction, img_file)

                try:
                    output_image.save(output_path, format="PNG")
                finally:
                    output_image.close()

                progress.advance(task)

            console.stop_progress(progress, f"Upscaled all {total_files} images to {real_output}")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            details = "\n".join(_describe_run(runtime_config, runtime))
            raise click.ClickException(
                f"The device ran out of memory while running {model_path.name}.\n"
                f"{details}\n"
                "Re-run with a smaller --tile-size, or drop --compile to use less reserved memory."
            ) from exc
        raise
    finally:
        runtime.close()

    if recursive:
        console.info(f"Finished processing {len(candidates)} folders.")
    return 0


@torch_group.command(
    name="to-onnx",
    help="Export torch model or safetensors model to ONNX format",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=False, disable_folder=False)
@click.option(
    "-qs",
    "--quant-size",
    "quant_size",
    type=click.Choice(["fp32", "fp16", "bf16"], case_sensitive=True),
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
def to_onnx(
    path_or_archive: Path,
    quant_size: Literal["fp32", "fp16", "bf16"],
    use_dynamo: bool,
    opset_version: int,
):
    """Export torch model or safetensors model to ONNX format"""

    file_candidates: list[Path] = []
    if not path_or_archive.is_file():
        for file in path_or_archive.glob("*"):
            if file.suffix in (".pt", ".pth", ".safetensors"):
                file_candidates.append(file)
    elif path_or_archive.suffix in (".pt", ".pth", ".safetensors"):
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
        "fp16": torch.float16,
    }

    torch_dtype = dtypes_maps.get(quant_size)
    if torch_dtype is None:
        raise click.ClickException(f"Unsupported quantization size: {quant_size}")

    if importlib.util.find_spec("safetensors") is None:
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

    model_sfx = f"{quant_size}_op{opset_version}"
    if use_dynamo:
        model_sfx += "_dynamo"

    for model_file in file_candidates:
        # Export only in fp32 for now
        onnx_path = model_file.parent / f"{model_file.stem}_{model_sfx}.onnx"
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

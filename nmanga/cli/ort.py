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

# ONNX Runtime commands: run an ONNX graph over a folder of images.

from __future__ import annotations

from pathlib import Path
from typing import Literal

import rich_click as click
from PIL import Image
from rich.table import Table

from .. import file_handler, term
from ..ml.backends import (
    AUTO_BACKEND_PRIORITY,
    InferenceBackend,
    describe_backends,
    detect_backend_status,
)
from ..ml.contracts import MLDataType
from ..ml.onnx_backend import denoise_single_image_with_overlap
from ..ml.onnx_session import prepare_model_runtime_builders
from . import options
from ._deco import check_config_first, time_program
from .base import NMangaCommandHandler

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)
console = term.get_console()

__all__ = ("ort_group",)


class BackendParamType(click.ParamType):
    """The inference backend, named after nmanga's vocabulary instead of the raw ORT providers."""

    name = "backend"

    def get_metavar(self, param: click.Parameter, ctx: click.Context) -> str | None:
        return "[{}]".format("|".join(backend.value for backend in InferenceBackend))

    def convert(self, value, param, ctx) -> InferenceBackend:
        if isinstance(value, InferenceBackend):
            return value
        try:
            return InferenceBackend.from_str(str(value))
        except ValueError as exc:
            self.fail(str(exc), param, ctx)


BACKEND = BackendParamType()


@click.group(
    name="ort",
    help="(Experimental) Run ONNX models with ONNX Runtime, on the backend of your choice",
    epilog="Supported backends:\n\n" + describe_backends(),
)
def ort_group():
    """Machine Learning commands that run on ONNX Runtime."""
    pass


@ort_group.command(
    name="backends",
    help="Show which inference backends this machine can actually run.",
    cls=NMangaCommandHandler,
)
@click.option(
    "--raw",
    "show_raw_providers",
    is_flag=True,
    default=False,
    help="Also list the raw ONNX Runtime provider and plugin device names",
)
def list_backends(show_raw_providers: bool):
    """Show which inference backends this machine can actually run."""

    # Imported here so `nmanga --help`, and every command that never touches a model, keep working
    # on an installation without ONNX Runtime, which is an optional extra.
    from ..ml.ort import get_onnxruntime

    console.info("Loading ONNX Runtime...")
    try:
        ort = get_onnxruntime()
    except ImportError as exc:
        raise click.ClickException(
            "ONNX Runtime is not installed. Install the `denoiser*` extra to use this command."
        ) from exc

    console.info(f"ONNX Runtime {ort.version}")
    available_providers = sorted(ort.get_available_providers())
    raw_devices = ort.get_ep_devices() if hasattr(ort, "get_ep_devices") else []
    ep_device_names = sorted({device.ep_name for device in raw_devices})
    statuses = detect_backend_status(available_providers, ep_device_names)

    table = Table(box=None, pad_edge=False, show_header=True)
    table.add_column("backend", style="bold cyan", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("provider", style="dim")
    for status in statuses:
        table.add_row(
            status.backend.value,
            "[green]available[/green]" if status.is_available else "[dim]not available[/dim]",
            status.provider or "",
        )
    console.console.print(table)

    by_backend = {status.backend: status for status in statuses}
    automatic = next((backend for backend in AUTO_BACKEND_PRIORITY if by_backend[backend].is_available), None)
    if automatic is None:
        console.warning("`auto` has no backend to use on this machine.")
    else:
        console.info(f"`auto` would use: {automatic.value} ({automatic.label})")

    if show_raw_providers:
        console.info(f"Provider names: {', '.join(available_providers) or 'none'}")
        console.info(f"Plugin devices: {', '.join(ep_device_names) or 'none'}")
    return 0


@ort_group.command(
    name="infer",
    help="(Experimental) Denoise or upscale images by running an ONNX model over them.",
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
    help="The ONNX model file to run, a x1 model denoises and a x2/x4 model upscales",
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
    "-be",
    "--backend",
    "backend",
    type=BACKEND,
    default=InferenceBackend.AUTO,
    show_default=True,
    help="The inference backend to use, naming one makes it mandatory (see `nmanga ort --help`)",
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
    "--tta",
    "tta",
    is_flag=True,
    default=False,
    show_default=True,
    help="Average each batch over its 8 flips/rotations (test-time augmentation), about 8x slower",
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
    help="The precision to build the engine for",
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
    help="Fail when `auto` cannot use the backend it picked, instead of running on whatever loaded",
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
@check_config_first
@time_program
def infer(
    path_or_archive: Path,
    dest_output: Path,
    model_path: Path,
    device_id: int,
    backend: InferenceBackend,
    batch_size: int,
    tile_size: int,
    tile_overlap: int,
    contrast_stretch: bool,
    tta: bool,
    background: Literal["black", "white"],
    quant_size: str,
    workspace_size: int | None,
    enable_cuda_graph: bool,
    strict_ep: bool,
    no_cpu_fallback: bool,
    recursive: bool,
):
    """Denoise or upscale images by running an ONNX model over them."""

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
        console.info(f"Found {len(candidates)} folders to process.")

    if not candidates and recursive:
        console.warning("No valid folders found to process.")
        return 1

    console.info(f"Output folder: {dest_output.resolve()} (recursive? {recursive!r})")
    console.info(f"Loading model: {model_path.name}...")
    console.info(f"Using device ID: {device_id}")

    if backend in (InferenceBackend.AUTO, InferenceBackend.TRT, InferenceBackend.TRT_RTX):
        console.info(
            f"Building/loading the engine for {model_path.name}. "
            "If it is building, the first run may take up to 20-25 minutes."
        )

    try:
        session = prepare_model_runtime_builders(
            model_path,
            device_id=device_id,
            is_verbose=console.debugged,
            tile_size=tile_size,
            batch_size=batch_size,
            data_type=MLDataType.from_str(quant_size),
            backend=backend,
            workspace_size=None if workspace_size is None else workspace_size * 1024 * 1024,
            enable_cuda_graph=enable_cuda_graph,
            strict_ep=strict_ep,
            disable_cpu=no_cpu_fallback,
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    # One command covers both jobs: a x1 model removes noise, anything bigger enlarges the page.
    scale_factor = getattr(session, "scale_factor", None) or 1
    action = "Upscaling" if scale_factor > 1 else "Denoising"
    if scale_factor > 1:
        console.info(f"Detected model scale factor: x{scale_factor}")

    console.info(f"Model loaded successfully. Starting the {action.lower()} process...")

    for candidate in candidates:
        if recursive:
            console.info(f"Processing: {candidate}")

        all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(candidate)]
        total_files = len(all_files)
        if total_files <= 0:
            console.warning(f"No images found in {candidate}, skipping.")
            continue

        # Keep the recursive folder identity instead of only its leaf name, so two same-named
        # nested folders cannot write into the same output directory.
        real_output = dest_output / candidate.relative_to(path_or_archive) if recursive else dest_output
        real_output.mkdir(parents=True, exist_ok=True)

        progress = console.make_progress()
        task = console.make_task(progress, f"{action} images...", total=total_files)

        for image_file in all_files:
            output_path = real_output / f"{image_file.stem}.png"

            img_file = Image.open(image_file)
            output_image = denoise_single_image_with_overlap(
                img_file,
                session,
                batch_size=batch_size,
                tile_size=tile_size,
                contrast_stretch=contrast_stretch,
                tta=tta,
                background=background,
                tile_overlap=tile_overlap,
                use_fp32=not session.use_halfp,
            )
            try:
                output_image.save(output_path, format="PNG")
            finally:
                img_file.close()
                output_image.close()

            progress.advance(task)

        console.stop_progress(progress, f"{action} finished for all {total_files} images to {real_output}")

    if recursive:
        console.info(f"Finished processing {len(candidates)} folders.")
    return 0


@ort_group.command(
    name="to-fp16",
    help="Re-export a FP32 model to FP16",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=False, disable_folder=False)
@click.option(
    "-t",
    "--tile-size",
    "tile_size",
    type=options.POSITIVE_INT,
    default=128,
    show_default=True,
    help="The tile size to use for processing images, higher values use more VRAM",
)
def into_fp16(path_or_archive: Path, tile_size: int):
    """Re-export a FP32 model to FP16"""

    file_candidates: list[Path] = []
    if not path_or_archive.is_file():
        for file in path_or_archive.glob("*"):
            if file.suffix in (".onnx"):
                file_candidates.append(file)
    elif path_or_archive.suffix in (".onnx"):
        file_candidates.append(path_or_archive)
    else:
        raise click.BadParameter(
            f"{path_or_archive} is not a valid model file. Please provide a .pt, .pth or .safetensors file.",
            param_hint="path_or_archive",
        )

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnxconverter_common.float16 import convert_float_to_float16
    except ImportError as exc:
        raise click.ClickException("Some packages are not installed. Please install it to use this command.") from exc

    providers = ["CPUExecutionProvider"]
    pix = np.random.default_rng(0).random((1, 3, tile_size, tile_size), dtype=np.float32)
    for onnx_file in file_candidates:
        dest = onnx_file.with_name(f"{onnx_file.stem}_fp16.onnx")
        console.info(f"Converting {onnx_file.name} to {dest.name}...")
        onnx.save(convert_float_to_float16(onnx.load(onnx_file)), dest)

        full = ort.InferenceSession(onnx_file, providers=providers)
        ref = np.asarray(full.run(None, {full.get_inputs()[0].name: pix})[0])
        half = ort.InferenceSession(dest, providers=providers)
        out = np.asarray(half.run(None, {half.get_inputs()[0].name: pix.astype(np.float16)})[0])
        if not np.isfinite(out).all():
            dest.unlink()
            console.warning(f"{dest.name}: NaN/inf in output, skipping")
            continue
        console.info(f"Converted {onnx_file.name} to {dest.name}")
        console.info(f"Max abs diff vs fp32: {np.abs(out.astype(np.float32) - ref).max():.5f}")
    return 0

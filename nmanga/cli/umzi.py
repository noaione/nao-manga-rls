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

# Umzi native commands: inspect a `.ummodel`, pack a checkpoint into one, or run one over images.

from __future__ import annotations

from pathlib import Path
from typing import Literal

import rich_click as click
from PIL import Image

from .. import file_handler, term
from ..ml.onnx_backend import denoise_single_image_with_overlap
from ..ml.onnx_cache import get_data_dir
from ..ml.umzi import (
    UMZI_CHECKPOINT_SUFFIXES,
    UMZI_FAMILIES,
    UMZI_PRECISIONS,
    UmziPackageInfo,
    build_kernel_options,
    create_umzi_session,
    describe_package,
    export_ummodel,
    parse_kernel_options,
    read_ummodel,
    resolve_precision,
)
from . import options
from ._deco import check_config_first, time_program
from .base import NMangaCommandHandler

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)
console = term.get_console()

__all__ = ("umzi_group",)


@click.group(
    name="umzi",
    help="(Experimental) Run native Umzi `.ummodel` packages on CUDA: inference, inspection and packing",
)
def umzi_group():
    """Native Umzi model commands, backed by the `umzi-native` runtime."""
    pass


def _print_package(info: UmziPackageInfo) -> None:
    """Print the report lines of one package, problems as warnings."""
    print_line = console.info if info.supported else console.warning
    for line in describe_package(info):
        print_line(line)


@umzi_group.command(
    name="peek",
    help="Read what a `.ummodel` package declares, without initializing CUDA.",
    cls=NMangaCommandHandler,
)
@click.argument(
    "models",
    nargs=-1,
    required=True,
    metavar="MODEL...",
    type=click.Path(exists=True, path_type=Path),
)
def peek(models: tuple[Path, ...]):
    """Read what a `.ummodel` package declares."""

    candidates: list[Path] = []
    for model in models:
        if model.is_dir():
            candidates.extend(sorted(model.glob("*.ummodel")))
        else:
            candidates.append(model)
    if not candidates:
        raise click.ClickException("No .ummodel packages found in the given paths.")

    failed = False
    for model in candidates:
        try:
            info = read_ummodel(model)
        except ImportError as exc:
            raise click.ClickException(str(exc)) from exc
        except (OSError, ValueError) as exc:
            console.error(f"{model}: {exc}")
            failed = True
            continue

        _print_package(info)
        failed = failed or not info.supported

    return 1 if failed else 0


@umzi_group.command(
    name="to-ummodel",
    help="Pack safetensors or PyTorch checkpoints into `.ummodel` packages.",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=False, disable_folder=False)
@click.option(
    "-k",
    "--kernel",
    "kernel",
    type=click.Choice(list(UMZI_FAMILIES)),
    default=None,
    help="Fail unless the checkpoint detects as this kernel family",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where to write the packages, defaults to each checkpoint's own folder",
)
@time_program
def to_ummodel(path_or_archive: Path, kernel: str | None, output_dir: Path | None):
    """Pack safetensors or PyTorch checkpoints into `.ummodel` packages."""

    accepted = ", ".join(UMZI_CHECKPOINT_SUFFIXES)
    candidates: list[Path] = []
    if path_or_archive.is_file():
        if path_or_archive.suffix.lower() not in UMZI_CHECKPOINT_SUFFIXES:
            raise click.BadParameter(
                f"{path_or_archive.name} is not one of: {accepted}",
                param_hint="path_or_archive",
            )
        candidates.append(path_or_archive)
    else:
        candidates.extend(
            sorted(
                file
                for file in path_or_archive.iterdir()
                if file.is_file() and file.suffix.lower() in UMZI_CHECKPOINT_SUFFIXES
            )
        )

    if not candidates:
        raise click.ClickException(f"No {accepted} checkpoints found in {path_or_archive}.")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    failed = False
    for checkpoint in candidates:
        destination = (output_dir or checkpoint.parent) / f"{checkpoint.stem}.ummodel"
        if destination.exists():
            console.info(f"{destination.name} already exists, skipping.")
            continue

        console.info(f"Packing {checkpoint.name} into {destination.name}...")
        try:
            info = export_ummodel(checkpoint, destination, kernel=kernel)
        except ImportError as exc:
            raise click.ClickException(str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            console.error(f"{checkpoint.name}: {exc}")
            failed = True
            continue

        _print_package(info)

    return 1 if failed else 0


@umzi_group.command(
    name="infer",
    help="Denoise or upscale images by running a `.ummodel` package over them.",
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
    help="The `.ummodel` package to run, a x1 model denoises and a x2/x4 model upscales",
)
@click.option(
    "-d",
    "--device",
    "device_id",
    type=options.ZERO_POSITIVE_INT,
    default=0,
    show_default=True,
    help="The CUDA device index to run the model on",
)
@click.option(
    "-b",
    "--batch-size",
    "batch_size",
    type=options.POSITIVE_INT,
    default=64,
    show_default=True,
    help="How many tiles to run per inference call, higher values use more VRAM",
)
@click.option(
    "-t",
    "--tile-size",
    "tile_size",
    type=options.POSITIVE_INT,
    default=128,
    show_default=True,
    help="The tile size to feed the model, the runtime pads it to the model's alignment",
)
@click.option(
    "-to",
    "--tile-overlap",
    "tile_overlap",
    type=options.ZERO_POSITIVE_INT,
    default=0,
    show_default=True,
    help="Total overlap between two neighbouring tiles (stride is tile-size minus this)",
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
    help="The background color to use for padding non-overlapping runs",
)
@click.option(
    "-p",
    "--precision",
    "precision",
    type=click.Choice(list(UMZI_PRECISIONS), case_sensitive=False),
    default="fp16",
    show_default=True,
    help="The precision the session runs at, the fp32 path only exists for GFISRv2, SMoSR, and FiGSR models",
)
@click.option(
    "-cd",
    "--cache-dir",
    "cache_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where to keep the tuned GPU plans, defaults to nmanga's data directory",
)
@click.option(
    "--no-cache",
    "no_cache",
    is_flag=True,
    default=False,
    show_default=True,
    help="Do not save or reuse the tuned GPU plans, every run tunes them again",
)
@click.option(
    "-cg",
    "--cuda-graph/--no-cuda-graph",
    "enable_cuda_graph",
    default=True,
    show_default=True,
    help="Capture the forward pass into a CUDA graph",
)
@click.option(
    "-ko",
    "--kernel-option",
    "kernel_options",
    multiple=True,
    metavar="NAME=VALUE",
    help="Per-family kernel switch that overrides the recommended one, repeatable",
)
@options.recursive
@check_config_first
@time_program
def infer(
    path_or_archive: Path,
    dest_output: Path,
    model_path: Path,
    device_id: int,
    batch_size: int,
    tile_size: int,
    tile_overlap: int,
    contrast_stretch: bool,
    tta: bool,
    background: Literal["black", "white"],
    precision: str,
    cache_dir: Path | None,
    no_cache: bool,
    enable_cuda_graph: bool,
    kernel_options: tuple[str, ...],
    recursive: bool,
):
    """Denoise or upscale images by running a `.ummodel` package over them."""

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    if model_path.suffix.lower() != ".ummodel":
        raise click.BadParameter(
            f"{model_path.name} is not a `.ummodel` package. Pack a checkpoint with `nmanga umzi to-ummodel` first.",
            param_hint="model_path",
        )

    if tile_overlap * 2 >= tile_size:
        raise click.BadParameter(
            "Tile overlap must be less than half the tile size.",
            param_hint="tile_overlap",
        )

    try:
        overrides = parse_kernel_options(kernel_options)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="kernel_option") from exc

    console.info(f"Reading model: {model_path.name}...")
    try:
        package = read_ummodel(model_path)
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Could not read {model_path.name}: {exc}") from exc

    if not package.supported:
        problems = "\n".join(f"- {problem}" for problem in package.problems)
        raise click.ClickException(f"{model_path.name} cannot run on this umzi-native build:\n{problems}")

    try:
        selected_precision = resolve_precision(package.family, precision)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="precision") from exc

    _print_package(package)

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
    console.info(f"Using GPU {device_id} at {selected_precision} precision")

    resolved_cache: Path | None = None
    if not no_cache:
        resolved_cache = cache_dir if cache_dir is not None else get_data_dir() / "umzi"
        resolved_cache.mkdir(parents=True, exist_ok=True)
        console.info(f"Plan cache: {resolved_cache}")

    used_options = build_kernel_options(package.family, overrides, precision=selected_precision)
    rendered_options = ", ".join(f"{name}={value}" for name, value in sorted(used_options.items()))
    console.info(f"Kernel options: {rendered_options or 'family defaults'}")

    console.info(f"Opening the {package.family} session, the first run also tunes it...")
    try:
        session = create_umzi_session(
            model_path,
            family=package.family,
            device_id=device_id,
            precision=selected_precision,
            cache_dir=resolved_cache,
            enable_cuda_graph=enable_cuda_graph,
            kernel_options=overrides,
        )
    except MemoryError as exc:
        raise click.ClickException(
            f"The GPU ran out of memory while preparing {model_path.name}. "
            "Re-run with a smaller --batch-size or --tile-size."
        ) from exc
    except OSError as exc:
        raise click.ClickException(
            f"Could not open {model_path.name}: {exc}\n"
            "The native runtime links cuBLASLt 13, cuDNN 9 (and cuFFT 12 for GFISRv2/FiGSR), which a CUDA build of "
            "torch (the `denoiser-nvidia` extra) loads into the process before the session opens."
        ) from exc
    except (ImportError, RuntimeError, ValueError) as exc:
        raise click.ClickException(f"Could not open {model_path.name}: {exc}") from exc

    scale_factor = session.scale_factor or 1
    action = "Upscaling" if scale_factor > 1 else "Denoising"
    console.info(f"Model opened successfully. Starting the {action.lower()} process...")

    try:
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

                with Image.open(image_file) as img_file:
                    img_file.load()
                    output_image = denoise_single_image_with_overlap(
                        img_file,
                        session,  # type: ignore[arg-type]  # the native session offers the ONNX tiler's contract
                        batch_size=batch_size,
                        tile_size=tile_size,
                        tile_overlap=tile_overlap,
                        contrast_stretch=contrast_stretch,
                        tta=tta,
                        background=background,
                        use_fp32=selected_precision == "fp32",
                        # The native session prepares one padded shape at a time, so a short last
                        # batch would make it re-tune for every page.
                        pad_final_batch=True,
                    )

                try:
                    output_image.save(output_path, format="PNG")
                finally:
                    output_image.close()

                progress.advance(task)

            console.stop_progress(progress, f"{action} finished for all {total_files} images to {real_output}")
    except MemoryError as exc:
        raise click.ClickException(
            f"The GPU ran out of memory while running {model_path.name}. "
            "Re-run with a smaller --batch-size or --tile-size."
        ) from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise click.ClickException(
                f"The GPU ran out of memory while running {model_path.name}. "
                "Re-run with a smaller --batch-size or --tile-size."
            ) from exc
        raise
    finally:
        session.close()

    if recursive:
        console.info(f"Finished processing {len(candidates)} folders.")
    return 0

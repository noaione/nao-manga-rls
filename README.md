# [nao] Manga Release Scripts

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/noaione/nao-manga-rls)](https://github.com/noaione/nao-manga-rls/blob/master/LICENSE)
![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fnoaione%2Fnao-manga-rls%2Frefs%2Fheads%2Fmaster%2Fpyproject.toml)

This repo contains stuff that I use to release and collect manga from a certain cat website.

All my release use one of the following tag:
- `[nao]`, default release tag for any of my stuff
- `(TooManyIsekai)` any release that has Isekai element in it. (Normal Fantasy does not count)
- `(oan)`, new tag for lower effort stuff that I use.
- `(naoX)`, tag for upscaled stuff that I did.

## Requirements
- Python 3.13+
- imagemagick (for `spreads join` command)
- exiftool (for `releases` and `tag`, optional)
- pingo (for `releases` and `optimize`, optional)

## Scripts
This repo also have a module or script to release and split stuff up.

To install, you need git since I'm not publishing this to PyPi.
After that you can run this:

```sh
pip install -U git+https://github.com/noaione/nao-manga-rls.git
```

This will install this project which then you can use the command `nmanga` to execute everything.

## Installation extras

The base install only has what the image and archive commands need. Everything else is optional:

| Extra | What it adds |
| --- | --- |
| `gui` | PySide6, for the `nmanga-gui` manual split GUI |
| `vapoursynth` | `vapoursynth` with the `imageseqs` and `nimages` plugins, used by `autolevel*`, `imops deblur`, `posterize*`, `lookup` and the orchestrator `leveling`/`posterize` actions |
| `epub-render` | Playwright, for the `epub render` command and the other `epub` subcommands |
| `dev` | `pytest`, `pytest-cov` and `ruff` |

The denoise and `upscale-trt` commands need the `denoiser` extra plus one execution provider.
The `torch infer` and `torch to-onnx` commands only need the `denoiser` extra: they load the
checkpoint with `resselt` and run it with PyTorch, so no ONNX Runtime provider is involved.
`torch infer --compile` additionally needs CUDA and Triton, which the `denoiser` extra provides on
Windows (elsewhere the torch wheel pulls it in).

| Extra | What it adds |
| --- | --- |
| `denoiser` | The vendor neutral stack: torch, ONNX, `onnxscript`, einops, resselt and the Windows ML (winrt) bindings |
| `denoiser-ort` | The stock ONNX Runtime wheel: CoreML on macOS, the CPU provider everywhere else |
| `denoiser-nvidia` | `onnxruntime-gpu`, the TensorRT RTX plugin and `tensorrt-cu13`, with a CUDA build of torch |
| `denoiser-intel` | The OpenVINO plugin EP (`onnxruntime-ep-openvino`), for Intel CPU, GPU and NPU |
| `denoiser-mlx` | The MLX plugin EP (`onnxruntime-ep-mlx`), for Apple Silicon on macOS 14+ |

- `denoiser-intel` and `denoiser-mlx` only add a plugin, so they are installed next to `denoiser-ort` or,
  for OpenVINO, next to `denoiser-nvidia`
- `denoiser-nvidia` replaces the stock runtime instead of extending it, so it cannot be combined with
  `denoiser-ort` or `denoiser-mlx`, which also makes `uv sync --all-extras` an error
- There is no AMD extra yet, DirectML support is expected to arrive through Windows ML

```sh
# NVIDIA
uv sync --extra denoiser --extra denoiser-nvidia

# NVIDIA, with the Intel GPU/NPU as a fallback provider
uv sync --extra denoiser --extra denoiser-nvidia --extra denoiser-intel

# Everything else: stock runtime with the CPU provider, plus MLX on Apple Silicon
uv sync --extra denoiser --extra denoiser-ort --extra denoiser-mlx
```

With `pip` the extras work the same way, but the CUDA torch wheel has to come from the PyTorch index:
`pip install "nmanga[denoiser,denoiser-nvidia]" --extra-index-url https://download.pytorch.org/whl/cu132`.

## Configuration

Configuration documentation moved to the [GitHub Wiki](https://github.com/noaione/nao-manga-rls/wiki/configuration).

## CLI commands

Command documentation moved to the [GitHub Wiki](https://github.com/noaione/nao-manga-rls/wiki).

## API Usage

You can use this module as an API too, if you want to follow what the CLI doing you can view the code and implement your own version.

For example, optimizing images in a folder:

```py
from pathlib import Path

from nmanga.common import optimize_images

target_dir = Path("target")
pingo_path = "pingo"

optimize_images(pingo_path, target_dir)
# Or aggresive: optimize_images(pingo_path, target_dir, True)
```

Creating Daiz-like filename formatting for archive/image filename

```py
from nmanga.common import ChapterRange, format_daiz_like_filename

filename, archive_name = format_daiz_like_filename(
    manga_title="Manga Title",
    manga_publisher="Real Publisher",
    manga_year=2023,
    chapter_info=ChapterRange(1, "Chapter 1", [0]),
    page_number="001",
    publication_type="digital",
    ripper_credit="nao",
    bracket_type="round",
    manga_volume="v01",
    extra_metadata="Cover",
    image_quality="HQ",
    rls_revision=2,
    fallback_volume_name="NA",
)
```

## Changelog

See here: [CHANGELOG.md](https://github.com/noaione/nao-manga-rls/blob/master/CHANGELOG.md)

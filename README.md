# [nao] Manga Release Scripts

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/noaione/nao-manga-rls)](https://github.com/noaione/nao-manga-rls/blob/master/LICENSE)
![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fnoaione%2Fnao-manga-rls%2Frefs%2Fheads%2Fmaster%2Fpyproject.toml)

This repo contains stuff that I use to release and collect manga from a certain cat website.

Every release uses one of these tags:
- `[nao]`, the default tag
- `(TooManyIsekai)`, for anything with an Isekai element (plain fantasy does not count)
- `(oan)`, lower effort releases
- `(naoX)`, upscaled releases

## Requirements
- Python 3.13 or 3.14
- imagemagick (for `spreads join`)
- exiftool (for `releases` and `tag`, optional)
- pingo (for `releases` and `optimize`, optional)

## Scripts
This repo also has modules you can call directly instead of going through the CLI.

Installing needs git, since this is not published to PyPI:

```sh
pip install -U git+https://github.com/noaione/nao-manga-rls.git
```

That puts the `nmanga` command on your PATH.

## Installation extras

The base install covers the image and archive commands. Everything else is optional.

| Extra | What it adds |
| --- | --- |
| `gui` | PySide6, for the `nmanga-gui` manual split GUI |
| `vapoursynth` | VapourSynth with the `imageseqs` and `nimages` plugins, used by `autolevel*`, `imops deblur`, `posterize*`, `lookup` and the orchestrator `leveling`/`posterize` actions |
| `epub-render` | Playwright, for `epub render` and the other `epub` subcommands |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `scipy-stubs` and `types-defusedxml` |

### Machine learning

`denoiser` is the vendor neutral half of the ML stack. It ships no execution provider, so pair it
with one of the `denoiser-*` extras.

| Extra | What it adds |
| --- | --- |
| `denoiser` | torch, `onnx`, `onnxscript`, einops, `resselt`, `safetensors`, Triton for `torch infer --compile`, and Windows ML on Windows |
| `denoiser-ort` | The stock `onnxruntime` wheel. CoreML on macOS, the CPU provider everywhere else |
| `denoiser-nvidia` | `onnxruntime-gpu`, the TensorRT RTX plugin and `tensorrt-cu13`, alongside a CUDA build of torch |
| `denoiser-intel` | The OpenVINO plugin, for Intel CPU, GPU and NPU |
| `denoiser-wgpu` | The WebGPU plugin, backed by Vulkan most of the time |
| `denoiser-mlx` | The MLX plugin, for Apple Silicon on macOS 14 or newer |

Which command needs what:

- `nmanga ort infer` runs an ONNX graph. It needs `denoiser` plus one runtime extra
- `nmanga ort backends` prints what the installed runtime can actually use
- `nmanga torch infer` and `nmanga torch to-onnx` need only `denoiser`. They load the checkpoint with
  `resselt` and run it on PyTorch, so no execution provider is involved

Which extras combine:

- `denoiser-intel` and `denoiser-mlx` add a plugin on top of `denoiser-ort`. OpenVINO also works on top
  of `denoiser-nvidia`
- `denoiser-nvidia` replaces the stock runtime instead of extending it, so it cannot be combined with
  `denoiser-ort` or `denoiser-mlx`. That also makes `uv sync --all-extras` fail
- `tensorrt-cu13` has no Python 3.14 wheel yet, so the TensorRT backends stop at 3.13
- There is no AMD extra yet. The `amdgpu` backend name is reserved, and no extra ships MIGraphX

```sh
# NVIDIA
uv sync --extra denoiser --extra denoiser-nvidia

# NVIDIA, with OpenVINO available as a second provider
uv sync --extra denoiser --extra denoiser-nvidia --extra denoiser-intel

# Stock runtime, plus MLX on Apple Silicon
uv sync --extra denoiser --extra denoiser-ort --extra denoiser-mlx
```

`pip` works the same way, but the CUDA torch wheel has to come from the PyTorch index.

```sh
pip install "nmanga[denoiser,denoiser-nvidia]" --extra-index-url https://download.pytorch.org/whl/cu132
```

## Configuration

Configuration documentation moved to the [GitHub Wiki](https://github.com/noaione/nao-manga-rls/wiki/configuration).

## CLI commands

Command documentation moved to the [GitHub Wiki](https://github.com/noaione/nao-manga-rls/wiki).

## API Usage

The same modules work as an API. Read the CLI code to see how each command is put together.

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

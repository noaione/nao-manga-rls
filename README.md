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

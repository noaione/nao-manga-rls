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

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from ._metadata import __author__, __name__, __version__
from .cli.archive import pack_releases, pack_releases_comment_archive, pack_releases_epub_mode
from .cli.auto_split import auto_split
from .cli.autolevel import autolevel, autolevel2, force_gray
from .cli.base import NMangaCommandHandler
from .cli.config import cli_config
from .cli.denoiser import denoiser, denoiser_trt, identify_denoise_candidates
from .cli.image_optimizer import image_jpegify, image_optimizer
from .cli.image_tagging import image_tagging, image_tagging_raw
from .cli.manual_split import manual_split
from .cli.merge_chapters import merge_chapters
from .cli.orchestrator import orchestractor
from .cli.pdfs import pdf_manager
from .cli.posterize import analyze_shades, auto_posterize, posterize_simple
from .cli.releases import prepare_releases, prepare_releases_chapter
from .cli.renamer import shift_renamer
from .cli.spreads_manager import spreads
from .cli.timewizard import timewizard_modify
from .cli.upscaler import upscale_tiled
from .term import get_console

console = get_console()


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
WORKING_DIR = Path.cwd().absolute()


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(
    __version__,
    "--version",
    "-V",
    prog_name=__name__,
    message="%(prog)s v%(version)s - Created by {}".format(__author__),
)
@click.option(
    "-v",
    "--verbose",
    "verbose",
    is_flag=True,
    required=False,
    help="Enable debug/verbose mode",
    default=False,
)
@click.pass_context
def main(ctx: click.Context, verbose: bool):
    """
    nmanga is a CLI tool for Processing pirated manga.
    """
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE_MODE"] = verbose
    if verbose:
        console.enable_debug()
    else:
        console.disable_debug()


@click.command(
    name="version",
    help="Show nmanga version information",
    cls=NMangaCommandHandler,
)
def show_version():
    """Show nmanga version information."""
    print(f"{__name__} version: {__version__}")
    print(f"Created by: {__author__}")
    print(f"Working directory: {WORKING_DIR}")
    print(f"Python version: {sys.version.replace(os.linesep, ' ')}")
    print(f"Platform: {sys.platform}")


main.add_command(auto_split)
main.add_command(cli_config)
main.add_command(manual_split)
main.add_command(merge_chapters)
main.add_command(pack_releases)
main.add_command(pack_releases_epub_mode)
main.add_command(pack_releases_comment_archive)
main.add_command(prepare_releases)
main.add_command(prepare_releases_chapter)
main.add_command(spreads)
main.add_command(image_tagging)
main.add_command(image_tagging_raw)
main.add_command(image_optimizer)
main.add_command(image_jpegify)
main.add_command(timewizard_modify)
main.add_command(autolevel)
main.add_command(autolevel2)
main.add_command(force_gray)
main.add_command(posterize_simple)
main.add_command(auto_posterize)
main.add_command(analyze_shades)
main.add_command(denoiser)
main.add_command(denoiser_trt)
main.add_command(upscale_tiled)
main.add_command(identify_denoise_candidates)
main.add_command(shift_renamer)
main.add_command(pdf_manager)
main.add_command(orchestractor)
main.add_command(show_version)


if __name__ == "__main__":
    main()

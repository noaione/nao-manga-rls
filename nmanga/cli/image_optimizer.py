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

# Image optimizer command
# This file is part of nmanga.

from pathlib import Path

import click

from .. import term
from . import options
from .base import CatchAllExceptionsCommand, is_executeable_global_path, test_or_find_pingo
from .common import optimize_images, time_program

console = term.get_console()


@click.command(
    name="optimize",
    help="Optimize images with pingo",
    cls=CatchAllExceptionsCommand,
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-ax",
    "--aggressive",
    "aggresive_mode",
    default=False,
    show_default=True,
)
@options.pingo_path
@time_program
def image_optimizer(
    path_or_archive: Path,
    aggresive_mode: bool,
    pingo_path: str,
):
    """
    Optimize images with pingo
    """

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    force_search = not is_executeable_global_path(pingo_path, "exiftool")
    pingo_exe = test_or_find_pingo(pingo_path, force_search)
    if pingo_exe is None:
        console.error("pingo not found, unable to optimize images!")
        raise click.exceptions.Exit(1)

    console.info(f"Using pingo at {pingo_exe}")
    console.info("Optimizing images...")
    optimize_images(pingo_exe, path_or_archive, aggresive_mode)

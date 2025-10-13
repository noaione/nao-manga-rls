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

# A full blown "automated" manga processing orchestrator

from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import signal
import subprocess as sp
from pathlib import Path
from typing import Literal

import click
from PIL import Image

from .. import file_handler, term
from ..orchestrator import *
from . import options
from ._deco import check_config_first, time_program
from .base import NMangaCommandHandler, test_or_find_magick

console = term.get_console()


def _init_worker():
    """Initialize worker processes to handle keyboard interrupts properly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@click.group(name="orchestra", help="Orchestrator for manga processing")
def orchestractor():
    pass


@orchestractor.command(
    name="gen",
    help="Generate a default orchestrator configuration file",
    cls=NMangaCommandHandler,
)
@click.argument(
    "output_file",
    metavar="OUTPUT_FILE",
    required=True,
    type=click.Path(resolve_path=True, file_okay=True, dir_okay=False, path_type=Path),
)
@options.manga_title
@options.manga_publisher
@options.rls_credit
@options.rls_email
@options.use_bracket_type
@check_config_first
@time_program
def orchestrator_generate(
    output_file: Path,
    manga_title: str,
    manga_publisher: str,
    rls_credit: str,
    rls_email: str,
    bracket_type: Literal["square", "round", "curly"],
):
    if output_file.exists():
        console.error(f"{output_file} already exists, please remove it first")
        raise click.Abort()

    config = OrchestratorConfig(
        title=manga_title,
        publisher=manga_publisher,
        base_path=Path("source"),
        credit=rls_credit,
        email=rls_email,
        bracket_type=bracket_type,
        volumes=[
            VolumeConfig(
                number=1,
                path=Path("v01"),
                chapters=[
                    ChapterConfig(number=1, start=0),
                ],
            )
        ],
        actions=[
            # Simple shift name
            ActionShiftName(start=0),
            # Then use daiz-renamer
            ActionRename(),
        ],
    )

    with output_file.open("w", encoding="utf-8") as f:
        f.write(config.model_dump_json(indent=4, exclude_none=True))

    console.info(f"Generated default orchestrator configuration to {output_file}")
    return 0


@orchestractor.command(
    name="run",
    help="Run the orchestrator with the given configuration file",
    cls=NMangaCommandHandler,
)
@click.argument(
    "input_file",
    metavar="INPUT_FILE",
    required=True,
    type=click.Path(resolve_path=True, file_okay=True, dir_okay=False, path_type=Path),
)
@check_config_first
@time_program
def orchestrator_runner(
    input_file: Path,
):
    if not input_file.exists():
        console.error(f"{input_file} does not exist, please provide a valid file")
        raise click.Abort()

    with input_file.open("r", encoding="utf-8") as f:
        config = OrchestratorConfig.model_validate_json(f.read(), strict=True)

    console.info(f"Running orchestrator for {config.title}...")
    full_base = input_file.resolve().parent

    input_dir = full_base / config.base_path
    for volume in config.volumes:
        chapter_path = input_dir / volume.path
        if not chapter_path.exists():
            console.warning(f"Volume path {chapter_path} does not exist, skipping...")
            continue
        if not chapter_path.is_dir():
            console.warning(f"Volume path {chapter_path} is not a directory, skipping...")
            continue

        console.info(f"Processing volume {volume.volume} {chapter_path}...")
        for action in config.actions:
            console.info(f" - Running action {action.kind.name}...")

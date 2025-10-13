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

# Prepare manga for releasing to the public

from pathlib import Path

import click

from nmanga.timewizard import modify_filetimestamp

from .. import config, file_handler, term
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler

console = term.get_console()
conf = config.get_config()


@click.command(
    name="timewizard",
    help="Uh oh, a time wizard is here to modify your file timestamp",
    cls=NMangaCommandHandler,
)
@options.path_or_archive()
@click.option("-t", "--timestamp", "timestamp", type=int, help="The timestamp to set the file to.", required=True)
@time_program
def timewizard_modify(path_or_archive: Path, timestamp: int):
    """
    Uh oh, a time wizard is here to modify your file timestamp
    """

    files_to_modify: list[Path] = []
    if path_or_archive.is_dir():
        with file_handler.MangaArchive(path_or_archive) as archive:
            for image, _ in archive:
                img_access = image.access()
                if isinstance(img_access, Path):
                    files_to_modify.append(img_access)
    elif path_or_archive.is_file():
        archive = file_handler.MangaArchive(path_or_archive)
        handler = archive.open()
        archive.close()
        if isinstance(handler, Path):
            raise click.ClickException(f"Failed to open archive: {archive}")
        files_to_modify.append(path_or_archive)
    else:
        raise click.ClickException(f"Invalid path: {path_or_archive}")

    if not files_to_modify:
        raise click.ClickException("No files to modify timestamp")

    total_files = len(files_to_modify)
    for idx, file in enumerate(files_to_modify, 1):
        console.status(f"Modifying timestamp ({idx}/{total_files})...")
        modify_filetimestamp(file, timestamp)
    console.stop_status(f"Modified timestamp ({total_files}/{total_files})... done!")

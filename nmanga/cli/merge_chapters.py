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

# Merge chapters together into a single chapter.

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import click

from .. import exporter, file_handler, term
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler

console = term.get_console()


def _clean_filename(output_name: Optional[Path]) -> Optional[Path]:
    if output_name is None:
        return None

    return output_name.stem


@click.command(
    "merge",
    help="Merge chapters together into a single chapter",
    cls=NMangaCommandHandler,
)
@click.argument(
    "archives",
    metavar="ARCHIVE_FILES",
    required=True,
    nargs=-1,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False, path_type=Path),
)
@options.dest_output(optional=True, dir_okay=False, file_okay=True)
@time_program
def merge_chapters(archives: List[Path], dest_output: Optional[Path] = None):  # pragma: no cover
    if len(archives) < 2:
        console.error("You must provide at least two archives to merge!")
        return 1

    first_dir = archives[0].parent
    output_name = _clean_filename(dest_output) or file_handler.random_name()
    output_path = first_dir / f"{output_name}.cbz"
    target_cbz = exporter.CBZMangaExporter(output_name, first_dir)

    for archive in archives:
        if not file_handler.is_archive(archive):
            console.error(f"{archive.name} is not an archive!")
            return 1

        console.info(f"[+] Merging: {archive.stem}")
        with file_handler.MangaArchive(archive) as archive_file:
            for image, _ in archive_file:
                image_name = Path(getattr(image, "filename", getattr(image, "name"))).name
                target_cbz.add_image(image_name, archive_file.read(image))
        console.info(f"[+] Merged: {archive.stem}")
        archive.unlink(missing_ok=True)

    console.info("[+] Writing output file...")
    target_cbz.close()
    actual_name = output_path.name
    if not dest_output:
        first_name = archives[0].name
        target_name = first_dir / first_name
        actual_name = first_name
        output_path.rename(target_name)

    console.info(f"[+] Output file: {actual_name}")

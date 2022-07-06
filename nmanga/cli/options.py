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

import os
from pathlib import Path

import click

path_or_archive = click.argument(
    "path_or_archive",
    metavar="FOLDER_OR_ARCHIVE_FILE",
    required=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=True, path_type=Path),
)
archive_file = click.argument(
    "archive_file",
    metavar="ARCHIVE_FILE",
    required=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False, path_type=Path),
)
output_dir = click.option(
    "-o",
    "--output",
    "output_dirpath",
    type=click.Path(
        exists=True, resolve_path=True, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    default=os.getcwd(),
    help="Existing folder to write the output [default: The current directory]",
)
magick_path = click.option(
    "-me",
    "--magick-exec",
    "magick_path",
    default="magick",
    help="Path to the magick executable",
    show_default=True,
)
exiftool_path = click.option(
    "-ee",
    "--exiftool-exec",
    "exiftool_path",
    default="exiftool",
    help="Path to the exiftool executable",
    show_default=True,
)

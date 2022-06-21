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
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True, writable=True, path_type=Path),
    default=os.getcwd(),
    help="Existing folder to write the output [default: The current directory]"
)
magick_path = click.option(
    "-me",
    "--magick-exec",
    "magick_path",
    default="magick",
    help="Path to the magick executable [default: magick]"
)

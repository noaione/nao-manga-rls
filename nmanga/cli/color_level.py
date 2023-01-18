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

# Color level for nmanga
# This file is part of nmanga.

import functools
import subprocess as sp
from os import path
from pathlib import Path

import click

from .. import exporter, file_handler, term
from . import options
from .base import CatchAllExceptionsCommand, test_or_find_magick
from .common import time_program

console = term.get_console()


def _is_default_path(path: str) -> bool:
    path = path.lower()
    if path == "magick":
        return True
    if path == "./magick":
        return True
    if path == ".\\magick":
        return True
    return False


def make_prefix_convert(magick_exe: str):
    name = path.splitext(path.basename(magick_exe))[0]
    if name.lower() == "convert":
        return ["convert"]
    return ["magick", "convert"]


def execute_level(
    magick_dir: str, min: float, max: float, input_file: Path, out_dir: Path, convert_gray: bool
):
    output_name = file_handler.random_name() + ".png"
    execute_this = make_prefix_convert(magick_dir) + [str(input_file)]
    if convert_gray:
        execute_this += ["-colorspace", "Gray"]
    if min > 0.0:
        level_data = f"{min:.2f}%,{max:.2f}%"
        execute_this += ["-level", level_data]
    if convert_gray:
        execute_this.append(f"{out_dir / output_name}")
    else:
        execute_this.append(f"PNG8:{out_dir / output_name}")
    try:
        sp.run(execute_this, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    except sp.CalledProcessError as e:
        console.error(f"Error: {e.output.decode('utf-8')}")
        raise e
    return output_name


@click.command(name="level", help="Batch color level a cbz file or folder", cls=CatchAllExceptionsCommand)
@options.path_or_archive()
@click.option(
    "-l",
    "--low",
    default=12.0,
    show_default=True,
    type=click.FloatRange(0.0, 100.0),
    help="The minimum value of the color level",
)
@click.option(
    "-h",
    "--high",
    default=100.0,
    show_default=True,
    type=click.FloatRange(1.0, 100.0),
    help="The maximum value of the color level",
)
@click.option(
    "-skip",
    "--skip-first",
    "skip_first",
    default=False,
    is_flag=True,
    help="Skip the first image of the file/folder",
)
@click.option(
    "--gray/--no-gray",
    "gray_mode",
    default=True,
    show_default=True,
    help="Convert image colorspaces to Gray8/Y",
)
@options.magick_path
@options.output_dir
@time_program
def color_level(
    path_or_archive: Path,
    low: float,
    high: float,
    skip_first: bool,
    gray_mode: bool,
    magick_path: str,
    output_dirpath: Path,
):
    """
    Batch color level a cbz file or folder
    """
    force_search = not _is_default_path(magick_path)
    magick_exe = test_or_find_magick(magick_path, force_search)
    if magick_exe is None:
        console.error("Could not find the magick executable")
        return 1
    console.info("Using magick executable: {}".format(magick_exe))
    # Create temporary folder
    temp_dir = file_handler.create_temp_dir()
    export_hdl: exporter.MangaExporter = None
    if file_handler.is_archive(path_or_archive):
        # Base filename
        base_name = path_or_archive.stem + "_levelled"
        export_hdl = exporter.CBZMangaExporter(base_name, output_dirpath)
    else:
        CWD = Path.cwd().absolute()
        output_dir = output_dirpath
        if CWD == output_dirpath:
            output_dir = output_dirpath / "nmanga"
        output_dir.mkdir(parents=True, exist_ok=True)
        export_hdl = exporter.MangaExporter(output_dir)

    wrapped_func = functools.partial(
        execute_level, magick_dir=magick_exe, min=low, max=high, out_dir=temp_dir, convert_gray=gray_mode
    )

    is_first_time = True
    console.status("Processing: 1/???")
    counter = 1
    with file_handler.MangaArchive(path_or_archive) as archive:
        for image, total_img in archive:
            console.status(f"Processing: {counter}/{total_img}")
            if is_first_time:
                is_first_time = False
                if skip_first:
                    image_bita = archive.read(image)
                    image_name: str = getattr(image, "name", getattr(image, "filename"))
                    export_hdl.add_image(path.basename(image_name), image_bita)

            if not image.is_archive():
                temp_output = wrapped_func(input_file=image)
                temp_output_file = temp_dir / temp_output
                export_hdl.add_image(image.stem + ".png", temp_output_file)
                temp_output_file.unlink(missing_ok=True)
            else:
                image_name = path.basename(str(image.filename))
                temp_file = temp_dir / image_name
                temp_file.write_bytes(archive.read(image))
                temp_output = wrapped_func(input_file=temp_file)
                temp_output_file = temp_dir / temp_output
                export_hdl.add_image(path.splitext(image_name)[0] + ".png", temp_output_file)
                temp_output_file.unlink(missing_ok=True)
                temp_file.unlink(missing_ok=True)
            counter += 1

    console.stop_status(f"Processed: {counter}/{counter}")
    export_hdl.close()
    console.info("Removing temp folder: {}".format(temp_dir))
    file_handler.remove_folder_and_contents(temp_dir)

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

# Collection of lookup commands

from __future__ import annotations

import shutil
from io import BytesIO
from pathlib import Path

import rich_click as click
from PIL import Image

from .. import file_handler, term
from ..common import lowest_or, threaded_worker
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)
console = term.get_console()


@click.group(
    name="lookup",
    help="Lookup information from files or directories",
)
def lookup_group():
    """Lookup information from files or directories"""
    pass


@lookup_group.command(
    name="imagesize",
    help="Lookup image sizes inside a manga archive or folder",
    cls=NMangaCommandHandler,
)
@options.path_or_archive()
@options.recursive
@time_program
def lookup_imagesize(path_or_archive: Path, recursive: bool):
    """
    Lookup image sizes inside a manga archive or folder
    """

    if recursive and not path_or_archive.is_dir():
        console.warning("The --recursive option is only applicable when a folder is given.")
        raise click.UsageError("The --recursive option is only applicable when a folder is given.")

    candidates: list[Path] = []
    if not recursive:
        console.info(f"Looking up image sizes in {path_or_archive}...")
        candidates.append(path_or_archive)
    else:
        console.info(f"Recursively looking up image sizes in {path_or_archive}...")
        for comic in file_handler.collect_all_comics(path_or_archive):
            candidates.append(comic)
        console.info(f"Found {len(candidates)} archives/folders to lookup.")

    if not candidates:
        console.warning("No valid archives or folders found to lookup.")
        return

    for candidate in candidates:
        if recursive:
            console.info(f"Processing: {candidate}")
        grouped_size: dict[str, list[str]] = {}
        with file_handler.MangaArchive(candidate) as archive:
            for image, _ in archive:
                img_read = archive.read(image)
                with Image.open(BytesIO(img_read)) as img:
                    img_size = f"{img.width}x{img.height}"
                    if img_size not in grouped_size:
                        grouped_size[img_size] = []
                    grouped_size[img_size].append(image.filename)

        console.info("Found the following image sizes:")
        for img_size, image_list in grouped_size.items():
            console.info(f" - {img_size}: {len(image_list)} images")
            if console.debugged:
                for img_name in image_list:
                    console.log(f"    - {img_name}")


def _batch_lookup_nongray(img_path: Path) -> tuple[bool, Path]:
    with Image.open(img_path) as img:
        if img.mode not in ("L", "LA"):
            return True, img_path
    return False, img_path


@lookup_group.command(
    name="nongray",
    help="Lookup non-grayscale images inside a folder, then split them into a separate folder",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.dest_output(optional=False)
@options.recursive
@options.threads
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Do move instead of copy existing files in the output directory.",
)
@time_program
def lookup_nongray_images(
    path_or_archive: Path,
    dest_output: Path,
    recursive: bool,
    threads: int,
    force: bool,
):
    """
    Lookup non-grayscale images inside a folder, then split them into a separate folder
    """

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    candidates: list[Path] = []
    if not recursive:
        console.info(f"Looking up images in {path_or_archive}...")
        candidates.append(path_or_archive)
    else:
        console.info(f"Recursively looking up images in {path_or_archive}...")
        for comic in file_handler.collect_all_comics(path_or_archive, dir_only=True):
            candidates.append(comic)
        console.info(f"Found {len(candidates)} folders to lookup.")

    if recursive and not candidates:
        console.warning("No valid folders found to lookup.")
        return

    dest_output.mkdir(parents=True, exist_ok=True)

    for path_real in candidates:
        if recursive:
            console.info(f"Processing: {path_real}")
        all_images = [file_path for file_path, _, _, _ in file_handler.collect_image_from_folder(path_real)]
        if not all_images:
            console.info(f"No images found in {path_real}, skipping.")
            continue

        real_output = dest_output
        if recursive:
            real_output = dest_output / path_real.name
            real_output.mkdir(parents=True, exist_ok=True)
        console.info(f"Using {threads} CPU threads for processing.")
        progress = console.make_progress()
        task_collect = progress.add_task(
            "Processing images...", finished_text="Processed images", total=len(all_images)
        )
        gray_images: list[Path] = []
        with threaded_worker(console, lowest_or(threads, all_images)) as (pool, _):
            for result, img_path in pool.imap_unordered(
                _batch_lookup_nongray,
                (image for image in all_images),
            ):
                if result:
                    gray_images.append(img_path)
                progress.update(task_collect, advance=1)

        if not gray_images:
            console.info("No non-grayscale images found in this folder.")
            console.stop_progress(progress)
            continue

        console.info(f"Found {len(gray_images)} non-grayscale images, copying to {real_output}...")
        task_copy = progress.add_task("Copying images...", finished_text="Copied images", total=len(gray_images))
        for img_path in gray_images:
            dest_file = real_output / img_path.name
            if force and dest_file.exists():
                dest_file.unlink()
            if force:
                shutil.move(img_path, dest_file)
            else:
                shutil.copy2(img_path, dest_file)
            progress.update(task_copy, advance=1)
        console.stop_progress(progress, f"Completed processing {path_real}")

    if recursive:
        console.info("Completed processing all folders.")


def _batch_lookup_broken(img_path: Path, interface: term.MessageQueue) -> tuple[bool, Path]:
    cnsl = term.with_thread_queue(interface)
    try:
        with Image.open(img_path) as img:
            img.load()
        return False, img_path
    except Exception as exc:
        cnsl.log(f"Image {img_path} is broken: {exc}")
        return True, img_path


def _batch_lookup_broken_star(args: tuple[Path, term.MessageQueue]) -> tuple[bool, Path]:
    return _batch_lookup_broken(*args)


@lookup_group.command(
    name="broken-images",
    help="Lookup for broken images inside a folder",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.recursive
@options.threads
@time_program
def lookup_broken_images(
    path_or_archive: Path,
    recursive: bool,
    threads: int,
):
    """
    Lookup for broken images inside a folder
    """

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    candidates: list[Path] = []
    if not recursive:
        console.info(f"Looking up images in {path_or_archive}...")
        candidates.append(path_or_archive)
    else:
        console.info(f"Recursively looking up images in {path_or_archive}...")
        for comic in file_handler.collect_all_comics(path_or_archive, dir_only=True):
            candidates.append(comic)
        console.info(f"Found {len(candidates)} folders to lookup.")

    if recursive and not candidates:
        console.warning("No valid folders found to lookup.")
        return

    for path_real in candidates:
        if recursive:
            console.info(f"Processing: {path_real}")
        all_images = [file_path for file_path, _, _, _ in file_handler.collect_image_from_folder(path_real)]
        if not all_images:
            console.info(f"No images found in {path_real}, skipping.")
            continue

        console.info(f"Using {threads} CPU threads for processing.")
        progress = console.make_progress()
        task_collect = progress.add_task(
            "Processing images...", finished_text="Processed images", total=len(all_images)
        )
        broken_images: list[Path] = []
        with threaded_worker(console, lowest_or(threads, all_images)) as (pool, log_q):
            for result, img_path in pool.imap_unordered(
                _batch_lookup_broken_star,
                [(image, log_q) for image in all_images],
            ):
                if result:
                    broken_images.append(img_path)
                progress.update(task_collect, advance=1)

        if not broken_images:
            console.info("No broken images found in this folder.")
            console.stop_progress(progress)
            continue

        console.warning(f"Found {len(broken_images)} broken images:")
        for img_path in broken_images:
            console.warning(f" - {img_path}")
        console.stop_progress(progress, f"Completed processing {path_real}")

    if recursive:
        console.info("Completed processing all folders.")

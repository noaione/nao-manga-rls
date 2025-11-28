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

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import rich_click as click

from .. import config, file_handler, term
from ..common import (
    ChapterRange,
    RegexCollection,
    format_archive_filename,
    format_daiz_like_filename,
    format_volume_text,
    inquire_chapter_ranges,
    lowest_or,
    optimize_images,
    safe_int,
    threaded_worker,
)
from ..constants import MangaPublication
from . import options
from ._deco import check_config_first, time_program
from .base import (
    NMangaCommandHandler,
    WithDeprecatedOption,
    is_executeable_global_path,
    test_or_find_exiftool,
    test_or_find_pingo,
)
from .image_tagging import _threaded_tagging_star

console = term.get_console()
conf = config.get_config()

__all__ = (
    "prepare_releases",
    "prepare_releases_chapter",
)


@dataclass
class SpecialNaming:
    page: int
    data: str


@click.command(
    name="releases",
    help="Prepare a release of a manga series.",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.manga_title
@options.manga_year
@options.manga_publisher
@options.manga_publication_type()
@options.rls_credit
@options.rls_email
@options.rls_revision
@options.rls_extra_metadata
@options.is_oneshot
@click.option(
    "-hq",
    "--is-high-quality",
    "is_high_quality",
    cls=WithDeprecatedOption,
    is_flag=True,
    help="Whether this is a high quality release",
    default=False,
    deprecated=True,
    preferred=["-mq", "--quality"],
    panel="Deprecated Options",
)
@click.option(
    "-mq",
    "--quality",
    "image_quality",
    type=click.Choice(["LQ", "HQ"]),
    default=None,
    help="Image quality of this release.",
    panel="Release Options",
)
@click.option(
    "--tag/--no-tag",
    "do_exif_tagging",
    default=True,
    show_default=True,
    help="Do exif metadata tagging on the files.",
)
@click.option(
    "--optimize/--no-optimize",
    "do_img_optimize",
    default=False,
    show_default=True,
    help="Optimize the images using pingo.",
)
@options.exiftool_path
@options.pingo_path
@options.use_bracket_type
@options.threads_alt
@check_config_first
@time_program
def prepare_releases(
    path_or_archive: Path,
    manga_title: str,
    manga_year: int | None,
    manga_publisher: str,
    manga_publication_type: MangaPublication,
    rls_credit: str,
    rls_email: str,
    rls_revision: int,
    rls_extra_metadata: str | None,
    is_oneshot: bool,
    is_high_quality: bool,
    image_quality: str | None,
    do_exif_tagging: bool,
    do_img_optimize: bool,
    exiftool_path: str,
    pingo_path: str,
    bracket_type: Literal["square", "round", "curly"],
    threads: int,
):  # pragma: no cover
    """
    Prepare a release of a manga series.
    """

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    current_pst = datetime.now(timezone(timedelta(hours=-8)))
    current_year = manga_year or current_pst.year

    force_search_exif = not is_executeable_global_path(exiftool_path, "exiftool")
    exiftool_exe = test_or_find_exiftool(exiftool_path, force_search_exif)
    if exiftool_exe is None and do_exif_tagging:
        console.warning("Exiftool not found, will skip tagging image with exif metadata!")
    force_search_pingo = not is_executeable_global_path(pingo_path, "pingo")
    pingo_exe = test_or_find_pingo(pingo_path, force_search_pingo)
    if pingo_exe is None and do_img_optimize:
        console.warning("Pingo not found, will skip optimizing image!")

    cmx_re = RegexCollection.cmx_re()
    console.status("Checking folder contents...")
    temp_image_count = 0
    for image, _, _, _ in file_handler.collect_image_from_folder(path_or_archive):
        title_match = cmx_re.match(image.name)
        if title_match is None:
            console.error("Unmatching file name: {}".format(image.name))
            return 1
        temp_image_count += 1
    console.stop_status("Checking folder contents... done!")
    if temp_image_count < 1:
        console.error("No image found in the folder!")
        return 1

    has_ch_title = console.confirm("Does this release have chapter titles?")
    rls_information = inquire_chapter_ranges(
        "Please input information regarding this release...",
        "Do you want to add another release?",
        has_ch_title,
    )
    rls_information.sort(key=lambda x: x.number)

    packing_extra: dict[int, list[ChapterRange]] = {}
    for info in rls_information:
        if info.base not in packing_extra:
            packing_extra[info.base] = []
        packing_extra[info.base].append(info)

    special_naming: dict[int, SpecialNaming] = {}
    do_special_get = console.confirm("Do you want to add some special naming?")
    if do_special_get:
        while True:
            page = console.inquire("Page number", lambda y: safe_int(y) is not None)
            page_number = int(page)

            data = console.inquire("Page type", lambda y: len(y.strip()) > 0)
            special_naming[page_number] = SpecialNaming(page_number, data)

            do_more = console.confirm("Do you want to add another naming?")
            if not do_more:
                break

    console.info("Preparing release...")
    console.info(f"Has {len(rls_information)} chapters")
    current = 1
    console.info("Processing: 1/???")
    image_titling: str | None = None
    vol_oshot_warn = False
    for image, _, total_img, _ in file_handler.collect_image_from_folder(path_or_archive):
        title_match = cmx_re.match(image.name)
        if title_match is None:
            console.error("Unmatching file name: {}".format(image.name))
            return 1

        p01 = title_match.group("a")
        p01_copy = int(title_match.group("a"))
        p02 = title_match.group("b")
        vol = title_match.group("vol")
        vol_ex = title_match.group("volex")
        if p02 is not None:
            p01 = f"{p01}-{p02}"
        vol_act: int | float | None
        if is_oneshot:
            vol_act = None
            if not vol_oshot_warn:
                vol_oshot_warn = True
                console.warning("Marked as Oneshot, using OShot as default for image and empty for archive name!")
        else:
            if vol is None:
                vol_act = None
                if not vol_oshot_warn:
                    vol_oshot_warn = True
                    console.warning(
                        "Volume is not specified, using Oneshot format as default for image and empty for archive name!"
                    )
            else:
                if vol.startswith("v"):
                    vol = vol[1:]
                vol_act = int(vol)

        if vol_act is not None and vol_ex is not None:
            if vol_ex.startswith("."):
                vol_ex = vol_ex[1:]
            vol_act = float(f"{vol_act}.{int(vol_ex)}")

        selected_range: ChapterRange | None = None
        for rls_info in rls_information:
            if rls_info.is_single:
                if p01_copy >= rls_info.range[0]:
                    selected_range = rls_info
                    break
            else:
                if p01_copy in rls_info.range:
                    selected_range = rls_info
                    break
        if selected_range is None:
            console.warning(f"Page {p01} are not included in any range, skipping!")
            current += 1
            continue

        extra_name = None
        if p01_copy in special_naming:
            extra_name = special_naming[p01_copy].data

        act_img_quality = "HQ" if is_high_quality else None
        if image_quality is not None:
            act_img_quality = image_quality

        image_filename, archive_filename = format_daiz_like_filename(
            manga_title=manga_title,
            manga_publisher=manga_publisher,
            manga_year=current_year,
            chapter_info=selected_range,
            page_number=p01,
            publication_type=manga_publication_type,
            ripper_credit=rls_credit,
            bracket_type=bracket_type,
            manga_volume=vol_act,
            extra_metadata=extra_name,
            image_quality=act_img_quality,
            rls_revision=rls_revision,
            chapter_extra_maps=packing_extra,
            extra_archive_metadata=rls_extra_metadata,
        )

        if not image_titling:
            image_titling = archive_filename

        image_filename += image.suffix
        new_name = image.parent / image_filename
        image.rename(new_name)
        console.status(f"Processing: {current}/{total_img}")
        current += 1

    console.stop_status(f"Processed {current - 1} images!")
    if image_titling is None:
        raise RuntimeError("No image titling generated, this is unexpected!")

    if pingo_exe is not None and do_img_optimize:
        console.info("Optimizing images...")
        optimize_images(pingo_exe, path_or_archive)
    if exiftool_exe is not None and do_exif_tagging:
        console.info("Tagging images with exif metadata...")
        progress = console.make_progress()
        precollect_images = [
            file_path for file_path, _, _, _ in file_handler.collect_image_from_folder(path_or_archive)
        ]
        precollect_images.sort(key=lambda p: p.name)

        task = progress.add_task("Tagging images...", finished_text="Tagged images", total=len(precollect_images))

        console.info(f"Using {threads} CPU threads for processing.")
        with threaded_worker(console, lowest_or(threads, precollect_images)) as (pool, log_q):
            for _ in pool.imap_unordered(
                _threaded_tagging_star,
                ((log_q, exiftool_exe, image, image_titling, rls_email) for image in precollect_images),
            ):
                progress.update(task, advance=1)
        console.stop_progress(progress, "Tagged all possible images with exif metadata.")


@click.command(
    name="releasesch",
    help="Prepare a release of a manga chapter.",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.manga_title
@options.manga_year
@options.manga_publisher
@options.manga_chapter
@options.manga_volume
@click.option(
    "-cht",
    "--chapter-title",
    "chapter_title",
    help="Chapter title that will be included between the publication type and publisher",
    default=None,
    required=False,
)
@options.manga_publication_type(chapter_mode=True)
@options.rls_credit
@options.rls_email
@options.rls_revision
@options.rls_extra_metadata
@click.option(
    "-hq",
    "--is-high-quality",
    "is_high_quality",
    cls=WithDeprecatedOption,
    is_flag=True,
    help="Whether this is a high quality release",
    default=False,
    deprecated=True,
    preferred=["-mq", "--quality"],
    panel="Deprecated Options",
)
@click.option(
    "-mq",
    "--quality",
    "image_quality",
    type=click.Choice(["LQ", "HQ"]),
    default=None,
    help="Image quality of this release.",
    panel="Release Options",
)
@click.option(
    "--tag/--no-tag",
    "do_exif_tagging",
    default=True,
    show_default=True,
    help="Do exif metadata tagging on the files.",
)
@click.option(
    "--optimize/--no-optimize",
    "do_img_optimize",
    default=False,
    show_default=True,
    help="Optimize the images using pingo.",
)
@options.exiftool_path
@options.pingo_path
@options.use_bracket_type
@options.threads_alt
@check_config_first
@time_program
def prepare_releases_chapter(
    path_or_archive: Path,
    manga_title: str,
    manga_year: int | None,
    manga_publisher: str,
    manga_chapter: int | float | None,
    manga_volume: int | None,
    chapter_title: str | None,
    manga_publication_type: MangaPublication,
    rls_credit: str,
    rls_email: str,
    rls_revision: int,
    rls_extra_metadata: str | None,
    is_high_quality: bool,
    image_quality: str | None,
    do_exif_tagging: bool,
    do_img_optimize: bool,
    exiftool_path: str,
    pingo_path: str,
    bracket_type: Literal["square", "round", "curly"],
    threads: int,
):  # pragma: no cover
    """
    Prepare a release of a manga chapter.
    """

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    if manga_chapter is None:
        raise click.BadParameter(
            "-ch/--chapter is required for this command.",
            param_hint="manga_chapter",
        )

    current_pst = datetime.now(timezone(timedelta(hours=-8)))
    current_year = manga_year or current_pst.year

    force_search_exif = not is_executeable_global_path(exiftool_path, "exiftool")
    exiftool_exe = test_or_find_exiftool(exiftool_path, force_search_exif)
    if exiftool_exe is None and do_exif_tagging:
        console.warning("Exiftool not found, will skip tagging image with exif metadata!")
    force_search_pingo = not is_executeable_global_path(pingo_path, "pingo")
    pingo_exe = test_or_find_pingo(pingo_path, force_search_pingo)
    if pingo_exe is None and do_img_optimize:
        console.warning("Pingo not found, will skip optimizing image!")

    page_re = RegexCollection.page_re()
    console.status("Checking folder contents...")
    temp_image_count = 0
    for image, _, _, _ in file_handler.collect_image_from_folder(path_or_archive):
        title_match = page_re.match(image.name)
        if title_match is None:
            console.error("Unmatching file name: {}".format(image.name))
            return 1
        temp_image_count += 1
    console.stop_status("Checking folder contents... done!")
    if temp_image_count < 1:
        console.error("No image found in the folder!")
        return 1

    console.info("Preparing release...")
    current = 1
    console.info("Processing: 1/???")
    image_titling: str | None = None
    for image, _, total_img, _ in file_handler.collect_image_from_folder(path_or_archive):
        title_match = page_re.match(image.name)
        if title_match is None:
            console.error("Unmatching file name: {}".format(image.name))
            return 1

        p01 = title_match.group("a")
        p02 = title_match.group("b")
        if p02 is not None:
            p01 = f"{p01}-{p02}"

        ch_range = ChapterRange(manga_chapter, chapter_title, [], True)
        act_img_quality = "HQ" if is_high_quality else None
        if image_quality is not None:
            act_img_quality = image_quality

        image_filename, _ = format_daiz_like_filename(
            manga_title=manga_title,
            manga_publisher=manga_publisher,
            manga_year=current_year,
            chapter_info=ch_range,
            page_number=p01,
            publication_type=manga_publication_type,
            ripper_credit=rls_credit,
            bracket_type=bracket_type,
            manga_volume=manga_volume,
            image_quality=act_img_quality,
            rls_revision=rls_revision,
            fallback_volume_name="NA",
        )

        image_filename += image.suffix
        new_name = image.parent / image_filename
        image.rename(new_name)
        console.status(f"Processing: {current}/{total_img}")
        current += 1
        if image_titling is None:
            image_titling = format_archive_filename(
                manga_title=manga_title,
                manga_year=current_year,
                publication_type=manga_publication_type,
                ripper_credit=rls_credit,
                bracket_type=bracket_type,
                manga_volume_text=format_volume_text(manga_volume=manga_volume, manga_chapter=manga_chapter),
                rls_revision=rls_revision,
                extra_metadata=rls_extra_metadata,
            )

    console.stop_status(f"Processed {current - 1} images!")
    if image_titling is None:
        raise RuntimeError("No image titling generated, this is unexpected!")

    if pingo_exe is not None and do_img_optimize:
        console.info("Optimizing images...")
        optimize_images(pingo_exe, path_or_archive)
    if exiftool_exe is not None and do_exif_tagging:
        console.info("Tagging images with exif metadata...")
        progress = console.make_progress()
        precollect_images = [
            file_path for file_path, _, _, _ in file_handler.collect_image_from_folder(path_or_archive)
        ]
        precollect_images.sort(key=lambda p: p.name)

        task = progress.add_task("Tagging images...", finished_text="Tagged images", total=len(precollect_images))

        console.info(f"Using {threads} CPU threads for processing.")
        with threaded_worker(console, lowest_or(threads, precollect_images)) as (pool, log_q):
            for _ in pool.imap_unordered(
                _threaded_tagging_star,
                ((log_q, exiftool_exe, image, image_titling, rls_email) for image in precollect_images),
            ):
                progress.update(task, advance=1)
        console.stop_progress(progress, "Tagged all possible images with exif metadata.")

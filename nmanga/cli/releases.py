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

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import click

from .. import config, exporter, file_handler, term
from . import options
from ._deco import check_config_first, time_program
from .base import (
    CatchAllExceptionsCommand,
    RegexCollection,
    is_executeable_global_path,
    test_or_find_exiftool,
    test_or_find_pingo,
)
from .common import (
    BRACKET_MAPPINGS,
    ChapterRange,
    inject_metadata,
    inquire_chapter_ranges,
    optimize_images,
    safe_int,
)

console = term.get_console()
conf = config.get_config()
TARGET_FORMAT = "{mt} - c{ch}{chex} ({vol}) - p{pg}{ex}[dig] [{t}] [{pb}] [{c}]"  # noqa
TARGET_FORMAT_ALT = "{mt} - c{ch}{chex} ({vol}) - p{pg}{ex}[dig] [{pb}] [{c}]"  # noqa
TARGET_TITLE = "{mt} {vol} ({year}) (Digital) {cpa}{c}{cpb}"
TARGET_TITLE_NOVEL = "{mt} {vol} [{source}] [{c}]"

__all__ = (
    "prepare_releases",
    "pack_releases",
    "pack_releases_epub_mode",
)


class SpecialNaming:
    def __init__(self, page: int, data: str):
        self.page = page
        self.data = data


@click.command(
    name="releases",
    help="Prepare a release of a manga series.",
    cls=CatchAllExceptionsCommand,
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-t",
    "--title",
    "manga_title",
    required=True,
    help="The title of the series",
)
@options.manga_year
@click.option(
    "-pub",
    "--publisher",
    "publisher",
    help="The publisher of the series",
    required=True,
)
@options.rls_credit
@options.rls_email
@options.rls_revision
@click.option(
    "-hq",
    "--is-high-quality",
    "is_high_quality",
    is_flag=True,
    help="Whether this is a high quality release",
    default=False,
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
@check_config_first
@time_program
def prepare_releases(
    path_or_archive: Path,
    manga_title: str,
    manga_year: Optional[int],
    publisher: str,
    rls_credit: str,
    rls_email: str,
    rls_revision: int,
    is_high_quality: bool,
    do_exif_tagging: bool,
    do_img_optimize: bool,
    exiftool_path: str,
    pingo_path: str,
    bracket_type: Literal["square", "round", "curly"],
):
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

    pair_left, pair_right = BRACKET_MAPPINGS.get(bracket_type.lower(), BRACKET_MAPPINGS["square"])

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
    for image, _, total_img, _ in file_handler.collect_image_from_folder(path_or_archive):
        title_match = cmx_re.match(image.name)
        if title_match is None:
            console.error("Unmatching file name: {}".format(image.name))
            return 1
    console.stop_status("Checking folder contents... done!")

    has_ch_title = console.confirm("Does this release have chapter titles?")
    rls_information = inquire_chapter_ranges(
        "Please input information regarding this release...",
        "Do you want to add another release?",
        has_ch_title,
    )
    rls_information.sort(key=lambda x: x.number)

    packing_extra: Dict[int, List[ChapterRange]] = {}
    for info in rls_information:
        if info.base not in packing_extra:
            packing_extra[info.base] = []
        packing_extra[info.base].append(info)

    special_naming: Dict[int, SpecialNaming] = {}
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
    image_titling: Optional[str] = None
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
        if p02 is not None:
            p01 = f"{p01}-{p02}"
        if vol is None:
            vol = "OShot"
            if not vol_oshot_warn:
                vol_oshot_warn = True
                console.warning("Volume is not specified, using OShot (Oneshot) as default!")

        # print(p01_copy)
        selected_range: ChapterRange = None
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
        extra_name = " "
        if p01_copy in special_naming:
            extra_name = f" [{special_naming[p01_copy].data}] "

        chapter_num = f"{selected_range.base:03d}"
        pack_data = packing_extra[selected_range.base]
        pack_data.sort(key=lambda x: x.number)
        chapter_ex_data = ""
        if len(pack_data) > 1:
            smallest = pack_data[1].floating
            for pack in pack_data:
                if pack.floating is not None and pack.floating < smallest:
                    smallest = pack.floating
            if smallest is not None and selected_range.floating is not None:
                # Check if we should append the custom float data
                if smallest >= 5:
                    # We don't need to append the float data
                    float_act = selected_range.floating - 4
                    chapter_num += f"x{float_act}"
                else:
                    idx = pack_data.index(selected_range)
                    chapter_ex_data = f" (c{chapter_num}.{selected_range.floating})"
                    chapter_num += f"x{idx}"
        else:
            floaty = selected_range.floating
            if floaty is not None:
                if floaty >= 5:
                    chapter_num += f"x{floaty - 4}"
                else:
                    chapter_ex_data = f" (c{chapter_num}.{floaty})"
                    chapter_num += "x1"

        ch_title_name = selected_range.name
        extension = image.suffix

        final_filename = TARGET_FORMAT_ALT.format(
            mt=manga_title,
            ch=chapter_num,
            chex=chapter_ex_data,
            vol=vol,
            pg=p01,
            ex=extra_name,
            pb=publisher,
            c=rls_credit,
        )
        if has_ch_title:
            final_filename = TARGET_FORMAT.format(
                mt=manga_title,
                ch=chapter_num,
                chex=chapter_ex_data,
                vol=vol,
                pg=p01,
                ex=extra_name,
                t=ch_title_name,
                pb=publisher,
                c=rls_credit,
            )
        if is_high_quality:
            final_filename += r" {HQ}"
        if rls_revision > 1:
            final_filename += " {r%d}" % rls_revision

        if not image_titling:
            image_titling = TARGET_TITLE.format(
                mt=manga_title,
                vol=vol,
                year=current_year,
                c=rls_credit,
                cpa=pair_left,
                cpb=pair_right,
            )
            if rls_revision > 1:
                image_titling += " (v%d)" % rls_revision

        final_filename += extension
        new_name = image.parent / final_filename
        image.rename(new_name)
        console.status(f"Processing: {current}/{total_img}")
        current += 1
    console.stop_status(f"Processed {current - 1} images!")

    if pingo_exe is not None and do_img_optimize:
        console.info("Optimizing images...")
        optimize_images(pingo_exe, path_or_archive)
    if exiftool_exe is not None and do_exif_tagging:
        console.info("Tagging images with exif metadata...")
        inject_metadata(exiftool_exe, path_or_archive, image_titling, rls_email)


@click.command(
    name="pack",
    help="Pack a release to a cbz archive.",
    cls=CatchAllExceptionsCommand,
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-t",
    "--title",
    "manga_title",
    required=True,
    help="The title of the series",
)
@options.manga_year
@options.manga_volume
@options.manga_chapter
@options.rls_credit
@options.rls_email
@options.rls_revision
@options.use_bracket_type
@check_config_first
@time_program
def pack_releases(
    path_or_archive: Path,
    manga_title: str,
    manga_year: Optional[int],
    manga_volume: Optional[int],
    manga_chapter: Optional[Union[int, float]],
    rls_credit: str,
    rls_email: str,
    rls_revision: int,
    bracket_type: Literal["square", "round", "curly"],
):
    """
    Pack a release to a cbz/cbr/cb7 archive.
    """

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    current_pst = datetime.now(timezone(timedelta(hours=-8)))
    current_year = manga_year or current_pst.year

    tag_sep = conf.defaults.ch_special_tag

    volume_text: Optional[str] = None
    if manga_chapter is not None:
        if isinstance(manga_chapter, float):
            float_string = str(manga_chapter)
            base_float, decimal_float = float_string.split(".")
            dec_float = int(decimal_float)
            if dec_float - 4 > 0:
                dec_float -= 4
            volume_text = f"{int(base_float):03d}{tag_sep}{dec_float}"
        else:
            volume_text = f"{manga_chapter:03d}"
        if conf.defaults.ch_add_c_prefix:
            volume_text = f"c{volume_text}"
    if manga_volume is not None:
        volume_text = f"v{manga_volume:02d}"
    if volume_text is None:
        raise click.BadParameter(
            "Please provide either a chapter or a volume number.",
            param_hint="manga_volume",
        )

    pair_left, pair_right = BRACKET_MAPPINGS.get(bracket_type.lower(), BRACKET_MAPPINGS["square"])
    console.info("Trying to pack release...")
    actual_filename = TARGET_TITLE.format(
        mt=manga_title,
        vol=volume_text,
        year=current_year,
        c=rls_credit,
        cpa=pair_left,
        cpb=pair_right,
    )
    if rls_revision > 1:
        actual_filename += " (v%d)" % rls_revision

    parent_dir = path_or_archive.parent
    cbz_target = exporter.CBZMangaExporter(actual_filename, parent_dir)

    cbz_target.set_comment(rls_email)
    console.status("Packing... (0/???)")
    idx = 1
    with file_handler.MangaArchive(path_or_archive) as archive:
        for image, total_count in archive:
            cbz_target.add_image(image.name, image.access())
            console.status(f"Packing... ({idx}/{total_count})")
            idx += 1
    console.stop_status(f"Packed ({idx - 1}/{total_count})")
    cbz_target.close()


@click.command(
    name="packepub",
    help="Pack a release to an epub archive.",
    cls=CatchAllExceptionsCommand,
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-t",
    "--title",
    "epub_title",
    required=True,
    help="The title of the series",
)
@click.option(
    "-s",
    "--source",
    "epub_source",
    required=True,
    help="The source where this is ripped from",
)
@options.manga_volume
@options.rls_credit
@check_config_first
@time_program
def pack_releases_epub_mode(
    path_or_archive: Path,
    epub_title: str,
    epub_source: str,
    manga_volume: Optional[int],
    rls_credit: str,
):
    """
    Pack a release to an epub archive.
    """

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )
    if manga_volume is None:
        raise click.BadParameter(
            "Please provide a volume number.",
            param_hint="manga_volume",
        )

    console.info("Trying to pack release...")
    actual_filename = TARGET_TITLE_NOVEL.format(
        mt=epub_title,
        c=rls_credit,
        vol=f"v{manga_volume:02d}",
        source=epub_source,
    )

    parent_dir = path_or_archive.parent
    save_target = parent_dir / f"{actual_filename}.epub"
    epub_target = ZipFile(save_target, "w", ZIP_DEFLATED)
    epub_target.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)

    # Check valid
    if not (path_or_archive / "META-INF").exists():
        raise click.BadParameter(
            f"{path_or_archive} is not a valid epub directory. Please provide a valid epub directory.",
            param_hint="path_or_archive",
        )

    console.status("Packing... (0/???)")
    idx = 1
    for path in path_or_archive.glob("**/*"):
        if path.name == "mimetype":
            continue
        console.status(f"Packing... ({idx}/???)")
        epub_target.write(path, path.relative_to(path_or_archive))
        idx += 1
    console.stop_status(f"Packed ({idx - 1}/{idx})")
    epub_target.close()

    MIMETYPE_MAGIC = b"mimetypeapplication/epub+zip"
    ZIP_MAGIC = b"PK\x03\x04"

    # Verify
    console.info("Verifying...")
    with save_target.open("rb") as fp:
        fp.seek(0)
        read_meta = fp.read(120)  # should be good enough
    if not read_meta.startswith(ZIP_MAGIC):
        console.error("Failed to pack EPUB. Please try again.")
        return 1

    if MIMETYPE_MAGIC not in read_meta:
        console.warning("We successfully packed the EPUB, but it is not a valid EPUB (mimetype is missing).")
        return 1


@click.command(
    name="packcomment",
    help="Comment an archive file.",
    cls=CatchAllExceptionsCommand,
)
@options.path_or_archive(disable_folder=True)
@click.option(
    "-c",
    "--comment",
    "archive_comment",
    required=False,
    default=None,
    help="The comment for this file.",
)
@click.option(
    "--remove",
    "remove_comment",
    is_flag=True,
    default=False,
    show_default=True,
    help="Remove the comment from the archive.",
)
@time_program
def pack_releases_comment_archive(
    path_or_archive: Path,
    archive_comment: Optional[str],
    remove_comment: bool,
):
    """ """

    if not path_or_archive.is_file():
        raise click.BadParameter(
            f"{path_or_archive} is not a file. Please provide a file!",
            param_hint="path_or_archive",
        )

    if archive_comment is None and not remove_comment:
        raise click.BadParameter(
            "Please provide a comment or use --remove to remove the comment.",
            param_hint="archive_comment",
        )

    stat_check = "Removing" if remove_comment else "Adding"
    archive = file_handler.MangaArchive(path_or_archive)
    console.status(f"{stat_check} comment...")
    if remove_comment:
        archive.comment = None
    else:
        archive.comment = archive_comment
    archive.close()

    console.stop_status(f"{stat_check} comment... done!")

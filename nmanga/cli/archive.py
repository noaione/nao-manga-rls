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
from typing import Literal, Optional, Union
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import click

from .. import config, exporter, file_handler, term
from . import options
from ._deco import check_config_first, time_program
from .base import NMangaCommandHandler
from .common import format_archive_filename
from .constants import MangaPublication

console = term.get_console()
conf = config.get_config()

TARGET_TITLE_NOVEL = "{mt} {vol} [{source}] [{c}]"


@click.command(
    name="pack",
    help="Pack a release to an archive.",
    cls=NMangaCommandHandler,
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
@options.manga_publication_type()
@options.rls_credit
@options.rls_email
@options.rls_revision
@options.use_bracket_type
@options.output_mode
@check_config_first
@time_program
def pack_releases(
    path_or_archive: Path,
    manga_title: str,
    manga_year: Optional[int],
    manga_volume: Optional[int],
    manga_chapter: Optional[Union[int, float]],
    manga_publication_type: MangaPublication,
    rls_credit: str,
    rls_email: str,
    rls_revision: int,
    bracket_type: Literal["square", "round", "curly"],
    output_mode: exporter.ExporterType,
):
    """
    Pack a release to an archive.
    """

    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    if output_mode == exporter.ExporterType.raw:
        raise click.BadParameter(
            f"{output_mode} cannot be `folder` or `raw` type. Use one of the archive mode.",
            param_hint="output_mode",
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

    console.info("Trying to pack release...")
    archive_filename = format_archive_filename(
        manga_title=manga_title,
        manga_year=current_year,
        publication_type=manga_publication_type,
        ripper_credit=rls_credit,
        bracket_type=bracket_type,
        manga_volume_text=volume_text,
        rls_revision=rls_revision,
    )

    parent_dir = path_or_archive.parent
    arc_target = exporter.exporter_factory(archive_filename, parent_dir, output_mode, manga_title=manga_title)

    if output_mode == exporter.ExporterType.epub:
        console.warning("Packing as EPUB, this will be a slower operation because of size checking!")

    arc_target.set_comment(rls_email)
    console.status("Packing... (0/???)")
    idx = 1
    with file_handler.MangaArchive(path_or_archive) as archive:
        for image, total_count in archive:
            arc_target.add_image(image.name, image.access())
            console.status(f"Packing... ({idx}/{total_count})")
            idx += 1
    console.stop_status(f"Packed ({idx - 1}/{total_count})")
    arc_target.close()


@click.command(
    name="packepub",
    help="Pack a release to an epub archive.",
    cls=NMangaCommandHandler,
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
    cls=NMangaCommandHandler,
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
    """Comment an archive file."""

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

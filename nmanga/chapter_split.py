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

# Reusable chapter splitting core shared by the CLI and the manual split GUI.
#
# This module is deliberately free of any terminal/UI dependency so it can be
# driven from a CLI, a test, or a GUI event loop alike.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Pattern

from . import exporter, file_handler, utils
from ._ntypes import VolumeNumberT
from .common import ChapterRange, RegexCollection, check_cbz_exist, format_volume_text

PAGE_NUMBER_RE = RegexCollection.page_re()
"""
The page number pattern used when no custom mapping or regex is given.

This is :meth:`nmanga.common.RegexCollection.page_re`, which understands the
standard naming used across this project: ``p003`` for a single page and
``p003-004`` for a spread, anywhere in the filename. So both ``p003`` and
``Title - c001 (v01) - p003 [dig] [nao]`` are understood out of the box.
"""

#: Matches a page digit run, either a single page or a ``003-004`` spread.
_DIGIT_RUN_RE = re.compile(r"(\d+)(?:[-~](\d+))?")

__all__ = (
    "PAGE_NUMBER_RE",
    "ChapterSplitError",
    "ChapterSplitObserver",
    "RangeCoverage",
    "SourcePage",
    "SplitReport",
    "build_chapter_archive_name",
    "coerce_number_page",
    "collect_source_pages",
    "extract_page_num",
    "find_chapter_for_page",
    "read_source_image",
    "split_chapters",
    "validate_chapter_coverage",
    "validate_source_coverage",
)


class ChapterSplitError(Exception):
    """Raised when a chapter split cannot be started or completed."""


class ChapterSplitObserver:
    """
    No-op observer for chapter split events.

    Subclass this and override the hooks you care about to receive progress
    updates while :func:`split_chapters` is running. Every hook is optional and
    is called with plain data so it can be forwarded to a log, a progress bar,
    or a GUI signal without any coupling to this module.
    """

    def on_chapter_start(self, chapter: str) -> None:
        """Called right before a new chapter archive starts being written."""

    def on_chapter_finish(self, chapter: str, target: Path) -> None:
        """Called after a chapter archive has been finalized."""

    def on_chapter_skip(self, chapter: str) -> None:
        """Called when a chapter is skipped because its output already exists."""

    def on_page_unassigned(self, page: int, filename: str) -> None:
        """Called when an image cannot be assigned to any chapter range."""


@dataclass
class SourcePage:
    """A single image of the source, with the page number(s) parsed from its name."""

    index: int
    """Zero-based position of the image in reading order."""

    filename: str
    page_numbers: list[int]

    @property
    def first_page(self) -> int:
        """The first page number this image occupies."""
        return self.page_numbers[0]

    @property
    def is_spread(self) -> bool:
        """Whether this image is a spread covering two pages (e.g. ``003-004``)."""
        return len(self.page_numbers) > 1


@dataclass
class SplitReport:
    """Result of a :func:`split_chapters` run."""

    created: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    unassigned_pages: list[SourcePage] = field(default_factory=list)
    pages_assigned: int = 0
    pages_total: int = 0

    @property
    def created_chapters(self) -> int:
        """How many chapter archives were written."""
        return len(self.created)

    @property
    def has_unassigned(self) -> bool:
        """Whether any page fell outside of every chapter range."""
        return len(self.unassigned_pages) > 0


@dataclass
class RangeCoverage:
    """Validation result for a set of chapter ranges against the available pages."""

    unassigned_pages: list[int] = field(default_factory=list)
    overlapping_pages: list[int] = field(default_factory=list)
    out_of_bounds_pages: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the chapter ranges cover the source exactly once."""
        return not (self.unassigned_pages or self.overlapping_pages or self.out_of_bounds_pages)


def _detect_page_num(filename: str) -> list[int] | None:
    """
    Detect the standard ``p003``/``p003-004`` page naming, anywhere in the filename.

    Returns ``None`` when the filename does not use that naming, so the caller can
    fall back to a bare page number.
    """
    matched = PAGE_NUMBER_RE.match(filename)
    if matched is None:
        return None
    # Re-scan from the first page digit to capture the full run: ``PAGE_NUMBER_RE``
    # caps each group at three digits and would truncate a padded ``p0001``.
    digits = _DIGIT_RUN_RE.match(filename, matched.start("a"))
    if digits is None:  # pragma: no cover (defensive: group "a" always holds a digit)
        return None
    first_part = digits.group(1)
    second_part = digits.group(2)
    if second_part is None:
        return [int(first_part)]
    return [int(first_part), int(second_part)]


def _parse_bare_page_number(filename: str) -> list[int]:
    """
    Parse a bare page number, optionally a spread (``"003"``, ``"003-004"``, ``"003~004"``).

    Raises
    ------
    ChapterSplitError
        If the filename is not a bare page number (or spread of them).
    """
    if filename.endswith("-"):
        filename = filename[:-1]
    try:
        matching_part = re.match(r"(\d+)[-~](\d+)", filename)
        if matching_part is None:
            return [int(filename)]
        first_part = matching_part.group(1)
        second_part = matching_part.group(2)
        return [int(first_part), int(second_part)]
    except ValueError:
        raise ChapterSplitError(
            f"Cannot read a page number from the filename {filename!r}. "
            "Rename it to a bare page number (e.g. `001`), use the standard "
            "`p001`/`p001-002` naming, or provide a page number regex/custom mapping."
        ) from None


def extract_page_num(
    filename: str, custom_data: dict[str, int] | None = None, regex_data: Pattern[str] | None = None
) -> list[int]:
    """
    Extract the page number(s) from an image filename.

    The page number is looked up in this order:

    1. ``custom_data``, a mapping of filename fragments to a fixed page number.
       The first fragment contained in the filename wins, so covers and other
       unnumbered files can be pinned to a page.
    2. ``regex_data``, a user supplied regex whose first two groups are the
       (possibly ranged) page numbers. Providing this replaces the built-in
       detection entirely.
    3. :data:`PAGE_NUMBER_RE`, the standard ``p003``/``p003-004`` naming, which
       is matched anywhere in the filename.
    4. A bare page number, optionally a spread (``"003"``, ``"003-004"``).

    Parameters
    ----------
    filename: :class:`str`
        The image filename stem, e.g. ``"p003"``, ``"003-004"``, or
        ``"Title - c001 (v01) - p003 [dig]"``.
    custom_data: :class:`dict`, optional
        Mapping of filename fragments to a fixed page number, checked first.
    regex_data: :class:`re.Pattern`, optional
        Regex whose first two groups are the (possibly ranged) page numbers.

    Returns
    -------
    :class:`list` of :class:`int`
        One number for a single page, two for a spread.

    Raises
    ------
    ChapterSplitError
        If no page number can be read from the filename.
    """
    for page_match, page_number in (custom_data or {}).items():
        if page_match in filename:
            return [page_number]

    if regex_data is not None:
        # Legacy behaviour: the regex rewrites the page portion into an "a-b" range.
        return _parse_bare_page_number(re.sub(regex_data, r"\1-\2", filename))

    detected = _detect_page_num(filename)
    if detected is not None:
        return detected

    return _parse_bare_page_number(filename)


def coerce_number_page(number: int | float) -> str:
    """
    Format a page number as a zero-padded string, keeping any decimal part.

    Parameters
    ----------
    number: :class:`int` or :class:`float`
        The page number to format.

    Returns
    -------
    :class:`str`
        ``3`` becomes ``"003"``, ``1.5`` becomes ``"001.5"``.
    """
    if isinstance(number, int):
        return f"{number:03d}"
    base, floating = str(number).split(".")
    return f"{int(base):03d}.{floating}"


def _chapter_extra_marker(chapter: ChapterRange) -> int | None:
    """
    Return the split/extra marker for a floating chapter number, if any.

    Mirrors :func:`nmanga.common.create_chapter`, where ``2.1`` and ``2.5`` both
    become the first extra (``.5``) and ``2.6`` the second (``.6``).
    """
    floating = chapter.floating
    if floating is None:
        return None
    if floating - 4 >= 1:
        floating -= 4
    return floating + 4


def build_chapter_archive_name(chapter: ChapterRange, volume: VolumeNumberT | None = None) -> str:
    """
    Build the output archive name (without extension) for a chapter range.

    The chapter part mirrors :func:`nmanga.common.create_chapter`: a zero-padded
    3-digit chapter number, an optional ``.N`` split marker, and the chapter
    title. The volume part additionally understands omnibus ranges, so
    ``ChapterRange(5, "Title", volume=(1, 2))`` becomes ``"01-02.005 - Title"``.

    Parameters
    ----------
    chapter: :class:`nmanga.common.ChapterRange`
        The chapter range to name. Its own ``volume`` takes precedence over the
        ``volume`` argument.
    volume: :class:`int`, :class:`float`, or :class:`tuple`, optional
        Fallback volume to use when the chapter range has none.

    Returns
    -------
    :class:`str`
        The archive name, e.g. ``"01.005 - Title"``.
    """
    actual_volume = chapter.volume if chapter.volume is not None else volume

    chapter_data = f"{chapter.base:03d}"
    extra_marker = _chapter_extra_marker(chapter)
    if extra_marker is not None:
        chapter_data += f".{extra_marker}"
    if chapter.name is not None:
        chapter_data += f" - {utils.clean_title(chapter.name)}"

    if actual_volume is None:
        return chapter_data

    volume_text = format_volume_text(manga_volume=actual_volume)
    if volume_text is None:  # pragma: no cover (defensive, ``format_volume_text`` always returns a string here)
        return chapter_data
    # ``format_volume_text`` always prefixes the volume with ``v``, drop it.
    return f"{volume_text[1:]}.{chapter_data}"


def find_chapter_for_page(chapters: Iterable[ChapterRange], page: int) -> ChapterRange | None:
    """
    Return the first chapter range that owns ``page``, if any.

    Ranges marked with ``is_single`` behave as open-ended: they match any page
    greater than or equal to their starting page, matching the CLI behavior.
    """
    for chapter in chapters:
        if not chapter.range:
            continue
        if chapter.is_single:
            if page >= chapter.range[0]:
                return chapter
        elif page in chapter.range:
            return chapter
    return None


def _validate_source(source: Path) -> None:
    if not source.exists():
        raise ChapterSplitError(f"Source path does not exist: {source}")
    if source.is_file() and not file_handler.is_archive(source):
        raise ChapterSplitError(f"Source is not a valid archive: {source}")


def collect_source_pages(
    source: Path,
    *,
    custom_data: dict[str, int] | None = None,
    regex_data: Pattern[str] | None = None,
) -> list[SourcePage]:
    """
    List every image of a source folder/archive with its parsed page number(s).

    This performs no writing and is intended for previewing a source, e.g. to
    build a page grid in a GUI before deciding on chapter ranges.

    Parameters
    ----------
    source: :class:`pathlib.Path`
        A folder of images, or a CBZ/CBR/CB7/tar archive.
    custom_data: :class:`dict`, optional
        Mapping of filename fragments to a fixed page number.
    regex_data: :class:`re.Pattern`, optional
        Regex used to extract page numbers from filenames. When omitted, the
        standard ``p003``/``p003-004`` naming is detected automatically and bare
        page numbers are used as a fallback.

    Returns
    -------
    :class:`list` of :class:`SourcePage`
        One entry per image, in reading order.
    """
    _validate_source(source)
    pages: list[SourcePage] = []
    with file_handler.MangaArchive(source) as archive:
        for index, (image, _) in enumerate(archive):
            image_path = Path(image.filename)
            pages.append(
                SourcePage(
                    index=index,
                    filename=image_path.name,
                    page_numbers=extract_page_num(image_path.stem, custom_data, regex_data),
                )
            )
    return pages


def read_source_image(source: Path, filename: str) -> bytes:
    """
    Read a single image out of a folder or archive by filename.

    Parameters
    ----------
    source: :class:`pathlib.Path`
        The folder or archive to read from.
    filename: :class:`str`
        The image filename to look for.

    Returns
    -------
    :class:`bytes`
        The raw image data.

    Raises
    ------
    ChapterSplitError
        If the image cannot be found in the source.
    """
    _validate_source(source)
    if source.is_dir():
        target = source / filename
        if not target.is_file():
            raise ChapterSplitError(f"Image {filename!r} not found in {source}")
        return target.read_bytes()

    with file_handler.MangaArchive(source) as archive:
        for image, _ in archive:
            if Path(image.filename).name == filename:
                return archive.read(image)
    raise ChapterSplitError(f"Image {filename!r} not found in {source}")


def _referenced_pages(chapters: Iterable[ChapterRange], largest_page: int | None) -> dict[int, int]:
    """
    Map every page claimed by a chapter range to the number of ranges claiming it.

    Ranges marked with ``is_single`` are open-ended, so they are expanded up to
    ``largest_page``, matching how :func:`find_chapter_for_page` and the split
    itself resolve them.

    Parameters
    ----------
    chapters: :class:`~collections.abc.Iterable` of :class:`nmanga.common.ChapterRange`
        The chapter ranges to expand.
    largest_page: :class:`int` or ``None``
        Last page of the source, used to close an open-ended range.

    Returns
    -------
    :class:`dict` of :class:`int` to :class:`int`
        How many chapter ranges own each referenced page.
    """
    coverage: dict[int, int] = {}
    for chapter in chapters:
        if not chapter.range:
            continue
        if chapter.is_single:
            end_page = largest_page if largest_page is not None else chapter.range[0]
            pages = set(range(chapter.range[0], end_page + 1))
        else:
            pages = set(chapter.range)
        for page in pages:
            coverage[page] = coverage.get(page, 0) + 1
    return coverage


def validate_chapter_coverage(chapters: Iterable[ChapterRange], available_pages: Iterable[int]) -> RangeCoverage:
    """
    Check a set of chapter ranges against the page numbers actually present.

    Useful for validating a manual split before writing anything, and for
    surfacing gaps/overlaps in a GUI. Prefer :func:`validate_source_coverage`
    when the images are known: a page that is only covered by the second half of
    a spread still has to be listed here, or it looks absent.

    Parameters
    ----------
    chapters: :class:`~collections.abc.Iterable` of :class:`nmanga.common.ChapterRange`
        The chapter ranges to validate.
    available_pages: :class:`~collections.abc.Iterable` of :class:`int`
        Every page number present in the source.

    Returns
    -------
    :class:`RangeCoverage`
        The uncovered, duplicated, and missing pages.
    """
    available = set(available_pages)
    coverage = _referenced_pages(chapters, max(available, default=None))
    return RangeCoverage(
        unassigned_pages=sorted(available - set(coverage)),
        overlapping_pages=sorted(page for page, count in coverage.items() if count > 1),
        out_of_bounds_pages=sorted(set(coverage) - available),
    )


def validate_source_coverage(chapters: Iterable[ChapterRange], pages: Iterable[SourcePage]) -> RangeCoverage:
    """
    Check a set of chapter ranges against the images actually present in a source.

    Unlike :func:`validate_chapter_coverage`, this understands spreads: an image
    named ``p002-003`` occupies pages 2 *and* 3, so a range ending on 3 is not
    reported as referencing a missing page, and page 3 is not reported as
    uncovered just because no image starts there.

    ``unassigned_pages`` therefore holds exactly the pages that
    :func:`split_chapters` would skip, identified by the first page of the image
    that owns them.

    Parameters
    ----------
    chapters: :class:`~collections.abc.Iterable` of :class:`nmanga.common.ChapterRange`
        The chapter ranges to validate.
    pages: :class:`~collections.abc.Iterable` of :class:`SourcePage`
        The images of the source, as returned by :func:`collect_source_pages`.

    Returns
    -------
    :class:`RangeCoverage`
        The uncovered, duplicated, and missing pages.
    """
    source_pages = list(pages)
    present = {number for page in source_pages for number in page.page_numbers}
    coverage = _referenced_pages(chapters, max(present, default=None))
    claimed = set(coverage)
    return RangeCoverage(
        # An image is skipped when no chapter range owns its first page.
        unassigned_pages=sorted({page.first_page for page in source_pages if page.first_page not in claimed}),
        overlapping_pages=sorted(page for page, count in coverage.items() if count > 1),
        out_of_bounds_pages=sorted(claimed - present),
    )


def split_chapters(
    source: Path,
    output_dir: Path,
    chapters: list[ChapterRange],
    *,
    volume: VolumeNumberT | None = None,
    custom_data: dict[str, int] | None = None,
    regex_data: Pattern[str] | None = None,
    overwrite: bool = False,
    observer: ChapterSplitObserver | None = None,
) -> SplitReport:
    """
    Split a folder or archive into chapter CBZ archives.

    Images are assigned to the first chapter range that owns their first page
    number; images that match no range are collected in the report instead of
    aborting the whole operation.

    Parameters
    ----------
    source: :class:`pathlib.Path`
        A folder of images, or a CBZ/CBR/CB7/tar archive.
    output_dir: :class:`pathlib.Path`
        Directory that will receive the chapter ``.cbz`` files.
    chapters: :class:`list` of :class:`nmanga.common.ChapterRange`
        The chapter ranges, each optionally carrying its own volume.
    volume: :class:`int`, :class:`float`, or :class:`tuple`, optional
        Default volume used by chapter ranges that do not carry one.
    custom_data: :class:`dict`, optional
        Mapping of filename fragments to a fixed page number.
    regex_data: :class:`re.Pattern`, optional
        Regex used to extract page numbers from filenames. When omitted, the
        standard ``p003``/``p003-004`` naming is detected automatically and bare
        page numbers are used as a fallback.
    overwrite: :class:`bool`, optional
        Re-create chapter archives that already exist instead of skipping them.
    observer: :class:`ChapterSplitObserver`, optional
        Receives progress events while the split runs.

    Returns
    -------
    :class:`SplitReport`
        Details about what was created, skipped, and left unassigned.

    Raises
    ------
    ChapterSplitError
        If no chapter is given, or the source is missing/not a valid archive.
    """
    if not chapters:
        raise ChapterSplitError("At least one chapter range is required.")
    _validate_source(source)

    notify = observer if observer is not None else ChapterSplitObserver()
    report = SplitReport()
    writers: dict[str, exporter.CBZMangaExporter] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with file_handler.MangaArchive(source) as archive:
            for image, _ in archive:
                image_path = Path(image.filename)
                page_numbers = extract_page_num(image_path.stem, custom_data, regex_data)
                first_page = page_numbers[0]
                report.pages_total += 1

                selected = find_chapter_for_page(chapters, first_page)
                if selected is None:
                    skipped_page = SourcePage(report.pages_total - 1, image_path.name, page_numbers)
                    report.unassigned_pages.append(skipped_page)
                    notify.on_page_unassigned(first_page, image_path.name)
                    continue

                chapter_name = build_chapter_archive_name(selected, volume)
                # The chapter was already found to exist, skip the remaining pages quietly.
                if chapter_name in report.skipped:
                    continue

                if chapter_name not in writers:
                    base_name = utils.unsecure_filename(utils.secure_filename(chapter_name))
                    if not overwrite and check_cbz_exist(output_dir, base_name):
                        report.skipped.append(chapter_name)
                        notify.on_chapter_skip(chapter_name)
                        continue
                    notify.on_chapter_start(chapter_name)
                    writers[chapter_name] = exporter.CBZMangaExporter(base_name, output_dir)

                writers[chapter_name].add_image(image_path.name, archive.read(image))
                report.pages_assigned += 1
    finally:
        for chapter_name, writer in writers.items():
            writer.close()
            report.created.append(writer.target_path)
            notify.on_chapter_finish(chapter_name, writer.target_path)

    return report

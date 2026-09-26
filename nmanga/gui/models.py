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

# Qt models backing the page grid of the manual split GUI.

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, QPersistentModelIndex, Qt
from PySide6.QtGui import QImage, QPixmap

from ..chapter_split import SourcePage, find_chapter_for_page
from ..common import ChapterRange

PAGE_LABEL_ROLE = int(Qt.ItemDataRole.UserRole) + 1
"""Formatted page number(s) of the image, e.g. ``003`` or ``003-004``."""

FILENAME_ROLE = int(Qt.ItemDataRole.UserRole) + 2
"""Original filename of the image."""

CHAPTER_LABEL_ROLE = int(Qt.ItemDataRole.UserRole) + 3
"""Label of the owning chapter, e.g. ``c012``; empty when unassigned."""

CHAPTER_INDEX_ROLE = int(Qt.ItemDataRole.UserRole) + 4
"""Position of the owning chapter in the chapter list, or ``-1`` when unassigned."""

THUMBNAIL_ROLE = int(Qt.ItemDataRole.UserRole) + 5
"""Decoded :class:`~PySide6.QtGui.QPixmap`, a null pixmap on failure, or ``None`` while loading."""

_INVALID_INDEX = QModelIndex()
"""Shared invalid index used as the default argument of the ``rowCount`` override."""


def format_page_label(page: SourcePage) -> str:
    """
    Format the page number(s) of a source page for display.

    Parameters
    ----------
    page: :class:`nmanga.chapter_split.SourcePage`
        The page to label.

    Returns
    -------
    :class:`str`
        ``003`` for a single page, ``003-004`` for a spread.
    """
    return "-".join(f"{number:03d}" for number in page.page_numbers)


def format_chapter_label(chapter: ChapterRange) -> str:
    """
    Format a chapter range as a short label such as ``c012`` or ``c012.5``.

    Parameters
    ----------
    chapter: :class:`nmanga.common.ChapterRange`
        The chapter to label.

    Returns
    -------
    :class:`str`
        The chapter number prefixed with ``c``.
    """
    return f"c{chapter.bnum}"


class PageGridModel(QAbstractListModel):
    """
    Expose the loaded :class:`~nmanga.chapter_split.SourcePage` list to the page grid.

    Besides the filename, each row reports its parsed page number, the chapter
    range that owns it, and its thumbnail. Chapter ownership is recomputed by
    :meth:`set_chapters`, so the grid recolors itself while ranges are edited.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pages: list[SourcePage] = []
        self._chapters: list[ChapterRange] = []
        self._assignment: list[int] = []
        self._thumbnails: dict[int, QPixmap] = {}

    @property
    def pages(self) -> list[SourcePage]:
        """Every page of the source, in reading order."""
        return self._pages

    @property
    def chapters(self) -> list[ChapterRange]:
        """The chapter ranges currently reflected in the grid."""
        return self._chapters

    def set_pages(self, pages: Sequence[SourcePage]) -> None:
        """
        Replace the content of the grid with a new source.

        Parameters
        ----------
        pages: :class:`~collections.abc.Sequence` of :class:`nmanga.chapter_split.SourcePage`
            The pages to display, already in reading order.
        """
        self.beginResetModel()
        self._pages = list(pages)
        self._chapters = []
        self._assignment = [-1] * len(self._pages)
        self._thumbnails = {}
        self.endResetModel()

    def set_chapters(self, chapters: Sequence[ChapterRange]) -> None:
        """
        Recompute chapter ownership and repaint the grid.

        Parameters
        ----------
        chapters: :class:`~collections.abc.Sequence` of :class:`nmanga.common.ChapterRange`
            The ranges to assign pages to.
        """
        self._chapters = list(chapters)
        assignment: list[int] = []
        for page in self._pages:
            owner = find_chapter_for_page(self._chapters, page.first_page)
            position = -1
            if owner is not None:
                for candidate, chapter in enumerate(self._chapters):
                    # Identity comparison: `ChapterRange.__eq__` only compares numbers.
                    if chapter is owner:
                        position = candidate
                        break
            assignment.append(position)
        self._assignment = assignment
        if self._pages:
            top_left = self.index(0, 0)
            bottom_right = self.index(len(self._pages) - 1, 0)
            self.dataChanged.emit(
                top_left, bottom_right, [CHAPTER_INDEX_ROLE, CHAPTER_LABEL_ROLE, Qt.ItemDataRole.ToolTipRole]
            )

    def set_thumbnail(self, row: int, image: QImage) -> None:
        """
        Store a decoded thumbnail and repaint its cell.

        Parameters
        ----------
        row: :class:`int`
            Row of the page the thumbnail belongs to.
        image: :class:`~PySide6.QtGui.QImage`
            The decoded image, or a null image when decoding failed.
        """
        if row < 0 or row >= len(self._pages):
            return
        self._thumbnails[row] = QPixmap.fromImage(image)
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [THUMBNAIL_ROLE])

    def page_at(self, row: int) -> SourcePage | None:
        """
        Return the page displayed at ``row``, if any.

        Parameters
        ----------
        row: :class:`int`
            Row to look up.

        Returns
        -------
        :class:`nmanga.chapter_split.SourcePage` or ``None``
            The page, or ``None`` when the row is out of range.
        """
        if 0 <= row < len(self._pages):
            return self._pages[row]
        return None

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = _INVALID_INDEX) -> int:  # ruff: ignore[invalid-function-name]
        """Number of pages in the grid."""
        if parent.isValid():
            return 0
        return len(self._pages)

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = int(Qt.ItemDataRole.DisplayRole)) -> object:
        """Return the data stored under ``role`` for the page at ``index``."""
        row = index.row()
        if not index.isValid() or not 0 <= row < len(self._pages):
            return None
        page = self._pages[row]
        position = self._assignment[row] if row < len(self._assignment) else -1

        if role == Qt.ItemDataRole.DisplayRole or role == FILENAME_ROLE:
            return page.filename
        if role == PAGE_LABEL_ROLE:
            return format_page_label(page)
        if role == CHAPTER_INDEX_ROLE:
            return position
        if role == CHAPTER_LABEL_ROLE:
            if 0 <= position < len(self._chapters):
                return format_chapter_label(self._chapters[position])
            return "unassigned"
        if role == THUMBNAIL_ROLE:
            return self._thumbnails.get(row)
        if role == Qt.ItemDataRole.ToolTipRole:
            chapter = "unassigned"
            if 0 <= position < len(self._chapters):
                chapter = format_chapter_label(self._chapters[position])
            return f"{page.filename}\npages {format_page_label(page)} · {chapter}"
        return None

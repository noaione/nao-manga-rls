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

# Chapter range editor for the manual split GUI.

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..chapter_split import SourcePage
from ..common import ChapterRange, int_or_float, parse_ch_ranges, parse_volume_number

ERROR_BRUSH = QBrush(QColor("#f5c6c6"))
"""Background used to flag a cell that could not be parsed."""


class ChapterEditorPanel(QWidget):
    """
    Table of chapter ranges: number, title, pages, and optional volume.

    The table is the single source of truth for the ranges: every read parses the
    cells, flags the ones that cannot be parsed, and skips them, so an in-progress
    edit never crashes a split. A bare page number in the pages column means
    "from that page to the end", matching the CLI.

    A per-row volume may be an omnibus range (``1-2``). Rows left blank use the
    default volume passed to :func:`nmanga.chapter_split.split_chapters`.
    """

    #: Emitted whenever a row was added, removed, reordered, or edited.
    chapters_changed = Signal()

    COLUMN_NUMBER = 0
    COLUMN_TITLE = 1
    COLUMN_PAGES = 2
    COLUMN_VOLUME = 3

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pages: list[SourcePage] = []
        self._suspend = 0

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Chapter", "Title", "Pages", "Volume"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(self.COLUMN_NUMBER, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COLUMN_TITLE, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self.COLUMN_PAGES, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COLUMN_VOLUME, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemChanged.connect(self._on_item_changed)

        hint = QLabel(
            "Pages accept `1-20`, a bare `21` for “21 to the end”, or a comma list such as `1,5-20` to skip pages. "
            "Volume accepts `1` or `1-2` for an omnibus."
        )
        hint.setWordWrap(True)

        self._status = QLabel("No chapters yet.")
        self._status.setWordWrap(True)

        auto_split_button = QPushButton("Split evenly in")
        auto_split_button.setToolTip("Distribute the loaded pages evenly over this many chapters")
        auto_split_button.clicked.connect(self._on_auto_split)
        self._auto_split_count = QSpinBox(self)
        self._auto_split_count.setRange(1, 99)
        self._auto_split_count.setValue(5)

        self._add_button = QPushButton("Add")
        self._add_button.setToolTip("Append a chapter range that starts on the next page no other chapter owns")
        self._add_button.clicked.connect(self.append_chapter)
        self._remove_button = QPushButton("Remove")
        self._remove_button.clicked.connect(self.remove_selected)
        self._up_button = QPushButton("Move up")
        self._up_button.clicked.connect(lambda: self.move_selected(-1))
        self._down_button = QPushButton("Move down")
        self._down_button.clicked.connect(lambda: self.move_selected(1))
        self._clear_button = QPushButton("Clear")
        self._clear_button.clicked.connect(self.clear_chapters)
        self._edit_buttons = (
            self._add_button,
            self._remove_button,
            self._up_button,
            self._down_button,
            self._clear_button,
        )

        buttons = QHBoxLayout()
        for button in self._edit_buttons:
            buttons.addWidget(button)

        auto_split_row = QHBoxLayout()
        auto_split_row.addWidget(auto_split_button)
        auto_split_row.addWidget(self._auto_split_count)
        auto_split_row.addWidget(QLabel("chapters"))
        auto_split_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Chapter ranges"))
        layout.addWidget(self._table, 1)
        layout.addLayout(auto_split_row)
        layout.addLayout(buttons)
        layout.addWidget(hint)
        layout.addWidget(self._status)

    # -- data in --------------------------------------------------------------
    def set_pages(self, pages: Sequence[SourcePage]) -> None:
        """
        Remember the loaded pages, used for the even auto-split.

        Parameters
        ----------
        pages: :class:`~collections.abc.Sequence` of :class:`nmanga.chapter_split.SourcePage`
            The pages of the loaded source.
        """
        self._pages = list(pages)

    # -- data out -------------------------------------------------------------
    def chapters(self) -> list[ChapterRange]:
        """
        Parse the table into chapter ranges.

        Rows that cannot be parsed are skipped; :meth:`_emit_changed` is what
        highlights them. This call is side-effect free so it can be used from a
        ``chapters_changed`` handler without recursing.

        Returns
        -------
        :class:`list` of :class:`nmanga.common.ChapterRange`
            One range per valid row, in table order.
        """
        ranges: list[ChapterRange] = []
        for row in range(self._table.rowCount()):
            chapter = self._parse_row(row, mark=False)
            if chapter is not None:
                ranges.append(chapter)
        return ranges

    def current_row(self) -> int:
        """
        Return the selected row.

        Returns
        -------
        :class:`int`
            The selected row, or ``-1`` when nothing is selected.
        """
        return self._table.currentRow()

    # -- mutations ------------------------------------------------------------
    def add_chapter(self, start_page: int, end_page: int, number: str | None = None, title: str = "") -> int:
        """
        Append a chapter range built from a page selection.

        Parameters
        ----------
        start_page: :class:`int`
            First page number of the range.
        end_page: :class:`int`
            Last page number of the range.
        number: :class:`str`, optional
            Chapter number to use; defaults to one past the highest in the table.
        title: :class:`str`, optional
            Chapter title.

        Returns
        -------
        :class:`int`
            The row that was added.
        """
        row = self._table.rowCount()
        with self._bulk():
            self._table.insertRow(row)
            self._write_row(row, number if number is not None else self._next_number(), title, start_page, end_page)
        self._table.setCurrentCell(row, self.COLUMN_PAGES)
        self._emit_changed()
        return row

    def append_chapter(self) -> int:
        """
        Append a chapter range that continues after the pages already covered.

        The new range starts on the first page no other row owns and stops before
        the next owned page, so pressing Add repeatedly carves the source up in
        reading order.

        Returns
        -------
        :class:`int`
            The row that was added.
        """
        start_page, end_page = self._next_free_range()
        return self.add_chapter(start_page, end_page)

    def apply_range(self, row: int, start_page: int, end_page: int) -> None:
        """
        Overwrite the page range of an existing row.

        Parameters
        ----------
        row: :class:`int`
            Row to update; ignored when out of range.
        start_page: :class:`int`
            First page number of the range.
        end_page: :class:`int`
            Last page number of the range.
        """
        if not 0 <= row < self._table.rowCount():
            return
        with self._bulk():
            self._set_cell(row, self.COLUMN_PAGES, f"{start_page:03d}-{end_page:03d}")
        self._emit_changed()

    def remove_selected(self) -> None:
        """Remove the selected row, if any."""
        row = self._table.currentRow()
        if row < 0:
            return
        with self._bulk():
            self._table.removeRow(row)
        self._emit_changed()

    def move_selected(self, offset: int) -> None:
        """
        Move the selected row up or down the table.

        Parameters
        ----------
        offset: :class:`int`
            ``-1`` to move up, ``1`` to move down.
        """
        row = self._table.currentRow()
        target = row + offset
        if row < 0 or not 0 <= target < self._table.rowCount():
            return
        with self._bulk():
            self._swap_rows(row, target)
        self._table.setCurrentCell(target, self.COLUMN_PAGES)
        self._emit_changed()

    def clear_chapters(self) -> None:
        """Remove every chapter row."""
        with self._bulk():
            self._table.setRowCount(0)
        self._emit_changed()

    def set_editable(self, editable: bool) -> None:
        """
        Enable or disable the editor.

        Parameters
        ----------
        editable: :class:`bool`
            Whether a source is loaded and chapters may be edited.
        """
        self._table.setEnabled(editable)
        for widget in (*self._edit_buttons, self._auto_split_count):
            widget.setEnabled(editable)

    # -- internals ------------------------------------------------------------
    @contextmanager
    def _bulk(self) -> Iterator[None]:
        """Suppress change notifications while several rows are touched at once."""
        self._suspend += 1
        try:
            yield
        finally:
            self._suspend -= 1

    def _on_item_changed(self, _item: QTableWidgetItem) -> None:
        if self._suspend:
            return
        self._emit_changed()

    def _emit_changed(self) -> None:
        """Re-validate the table, repaint the error rows, and notify listeners."""
        with self._bulk():
            invalid = sum(1 for row in range(self._table.rowCount()) if self._parse_row(row, mark=True) is None)
        valid = self._table.rowCount() - invalid
        if invalid:
            self._status.setText(f"{valid} valid chapter(s), {invalid} row(s) need fixing (highlighted in red).")
            self._status.setStyleSheet("color: #b3261e;")
        else:
            self._status.setText(f"{valid} chapter(s) ready." if valid else "No chapters yet.")
            self._status.setStyleSheet("")
        self.chapters_changed.emit()

    def _parse_row(self, row: int, *, mark: bool) -> ChapterRange | None:
        """
        Parse one row into a chapter range.

        Parameters
        ----------
        row: :class:`int`
            Row to parse.
        mark: :class:`bool`
            Highlight the cells that could not be parsed.

        Returns
        -------
        :class:`nmanga.common.ChapterRange` or ``None``
            The parsed range, or ``None`` when the row is incomplete or invalid.
        """
        number = int_or_float(self._cell_text(row, self.COLUMN_NUMBER))
        if number is None:
            if mark:
                self._mark(row, self.COLUMN_NUMBER, "Chapter number must be a number, e.g. `12` or `12.5`.")
            return None

        pages_text = self._cell_text(row, self.COLUMN_PAGES)
        try:
            pages, is_single = parse_ch_ranges(pages_text)
        except ValueError:
            if mark:
                self._mark(
                    row,
                    self.COLUMN_PAGES,
                    "Pages must be `1-20`, a bare `21` for “21 to the end”, or a comma list such as `1,5-20`.",
                )
            return None
        if not pages:
            if mark:
                self._mark(row, self.COLUMN_PAGES, "Pages range cannot be empty.")
            return None

        volume_text = self._cell_text(row, self.COLUMN_VOLUME).strip()
        volume = None
        if volume_text:
            volume = parse_volume_number(volume_text)
            if volume is None:
                if mark:
                    self._mark(row, self.COLUMN_VOLUME, "Volume must be `1`, `1.5`, or `1-2` for an omnibus.")
                return None

        if mark:
            for column in (self.COLUMN_NUMBER, self.COLUMN_PAGES, self.COLUMN_VOLUME):
                self._mark(row, column, None)
        title = self._cell_text(row, self.COLUMN_TITLE).strip()
        return ChapterRange(number, title or None, pages, is_single, volume=volume)

    def _mark(self, row: int, column: int, error: str | None) -> None:
        """Flag or clear a cell that could not be parsed."""
        item = self._table.item(row, column)
        if item is None:
            return
        item.setBackground(ERROR_BRUSH if error else QBrush(Qt.GlobalColor.transparent))
        item.setToolTip(error or "")

    def _write_row(self, row: int, number: str, title: str, start_page: int, end_page: int) -> None:
        """Fill a freshly inserted row with parsed defaults."""
        self._set_cell(row, self.COLUMN_NUMBER, number)
        self._set_cell(row, self.COLUMN_TITLE, title)
        self._set_cell(row, self.COLUMN_PAGES, f"{start_page:03d}-{end_page:03d}")
        self._set_cell(row, self.COLUMN_VOLUME, "")

    def _set_cell(self, row: int, column: int, text: str) -> None:
        """Write a cell, creating the item if the row was just inserted."""
        item = self._table.item(row, column)
        if item is None:
            self._table.setItem(row, column, QTableWidgetItem(text))
            return
        item.setText(text)

    def _swap_rows(self, first: int, second: int) -> None:
        """Exchange the content of two rows."""
        for column in range(self._table.columnCount()):
            left = self._cell_text(first, column)
            right = self._cell_text(second, column)
            self._set_cell(first, column, right)
            self._set_cell(second, column, left)

    def _cell_text(self, row: int, column: int) -> str:
        """Read a cell as text, tolerating a row that was never filled in."""
        item = self._table.item(row, column)
        return item.text() if item is not None else ""

    def _next_free_range(self) -> tuple[int, int]:
        """
        Page range a new chapter should cover by default.

        Returns
        -------
        :class:`tuple` of :class:`int`
            The first page no row owns, and the last page before the next owned
            page (or the last page of the source when nothing is owned after it).
        """
        numbers = [number for page in self._pages for number in page.page_numbers]
        if not numbers:
            return 1, 1
        last_page = numbers[-1]
        taken: set[int] = set()
        for chapter in self.chapters():
            if not chapter.range:
                continue
            if chapter.is_single:
                taken.update(range(chapter.range[0], last_page + 1))
            else:
                taken.update(chapter.range)
        start_page = next((number for number in numbers if number not in taken), last_page)
        later_owned = [number for number in numbers if number > start_page and number in taken]
        end_page = min(later_owned) - 1 if later_owned else last_page
        return start_page, max(start_page, end_page)

    def _next_number(self) -> str:
        """Number to give a newly added chapter: one past the highest so far."""
        highest: int | float | None = None
        for row in range(self._table.rowCount()):
            number = int_or_float(self._cell_text(row, self.COLUMN_NUMBER))
            if number is None:
                continue
            highest = number if highest is None else max(highest, number)
        if highest is None:
            return "1"
        if isinstance(highest, int):
            return str(highest + 1)
        return str(math.floor(highest) + 1)

    def _on_auto_split(self) -> None:
        """Replace every row with an even split of the loaded pages."""
        self.auto_split(self._auto_split_count.value())

    def auto_split(self, count: int) -> None:
        """
        Distribute the loaded pages evenly over ``count`` chapters.

        Parameters
        ----------
        count: :class:`int`
            Number of chapters to create; ignored when no source is loaded.
        """
        if not self._pages or count < 1:
            return
        chunk = math.ceil(len(self._pages) / count)
        with self._bulk():
            self._table.setRowCount(0)
            for index in range(count):
                part = self._pages[index * chunk : (index + 1) * chunk]
                if not part:
                    break
                row = self._table.rowCount()
                self._table.insertRow(row)
                self._write_row(row, str(index + 1), "", part[0].first_page, part[-1].page_numbers[-1])
        self._emit_changed()

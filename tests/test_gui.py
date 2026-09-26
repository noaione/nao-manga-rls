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

# Tests for the optional PySide6 manual split GUI.
#
# The whole module is skipped when PySide6 is not installed (it is an optional
# `gui` extra), and the widgets run against Qt's offscreen platform so no
# display is needed.

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any, Generator, cast

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent
from PySide6.QtWidgets import QApplication, QMessageBox, QTableWidgetItem

from nmanga.chapter_split import SourcePage
from nmanga.common import ChapterRange, format_page_numbers
from nmanga.gui.chapters import ChapterEditorPanel
from nmanga.gui.models import (
    CHAPTER_LABEL_ROLE,
    FILENAME_ROLE,
    PAGE_LABEL_ROLE,
    PageGridModel,
)
from nmanga.gui.thumbnails import (
    MAX_THUMBNAIL_WIDTH,
    MIN_THUMBNAIL_WIDTH,
    ThumbnailLoader,
    pick_thumbnail_width,
)
from nmanga.gui.window import MainWindow

_APP: QApplication | None = None
"""Strong reference to the QApplication, created once for the whole module."""


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    global _APP
    _APP = QApplication.instance() or QApplication([])  # pyright: ignore[reportAssignmentType]
    return cast(QApplication, _APP)


@pytest.fixture
def pages() -> list[SourcePage]:
    return [SourcePage(index=index, filename=f"p{index + 1:03d}.png", page_numbers=[index + 1]) for index in range(20)]


@pytest.fixture
def window(qt_app) -> Generator[MainWindow, Any, Any]:
    """A main window with nothing loaded, torn down after the test."""
    win = MainWindow()
    try:
        yield win
    finally:
        win.close()
        win.deleteLater()
        qt_app.processEvents()


class _Drag:
    """
    A file-manager drag payload, together with the events built from it.

    A ``QDropEvent`` only borrows the :class:`~PySide6.QtCore.QMimeData` pointer,
    so the payload has to outlive every event created from it.
    """

    def __init__(self, *paths: Path) -> None:
        self.mime = QMimeData()
        self.mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])

    def enter(self) -> QDragEnterEvent:
        """A drag-enter event carrying the payload."""
        return QDragEnterEvent(
            QPoint(10, 10),
            Qt.DropAction.CopyAction,
            self.mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

    def drop(self) -> QDropEvent:
        """A drop event carrying the payload."""
        return QDropEvent(
            QPointF(10.0, 10.0),
            Qt.DropAction.CopyAction,
            self.mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )


def test_pick_thumbnail_width_stays_within_bounds():
    assert pick_thumbnail_width(0) == MAX_THUMBNAIL_WIDTH
    assert pick_thumbnail_width(200) == MAX_THUMBNAIL_WIDTH
    assert MIN_THUMBNAIL_WIDTH <= pick_thumbnail_width(5000) < MAX_THUMBNAIL_WIDTH
    assert pick_thumbnail_width(1_000_000) == MIN_THUMBNAIL_WIDTH


def test_thumbnail_loader_picks_a_width_for_the_batch(qt_app, tmp_path, pages):
    loader = ThumbnailLoader()
    try:
        assert loader.start(tmp_path, pages) == MAX_THUMBNAIL_WIDTH
    finally:
        loader.cancel()


def test_grid_model_labels_pages_and_chapters(qt_app, pages):
    model = PageGridModel()
    model.set_pages(pages)
    model.set_chapters([
        ChapterRange(1, "start", list(range(1, 11))),
        ChapterRange(2, "end", list(range(11, 21))),
    ])

    assert model.rowCount() == len(pages)
    assert model.data(model.index(0, 0), FILENAME_ROLE) == "p001.png"
    assert model.data(model.index(0, 0), PAGE_LABEL_ROLE) == "001"
    assert model.data(model.index(0, 0), CHAPTER_LABEL_ROLE) == "c001"
    assert model.data(model.index(15, 0), CHAPTER_LABEL_ROLE) == "c002"


def test_grid_model_marks_unassigned_pages(qt_app, pages):
    model = PageGridModel()
    model.set_pages(pages)
    model.set_chapters([ChapterRange(1, None, list(range(1, 6)))])

    assert model.data(model.index(4, 0), CHAPTER_LABEL_ROLE) == "c001"
    assert model.data(model.index(5, 0), CHAPTER_LABEL_ROLE) == "unassigned"


def test_editor_parses_added_chapters(qt_app, pages):
    panel = ChapterEditorPanel()
    panel.set_pages(pages)

    panel.add_chapter(1, 5)
    panel.add_chapter(6, 10, title="Beach")

    chapters = panel.chapters()
    assert [(chapter.number, chapter.name) for chapter in chapters] == [(1, None), (2, "Beach")]
    assert chapters[0].range == [1, 2, 3, 4, 5]
    assert chapters[1].is_single is False


def test_editor_skips_unparseable_rows(qt_app, pages):
    panel = ChapterEditorPanel()
    panel.set_pages(pages)
    panel.add_chapter(1, 5)

    panel._table.setItem(0, ChapterEditorPanel.COLUMN_PAGES, QTableWidgetItem("not a range"))

    assert panel.chapters() == []
    assert "need fixing" in panel._status.text()


def test_editor_keeps_per_chapter_volume_for_omnibus(qt_app, pages):
    panel = ChapterEditorPanel()
    panel.set_pages(pages)
    panel.add_chapter(1, 5)
    panel._table.setItem(0, ChapterEditorPanel.COLUMN_VOLUME, QTableWidgetItem("1-2"))

    chapter = panel.chapters()[0]
    assert chapter.volume == (1, 2)
    assert chapter.is_omnibus is True


def test_editor_auto_split_covers_every_page_once(qt_app, pages):
    panel = ChapterEditorPanel()
    panel.set_pages(pages)

    panel.auto_split(4)

    chapters = panel.chapters()
    assert [chapter.range[0] for chapter in chapters] == [1, 6, 11, 16]
    assert [chapter.range[-1] for chapter in chapters] == [5, 10, 15, 20]


def test_editor_add_button_appends_after_the_last_range(qt_app, pages):
    panel = ChapterEditorPanel()
    panel.set_pages(pages)
    panel.add_chapter(1, 5)

    panel._add_button.click()

    chapters = panel.chapters()
    assert [chapter.number for chapter in chapters] == [1, 2]
    assert [chapter.range[0] for chapter in chapters] == [1, 6]
    assert chapters[-1].range == list(range(6, 21))


def test_editor_add_button_fills_the_first_gap(qt_app, pages):
    panel = ChapterEditorPanel()
    panel.set_pages(pages)
    panel.add_chapter(1, 5)
    panel.add_chapter(11, 20)

    panel._add_button.click()

    chapters = panel.chapters()
    assert [chapter.number for chapter in chapters] == [1, 2, 3]
    assert chapters[-1].range == list(range(6, 11))


def test_editor_add_button_starts_at_page_one_without_a_source(qt_app):
    panel = ChapterEditorPanel()

    panel._add_button.click()

    assert [chapter.range for chapter in panel.chapters()] == [[1]]


def test_editor_add_button_ignores_unparseable_rows(qt_app, pages):
    panel = ChapterEditorPanel()
    panel.set_pages(pages)
    panel.add_chapter(1, 5)
    panel._table.setItem(0, ChapterEditorPanel.COLUMN_PAGES, QTableWidgetItem("not a range"))

    panel._add_button.click()

    chapters = panel.chapters()
    assert len(chapters) == 1
    assert chapters[0].range == list(range(1, 21))


def test_editor_accepts_a_comma_list_and_skips_pages(qt_app, pages):
    panel = ChapterEditorPanel()
    panel.set_pages(pages)
    panel.add_chapter(1, 1)

    panel._table.setItem(0, ChapterEditorPanel.COLUMN_PAGES, QTableWidgetItem("1,5-7"))

    chapters = panel.chapters()
    assert len(chapters) == 1
    assert chapters[0].range == [1, 5, 6, 7]
    # A comma means manual selection, so no open-ended "to the end" range.
    assert chapters[0].is_single is False


def test_editor_rejects_a_comma_list_with_a_trailing_comma(qt_app, pages):
    panel = ChapterEditorPanel()
    panel.set_pages(pages)
    panel.add_chapter(1, 5)

    panel._table.setItem(0, ChapterEditorPanel.COLUMN_PAGES, QTableWidgetItem("1,5-"))

    assert panel.chapters() == []
    assert "need fixing" in panel._status.text()


def test_window_accepts_a_dropped_folder(qt_app, window, tmp_path):
    source = tmp_path / "v01"
    source.mkdir()
    for page in range(1, 4):
        # Listing a folder source only looks at the extension, so empty files are enough.
        (source / f"p{page:03d}.png").write_bytes(b"")

    drag = _Drag(source)
    enter = drag.enter()
    window.dragEnterEvent(enter)
    assert enter.isAccepted()
    assert "Drop to load the folder v01" in window._status_label.text()

    drop = drag.drop()
    window.dropEvent(drop)
    assert drop.isAccepted()
    assert window._load_worker is not None
    assert window._load_worker.wait(10000)
    qt_app.processEvents()
    window._thumbnails.cancel()

    assert window._source_edit.text() == str(source)
    assert [page.filename for page in window._model.pages] == ["p001.png", "p002.png", "p003.png"]
    assert window._status_label.text() == "3 pages · 0 chapter(s) · add a chapter to split"


def test_window_accepts_a_dropped_archive(qt_app, window, tmp_path):
    archive = tmp_path / "v01.cbz"
    with zipfile.ZipFile(archive, "w") as cbz:
        cbz.writestr("p001.png", b"")
        cbz.writestr("p002.png", b"")

    drag = _Drag(archive)
    enter = drag.enter()
    window.dragEnterEvent(enter)
    assert enter.isAccepted()
    assert "Drop to load the archive v01.cbz" in window._status_label.text()

    window.dropEvent(drag.drop())
    assert window._load_worker is not None
    assert window._load_worker.wait(10000)
    qt_app.processEvents()
    window._thumbnails.cancel()

    assert window._source_edit.text() == str(archive)
    assert len(window._model.pages) == 2


def test_window_rejects_a_dropped_image(qt_app, window, tmp_path):
    image = tmp_path / "p001.png"
    image.write_bytes(b"")

    drag = _Drag(image)
    enter = drag.enter()
    window.dragEnterEvent(enter)
    assert not enter.isAccepted()

    drop = drag.drop()
    window.dropEvent(drop)
    assert not drop.isAccepted()
    assert window._source_edit.text() == ""
    assert window._load_worker is None


def test_window_rejects_a_drop_of_several_paths(qt_app, window, tmp_path):
    first = tmp_path / "v01"
    second = tmp_path / "v02"
    first.mkdir()
    second.mkdir()

    drag = _Drag(first, second)
    enter = drag.enter()
    window.dragEnterEvent(enter)
    assert not enter.isAccepted()
    assert "Drop to load" not in window._status_label.text()


def test_window_clears_the_drop_hint_on_leave(qt_app, window, tmp_path):
    source = tmp_path / "v01"
    source.mkdir()

    drag = _Drag(source)
    window.dragEnterEvent(drag.enter())
    assert window._source_edit.styleSheet() != ""

    window.dragLeaveEvent(QDragLeaveEvent())
    assert window._source_edit.styleSheet() == ""
    assert window._status_label.text() == "Load a source to start."


class TestFormatPageNumbers:
    def test_collapses_runs(self):
        assert format_page_numbers([3, 4, 5, 9]) == "003-005, 009"
        assert format_page_numbers([1]) == "001"
        assert format_page_numbers([]) == ""

    def test_sorts_and_deduplicates(self):
        assert format_page_numbers([5, 3, 4, 3]) == "003-005"

    def test_limit_summarizes_the_rest(self):
        assert format_page_numbers([1, 3, 5, 7], limit=2) == "001, 003, … (+2 more)"

    def test_limit_keeps_everything_that_fits(self):
        assert format_page_numbers([1, 3, 5], limit=3) == "001, 003, 005"


def _load_pages(window: MainWindow, pages: list[SourcePage]) -> None:
    """Put a page list into the window without going through the loader worker."""
    window._pages = list(pages)
    window._model.set_pages(pages)
    window._editor.set_pages(pages)


def _numbered_pages(count: int) -> list[SourcePage]:
    """`count` single-page images numbered from 1."""
    return [
        SourcePage(index=index, filename=f"p{index + 1:03d}.png", page_numbers=[index + 1]) for index in range(count)
    ]


def test_window_does_not_report_a_spread_page_as_missing(qt_app, window):
    # Regression: page 3 is only reachable through the 002-003 spread, so a range
    # ending on 3 used to be flagged as referencing a page missing from the source.
    _load_pages(
        window,
        [
            SourcePage(index=0, filename="p001.png", page_numbers=[1]),
            SourcePage(index=1, filename="p002-003.png", page_numbers=[2, 3]),
            SourcePage(index=2, filename="p004.png", page_numbers=[4]),
        ],
    )
    window._editor.add_chapter(1, 4)

    assert window._status_label.text() == "3 pages · 1 chapter(s) · fully covered"
    assert window._status_label.toolTip() == ""


def test_window_reports_the_uncovered_page_numbers(qt_app, window):
    _load_pages(window, _numbered_pages(10))
    window._editor.add_chapter(1, 5)

    assert window._status_label.text() == "10 pages · 1 chapter(s) · 5 page(s) uncovered (006-010)"
    assert "No chapter range owns them: 006-010" in window._status_label.toolTip()


def test_window_reports_the_pages_missing_from_the_source(qt_app, window):
    _load_pages(window, _numbered_pages(5))
    window._editor.add_chapter(1, 8)

    assert "3 page(s) missing from the source (006-008)" in window._status_label.text()
    assert "absent from the source: 006-008" in window._status_label.toolTip()


def test_window_reports_pages_skipped_by_a_comma_list_as_uncovered(qt_app, window):
    _load_pages(window, _numbered_pages(10))
    window._editor.add_chapter(1, 1)
    window._editor._table.setItem(0, ChapterEditorPanel.COLUMN_PAGES, QTableWidgetItem("1,5-7"))

    assert window._status_label.text() == "10 pages · 1 chapter(s) · 6 page(s) uncovered (002-004, 008-010)"
    assert "No chapter range owns them: 002-004, 008-010" in window._status_label.toolTip()


def test_window_marks_a_skipped_page_as_unassigned_in_the_grid(qt_app, window):
    _load_pages(window, _numbered_pages(8))
    window._editor.add_chapter(1, 1)
    window._editor._table.setItem(0, ChapterEditorPanel.COLUMN_PAGES, QTableWidgetItem("1,5-8"))

    labels = [window._model.data(window._model.index(row, 0), CHAPTER_LABEL_ROLE) for row in range(8)]
    assert labels == ["c001", "unassigned", "unassigned", "unassigned", "c001", "c001", "c001", "c001"]


def test_window_reports_every_page_number_in_the_split_dialog(qt_app, window, tmp_path, monkeypatch):
    asked: list[str] = []

    def _question(*args: Any, **kwargs: Any) -> QMessageBox.StandardButton:
        asked.append(str(args[2]))
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_question))
    source = tmp_path / "v01"
    source.mkdir()
    (source / "p001.png").write_bytes(b"")
    _load_pages(window, _numbered_pages(10))
    window._editor.add_chapter(1, 5)
    window._source_edit.setText(str(source))
    window._output_edit.setText(str(tmp_path))

    window._split()

    assert asked, "the split dialog should have listed the coverage problems"
    assert "5 page(s) belong to no chapter and will be skipped: 006-010" in asked[0]
    assert window._split_worker is None


def test_window_clears_a_stale_source_after_a_failed_load(qt_app, window, monkeypatch):
    # Regression: a failed load used to keep reporting the previous source.
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *args, **kwargs: None))
    _load_pages(window, _numbered_pages(2))
    window._editor.add_chapter(1, 2)
    assert window._split_button.isEnabled()

    window._on_source_failed("not a readable archive")

    assert window._pages == []
    assert window._chapters == []
    assert window._model.pages == []
    assert window._status_label.text() == "Load a source to start."
    assert not window._split_button.isEnabled()
    assert not window._editor._table.isEnabled()


def test_window_splits_a_folder_with_a_spread(qt_app, window, tmp_path, monkeypatch):
    # The chapter ends on page 3, which only exists inside the p002-003 spread.
    # That is a complete cover, so nothing should interrupt the split.
    asked: list[str] = []

    def _question(*args: Any, **kwargs: Any) -> QMessageBox.StandardButton:
        asked.append(str(args[2]))
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_question))
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Close)
    )

    source = tmp_path / "v01"
    source.mkdir()
    for name in ("p001.png", "p002-003.png", "p004.png"):
        (source / name).write_bytes(b"image")
    output = tmp_path / "out"
    output.mkdir()

    window._source_edit.setText(str(source))
    window._output_edit.setText(str(output))
    window._load_source()
    assert window._load_worker is not None
    assert window._load_worker.wait(10000)
    qt_app.processEvents()
    window._thumbnails.cancel()
    assert [page.first_page for page in window._pages] == [1, 2, 4]

    window._editor.add_chapter(1, 3)
    window._editor.add_chapter(4, 4)
    assert window._status_label.text() == "3 pages · 2 chapter(s) · fully covered"

    window._split()
    assert window._split_worker is not None
    assert window._split_worker.wait(10000)
    qt_app.processEvents()

    assert asked == []
    assert "Wrote 2 chapter(s) from 3/3 pages." in window._log_view.toPlainText()
    created = sorted(output.glob("*.cbz"))
    assert len(created) == 2
    with zipfile.ZipFile(created[0]) as archive:
        # The spread image travels with page 3, which the first chapter owns.
        assert sorted(archive.namelist()) == ["p001.png", "p002-003.png"]
    with zipfile.ZipFile(created[1]) as archive:
        assert sorted(archive.namelist()) == ["p004.png"]

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

# Manual split GUI: pick a source, mark the chapter ranges, write the CBZ files.

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QMimeData, QSize, Qt, QUrl
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QDesktopServices,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import file_handler
from ..chapter_split import PAGE_NUMBER_RE, RangeCoverage, SourcePage, SplitReport, validate_source_coverage
from ..common import ChapterRange, format_page_numbers, parse_volume_number, safe_int
from .chapters import ChapterEditorPanel
from .delegate import CELL_SIZE, PageItemDelegate
from .models import PageGridModel
from .thumbnails import ThumbnailLoader
from .worker import SourceLoadWorker, SplitWorker

ARCHIVE_FILTER = "Comic archives (*.cbz *.cbr *.cb7 *.zip *.rar *.7z *.tar);;All files (*)"
"""File dialog filter for the supported archive sources."""

INLINE_PAGE_LIMIT = 6
"""How many page entries the one-line status bar spells out before summarizing."""


def _inline_pages(numbers: Sequence[int]) -> str:
    """Format page numbers for the one-line status bar."""
    return format_page_numbers(numbers, limit=INLINE_PAGE_LIMIT)


class MainWindow(QMainWindow):
    """
    Whole manual split workflow in one window.

    Pick a folder or archive, let it load a thumbnail grid of the pages, mark
    the chapter ranges (by dragging a selection and adding it as a chapter, or by
    typing ranges), then write one CBZ per chapter. Chapter ranges carry an
    optional volume so an omnibus can spread its chapters over ``v01-02``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("nmanga manual split")

        self._pages: list[SourcePage] = []
        self._chapters: list[ChapterRange] = []
        self._busy = False
        self._load_worker: SourceLoadWorker | None = None
        self._split_worker: SplitWorker | None = None

        self._model = PageGridModel(self)
        self._thumbnails = ThumbnailLoader(self)
        self._editor = ChapterEditorPanel(self)

        self.setCentralWidget(self._build_central())
        self._status_label = QLabel("Load a source to start.")
        self.statusBar().addPermanentWidget(self._status_label, 1)
        self._configure_drag_and_drop()
        self._wire_signals()
        self._set_busy(False)
        self.resize(1440, 920)

    # -- construction ---------------------------------------------------------
    def _build_central(self) -> QWidget:
        """Assemble the source form, the page grid, the chapter editor, and the log."""
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.addWidget(self._build_source_group())

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.addWidget(self._build_grid_panel())
        splitter.addWidget(self._editor)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        self._progress = QProgressBar(central)
        self._progress.setVisible(False)
        self._split_button = QPushButton("Split into chapters", central)
        self._split_button.clicked.connect(self._split)
        run_row = QHBoxLayout()
        run_row.addWidget(self._progress, 1)
        run_row.addWidget(self._split_button)
        layout.addLayout(run_row)

        self._log_view = QPlainTextEdit(central)
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(2000)
        self._log_view.setMaximumHeight(150)
        layout.addWidget(self._log_view)
        return central

    def _build_source_group(self) -> QWidget:
        """Build the source form, with the advanced page-numbering options."""
        group = QGroupBox("Source", self)
        form = QFormLayout(group)

        self._source_edit = QLineEdit(group)
        self._source_edit.setReadOnly(True)
        self._source_edit.setPlaceholderText("A folder of images, or a CBZ/CBR/CB7 archive")
        self._source_edit.setToolTip(
            "Choose a folder or archive, or drag one from your file manager and drop it anywhere on this window."
        )
        self._source_button = QToolButton(group)
        self._source_button.setText("Choose…")
        self._source_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self._source_button)
        folder_action = QAction("Folder…", menu)
        folder_action.triggered.connect(self._browse_source_folder)
        archive_action = QAction("Archive…", menu)
        archive_action.triggered.connect(self._browse_source_archive)
        menu.addAction(folder_action)
        menu.addAction(archive_action)
        self._source_button.setMenu(menu)
        source_row = QHBoxLayout()
        source_row.addWidget(self._source_edit, 1)
        source_row.addWidget(self._source_button)
        form.addRow("Folder or archive", source_row)

        self._output_edit = QLineEdit(group)
        self._output_edit.setPlaceholderText("Defaults to the folder next to the source")
        self._output_button = QToolButton(group)
        self._output_button.setText("Browse…")
        self._output_button.clicked.connect(self._browse_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_edit, 1)
        output_row.addWidget(self._output_button)
        form.addRow("Output folder", output_row)

        self._volume_edit = QLineEdit(group)
        self._volume_edit.setPlaceholderText("e.g. 1, or 1-2 for an omnibus range")
        self._volume_edit.editingFinished.connect(self._validate_volume_field)
        form.addRow("Default volume", self._volume_edit)

        self._overwrite_check = QCheckBox("Overwrite existing chapter archives", group)
        form.addRow("", self._overwrite_check)

        self._load_button = QPushButton("Load source", group)
        self._load_button.clicked.connect(self._load_source)
        form.addRow("", self._load_button)

        advanced = QGroupBox("Page numbering (advanced)", self)
        advanced_form = QFormLayout(advanced)
        self._regex_edit = QLineEdit(advanced)
        self._regex_edit.setPlaceholderText(PAGE_NUMBER_RE.pattern)
        self._regex_edit.setToolTip(
            "Optional. Needs two capture groups: the first page and the optional second page of a spread.\n"
            "Leave empty to auto-detect the standard p003/p003-004 naming and bare page numbers."
        )
        advanced_form.addRow("Page regex", self._regex_edit)

        self._custom_edit = QPlainTextEdit(advanced)
        self._custom_edit.setPlaceholderText("cover=1\ncredits=2")
        self._custom_edit.setMaximumHeight(70)
        self._custom_edit.setToolTip(
            "Optional, one `filename fragment=page number` per line.\n"
            "Useful for pages that carry no number, such as a cover or a credits page."
        )
        advanced_form.addRow("Custom page mapping", self._custom_edit)

        wrapper = QWidget(self)
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(group)
        wrapper_layout.addWidget(advanced)
        return wrapper

    def _build_grid_panel(self) -> QWidget:
        """Build the page grid with its selection-driven chapter buttons."""
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        self._grid = QListView(panel)
        self._grid.setModel(self._model)
        self._grid.setItemDelegate(PageItemDelegate(self._grid))
        self._grid.setViewMode(QListView.ViewMode.IconMode)
        self._grid.setResizeMode(QListView.ResizeMode.Adjust)
        self._grid.setMovement(QListView.Movement.Static)
        self._grid.setUniformItemSizes(True)
        self._grid.setWrapping(True)
        self._grid.setGridSize(QSize(CELL_SIZE.width() + 8, CELL_SIZE.height() + 8))
        self._grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._grid.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)

        self._add_button = QPushButton("Add chapter from selection", panel)
        self._add_button.setToolTip("Append a chapter range covering the selected pages")
        self._add_button.clicked.connect(self._add_chapter_from_selection)
        self._range_button = QPushButton("Set range of selected chapter", panel)
        self._range_button.setToolTip("Point the chapter row selected on the right at the selected pages")
        self._range_button.clicked.connect(self._set_range_from_selection)
        self._selection_label = QLabel("No pages selected", panel)

        header = QHBoxLayout()
        header.addWidget(self._add_button)
        header.addWidget(self._range_button)
        header.addWidget(self._selection_label, 1)
        layout.addLayout(header)
        layout.addWidget(self._grid, 1)
        return panel

    def _wire_signals(self) -> None:
        """Connect the model, loader, editor, and grid together."""
        self._thumbnails.loaded.connect(self._model.set_thumbnail)
        self._editor.chapters_changed.connect(self._on_chapters_changed)
        selection = self._grid.selectionModel()
        if selection is not None:
            selection.selectionChanged.connect(self._update_selection_label)

    # -- drag and drop --------------------------------------------------------
    def _configure_drag_and_drop(self) -> None:
        """
        Accept a folder or archive dropped anywhere on the window.

        Every descendant is told to ignore drops, so that a drop bubbles up to
        the window instead of being swallowed as text by a line edit or the log.
        """
        self.setAcceptDrops(True)
        for widget in self.findChildren(QWidget):
            widget.setAcceptDrops(False)

    def _dropped_source(self, mime_data: QMimeData | None) -> Path | None:
        """
        The single folder or archive carried by a drop, when it is usable.

        Parameters
        ----------
        mime_data: :class:`~PySide6.QtCore.QMimeData` or ``None``
            Payload of the drag/drop event.

        Returns
        -------
        :class:`pathlib.Path` or ``None``
            The source to load, or ``None`` when the drop is not exactly one
            folder or one supported archive.
        """
        if mime_data is None or not mime_data.hasUrls():
            return None
        urls = mime_data.urls()
        if len(urls) != 1 or not urls[0].isLocalFile():
            return None
        path = Path(urls[0].toLocalFile())
        if path.is_dir() or (path.is_file() and file_handler.is_archive(path)):
            return path
        return None

    def _set_drop_hint(self, path: Path | None) -> None:
        """Preview the dragged source in the status bar, or clear the hint."""
        if path is None:
            self._source_edit.setStyleSheet("")
            self._refresh_coverage()
            return
        kind = "folder" if path.is_dir() else "archive"
        self._status_label.setText(f"Drop to load the {kind} {path.name}")
        self._status_label.setStyleSheet("color: #0b5cad; font-weight: 600;")
        self._source_edit.setStyleSheet("border: 2px solid #0b5cad;")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # ruff: ignore[invalid-function-name]
        """Highlight the window when a usable folder or archive is dragged over it."""
        source = self._dropped_source(event.mimeData())
        if self._busy or source is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self._set_drop_hint(source)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # ruff: ignore[invalid-function-name]
        """Keep accepting the drag while it travels across the window."""
        if self._busy or self._dropped_source(event.mimeData()) is None:
            event.ignore()
            return
        event.acceptProposedAction()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:  # ruff: ignore[invalid-function-name]
        """Clear the highlight once the drag leaves the window."""
        self._set_drop_hint(None)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # ruff: ignore[invalid-function-name]
        """Load the dropped folder or archive as the source."""
        self._set_drop_hint(None)
        source = self._dropped_source(event.mimeData())
        if self._busy or source is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self._set_source(source)
        self._load_source()

    # -- source loading -------------------------------------------------------
    def _browse_source_folder(self) -> None:
        """Ask for a folder source."""
        start = self._source_edit.text() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Select source folder", start)
        if chosen:
            self._set_source(Path(chosen))

    def _browse_source_archive(self) -> None:
        """Ask for an archive source."""
        start = self._source_edit.text() or str(Path.home())
        chosen, _ = QFileDialog.getOpenFileName(self, "Select source archive", start, ARCHIVE_FILTER)
        if chosen:
            self._set_source(Path(chosen))

    def _set_source(self, source: Path) -> None:
        """Record a new source path and guess the matching output folder."""
        self._source_edit.setText(str(source))
        if not self._output_edit.text().strip():
            self._output_edit.setText(str(source.parent))
        self._log(f"Source set to {source}")

    def _browse_output(self) -> None:
        """Ask for the folder the chapter archives are written to."""
        start = self._output_edit.text() or self._source_edit.text() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Select output folder", start)
        if chosen:
            self._output_edit.setText(chosen)

    def _load_source(self) -> None:
        """Read the page list of the configured source in the background."""
        source = self._current_source()
        if source is None:
            return
        try:
            regex = self._compile_regex()
            custom_data = self._parse_custom_data()
        except ValueError as exc:
            self._warn(str(exc))
            return

        if not self._output_edit.text().strip():
            self._output_edit.setText(str(source.parent))

        self._set_busy(True, f"Reading {source.name}…")
        self._log(f"Reading pages from {source}…")
        self._load_worker = SourceLoadWorker(source, self, regex=regex, custom_data=custom_data)
        self._load_worker.succeeded.connect(self._on_source_loaded)
        self._load_worker.failed.connect(self._on_source_failed)
        self._load_worker.start()

    def _on_source_loaded(self, pages: list[SourcePage]) -> None:
        """Fill the grid with a freshly read source."""
        self._pages = list(pages)
        self._model.set_pages(self._pages)
        self._editor.set_pages(self._pages)
        source = self._current_source()
        if source is not None:
            width = self._thumbnails.start(source, self._pages)
            self._log(f"Decoding {len(self._pages)} thumbnail(s) at {width}px…")
        self._editor.clear_chapters()
        self._set_busy(False)
        self._editor.set_editable(bool(self._pages))
        if not self._pages:
            self._warn("No images were found in the selected source.")
            self._log("No images found.")
        else:
            self._log(f"Loaded {len(self._pages)} page(s). Select pages and add them as a chapter.")
        self._refresh_coverage()

    def _on_source_failed(self, message: str) -> None:
        """Report a source that could not be read, and drop the stale page list."""
        # The configured source could not be read, so the previously loaded pages
        # no longer describe what the window would split.
        self._clear_source_state()
        self._set_busy(False)
        self._log(f"Failed to read the source: {message}")
        self._warn(message)
        self._refresh_coverage()

    def _clear_source_state(self) -> None:
        """Forget the loaded pages and chapter ranges."""
        self._pages = []
        self._model.set_pages([])
        self._editor.set_pages([])
        self._editor.clear_chapters()

    # -- chapter editing ------------------------------------------------------
    def _selected_rows(self) -> list[int]:
        """Rows of the pages currently selected in the grid, in reading order."""
        selection = self._grid.selectionModel()
        if selection is None:
            return []
        return sorted(index.row() for index in selection.selectedIndexes())

    def _selected_page_range(self) -> tuple[int, int] | None:
        """First and last page number covered by the grid selection, if any."""
        rows = self._selected_rows()
        if not rows:
            return None
        first = self._model.page_at(rows[0])
        last = self._model.page_at(rows[-1])
        if first is None or last is None:
            return None
        return first.first_page, last.page_numbers[-1]

    def _add_chapter_from_selection(self) -> None:
        """Append a chapter range covering the selected pages."""
        page_range = self._selected_page_range()
        if page_range is None:
            self._warn("Select the pages of a chapter in the grid first.")
            return
        start_page, end_page = page_range
        row = self._editor.add_chapter(start_page, end_page)
        self._log(f"Added chapter row {row + 1} for pages {start_page:03d}-{end_page:03d}.")

    def _set_range_from_selection(self) -> None:
        """Point the selected chapter row at the selected pages."""
        page_range = self._selected_page_range()
        if page_range is None:
            self._warn("Select the pages of a chapter in the grid first.")
            return
        row = self._editor.current_row()
        if row < 0:
            self._warn("Select the chapter row to update in the chapter table first.")
            return
        start_page, end_page = page_range
        self._editor.apply_range(row, start_page, end_page)
        self._log(f"Chapter row {row + 1} now covers pages {start_page:03d}-{end_page:03d}.")

    def _update_selection_label(self) -> None:
        """Show how many pages and which range the selection covers."""
        rows = self._selected_rows()
        page_range = self._selected_page_range()
        if page_range is None:
            self._selection_label.setText("No pages selected")
            return
        self._selection_label.setText(f"{len(rows)} page(s) selected · pages {page_range[0]:03d}-{page_range[1]:03d}")

    def _on_chapters_changed(self) -> None:
        """Follow the editor: recolor the grid and re-check the coverage."""
        self._chapters = self._editor.chapters()
        self._model.set_chapters(self._chapters)
        self._refresh_coverage()

    def _coverage(self) -> RangeCoverage:
        """Validate the chapter ranges against the images of the loaded source."""
        return validate_source_coverage(self._chapters, self._pages)

    def _refresh_coverage(self) -> None:
        """Describe the current chapter coverage in the status bar."""
        if not self._pages:
            self._status_label.setText("Load a source to start.")
            self._status_label.setStyleSheet("")
            self._status_label.setToolTip("")
            self._split_button.setEnabled(False)
            return

        summary = f"{len(self._pages)} pages · {len(self._chapters)} chapter(s)"
        problems: list[str] = []
        details: list[str] = []
        if self._chapters:
            coverage = self._coverage()
            if coverage.unassigned_pages:
                count = len(coverage.unassigned_pages)
                problems.append(f"{count} page(s) uncovered ({_inline_pages(coverage.unassigned_pages)})")
                details.append(f"No chapter range owns them: {format_page_numbers(coverage.unassigned_pages)}")
            if coverage.overlapping_pages:
                count = len(coverage.overlapping_pages)
                problems.append(f"{count} page(s) in two chapters ({_inline_pages(coverage.overlapping_pages)})")
                details.append(f"Owned by two chapter ranges: {format_page_numbers(coverage.overlapping_pages)}")
            if coverage.out_of_bounds_pages:
                count = len(coverage.out_of_bounds_pages)
                problems.append(
                    f"{count} page(s) missing from the source ({_inline_pages(coverage.out_of_bounds_pages)})"
                )
                details.append(
                    "Referenced by a chapter range but absent from the source: "
                    f"{format_page_numbers(coverage.out_of_bounds_pages)}"
                )
        if problems:
            self._status_label.setText(f"{summary} · " + ", ".join(problems))
            self._status_label.setStyleSheet("color: #b3261e;")
            self._status_label.setToolTip("\n".join(details))
        else:
            suffix = "fully covered" if self._chapters else "add a chapter to split"
            self._status_label.setText(f"{summary} · {suffix}")
            self._status_label.setStyleSheet("color: #1b5e20;" if self._chapters else "")
            self._status_label.setToolTip("")
        self._split_button.setEnabled(bool(self._chapters) and not self._busy)

    # -- splitting ------------------------------------------------------------
    def _split(self) -> None:
        """Write one chapter archive per chapter range."""
        source = self._current_source()
        if source is None or not self._chapters:
            self._warn("Load a source and add at least one chapter range first.")
            return

        output_text = self._output_edit.text().strip()
        if not output_text:
            self._warn("Choose an output folder first.")
            return
        volume_text = self._volume_edit.text().strip()
        volume = parse_volume_number(volume_text) if volume_text else None
        if volume_text and volume is None:
            self._warn("Default volume must be a number such as `1`, `1.5`, or an omnibus range such as `1-2`.")
            return
        try:
            custom_data = self._parse_custom_data()
            regex = self._compile_regex()
        except ValueError as exc:
            self._warn(str(exc))
            return

        coverage = self._coverage()
        if not coverage.ok:
            details = []
            if coverage.unassigned_pages:
                details.append(
                    f"{len(coverage.unassigned_pages)} page(s) belong to no chapter and will be skipped: "
                    f"{format_page_numbers(coverage.unassigned_pages)}"
                )
            if coverage.overlapping_pages:
                details.append(
                    f"{len(coverage.overlapping_pages)} page(s) belong to two chapters, the first one wins: "
                    f"{format_page_numbers(coverage.overlapping_pages)}"
                )
            if coverage.out_of_bounds_pages:
                details.append(
                    f"{len(coverage.out_of_bounds_pages)} referenced page(s) do not exist in the source: "
                    f"{format_page_numbers(coverage.out_of_bounds_pages)}"
                )
            answer = QMessageBox.question(
                self,
                "Chapter ranges have problems",
                "\n".join(f"• {detail}" for detail in details) + "\n\nSplit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        output_dir = Path(output_text)
        self._set_busy(True, f"Writing chapters to {output_dir}…")
        self._log(f"Splitting {source.name} into {len(self._chapters)} chapter(s) → {output_dir}")
        self._split_worker = SplitWorker(
            source,
            output_dir,
            self._chapters,
            self,
            volume=volume,
            regex=regex,
            custom_data=custom_data,
            overwrite=self._overwrite_check.isChecked(),
        )
        self._split_worker.signals.chapter_started.connect(lambda name: self._log(f"+ {name}"))
        self._split_worker.signals.chapter_skipped.connect(lambda name: self._log(f"= {name} already exists, skipped"))
        self._split_worker.signals.page_unassigned.connect(
            lambda page, filename: self._log(f"! page {page:03d} ({filename}) is in no chapter range")
        )
        self._split_worker.succeeded.connect(self._on_split_finished)
        self._split_worker.failed.connect(self._on_split_failed)
        self._split_worker.start()

    def _on_split_finished(self, report: SplitReport) -> None:
        """Report what the split wrote and offer to open the output folder."""
        self._set_busy(False)
        lines = [f"Wrote {report.created_chapters} chapter(s) from {report.pages_assigned}/{report.pages_total} pages."]
        if report.skipped:
            lines.append(f"Skipped {len(report.skipped)} existing chapter(s).")
        if report.has_unassigned:
            lines.append(f"{len(report.unassigned_pages)} page(s) were in no chapter range and were skipped.")
        for line in lines:
            self._log(line)
        self._refresh_coverage()

        answer = QMessageBox.information(
            self,
            "Split finished",
            "\n".join(lines) + "\n\nOpen the output folder?",
            QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Close,
            QMessageBox.StandardButton.Close,
        )
        if answer == QMessageBox.StandardButton.Open:
            self._open_output_folder()

    def _on_split_failed(self, message: str) -> None:
        """Report a split that could not run."""
        self._set_busy(False)
        self._log(f"Split failed: {message}")
        self._warn(message)
        self._refresh_coverage()

    def _open_output_folder(self) -> None:
        """Open the output folder in the desktop file browser."""
        output_text = self._output_edit.text().strip()
        if output_text:
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_text))

    # -- helpers --------------------------------------------------------------
    def _current_source(self) -> Path | None:
        """The configured source, or ``None`` with a warning when it is unusable."""
        text = self._source_edit.text().strip()
        if not text:
            self._warn("Choose a source folder or archive first.")
            return None
        source = Path(text)
        if not source.exists():
            self._warn(f"{source} does not exist.")
            return None
        return source

    def _compile_regex(self) -> re.Pattern[str] | None:
        """
        Compile the optional page number regex.

        Returns
        -------
        :class:`re.Pattern` or ``None``
            The compiled regex, or ``None`` to let the core auto-detect the naming.

        Raises
        ------
        ValueError
            If the regex is invalid or lacks the two expected capture groups.
        """
        text = self._regex_edit.text().strip()
        if not text:
            return None
        try:
            pattern = re.compile(text)
        except re.error as exc:
            raise ValueError(f"Invalid page regex: {exc}") from exc
        if pattern.groups < 2:
            raise ValueError(
                "The page regex needs two capture groups: the first page, and the second page of a spread."
            )
        return pattern

    def _parse_custom_data(self) -> dict[str, int]:
        """
        Parse the custom page mapping field.

        Returns
        -------
        :class:`dict`
            Filename fragment to fixed page number.

        Raises
        ------
        ValueError
            If a line is not in ``fragment=page`` form.
        """
        custom_data: dict[str, int] = {}
        for line in self._custom_edit.toPlainText().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fragment, _, page = line.partition("=")
            fragment = fragment.strip()
            page_number = safe_int(page.strip())
            if not fragment or page_number is None:
                raise ValueError(f"Custom page mapping {line!r} must look like `cover=1`.")
            custom_data[fragment] = page_number
        return custom_data

    def _validate_volume_field(self) -> None:
        """Flag the default volume field when it cannot be parsed."""
        text = self._volume_edit.text().strip()
        invalid = bool(text) and parse_volume_number(text) is None
        self._volume_edit.setStyleSheet("background: #f5c6c6;" if invalid else "")
        self._volume_edit.setToolTip("Volume must be `1`, `1.5`, or `1-2` for an omnibus." if invalid else "")

    def _set_busy(self, busy: bool, message: str = "") -> None:
        """Disable the editing surface while a background worker runs."""
        self._busy = busy
        self._progress.setVisible(busy)
        if busy:
            self._progress.setRange(0, 0)
            self._status_label.setText(message)
        else:
            self._progress.setRange(0, 1)
            self._progress.setValue(0)
        for widget in (
            self._load_button,
            self._source_button,
            self._output_button,
            self._output_edit,
            self._grid,
            self._add_button,
            self._range_button,
        ):
            widget.setEnabled(not busy)
        self._editor.set_editable(not busy and bool(self._pages))
        self._split_button.setEnabled(not busy and bool(self._chapters))

    def _log(self, message: str) -> None:
        """Append a line to the progress log."""
        self._log_view.appendPlainText(message)

    def _warn(self, message: str) -> None:
        """Show a non-fatal problem to the user."""
        QMessageBox.warning(self, "Manual split", message)

    def closeEvent(self, event: QCloseEvent) -> None:  # ruff: ignore[invalid-function-name] (Qt override)
        """Refuse to quit mid-split, and stop thumbnail decoding on the way out."""
        if self._split_worker is not None and self._split_worker.isRunning():
            QMessageBox.information(
                self,
                "Split in progress",
                "A split is still running. Wait for it to finish before closing the window.",
            )
            event.ignore()
            return
        self._thumbnails.cancel()
        if self._load_worker is not None and self._load_worker.isRunning():
            self._load_worker.wait(3000)
        super().closeEvent(event)

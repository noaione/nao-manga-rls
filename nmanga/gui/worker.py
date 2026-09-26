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

# Background workers for the manual split GUI.
#
# Both workers only wrap the UI-agnostic core (`collect_source_pages` and
# `split_chapters`) in a thread and translate the results into Qt signals, so all
# of the actual chapter logic stays testable without a display.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from re import Pattern

from PySide6.QtCore import QObject, QThread, Signal

from .._ntypes import VolumeNumberT
from ..chapter_split import (
    ChapterSplitObserver,
    SplitReport,
    collect_source_pages,
    split_chapters,
)
from ..common import ChapterRange


class SourceLoadWorker(QThread):
    """Read the page list of a source folder/archive without blocking the GUI."""

    succeeded = Signal(list)
    """Emitted with the loaded ``list[SourcePage]``."""

    failed = Signal(str)
    """Emitted with a human-readable reason when the source cannot be read."""

    def __init__(
        self,
        source: Path,
        parent: QObject | None = None,
        *,
        regex: Pattern[str] | None = None,
        custom_data: dict[str, int] | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._regex = regex
        self._custom_data = custom_data

    def run(self) -> None:
        """Load the page list and report it back to the GUI thread."""
        try:
            pages = collect_source_pages(self._source, custom_data=self._custom_data, regex_data=self._regex)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(pages)


class SplitSignals(QObject):
    """Progress signals of a running :class:`SplitWorker`."""

    chapter_started = Signal(str)
    chapter_finished = Signal(str, str)
    chapter_skipped = Signal(str)
    page_unassigned = Signal(int, str)


class QtSplitObserver(ChapterSplitObserver):
    """
    Bridge the core observer protocol onto Qt signals.

    A plain subclass rather than a ``QObject`` mixin: keeping the signals on
    :class:`SplitSignals` avoids Qt's multiple-inheritance initialization rules.
    """

    def __init__(self, signals: SplitSignals) -> None:
        self._signals = signals

    def on_chapter_start(self, chapter: str) -> None:
        """Report that a chapter archive is being written."""
        self._signals.chapter_started.emit(chapter)

    def on_chapter_finish(self, chapter: str, target: Path) -> None:
        """Report that a chapter archive was written."""
        self._signals.chapter_finished.emit(chapter, str(target))

    def on_chapter_skip(self, chapter: str) -> None:
        """Report that an existing chapter archive was left alone."""
        self._signals.chapter_skipped.emit(chapter)

    def on_page_unassigned(self, page: int, filename: str) -> None:
        """Report a page that no chapter range covers."""
        self._signals.page_unassigned.emit(page, filename)


class SplitWorker(QThread):
    """Run :func:`nmanga.chapter_split.split_chapters` off the GUI thread."""

    succeeded = Signal(object)
    """Emitted with the resulting :class:`nmanga.chapter_split.SplitReport`."""

    failed = Signal(str)
    """Emitted with a human-readable reason when the split cannot run."""

    def __init__(
        self,
        source: Path,
        output_dir: Path,
        chapters: Sequence[ChapterRange],
        parent: QObject | None = None,
        *,
        volume: VolumeNumberT | None = None,
        regex: Pattern[str] | None = None,
        custom_data: dict[str, int] | None = None,
        overwrite: bool = False,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._output_dir = output_dir
        self._chapters = list(chapters)
        self._volume = volume
        self._regex = regex
        self._custom_data = custom_data
        self._overwrite = overwrite
        self.signals = SplitSignals(self)

    def run(self) -> None:
        """Split the source, translating the observer callbacks into signals."""
        try:
            report: SplitReport = split_chapters(
                self._source,
                self._output_dir,
                self._chapters,
                volume=self._volume,
                custom_data=self._custom_data,
                regex_data=self._regex,
                overwrite=self._overwrite,
                observer=QtSplitObserver(self.signals),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(report)


__all__ = (
    "QtSplitObserver",
    "SourceLoadWorker",
    "SplitSignals",
    "SplitWorker",
)

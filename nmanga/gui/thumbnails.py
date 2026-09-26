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

# Background thumbnail decoding for the page grid.
#
# Decoding happens on a QThreadPool so a large or broken image never blocks the
# event loop; results are delivered back to the GUI thread through a signal.
#
# Note: each image is read with `read_source_image`, which opens the source once
# per image. That is cheap for a folder source (the primary use case) but does
# repeat the archive lookup for CBZ/CBR/CB7 sources.

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThread, QThreadPool, Signal
from PySide6.QtGui import QImage

from ..chapter_split import ChapterSplitError, SourcePage, read_source_image

MAX_THUMBNAIL_WIDTH = 180
"""Preferred width in pixels of the thumbnails decoded for the page grid."""

MIN_THUMBNAIL_WIDTH = 72
"""Floor for the auto-picked width, below which a page is no longer readable."""

THUMBNAIL_MEMORY_BUDGET = 256 * 1024 * 1024
"""Rough ceiling on decoded thumbnails, so a huge omnibus cannot exhaust memory."""

_ASPECT_RATIO = 1.45
"""Assumed height/width ratio of a manga page, used for the memory estimate."""


def pick_thumbnail_width(page_count: int) -> int:
    """
    Pick a thumbnail width that keeps the decoded pages within the memory budget.

    Thumbnails are held in memory for every page of the source, so a 5000-page
    omnibus at the preferred width would cost several hundred megabytes. The
    width shrinks for very large sources instead of the app becoming unusable.

    Parameters
    ----------
    page_count: :class:`int`
        Number of pages that will be decoded.

    Returns
    -------
    :class:`int`
        The width to decode at, between :data:`MIN_THUMBNAIL_WIDTH` and
        :data:`MAX_THUMBNAIL_WIDTH`.
    """
    if page_count < 1:
        return MAX_THUMBNAIL_WIDTH
    per_page_bytes = _ASPECT_RATIO * 4  # RGBA, ignoring the width factor
    affordable = math.sqrt(THUMBNAIL_MEMORY_BUDGET / (page_count * per_page_bytes))
    return max(MIN_THUMBNAIL_WIDTH, min(MAX_THUMBNAIL_WIDTH, int(affordable)))


class ThumbnailTask(QRunnable):
    """
    Decode a single source image into a scaled :class:`~PySide6.QtGui.QImage`.

    Failures are delivered as a null image, which the delegate paints as a
    placeholder rather than leaving the cell blank forever.
    """

    def __init__(self, source: Path, row: int, filename: str, width: int, sink: ThumbnailLoader, token: int) -> None:
        super().__init__()
        self._source = source
        self._row = row
        self._filename = filename
        self._width = width
        self._sink = sink
        self._token = token
        self.setAutoDelete(True)

    def run(self) -> None:
        """Read, scale, and hand the image back to the loader."""
        image = QImage()
        try:
            image = QImage.fromData(read_source_image(self._source, self._filename))
        except (ChapterSplitError, OSError):
            image = QImage()
        if not image.isNull() and image.width() > self._width:
            image = image.scaledToWidth(self._width, Qt.TransformationMode.SmoothTransformation)
        self._sink.deliver(self._token, self._row, image)


class ThumbnailLoader(QObject):
    """
    Load page thumbnails on a worker pool and emit them one at a time.

    Every batch carries a token: :meth:`start` bumps the token so results from a
    previous source are dropped instead of leaking into a freshly loaded grid.
    """

    loaded = Signal(int, QImage)
    """Emitted with ``(row, image)`` for each decoded page; null images failed to decode."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._token = 0
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(2, min(6, QThread.idealThreadCount() - 1)))

    def start(self, source: Path, pages: Iterable[SourcePage], width: int | None = None) -> int:
        """
        Queue every page of ``source`` for thumbnail decoding.

        Parameters
        ----------
        source: :class:`pathlib.Path`
            The folder or archive the pages are read from.
        pages: :class:`~collections.abc.Iterable` of :class:`nmanga.chapter_split.SourcePage`
            The pages to decode.
        width: :class:`int`, optional
            Decode width; defaults to a width picked from the page count by
            :func:`pick_thumbnail_width`.

        Returns
        -------
        :class:`int`
            The width the thumbnails will be decoded at.
        """
        self.cancel()
        token = self._token
        page_list = list(pages)
        if width is None:
            width = pick_thumbnail_width(len(page_list))
        for page in page_list:
            self._pool.start(ThumbnailTask(source, page.index, page.filename, width, self, token))
        return width

    def cancel(self) -> None:
        """Drop queued work and ignore the results of the batch in flight."""
        self._token += 1
        self._pool.clear()

    def deliver(self, token: int, row: int, image: QImage) -> None:
        """
        Accept a decoded image from a worker thread.

        Called by :class:`ThumbnailTask`; safe to call from any thread since the
        emitted signal is delivered to the GUI thread by the event loop.
        """
        if token == self._token:
            self.loaded.emit(row, image)

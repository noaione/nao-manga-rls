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

# Painting of a single page card in the manual split page grid.

from __future__ import annotations

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, QPoint, QRect, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem, QWidget

from .models import CHAPTER_INDEX_ROLE, CHAPTER_LABEL_ROLE, PAGE_LABEL_ROLE, THUMBNAIL_ROLE

CHAPTER_COLORS: tuple[QColor, ...] = (
    QColor("#4c8bf5"),
    QColor("#f59f4c"),
    QColor("#4cc38a"),
    QColor("#c44cf5"),
    QColor("#f54c6b"),
    QColor("#4cf5e6"),
    QColor("#f5e04c"),
    QColor("#8b7cf5"),
)
"""Palette cycled through for chapter ranges, so a chapter reads the same everywhere."""

UNASSIGNED_COLOR = QColor("#9a9a9a")
"""Color used for pages that no chapter range owns."""

CELL_SIZE = QSize(188, 250)
"""Preferred size of one page cell."""

FOOTER_HEIGHT = 26
"""Height of the chapter label strip at the bottom of a cell."""


def chapter_color(position: int) -> QColor:
    """
    Return the color representing a chapter position.

    Parameters
    ----------
    position: :class:`int`
        Index of the chapter in the chapter list, or ``-1`` when unassigned.

    Returns
    -------
    :class:`~PySide6.QtGui.QColor`
        A stable color for that chapter, grey when unassigned.
    """
    if position < 0:
        return UNASSIGNED_COLOR
    return CHAPTER_COLORS[position % len(CHAPTER_COLORS)]


class PageItemDelegate(QStyledItemDelegate):
    """
    Draw a page as a thumbnail card.

    Each card shows the parsed page number as a badge, the chapter that owns the
    page as a colored footer, and the chapter color as a tint so ranges can be
    eyeballed at a glance. Unassigned pages are grey and marked.
    """

    def __init__(self, parent: QWidget | None = None, cell_size: QSize = CELL_SIZE) -> None:
        super().__init__(parent)
        self._cell_size = cell_size

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex) -> QSize:  # ruff: ignore[invalid-function-name] (Qt override)
        """A fixed cell size keeps the grid uniform regardless of image aspect."""
        return self._cell_size

    def paint(
        self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex
    ) -> None:
        """Paint the tint, thumbnail, page badge, and chapter footer of one page."""
        painter.save()
        rect = option.rect.adjusted(2, 2, -2, -2)

        raw_position = index.data(CHAPTER_INDEX_ROLE)
        position = raw_position if isinstance(raw_position, int) else -1
        color = chapter_color(position)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)

        tint = QColor(color)
        tint.setAlpha(46 if position >= 0 else 18)
        painter.fillRect(rect, tint)
        painter.setPen(QPen(color, 2 if selected else 1))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4.0, 4.0)

        content = rect.adjusted(6, 6, -6, -FOOTER_HEIGHT - 2)
        self._paint_thumbnail(painter, content, index)
        self._paint_page_badge(painter, content, index)

        raw_label = index.data(CHAPTER_LABEL_ROLE)
        footer = QRect(rect.left() + 4, rect.bottom() - FOOTER_HEIGHT + 2, rect.width() - 8, FOOTER_HEIGHT - 6)
        font = painter.font()
        font.setBold(position >= 0)
        painter.setFont(font)
        painter.setPen(color)
        painter.drawText(footer, Qt.AlignmentFlag.AlignCenter, str(raw_label or "unassigned"))
        painter.restore()

    def _paint_thumbnail(self, painter: QPainter, content: QRect, index: QModelIndex | QPersistentModelIndex) -> None:
        """Draw the decoded image centered in ``content``, or a placeholder."""
        raw_pixmap = index.data(THUMBNAIL_ROLE)
        if raw_pixmap is None:
            self._paint_placeholder(painter, content, "…", QColor("#b0b0b0"))
            return
        if not isinstance(raw_pixmap, QPixmap) or raw_pixmap.isNull():
            self._paint_placeholder(painter, content, "?", UNASSIGNED_COLOR)
            return

        scaled = raw_pixmap.scaled(
            content.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        painter.drawPixmap(
            content.left() + (content.width() - scaled.width()) // 2,
            content.top() + (content.height() - scaled.height()) // 2,
            scaled,
        )

    def _paint_placeholder(self, painter: QPainter, content: QRect, text: str, color: QColor) -> None:
        """Draw a marker for a page whose thumbnail is missing or unreadable."""
        placeholder = QRect(content.left(), content.top(), content.width(), content.height())
        painter.fillRect(placeholder, QColor(0, 0, 0, 24))
        painter.setPen(color)
        painter.drawText(placeholder, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_page_badge(self, painter: QPainter, content: QRect, index: QModelIndex | QPersistentModelIndex) -> None:
        """Draw the parsed page number as a badge over the top-left of the thumbnail."""
        label = str(index.data(PAGE_LABEL_ROLE) or "")
        if not label:
            return
        font = painter.font()
        font.setBold(True)
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.0))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        badge = metrics.boundingRect(label).adjusted(-6, -3, 6, 3)
        badge.moveTopLeft(content.topLeft() + QPoint(4, 4))
        painter.fillRect(badge, QColor(0, 0, 0, 165))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, label)

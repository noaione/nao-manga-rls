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

# Entry point of the optional manual split GUI.

from __future__ import annotations

import sys

__all__ = ("main",)


def _report_missing_pyside() -> int:
    """Explain the missing optional dependency through the regular console."""
    from .. import term

    console = term.get_console()
    console.error("The nmanga GUI needs PySide6, which is not installed.")
    console.info("Install it with `uv sync --extra gui` or `pip install 'nmanga[gui]'`.")
    return 1


def main() -> int:
    """
    Launch the manual split GUI.

    PySide6 is optional, so it is imported here rather than at module level: the
    CLI keeps working without it, and importing :mod:`nmanga.gui` never fails.

    Returns
    -------
    :class:`int`
        The Qt event loop exit code, or ``1`` when PySide6 is not installed.
    """
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return _report_missing_pyside()

    from .window import MainWindow

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("nmanga")
    app.setApplicationDisplayName("nmanga manual split")  # pyright: ignore[reportAttributeAccessIssue]
    app.setOrganizationName("noaione")

    window = MainWindow()
    window.show()
    return app.exec()

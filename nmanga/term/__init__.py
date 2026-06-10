"""
nmanga.term
~~~~~~~~~~~
Terminal utilities and classes for nmanga.

:copyright: (c) 2022-present noaione
:license: MIT, see LICENSE for more details.
"""

from __future__ import annotations

from .logger import (
    Console,
    ConsoleChoice,
    ConsoleInterface,
    MessageOrInterface,
    MessageQueue,
    ThreadConsoleQueue,
    get_console,
    get_logger,
    thread_queue_callback,
    with_thread_queue,
)
from .shell import SupportedShell, detect_shell

__all__ = (
    "Console",
    "ConsoleChoice",
    "ConsoleInterface",
    "MessageOrInterface",
    "MessageQueue",
    "SupportedShell",
    "ThreadConsoleQueue",
    "detect_shell",
    "get_console",
    "get_logger",
    "thread_queue_callback",
    "with_thread_queue",
)

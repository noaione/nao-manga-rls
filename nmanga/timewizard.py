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

import os
from pathlib import Path

try:
    from ctypes import byref, windll, wintypes

    kernel32 = windll.kernel32
    FILETIME = wintypes.FILETIME
    HANDLE = wintypes.HANDLE
    BOOL = wintypes.BOOL

    IS_WINDOWS_CTIME_SUPPORTED = True
except (ImportError, AttributeError, OSError, ValueError):
    IS_WINDOWS_CTIME_SUPPORTED = False

__all__ = ("modify_filetimestamp",)


def modify_filetimestamp(file_path: Path, epoch: int) -> None:
    """Modify the timestamp of a file to the given epoch time.

    On Windows, this will also modify the ctime (creation time) of the file.

    The code to modify ctime is adapted from:
    https://stackoverflow.com/a/56805533

    Parameters
    ----------
    file_path: :class:`pathlib.Path`
        The path to the file.
    epoch: :class:`int`
        The epoch time to set the file to.
    """

    # Do modify atime and mtime
    os.utime(file_path, (epoch, epoch))

    # Modify ctime (on Windows)
    if not IS_WINDOWS_CTIME_SUPPORTED:
        return

    timestamp = int((epoch * 10000000) + 116444736000000000)
    ctime = FILETIME(timestamp & 0xFFFFFFFF, timestamp >> 32)

    handle = kernel32.CreateFileW(str(file_path), 256, 0, None, 3, 128, None)
    if handle.value == HANDLE(-1).value:
        return
    kernel32.SetFileTime(handle, byref(ctime), None, None)
    kernel32.CloseHandle(handle)

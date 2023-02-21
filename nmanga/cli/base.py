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
import re
import subprocess as sp
import sys
import traceback
from typing import List, Optional, Pattern, Union, overload

import click

from .. import term

console = term.get_console()
__all__ = (
    "CatchAllExceptionsCommand",
    "RegexCollection",
    "UnrecoverableNMangaError",
    "test_or_find_magick",
    "test_or_find_exiftool",
    "test_or_find_pingo",
    "is_executeable_global_path",
)


def _test_exec(arguments: list):
    try:
        exc = sp.check_call(arguments, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        if exc != 0:
            return False
        return True
    except OSError:
        return False


def _find_exec_path(exec_name: Union[str, List[str]], test_cmd: Optional[str] = None) -> str:
    if isinstance(exec_name, str):
        exec_name = [exec_name]
    path_env = os.environ.get("PATH", "")
    console.status("Falling back to finding in PATH")
    # find magick in PATH
    for path in path_env.split(os.pathsep):
        path = path.strip('"')
        for exec in exec_name:
            exec_path = os.path.join(path, exec)
            exec_cmd = [exec_path]
            if test_cmd is not None:
                exec_cmd.append(test_cmd)
            if _test_exec(exec_cmd):
                console.stop_status()
                return exec_path
    console.stop_status()


def test_or_find_magick(magick_path: str, force_search: bool = True) -> Optional[str]:
    try:
        success = _test_exec([magick_path, "-version"])
        if not success:
            return None if not force_search else _find_exec_path(["magick", "convert"], "-version")
        return magick_path or (
            None if not force_search else _find_exec_path(["magick", "convert"], "-version")
        )
    except OSError:
        return None if not force_search else _find_exec_path(["magick", "convert"], "-version")


def test_or_find_exiftool(exiftool_path: str, force_search: bool = True) -> Optional[str]:
    try:
        success = _test_exec([exiftool_path])
        if not success:
            return None if not force_search else _find_exec_path("exiftool")
        return exiftool_path or (None if not force_search else _find_exec_path("exiftool"))
    except OSError:
        return None if not force_search else _find_exec_path("exiftool")


def test_or_find_pingo(pingo_path: str, force_search: bool = True) -> Optional[str]:
    try:
        success = _test_exec([pingo_path])
        if not success:
            return None if not force_search else _find_exec_path("pingo")
        return pingo_path or (None if not force_search else _find_exec_path("pingo"))
    except OSError:
        return None if not force_search else _find_exec_path("pingo")


def is_executeable_global_path(path: str, executable: str) -> bool:
    path = path.lower()
    if path == executable:
        return True
    if path == f"./{executable}":
        return True
    if path == f".\\{executable}":
        return True
    return False


class CatchAllExceptionsCommand(click.Command):
    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except Exception as ex:
            raise UnrecoverableNMangaError(str(ex), sys.exc_info())


class UnrecoverableNMangaError(click.ClickException):
    def __init__(self, message, exc_info):
        super().__init__(message)
        self.exc_info = exc_info

    def show(self):
        emoji = ""
        if console.is_advanced():
            emoji = "\u274C "
        console.error(f"*** {emoji}An unrecoverable error occured ***")
        console.error(self.message)
        # Make traceback
        traceback.print_exception(*self.exc_info)


class RegexCollection:
    _VolumeRegex = r"CHANGETHIS v(\d+).*"
    _OneShotRegex = r"CHANGETHIS .*"
    # fmt: off
    _ChapterTitleRe = r"CHANGETHIS - c(?P<ch>\d+)(?P<ex>[\#x.][\d]{1,2})? \(?c?(?P<actual>[\d]{1,3}[\.][\d]{1,3})?\)?" \
                      r" ?\(?(?P<vol>v[\d]+|[Oo][Ss]hot|[Oo]ne[ -]?[Ss]hot|[Nn][Aa])?\)? ?- p[\d]+x?[\d]?\-?[\d]+x?" \
                      r"[\d]? .*\[(?:dig|web|c2c|mag|scan|paper)] (?:\[(?P<title>.*)\] )?\[CHANGEPUBLISHER.*"
    _ChapterBasicRe = r"CHANGETHIS - c(?P<ch>\d+)(?P<ex>[\#x.][\d]{1,2})? \(?c?(?P<actual>[\d]{1,3}[\.][\d]{1,3})?\)?" \
                      r" ?\(?(?P<vol>v[\d]+|[Oo][Ss]hot|[Oo]ne[ -]?[Ss]hot|[Nn][Aa])?\)? ?- p[\d]+x?[\d]?\-?[\d]+x?" \
                      r"[\d]?.*"
    # fmt: on

    @classmethod
    def volume_re(cls, title: str, limit_credit: Optional[str] = None) -> Pattern[str]:
        re_fmt = cls._VolumeRegex.replace("CHANGETHIS", re.escape(title))
        if limit_credit is not None:
            re_fmt += r"[\[\(]" + limit_credit + r".*"
        return re.compile(re_fmt)

    @overload
    def chapter_re(self, title: str) -> Pattern[str]:
        ...

    @overload
    def chapter_re(self, title: str, publisher: str) -> Pattern[str]:
        ...

    @classmethod
    def chapter_re(cls, title: str, publisher: Optional[str] = None) -> Pattern[str]:
        if publisher is None:
            return re.compile(cls._ChapterBasicRe.replace("CHANGETHIS", re.escape(title)))
        return re.compile(
            cls._ChapterTitleRe.replace("CHANGETHIS", re.escape(title)).replace(
                "CHANGEPUBLISHER", re.escape(publisher)
            )
        )

    @classmethod
    def cmx_re(cls) -> Pattern[str]:
        return re.compile(
            r"(?P<t>[\w\W\D\d\S\s]+?)(?:\- (?P<vol>v[\d]{1,3}))? \- p(?P<a>[\d]{1,3})\-?(?P<b>[\d]{1,3})?"
        )

    @classmethod
    def page_re(cls) -> Pattern[str]:
        return re.compile(r"(?P<any>.*)p(?P<a>[\d]{1,3})\-?(?P<b>[\d]{1,3})?(?P<anyback>.*)")

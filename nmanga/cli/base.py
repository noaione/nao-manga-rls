import os
import re
import subprocess as sp
import sys
import traceback
from typing import Optional, Pattern, overload

import click

from .. import term

console = term.get_console()
__all__ = (
    "CatchAllExceptionsCommand",
    "RegexCollection",
    "UnrecoverableNMangaError",
    "test_or_find_magick",
)


def _test_magick(magick_path: str):
    try:
        exc = sp.check_call([magick_path, "-version"], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        if exc != 0:
            return False
        return True
    except OSError:
        return False


def _find_magick_path():
    path_env = os.environ.get("PATH", "")
    console.status("Falling back to finding magick in PATH")
    # find magick in PATH
    for path in path_env.split(os.pathsep):
        path = path.strip('"')
        magick_path = os.path.join(path, "magick")
        convert_path = os.path.join(path, "convert")
        if _test_magick(magick_path):
            console.stop_status()
            return magick_path
        if _test_magick(convert_path):
            console.stop_status()
            return convert_path
    console.stop_status()
    return None


def test_or_find_magick(magick_path: str, force_search: bool = True) -> Optional[str]:
    try:
        success = _test_magick(magick_path)
        if not success:
            return None if not force_search else _find_magick_path()
        return magick_path or (None if not force_search else _find_magick_path())
    except OSError:
        return None if not force_search else _find_magick_path()


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
    _ChapterTitleRe = r"CHANGETHIS - c(?P<ch>\d+)(?P<ex>x[\d]{1,2})? \((?P<vol>v[\d]+|[Oo][Ss]hot|[Oo]ne[ -]?[Ss]hot" \
                      r"|[Nn][Aa])\) - p[\d]+x?[\d]?\-?[\d]+x?[\d]? .*\[dig] (?:\[(?P<title>.*)\] )?\[CHANGEPUBLISHER.*"
    _ChapterBasicRe = r"CHANGETHIS - c(?P<ch>\d+)(?P<ex>[\#x][\d]{1,2})? \((?P<vol>v[\d]+|[Oo][Ss]hot|[Oo]ne[ -]?[Ss]" \
                      r"hot|[Nn][Aa])\) - p[\d]+x?[\d]?\-?[\d]+x?[\d]?.*"
    # fmt: on

    @classmethod
    def volume_re(cls, title: str) -> Pattern[str]:
        return re.compile(cls._VolumeRegex.replace("CHANGETHIS", title))

    @overload
    def chapter_re(self, title: str) -> Pattern[str]:
        ...

    @overload
    def chapter_re(self, title: str, publisher: str) -> Pattern[str]:
        ...

    @classmethod
    def chapter_re(cls, title: str, publisher: Optional[str] = None) -> Pattern[str]:
        if publisher is None:
            return re.compile(cls._ChapterBasicRe.replace("CHANGETHIS", title))
        return re.compile(
            cls._ChapterTitleRe.replace("CHANGETHIS", title).replace("CHANGEPUBLISHER", publisher)
        )

    @classmethod
    def cmx_re(cls) -> Pattern[str]:
        return re.compile(r"(?P<t>.*)\- \(?(?P<vol>v[\d]{1,2})\)? - p(?P<a>[\d]{1,3})\-?(?P<b>[\d]{1,3})?")

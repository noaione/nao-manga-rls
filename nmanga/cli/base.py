import sys
import os
import subprocess as sp
import traceback
from typing import Optional

import click

from .. import term

console = term.get_console()
__all__ = ("CatchAllExceptionsCommand", "UnrecoverableNMangaError", "test_or_find_magick",)


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

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
import subprocess as sp
import sys
import traceback
import warnings
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Pattern, Tuple, Union, cast, overload

import click
from click.core import Context
from click.parser import Option as ParserOption
from click.parser import OptionParser

from .. import term
from ..common import RegexCollection as _RegexCollection  # noqa: F401, RUF100

if TYPE_CHECKING:
    from click.parser import ParsingState

console = term.get_console()
__all__ = (
    "NMangaCommandHandler",
    "UnrecoverableNMangaError",
    "test_or_find_magick",
    "test_or_find_exiftool",
    "test_or_find_pingo",
    "is_executeable_global_path",
)


def _test_exec(
    arguments: list,
    *,
    extra_check: Optional[Callable[[str, str], bool]] = None,
):
    try:
        proc = sp.Popen(arguments, stdout=sp.PIPE, stderr=sp.PIPE)
        proc.wait(5.0)
        if proc.returncode != 0:
            if callable(extra_check):
                return bool(extra_check(proc.stdout.read().decode("utf-8"), proc.stderr.read().decode("utf-8")))
            return False
        return True
    except sp.TimeoutExpired:
        # Pingo 1.x will hang if no -help parameter is given
        return True
    except OSError:
        return False


def _find_exec_path(
    exec_name: Union[str, List[str]],
    test_cmd: Optional[str] = None,
    *,
    extra_check: Optional[Callable[[str, str], bool]] = None,
) -> str:
    if isinstance(exec_name, str):
        exec_name = [exec_name]
    path_env = os.environ.get("PATH", "")
    console.status("Falling back to finding in PATH")
    # find magick in PATH
    for path in path_env.split(os.pathsep):
        path = path.strip('"')
        for exec in exec_name:
            exec_path = os.path.join(path, exec)  # noqa: PTH118
            exec_cmd = [exec_path]
            if test_cmd is not None:
                exec_cmd.append(test_cmd)
            if _test_exec(exec_cmd, extra_check=extra_check):
                console.stop_status()
                return exec_path
    console.stop_status()


def test_or_find_magick(magick_path: str, force_search: bool = True) -> Optional[str]:
    try:
        success = _test_exec([magick_path, "-version"])
        if not success:
            return None if not force_search else _find_exec_path(["magick", "convert"], "-version")
        return magick_path or (None if not force_search else _find_exec_path(["magick", "convert"], "-version"))
    except OSError:
        return None if not force_search else _find_exec_path(["magick", "convert"], "-version")


def test_or_find_exiftool(exiftool_path: str, force_search: bool = True) -> Optional[str]:
    try:
        success = _test_exec([exiftool_path, "-ver"])
        if not success:
            return None if not force_search else _find_exec_path("exiftool", ["-ver"])
        return exiftool_path or (None if not force_search else _find_exec_path("exiftool", ["-ver"]))
    except OSError:
        return None if not force_search else _find_exec_path("exiftool", ["-ver"])


def _is_pingo_validity_check(stdout: str, stderr: str):
    if "bad command. type 'pingo'" in (stdout := stdout.strip().lower()):
        return True
    if "bad command. type 'pingo'" in (stderr := stderr.strip().lower()):
        return True
    return False


def test_or_find_pingo(pingo_path: str, force_search: bool = True) -> Optional[str]:
    try:
        success = _test_exec([pingo_path, "-help"], extra_check=_is_pingo_validity_check)
        if not success:
            return (
                None if not force_search else _find_exec_path("pingo", ["-help"], extra_check=_is_pingo_validity_check)
            )
        return pingo_path or (
            None if not force_search else _find_exec_path("pingo", ["-help"], extra_check=_is_pingo_validity_check)
        )
    except OSError:
        return None if not force_search else _find_exec_path("pingo", ["-help"], extra_check=_is_pingo_validity_check)


def is_executeable_global_path(path: str, executable: str) -> bool:
    path = path.lower()
    if path == executable:
        return True
    if path == f"./{executable}":
        return True
    if path == f".\\{executable}":
        return True
    return False


class WithDeprecatedOption(click.Option):
    def __init__(self, *args, **kwargs):
        self.is_deprecated = bool(kwargs.pop("deprecated", False))

        preferred: Optional[Union[Tuple[str], List[str], str]] = kwargs.pop("preferred", None)
        preferred_list = []
        if preferred is not None:
            if isinstance(preferred, str):
                preferred_list = [preferred]
            elif isinstance(preferred_list, (tuple, list)):
                preferred_list: List[str] = []
                for pref in preferred:
                    if not isinstance(pref, str):
                        raise ValueError(f"The following prefered option is not a string! `{pref!r}`")
                    preferred_list.append(pref)
        self.preferred: List[str] = preferred_list
        super(WithDeprecatedOption, self).__init__(*args, **kwargs)

    def get_help_record(self, ctx: Context) -> Optional[Tuple[str, str]]:
        parent = super().get_help_record(ctx)
        if parent is None:
            return parent

        if self.is_deprecated:
            opts_thing, help = parent
            return (opts_thing, f"(DEPRECATED) {help}")
        return parent


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


def _fmt_pref_text(preferred: List[str]):
    return "`" + "` or `".join(preferred) + "`"


class NMangaCommandHandler(click.Command):
    def make_parser(self, ctx: Context) -> OptionParser:
        """
        Hook the process of making parser to handle deprecated options.

        https://stackoverflow.com/a/50402799
        """
        parser = super(NMangaCommandHandler, self).make_parser(ctx)

        options = set(parser._short_opt.values())
        options |= set(parser._long_opt.values())

        for option in options:
            if not isinstance(option.obj, WithDeprecatedOption):
                continue

            def make_process(opt: ParserOption):
                orig_process = option.process

                def _process_intercept(
                    value: str,
                    state: "ParsingState",
                    upper_opt: ParserOption,
                    original_func: callable,
                ):
                    is_deprecated = cast(bool, getattr(upper_opt.obj, "is_deprecated", False))
                    preferred = cast(List[str], getattr(upper_opt.obj, "preferred", []))

                    opt_short = upper_opt._short_opts
                    opt_long = upper_opt._long_opts

                    merged_short = "/".join(opt_short)
                    merged_long = "/".join(opt_long)

                    merged_opt = ""
                    if merged_short:
                        merged_opt += merged_short
                    if merged_long:
                        if merged_opt:
                            merged_opt += "/"
                        merged_opt += merged_long

                    if is_deprecated is not None:
                        warn_msg = f"Option `{merged_opt}` is deprecated!"
                        if len(preferred) > 0:
                            warn_msg += f" Use {_fmt_pref_text(preferred)} instead!"
                        else:
                            warn_msg += " This option will be removed in the future!"
                        console.warning(warn_msg)

                    return original_func(value, state)

                return partial(_process_intercept, upper_opt=opt, original_func=orig_process)

            option.process = make_process(option)

        return parser

    def invoke(self, ctx: Context):
        try:
            return super().invoke(ctx)
        except Exception as ex:
            # Invoke error handler
            raise UnrecoverableNMangaError(str(ex), sys.exc_info())


class RegexCollection:
    @classmethod
    def volume_re(cls, title: str, limit_credit: Optional[str] = None) -> Pattern[str]:
        warnings.warn(
            "nmanga.cli.base.RegexCollection.volume_re is deprecated, "
            "use nmanga.common.RegexCollection.volume_re instead",
            DeprecationWarning,
        )
        return _RegexCollection.volume_re(title, limit_credit)

    @overload
    @classmethod
    def chapter_re(cls, title: str) -> Pattern[str]:
        ...

    @overload
    @classmethod
    def chapter_re(cls, title: str, publisher: str) -> Pattern[str]:
        ...

    @classmethod
    def chapter_re(cls, title: str, publisher: Optional[str] = None) -> Pattern[str]:
        warnings.warn(
            "nmanga.cli.base.RegexCollection.chapter_re is deprecated, "
            "use nmanga.common.RegexCollection.chapter_re instead",
            DeprecationWarning,
        )
        return _RegexCollection.chapter_re(title, publisher)

    @classmethod
    def cmx_re(cls) -> Pattern[str]:
        warnings.warn(
            "nmanga.cli.base.RegexCollection.cmx_re is deprecated, use nmanga.common.RegexCollection.cmx_re instead",
            DeprecationWarning,
        )
        return _RegexCollection.cmx_re()

    @classmethod
    def page_re(cls) -> Pattern[str]:
        warnings.warn(
            "nmanga.cli.base.RegexCollection.page_re is deprecated, use nmanga.common.RegexCollection.cmx_re instead",
            DeprecationWarning,
        )
        return _RegexCollection.page_re()

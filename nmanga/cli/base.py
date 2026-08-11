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

from __future__ import annotations

import os
import subprocess as sp
import sys
import traceback
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Mapping, Pattern, cast, overload

import rich_click as click
from click.core import Context
from rich.columns import Columns
from rich.text import Text
from rich_click.rich_click_theme import RichClickTheme

from .. import term
from ..common import RegexCollection as _RegexCollection

if TYPE_CHECKING:
    from click.parser import _Option, _OptionParser, _ParsingState

console = term.get_console()
__all__ = (
    "NMangaCommandHandler",
    "UnrecoverableNMangaError",
    "WithDeprecatedOption",
    "WithMutuallyExclusiveOption",
    "is_executeable_global_path",
    "test_or_find_cjpegli",
    "test_or_find_cjxl",
    "test_or_find_exiftool",
    "test_or_find_magick",
    "test_or_find_pingo",
    "test_or_find_w2x_trt",
)


def _test_exec(
    arguments: list,
    *,
    extra_check: Callable[[str, str], bool] | None = None,
):
    try:
        proc = sp.Popen(arguments, stdout=sp.PIPE, stderr=sp.PIPE)
        proc.wait(5.0)

        if proc.stdout is None or proc.stderr is None:
            raise RuntimeError("Failed to capture stdout/stderr")

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
    exec_name: str | list[str],
    test_cmd: str | list[str] | None = None,
    *,
    extra_check: Callable[[str, str], bool] | None = None,
) -> str | None:
    if isinstance(exec_name, str):
        exec_name = [exec_name]
    test_cmd_list: list[str] = []
    if isinstance(test_cmd, str):
        test_cmd_list = [test_cmd]
    elif isinstance(test_cmd, list):
        test_cmd_list = test_cmd
    path_env = os.environ.get("PATH", "")
    console.status("Falling back to finding in PATH")
    # find magick in PATH
    for path in path_env.split(os.pathsep):
        path = path.strip('"')
        for exec in exec_name:
            exec_path = os.path.join(path, exec)  # ruff: ignore[os-path-join]
            exec_cmd = [exec_path]
            for test_cmd in test_cmd_list:
                exec_cmd.append(test_cmd)
            if _test_exec(exec_cmd, extra_check=extra_check):
                console.stop_status(f"Found executable at {exec_path}")
                return exec_path
    console.stop_status()


def test_or_find_magick(magick_path: str, force_search: bool = True) -> str | None:
    try:
        success = _test_exec([magick_path, "-version"])
        if not success:
            return None if not force_search else _find_exec_path(["magick"], "-version")
        return magick_path or (None if not force_search else _find_exec_path(["magick"], "-version"))
    except OSError:
        return None if not force_search else _find_exec_path(["magick"], "-version")


def test_or_find_exiftool(exiftool_path: str, force_search: bool = True) -> str | None:
    try:
        success = _test_exec([exiftool_path, "-ver"])
        if not success:
            return None if not force_search else _find_exec_path("exiftool", ["-ver"])
        return exiftool_path or (None if not force_search else _find_exec_path("exiftool", ["-ver"]))
    except OSError:
        return None if not force_search else _find_exec_path("exiftool", ["-ver"])


def test_or_find_cjpegli(cjpegli: str, force_search: bool = True) -> str | None:
    try:
        success = _test_exec([cjpegli, "-h"])
        if not success:
            return None if not force_search else _find_exec_path("cjpegli", ["-h"])
        return cjpegli or (None if not force_search else _find_exec_path("cjpegli", ["-h"]))
    except OSError:
        return None if not force_search else _find_exec_path("cjpegli", ["-h"])


def test_or_find_cjxl(cjxl: str, force_search: bool = True) -> str | None:
    try:
        success = _test_exec([cjxl, "-h"])
        if not success:
            return None if not force_search else _find_exec_path("cjxl", ["-h"])
        return cjxl or (None if not force_search else _find_exec_path("cjxl", ["-h"]))
    except OSError:
        return None if not force_search else _find_exec_path("cjxl", ["-h"])


def test_or_find_w2x_trt(w2x_trt_path: str | None, force_search: bool = True) -> str | None:
    if not w2x_trt_path:
        return None if not force_search else _find_exec_path("waifu2x-tensorrt", ["-h"])
    try:
        success = _test_exec([w2x_trt_path, "-h"])
        if not success:
            return None if not force_search else _find_exec_path("waifu2x-tensorrt", ["-h"])
        return w2x_trt_path or (None if not force_search else _find_exec_path("waifu2x-tensorrt", ["-h"]))
    except OSError:
        return None if not force_search else _find_exec_path("waifu2x-tensorrt", ["-h"])


def _is_pingo_validity_check(stdout: str, stderr: str) -> bool:
    if "bad command. type 'pingo'" in (stdout := stdout.strip().lower()):
        return True
    if "bad command. type 'pingo'" in (stderr := stderr.strip().lower()):
        return True
    return False


def test_or_find_pingo(pingo_path: str, force_search: bool = True) -> str | None:
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


class WithDeprecatedOption(click.RichOption):
    def __init__(self, *args, **kwargs):
        self.is_deprecated = bool(kwargs.pop("deprecated", False))

        preferred: tuple[str] | list[str] | str | None = kwargs.pop("preferred", None)
        preferred_list = []
        if preferred is not None:
            if isinstance(preferred, str):
                preferred_list = [preferred]
            elif isinstance(preferred_list, (tuple, list)):
                preferred_list: list[str] = []
                for pref in preferred:
                    if not isinstance(pref, str):
                        raise ValueError(f"The following prefered option is not a string! `{pref!r}`")
                    preferred_list.append(pref)
        self.preferred: list[str] = preferred_list
        super(WithDeprecatedOption, self).__init__(*args, **kwargs)

    def _get_bracket_text(self, formatter: click.RichHelpFormatter) -> tuple[tuple[str, str], str]:
        the_theme = formatter.config.theme
        if isinstance(the_theme, str):
            if "-nu" in the_theme:
                return ("(", ")"), "Deprecated"
            elif "-robo" in the_theme:
                return ("❮", "❯"), "deprecated"  # ruff: ignore[ambiguous-unicode-character-string]
        elif isinstance(the_theme, RichClickTheme):
            if "nu" in the_theme.name:
                return ("(", ")"), "dim blue"
            elif "robo" in the_theme.name:
                return ("❮", "❯"), "dim blue"  # ruff: ignore[ambiguous-unicode-character-string]
        return ("[", "]"), "dim blue"

    def get_rich_help(self, ctx: click.RichContext, formatter: click.RichHelpFormatter) -> Columns:
        cols = super().get_rich_help(ctx, formatter)
        brackets, depre_text = self._get_bracket_text(formatter)
        if self.is_deprecated:
            render = Text.assemble((brackets[0], "dim blue"), (depre_text, "dim blue bold"), (brackets[1], "dim blue"))
            # Iterate through all existing text render
            for rr in cols.renderables:
                if isinstance(rr, Text):
                    rr.style = "strike"
            cols.add_renderable(render)
            if len(self.preferred) > 0:
                pref_text = _fmt_pref_text(self.preferred)
                render_pref = Text.assemble(
                    (brackets[0], "dim"),
                    ("Preferred option(s): ", "dim"),
                    (pref_text, "dim bold"),
                    (brackets[1], "dim"),
                )
                cols.add_renderable(render_pref)
        return cols


class WithMutuallyExclusiveOption(click.RichOption):
    def __init__(self, *args, **kwargs):
        mutex_with: tuple[str | click.Option] | list[str | click.Option] | str | click.Option | None = kwargs.pop(
            "mutex_with", None
        )
        mutex_list: list[str] = []
        if mutex_with is not None:
            if isinstance(mutex_with, (str, click.Option)):
                mutex_list = [self._normalize_mutex_name(mutex_with)]
            elif isinstance(mutex_with, (tuple, list)):
                mutex_list = []
                for mw in mutex_with:
                    if not isinstance(mw, (str, click.Option)):
                        raise ValueError(
                            f"The following mutually exclusive option is not a string or click.Option! `{mw!r}`"
                        )
                    mutex_list.append(self._normalize_mutex_name(mw))
        self.mutex_with: list[str] = mutex_list
        self.is_mutex: bool = len(mutex_list) > 0
        super(WithMutuallyExclusiveOption, self).__init__(*args, **kwargs)

    @staticmethod
    def _normalize_mutex_name(name_or_opt: str | click.Option) -> str:
        """Normalize a mutex reference to its long-form option name.

        Accepts either a raw token (``"bar"``, ``"-b"``, ``"--bar"``) or a
        :class:`click.Option` instance. For options, the long form (``--name``)
        is preferred when available; otherwise the first declared opt is used.
        For strings, the token is normalized to the long form when no leading
        dash is present, otherwise returned unchanged.
        """
        if isinstance(name_or_opt, click.Option):
            long_opts = [o for o in name_or_opt.opts if o.startswith("--")]
            if long_opts:
                return long_opts[0]
            if name_or_opt.opts:
                return name_or_opt.opts[0]
            # Fallback: synthesize from the canonical parameter name.
            return f"--{name_or_opt.name}"
        # String token
        s = name_or_opt.strip()
        if not s.startswith("-"):
            return f"--{s}"
        return s

    @staticmethod
    def _is_value_set(value, default) -> bool:
        """Return True when ``value`` looks user-provided rather than default.

        Treats ``None``, empty containers, and values equal to the parameter
        default as "not set" so that mutually-exclusive checks do not fire on
        options the user never supplied.
        """
        if value is None:
            return False
        if isinstance(value, (list, tuple)) and len(value) == 0:
            return False
        if value == default:
            return False
        return True

    def _resolve_partner(self, ctx: click.Context, partner_name: str) -> tuple[str | None, object]:
        """Resolve a partner's normalized long-form name to ``(param_name, default)``.

        Looks up the partner ``click.Option`` in the current command's params
        so the parse-time conflict check can compare against its parsed value
        while accounting for its default.
        """
        bare = partner_name.lstrip("-")
        for p in ctx.command.params:
            if not isinstance(p, click.Option) or not p.name:
                continue
            if partner_name in p.opts or bare == p.name:
                return p.name, p.default
        return None, None

    def handle_parse_result(self, ctx: click.Context, opts: Mapping[str, Any], args: list) -> tuple[dict, list]:
        """Click hook: enforce mutex relationships against already-parsed options.

        Raises :class:`click.UsageError` when this option was set by the user
        and any of its ``mutex_with`` partners was also set. Options are
        processed in declaration order, so a conflict raised on the later
        option references both clashing option names in the error message.
        """
        if not self.is_mutex or self.name not in opts:
            return super().handle_parse_result(ctx, opts, args)
        if not self._is_value_set(opts[self.name], self.default):
            return super().handle_parse_result(ctx, opts, args)

        for partner_token in self.mutex_with:
            partner_param, partner_default = self._resolve_partner(ctx, partner_token)
            if partner_param is None or partner_param == self.name:
                continue
            if partner_param not in opts:
                continue
            if not self._is_value_set(opts[partner_param], partner_default):
                continue
            self_name = next((o for o in self.opts if o.startswith("--")), None) or next(
                iter(self.opts), f"--{self.name}"
            )
            raise click.UsageError(f"Illegal usage: `{self_name}` is mutually exclusive with `{partner_token}`.")

        return super().handle_parse_result(ctx, opts, args)

    def _get_bracket_text(self, formatter: click.RichHelpFormatter) -> tuple[tuple[str, str], str]:
        the_theme = formatter.config.theme
        if isinstance(the_theme, str):
            if "-nu" in the_theme:
                return ("(", ")"), "Mutually exclusive"
            elif "-robo" in the_theme:
                return ("❮", "❯"), "mutually exclusive"  # ruff: ignore[ambiguous-unicode-character-string]
        elif isinstance(the_theme, RichClickTheme):
            if "nu" in the_theme.name:
                return ("(", ")"), "yellow"
            elif "robo" in the_theme.name:
                return ("❮", "❯"), "yellow"  # ruff: ignore[ambiguous-unicode-character-string]
        return ("[", "]"), "yellow"

    def get_rich_help(self, ctx: click.RichContext, formatter: click.RichHelpFormatter) -> Columns:
        cols = super().get_rich_help(ctx, formatter)
        if not self.is_mutex:
            return cols
        brackets, _ = self._get_bracket_text(formatter)
        if len(self.mutex_with) > 0:
            mutex_with_text = _fmt_pref_text(self.mutex_with)
            render_mutex = Text.assemble(
                (brackets[0], "yellow"),
                ("Mutex with: ", "yellow"),
                (mutex_with_text, "yellow bold"),
                (brackets[1], "yellow"),
            )
            cols.add_renderable(render_mutex)
        return cols


class UnrecoverableNMangaError(click.ClickException):
    def __init__(self, message, exc_info):
        super().__init__(message)
        self.exc_info = exc_info
        self.message = message  # type: ignore

    def format_message(self) -> str:
        error_msg = "*** An unrecoverable error occured ***\n"
        error_msg += self.message
        # Includde traceback
        tb_lines = traceback.format_exception(*self.exc_info)
        error_msg += "\n\n"
        error_msg += "".join(tb_lines)
        return error_msg

    def show(self, file=None):
        emoji = ""
        if console.is_advanced():
            emoji = "\u274c "
        console.error(f"*** {emoji}An unrecoverable error occured ***")
        console.error(self.message)
        # Make traceback
        traceback.print_exception(*self.exc_info)


def _fmt_pref_text(preferred: list[str]):
    return "`" + "` or `".join(preferred) + "`"


class NMangaCommandHandler(click.RichCommand):
    def make_parser(self, ctx: Context) -> "_OptionParser":
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

            orig_process = option.process

            def make_process(opt: "_Option", orig_process: Callable[..., None]):
                def _process_intercept(
                    value: str,
                    state: "_ParsingState",
                    upper_opt: "_Option",
                    original_func: Callable[[str, "_ParsingState"], None],
                ):
                    is_deprecated = cast(bool, getattr(upper_opt.obj, "is_deprecated", False))
                    preferred = cast(list[str], getattr(upper_opt.obj, "preferred", []))

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

            option.process = make_process(option, orig_process)

        return parser

    def invoke(self, ctx: Context):
        try:
            # Try to see if command is being run with deprecated options
            return super().invoke(ctx)
        except Exception as ex:
            # Invoke error handler
            console.stop_status()  # So it doesn't consume the indicator in iTerm (and maybe more)
            console.stop_current_progress(skip_total=True)
            console.console.show_cursor()
            raise UnrecoverableNMangaError(str(ex), sys.exc_info()) from ex


class RegexCollection:
    @classmethod
    def volume_re(cls, title: str, limit_credit: str | None = None) -> Pattern[str]:
        warnings.warn(
            "nmanga.cli.base.RegexCollection.volume_re is deprecated, "
            "use nmanga.common.RegexCollection.volume_re instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return _RegexCollection.volume_re(title, limit_credit)

    @overload
    @classmethod
    def chapter_re(cls, title: str) -> Pattern[str]: ...

    @overload
    @classmethod
    def chapter_re(cls, title: str, publisher: str) -> Pattern[str]: ...

    @classmethod
    def chapter_re(cls, title: str, publisher: str | None = None) -> Pattern[str]:
        warnings.warn(
            "nmanga.cli.base.RegexCollection.chapter_re is deprecated, "
            "use nmanga.common.RegexCollection.chapter_re instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return _RegexCollection.chapter_re(title, publisher)

    @classmethod
    def cmx_re(cls) -> Pattern[str]:
        warnings.warn(
            "nmanga.cli.base.RegexCollection.cmx_re is deprecated, use nmanga.common.RegexCollection.cmx_re instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return _RegexCollection.cmx_re()

    @classmethod
    def page_re(cls) -> Pattern[str]:
        warnings.warn(
            "nmanga.cli.base.RegexCollection.page_re is deprecated, use nmanga.common.RegexCollection.cmx_re instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return _RegexCollection.page_re()

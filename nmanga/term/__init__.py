"""
nmanga.term
~~~~~~~~~~~
Terminal utilities and classes for nmanga.

:copyright: (c) 2022-present noaione
:license: MIT, see LICENSE for more details.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, TypeAlias, TypeVar, cast, overload

import inquirer
from rich.console import Console as RichConsole
from rich.progress import (
    Progress,
    TaskID,
)
from rich.theme import Theme as RichTheme

from .progress import NMProgress

if TYPE_CHECKING:
    from rich.status import Status as RichStatus

__all__ = ("ConsoleChoice", "get_console")

rich_theme = RichTheme({
    "success": "green bold",
    "warning": "yellow bold",
    "error": "red bold",
    "highlight": "magenta bold",
    "info": "cyan bold",
    "tracker-bar.pulse": "yellow bold",
    "tracker-bar.remaining": "grey23",
    "tracker-bar.complete": "cyan bold",
    "tracker-bar.finished": "green bold",
    "tracker-bar.outer": "bold",
})
AnyType = TypeVar("AnyType", str, bytes, int, float)
ValidateFunc: TypeAlias = Callable[[str], bool]
ValidationType: TypeAlias = ValidateFunc | str


@dataclass
class ConsoleChoice:
    name: str
    value: str | None = None

    def __post_init__(self):
        self.value = self.value or self.name


class Console:
    def __init__(self, debug_mode: bool = False):
        self.__debug_mode = debug_mode
        self.console = RichConsole(highlight=False, theme=rich_theme, soft_wrap=True, width=None)
        self._status: "RichStatus | None" = None
        self.__last_known_status: str | None = None

        self.__current_progress: NMProgress | None = None

        self.shift_space = 0

    def enable_debug(self):
        self.__debug_mode = True

    def disable_debug(self):
        self.__debug_mode = False

    @property
    def debugged(self) -> bool:
        return self.__debug_mode

    def __wrap_theme(self, text: str, theme: str):
        if self.__debug_mode:
            return f"[{text}]"
        return f"[[{theme}]{text}[/{theme}]]"

    def set_space(self, space: int):
        self.shift_space = space

    def info(self, *args, **kwargs):
        space = " " * self.shift_space
        if not space:
            self.console.print(self.__wrap_theme("INFO", "info"), *args, **kwargs)
        else:
            self.console.print(self.__wrap_theme("INFO", "info"), space, *args, **kwargs)

    def warning(self, *args, **kwargs):
        space = " " * self.shift_space
        if not space:
            self.console.print(self.__wrap_theme("WARN", "warning"), *args, **kwargs)
        else:
            self.console.print(self.__wrap_theme("WARN", "warning"), space, *args, **kwargs)

    def error(self, *args, **kwargs):
        space = " " * self.shift_space
        if not space:
            self.console.print(self.__wrap_theme("ERROR", "error"), *args, **kwargs)
        else:
            self.console.print(self.__wrap_theme("ERROR", "error"), space, *args, **kwargs)

    def __beautiful_status(self, message: str, **kwargs_spinner_style):
        if not self._status:
            default = {}
            if self.is_advanced():
                default = {"spinner": "dots"}
            else:
                default = {"spinner": "line", "refresh_per_second": 6}
            st_args = {**default, **kwargs_spinner_style}

            self._status = self.console.status(message, **st_args)
            self._status.start()
        else:
            self._status.update(message, **kwargs_spinner_style)

    def __debug_status(self, message: str):
        # Rewrite message in same line, wipe out the previous message
        self.console.print(message + "\r", end="")

    def status(self, message: str, **kwargs_spinner_style):
        self.__last_known_status = message
        if not self.__debug_mode:
            self.__beautiful_status(message, **kwargs_spinner_style)
        else:
            self.__debug_status(message)

    def make_progress(self) -> NMProgress:
        if self.__current_progress is not None and self.__current_progress.live.is_started:
            # If already started, return the current one
            return self.__current_progress
        progress = NMProgress(
            *NMProgress.get_default_columns(),  # Use custom default columns
            console=self.console,
            expand=True,
        )
        self.__current_progress = progress
        # Auto-start
        progress.start()
        return progress

    def make_task(
        self, progress: NMProgress, description: str, total: int | None = None, *, finished_text: str | None = None
    ) -> TaskID:
        task = progress.add_task(description, total=total, finished_text=finished_text)
        return task

    def stop_progress(self, progress: Progress, text: str | None = None, *, skip_total: bool = False) -> None:
        for task in progress.tasks:
            if task.total is not None and not skip_total:
                progress.update(task.id, completed=task.total)
        progress.stop()
        if text is not None:
            self.info(text)

    @overload
    def stop_status(self) -> None: ...

    @overload
    def stop_status(self, final_text: str) -> None: ...

    def stop_status(self, final_text: str | None = None) -> None:
        final_text = final_text or self.__last_known_status
        if self._status:
            if final_text is not None:
                self._status.update(final_text)
            self._status.stop()
            self._status = None
            if final_text is not None:
                self.info(final_text)
        elif self.__debug_mode:
            if final_text is not None:
                self.__debug_status(final_text)
            # Print new line
            self.console.print()

    def log(self, *args, **kwargs):
        if self.__debug_mode:
            self.console.log(self.__wrap_theme("LOG", "highlight"), *args, **kwargs)

    def sleep(self, duration: int):
        time.sleep(duration)

    def is_advanced(self):
        return not self.console.legacy_windows

    @overload
    def choice(
        self,
        message: str,
        choices: list[ConsoleChoice],
        default: ConsoleChoice | None = ...,
    ) -> ConsoleChoice: ...

    @overload
    def choice(
        self,
        message: str | None = ...,
        choices: list[str] = ...,
        default: str | None = ...,
    ) -> str: ...

    def choice(
        self,
        message: str | None = None,
        choices: list[str] | list[ConsoleChoice] | None = None,
        default: str | ConsoleChoice | None = None,
    ) -> str | ConsoleChoice:
        if not choices:
            raise ValueError("No choices provided")
        message = message or "Please choose an option"
        console_choice = []
        any_cchoice = False
        for choice in choices:
            if isinstance(choice, ConsoleChoice):
                console_choice.append(choice.value)
                any_cchoice = True
            else:
                console_choice.append(choice)
        default_val = None
        if isinstance(default, ConsoleChoice):
            default_val = default.value
        else:
            default_val = default
        answers = inquirer.list_input(message, choices=console_choice, default=default_val)
        if any_cchoice:
            return choices[console_choice.index(answers)]
        return answers

    def _internal_validation(self, text_input: str, validation: ValidateFunc):
        try:
            is_valid = validation(text_input)
            return bool(is_valid)  # Coerce None to bool to avoid error
        except Exception:
            return False

    @overload
    def inquire(
        self,
        prompt: str,
        validation: ValidateFunc | None = ...,
    ) -> str: ...

    @overload
    def inquire(
        self,
        prompt: str,
        validation: ValidateFunc | None = ...,
        default: None = ...,
    ) -> str: ...

    @overload
    def inquire(
        self,
        prompt: str,
        validation: ValidateFunc | None = ...,
        default: str = ...,
    ) -> str: ...

    def inquire(self, prompt: str, validation: ValidateFunc | None = None, default: str | None = None) -> str | None:
        # Custom inquirer
        inquired_text = default
        while True:
            print_this = f"{self.__wrap_theme('?', 'warning')} {prompt}"
            if default is not None:
                print_this += f" [default: {default}]"
            input_temp = self.console.input(print_this + ": ")
            if not input_temp and default is not None:
                # Ignore validation
                break

            if validation is not None:
                if self._internal_validation(input_temp, validation):
                    inquired_text = input_temp
                    break
                else:
                    # Show error temporary and ask again
                    error_text = input_temp or "[No Input]"
                    self.console.print(
                        f"{self.__wrap_theme('ERROR', 'error')} Failed to validate input:",
                        error_text,
                    )
                    continue
            inquired_text = input_temp or ""
            break

        return inquired_text

    def confirm(self, prompt: str | None = None) -> bool:
        prompt = prompt or "Are you sure?"
        return inquirer.confirm(prompt, default=False)

    def enter(self):
        self.console.print()

    # Aliases
    warn = warning
    verbose = log
    debug = log
    i = info
    w = warning
    e = error
    d = log
    v = log


ROOT_CONSOLE = Console()


def get_console():
    return ROOT_CONSOLE

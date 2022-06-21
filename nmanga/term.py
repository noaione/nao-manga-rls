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

from dataclasses import dataclass
from typing import Callable, List, Optional, Union, overload

import inquirer
from rich.console import Console as RichConsole
from rich.theme import Theme as RichTheme

__all__ = ("get_console", "ConsoleChoice")

rich_theme = RichTheme(
    {
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "highlight": "magenta",
    }
)
AnyType = Union[str, bytes, int, float]
ValidateA = Callable[[dict, str], bool]
ValidateB = Callable[[dict], bool]
ValidationType = Union[ValidateA, ValidateB, str]


@dataclass
class ConsoleChoice:
    name: str
    value: Optional[str] = None

    def __post_init__(self):
        self.value = self.value or self.name


class Console:
    def __init__(self):
        self.console = RichConsole(highlight=False, theme=rich_theme, soft_wrap=True)
        self._status = None

    def info(self, *args, **kwargs):
        self.console.print("[INFO]", *args, **kwargs)

    def warning(self, *args, **kwargs):
        self.console.print("[WARN]", *args, **{"style": "warning", **kwargs})

    def error(self, *args, **kwargs):
        self.console.print("[ERROR]", *args, **{"style": "error", **kwargs})

    def status(self, message: str, **kwargs_spinner_style):
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

    def stop_status(self):
        if self._status:
            self._status.stop()

    def log(self, *args, **kwargs):
        self.console.log("[LOG]", *args, **kwargs)

    def is_advanced(self):
        return not self.console.legacy_windows

    @overload
    def choice(self, message: Optional[str] = ..., choices: List[AnyType] = ...) -> str:
        ...

    @overload
    def choice(self, message: Optional[str] = ..., choices: List[ConsoleChoice] = ...) -> ConsoleChoice:
        ...

    def choice(
        self, message: Optional[str] = None, choices: Union[List[AnyType], List[ConsoleChoice]] = []
    ) -> Union[AnyType, ConsoleChoice]:
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
        answers = inquirer.list_input(message, choices=console_choice)
        if any_cchoice:
            return choices[console_choice.index(answers)]
        return answers

    @overload
    def inquire(
        self,
        prompt: str,
        validation: ValidateA = ...,
        default: Optional[AnyType] = ...,
    ) -> AnyType:
        ...

    @overload
    def inquire(
        self,
        prompt: str,
        validation: ValidateB = ...,
        default: Optional[AnyType] = ...,
    ) -> AnyType:
        ...

    @overload
    def inquire(
        self,
        prompt: str,
        validation: str = ...,
        default: Optional[AnyType] = ...,
    ) -> AnyType:
        ...

    def inquire(
        self, prompt: str, validation: Optional[ValidationType] = None, default: Optional[AnyType] = None
    ) -> AnyType:
        return inquirer.text(prompt, default=default, validate=validation)

    def confirm(self, prompt: Optional[str] = None) -> bool:
        prompt = prompt or "Are you sure?"
        return inquirer.confirm(prompt, default=False)

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

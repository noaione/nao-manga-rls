"""
nmanga.term
~~~~~~~~~~~
Terminal utilities and classes for nmanga.

:copyright: (c) 2022-present noaione
:license: MIT, see LICENSE for more details.
"""

from __future__ import annotations

import abc
import logging
import queue
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeAlias, TypeVar, Union, cast, overload
from uuid import uuid4

import inquirer
from rich.console import Console as RichConsole
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    TaskID,
)
from rich.theme import Theme as RichTheme

from .._ntypes import STOP_SIGNAL
from .progress import NMProgress, ProgressStopState

if TYPE_CHECKING:
    from rich.status import Status as RichStatus

__all__ = (
    "Console",
    "ConsoleChoice",
    "MessageOrInterface",
    "MessageQueue",
    "ThreadConsoleQueue",
    "get_console",
    "get_logger",
    "thread_queue_callback",
    "with_thread_queue",
)

rich_theme = RichTheme({
    "success": "green bold",
    "warning": "yellow bold",
    "error": "red bold",
    "highlight": "magenta bold",
    "info": "cyan bold",
    "tracker-bar.pulse": "yellow bold",
    "tracker-bar.remaining": "grey23",
    "tracker-bar.failure": "red bold",
    "tracker-bar.complete": "cyan bold",
    "tracker-bar.finished": "green bold",
    "tracker-bar.outer": "bold",
})
AnyType = TypeVar("AnyType", str, bytes, int, float)
ValidateFunc: TypeAlias = Callable[[str], bool]
ValidationType: TypeAlias = ValidateFunc | str
MessageQueue: TypeAlias = queue.Queue[Any]
MessageOrInterface: TypeAlias = Union["Console", MessageQueue]


def attach_logger(console: "RichConsole", *, is_debug: bool = False) -> None:
    log_lvl = logging.DEBUG if is_debug else logging.INFO

    logging.basicConfig(
        level=log_lvl,
        datefmt="[%X]",
        format="[bold][[bright_magenta]%(name)s[/bright_magenta]][/bold] %(message)s",
        handlers=[
            RichHandler(
                console=console,
                markup=True,
                rich_tracebacks=True,
            )
        ],
    )

    logging.getLogger("rich").setLevel(logging.WARNING)
    logging.getLogger("multiprocessing").setLevel(logging.WARNING)


@dataclass
class ConsoleChoice:
    name: str
    value: str | None = None

    def __post_init__(self):
        self.value = self.value or self.name


class ConsoleInterface(abc.ABC):
    @abc.abstractmethod
    def info(self, *args, **kwargs): ...
    @abc.abstractmethod
    def warning(self, *args, **kwargs): ...
    @abc.abstractmethod
    def error(self, *args, **kwargs): ...
    @abc.abstractmethod
    def log(self, *args, **kwargs): ...
    @abc.abstractmethod
    def enter(self): ...
    @abc.abstractmethod
    def new_task(self, description: str, total: int | None = None, *, finished_text: str | None = None) -> TaskID: ...

    @abc.abstractmethod
    def update_progress(
        self,
        task_id: TaskID,
        total: float | None = None,
        completed: float | None = None,
        advance: float | None = None,
        description: str | None = None,
        visible: bool | None = None,
        refresh: bool = False,
        **fields: Any,
    ) -> None: ...


class Console(ConsoleInterface):
    def __init__(self, debug_mode: bool = False):
        self.__debug_mode = debug_mode

        # Init logger
        self.console = RichConsole(highlight=False, theme=rich_theme, soft_wrap=True, width=None)
        attach_logger(self.console, is_debug=debug_mode)

        self._status: "RichStatus | None" = None
        self.__last_known_status: str | None = None

        self.__current_progress: NMProgress | None = None

        self.shift_space = 0

    def enable_debug(self):
        self.__debug_mode = True
        logging.getLogger().setLevel(logging.DEBUG)

    def disable_debug(self):
        self.__debug_mode = False
        logging.getLogger().setLevel(logging.INFO)

    @property
    def debugged(self) -> bool:
        return self.__debug_mode

    def __wrap_theme(self, text: str, theme: str):
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

    def stop_progress(self, progress: NMProgress, text: str | None = None, *, skip_total: bool = False) -> None:
        for task in progress.tasks:
            if task.total is not None and not skip_total:
                run_state = (
                    ProgressStopState.COMPLETED if task.completed >= task.total else ProgressStopState.EARLY_STOP
                )
                progress.update(task.id, completed=task.total, run_state=run_state)
        progress.stop()
        if text is not None:
            self.info(text)
        if progress == self.__current_progress:
            self.__current_progress = None

    def stop_current_progress(self, text: str | None = None, *, skip_total: bool = False) -> None:
        if self.__current_progress is not None:
            self.stop_progress(self.__current_progress, text, skip_total=skip_total)
            self.__current_progress = None

    def new_task(self, description: str, total: int | None = None, *, finished_text: str | None = None) -> TaskID:
        if self.__current_progress is None:
            progress = self.make_progress()
        else:
            progress = self.__current_progress
        return progress.add_task(description, total=total, finished_text=finished_text)

    def update_progress(
        self,
        task_id: TaskID,
        total: float | None = None,
        completed: float | None = None,
        advance: float | None = None,
        description: str | None = None,
        visible: bool | None = None,
        refresh: bool = False,
        **fields: Any,
    ) -> None:
        if self.__current_progress is None:
            return
        if task_id < 0:
            return  # In case of invalid task id

        self.__current_progress.update(
            task_id=task_id,
            total=total,
            completed=completed,
            advance=advance,
            description=description,
            visible=visible,
            refresh=refresh,
            **fields,
        )

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


class ThreadConsoleQueue(ConsoleInterface):
    """
    A thread-safe console queue for logging from multiple threads.
    """

    def __init__(self, queue: MessageQueue) -> None:
        self.queue = queue

    def info(self, message: str) -> None:
        self.queue.put(("info", message))

    def warning(self, message: str) -> None:
        self.queue.put(("warning", message))

    def error(self, message: str) -> None:
        self.queue.put(("error", message))

    def log(self, message: str) -> None:
        self.queue.put(("log", message))

    def close(self) -> None:
        self.queue.put_nowait(STOP_SIGNAL)

    def enter(self) -> None:
        self.queue.put(("enter", ""))

    def new_task(self, description: str, total: int | None = None, *, finished_text: str | None = None) -> TaskID:
        # Put then wait for response
        consistent_id = str(uuid4())
        self.queue.put((
            "new_task",
            {
                "description": description,
                "total": total,
                "finished_text": finished_text,
                "consistent_id": consistent_id,
            },
        ))

        close_signal = False
        while True:
            raw_q = self.queue.get()
            if raw_q is None:
                continue

            if isinstance(raw_q, logging.LogRecord):
                logger = logging.getLogger(raw_q.name)
                logger.handle(raw_q)
                continue
            if raw_q is STOP_SIGNAL:
                close_signal = True
                break  # Exit loop if we got any close signal

            method, resp_data = raw_q
            if method == "new_task_response":
                resp_data = cast(dict[str, Any], resp_data)
                if resp_data.get("consistent_id") == consistent_id:
                    return TaskID(resp_data.get("task_id", -1))

            # Re-queue if not matched
            self.queue.put_nowait(raw_q)

        if close_signal:
            self.close()
        return TaskID(-1)  # In case of failure

    def update_progress(
        self,
        task_id: TaskID,
        total: float | None = None,
        completed: float | None = None,
        advance: float | None = None,
        description: str | None = None,
        visible: bool | None = None,
        refresh: bool = False,
        **fields: Any,
    ) -> None:
        self.queue.put((
            "update_progress",
            {
                "task_id": task_id,
                "total": total,
                "completed": completed,
                "advance": advance,
                "description": description,
                "visible": visible,
                "refresh": refresh,
                "fields": fields,
            },
        ))


def thread_queue_callback(log_q: MessageQueue, console: Console) -> None:
    """
    Where the queue is processed and messages are printed to the console.

    :param console: The console to print messages to.
    """

    while True:
        try:
            item = log_q.get()
            if item is None:
                continue

            if isinstance(item, logging.LogRecord):
                logger = logging.getLogger(item.name)
                logger.handle(item)
                continue
            if item is STOP_SIGNAL:
                break
            if not isinstance(item, tuple) or len(item) != 2:
                continue

            level, message = item

            match level:
                case "info":
                    console.info(message)
                case "warning":
                    console.warning(message)
                case "error":
                    console.error(message)
                case "log":
                    console.log(message)
                case "enter":
                    console.enter()
                case "update_progress":
                    params = cast(dict[str, Any], message)
                    console.update_progress(
                        task_id=params.get("task_id", -1),
                        total=params.get("total"),
                        completed=params.get("completed"),
                        advance=params.get("advance"),
                        description=params.get("description"),
                        visible=params.get("visible"),
                        refresh=params.get("refresh", False),
                        **params.get("fields", {}),
                    )
                case "new_task":
                    params = cast(dict[str, Any], message)
                    consist_id = params.get("consistent_id", "")
                    task_id = console.new_task(
                        description=params.get("description", ""),
                        total=params.get("total"),
                        finished_text=params.get("finished_text"),
                    )
                    if not consist_id:
                        continue
                    log_q.put_nowait((
                        "new_task_response",
                        {
                            "consistent_id": consist_id,
                            "task_id": task_id,
                        },
                    ))
                case "new_task_response":
                    # Re-queue
                    log_q.put_nowait((level, message))
        except queue.Empty:
            break
        except Exception as exc:
            console.error("Error in ThreadConsoleQueue callback", exc)


ROOT_CONSOLE = Console()


def get_console():
    return ROOT_CONSOLE


def get_logger(child_name: str | None = None, *, override_name: str | None = None) -> logging.Logger:
    log_name = "nmanga"
    if child_name is not None:
        child_name = child_name.lstrip(".")
        log_name += f".{child_name}"
    return logging.getLogger(override_name or log_name)


def with_thread_queue(queue: MessageOrInterface) -> ConsoleInterface:
    """
    Create a ThreadConsoleQueue from a standard queue.

    This will wrap the given queue into a ThreadConsoleQueue if it is not already a Console.

    Note: This is only necessary when using threading and logging from multiple threads.

    :param queue: The queue to use.
    :return: A ConsoleInterface instance.
    """

    if isinstance(queue, Console):
        return queue
    return ThreadConsoleQueue(queue)

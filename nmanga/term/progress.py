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

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Union, cast

from rich.console import Console as RichConsole
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    ProgressSample,
    SpinnerColumn,
    TaskID,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.progress import (
    Task as RichTask,
)
from rich.style import StyleType
from rich.table import Column, Table
from rich.text import Text

__all__ = (
    "BetterDescriptionColumn",
    "BetterSpinnerColumn",
    "MaybePercentageColumn",
    "NMProgress",
    "NMRichTask",
    "ProgressStopState",
    "TrackerBarColumn",
)


class _FakeTrackerBarColumn(ProgressColumn):
    """A fake bar column that does not render anything."""

    def __init__(self, min_segments: int = 20, table_column: Column | None = None) -> None:
        super().__init__(table_column or Column(ratio=1, min_width=min_segments, no_wrap=True))

    def render(self, task) -> Text:
        return Text("[xxx]")


class ProgressStopState(int, Enum):
    """Enum for progress stop states."""

    RUNNING = 0
    """The task is still running."""
    COMPLETED = 1
    """The task has completed successfully."""
    EARLY_STOP = 2
    """The task was stopped early due to an error or interruption."""


class TrackerBarColumn(ProgressColumn):
    """Custom progress bar column that fills all available width."""

    def __init__(
        self,
        remain_style: StyleType = "tracker-bar.remaining",
        complete_style: StyleType = "tracker-bar.complete",
        finished_style: StyleType = "tracker-bar.finished",
        failure_style: StyleType = "tracker-bar.failure",
        head_style: StyleType = "tracker-bar.pulse",
        outer_style: StyleType = "tracker-bar.outer",
        complete_glyph: str = "#",
        pending_glyph: str = "#",
        head_glyph: str = ">",
        min_segments: int = 20,
        table_column: Column | None = None,
    ):
        super().__init__(table_column=table_column or Column(ratio=1, min_width=min_segments, no_wrap=True))
        self.remain_style = remain_style
        self.complete_style = complete_style
        self.finished_style = finished_style
        self.pulse_style = head_style
        self.outer_style = outer_style
        self.complete_glyph = complete_glyph
        self.failure_style = failure_style
        self.pending_glyph = pending_glyph
        self.head_style = head_glyph
        self.min_segments = min_segments

    def _fake_render(self, progress_bar: "NMProgress") -> int | None:
        """
        Render a fake version of the progress bar to calculate width.

        Used to determine the available width for the actual bar.
        """
        table_columns: list[Column] = []
        for column in progress_bar.columns:
            if isinstance(column, ProgressColumn):
                table_columns.append(column.get_table_column().copy())
            else:
                table_columns.append(Column(no_wrap=True))

        table = Table.grid(*table_columns, padding=(0, 1), expand=progress_bar.expand)
        prefer_index = -1
        for task in progress_bar.tasks:
            if task.visible:
                all_columns = []
                for idx, column in enumerate(progress_bar.columns):
                    if isinstance(column, TrackerBarColumn):
                        remade = _FakeTrackerBarColumn(
                            min_segments=self.min_segments,
                            table_column=column.get_table_column().copy(),
                        )
                        prefer_index = idx
                        all_columns.append(remade(task))
                    elif isinstance(column, ProgressColumn):
                        all_columns.append(column(task))
                    elif isinstance(column, str):
                        all_columns.append(column.format(task))
                table.add_row(*all_columns)

        # Get each column width
        console_options = progress_bar.console.options
        measurement = table._calculate_column_widths(progress_bar.console, console_options)
        if prefer_index >= 0:
            task_column_width = measurement[prefer_index] - 3  # account for padding + brackets
            return task_column_width
        return None

    def render(self, task: Union["NMRichTask", "RichTask"]) -> Text:
        """Render the tracker bar for a given task."""
        if not hasattr(self, "progress"):
            raise RuntimeError("`TrackerBarColumn` must be used within a `NMProgress` instance.")

        progress_bar = cast(NMProgress, self.progress)  # pyright: ignore[reportAttributeAccessIssue]
        width = self._fake_render(progress_bar) or self.min_segments

        if not task.total:
            if task.finished:
                bar_text = Text("[", style=self.outer_style)
                bar_text.append(self.complete_glyph * width, style=self.finished_style)
                bar_text.append("]", style=self.outer_style)
                return bar_text
            # Indeterminate state with moving pulse
            pulse_width = max(3, width // 8)
            frame_buffer = math.ceil(pulse_width / 2)
            frame = time.monotonic() * frame_buffer
            pos = frame % width

            bar_text = Text("[", style=self.outer_style)
            for i in range(width):
                dist = (i - pos) % width
                if dist < pulse_width:
                    bar_text.append(self.complete_glyph, style=self.pulse_style)
                else:
                    bar_text.append(self.pending_glyph, style=self.remain_style)
            bar_text.append("]", style=self.outer_style)
            return bar_text

        completed = min(task.total, max(0, task.completed))
        complete_halves = int(width * 2 * completed / task.total)
        bar_count = complete_halves // 2
        half_bar_count = complete_halves % 2
        is_finished = task.completed >= task.total

        complete_style = self.finished_style if is_finished else self.complete_style
        is_runner_stop = False
        if isinstance(task, NMRichTask):
            if task.stop_state == ProgressStopState.EARLY_STOP:
                complete_style = self.failure_style
                is_runner_stop = True

        bar_text = Text("[", style=self.outer_style)
        if bar_count:
            bar_text.append(self.complete_glyph * bar_count, style=complete_style)
        if half_bar_count:
            bar_text.append(self.complete_glyph, style=complete_style)
        remaining_bars = width - bar_count - half_bar_count
        if not is_finished and not is_runner_stop:
            bar_text.append(self.head_style, style=self.pulse_style)
            remaining_bars -= 1
        if remaining_bars:
            if remaining_bars > 0:
                bar_text.append(self.pending_glyph * remaining_bars, style=self.remain_style)
        bar_text.append("]", style=self.outer_style)
        return bar_text


class MaybePercentageColumn(ProgressColumn):
    def render(self, task) -> Text:
        if not task.total:
            return Text("???%", style="progress.percentage")
        return Text(f"{task.percentage:>3.0f}%", style="progress.percentage")


class BetterDescriptionColumn(ProgressColumn):
    def render(self, task: RichTask | NMRichTask) -> Text:
        """Render the description column for a given task."""

        if isinstance(task, NMRichTask):
            return Text(task.finish_or_description, style="progress.description")
        return Text(task.description, style="progress.description")


class BetterSpinnerColumn(SpinnerColumn):
    def __init__(
        self,
        spinner_name: str = "dots",
        style: str | StyleType | None = "progress.spinner",
        speed: float = 1,
        finished_text: str | Text = " ",
        failure_text: str | Text = " ",
        table_column: Column | None = None,
    ):
        super().__init__(spinner_name, style, speed, finished_text, table_column)
        self.failure_text = Text.from_markup(failure_text) if isinstance(failure_text, str) else failure_text

    def render(self, task: RichTask | NMRichTask):
        if isinstance(task, NMRichTask):
            if task.stop_state == ProgressStopState.EARLY_STOP:
                return self.failure_text
        return super().render(task)


@dataclass
class NMRichTask(RichTask):
    finished_text: str | None = None
    """:class:`str`: Custom finished text to display when the task is completed."""
    failure_text: str | None = None
    """:class:`str`: Custom failure text to display when the task is stopped early."""
    stop_state: ProgressStopState = ProgressStopState.RUNNING
    """:class:`bool`: Whether to force the task to be considered complete."""

    @property
    def finish_or_description(self) -> str:
        """:class:`str`: finished text if available and task is completed, else description."""
        if self.failure_text is not None and self.stop_state == ProgressStopState.EARLY_STOP:
            return self.failure_text
        if self.finished_text is not None and self.finished:
            return self.finished_text
        return self.description


class NMProgress(Progress):
    """
    A custom progress bar that supports making custom finish texts.
    """

    def __init__(
        self,
        *columns: str | ProgressColumn,
        console: RichConsole | None = None,
        auto_refresh: bool = True,
        refresh_per_second: float = 10,
        speed_estimate_period: float = 30,
        transient: bool = False,
        redirect_stdout: bool = True,
        redirect_stderr: bool = True,
        get_time: Callable[[], float] | None = None,
        disable: bool = False,
        expand: bool = True,
    ) -> None:
        super().__init__(
            *columns,
            console=console,
            auto_refresh=auto_refresh,
            refresh_per_second=refresh_per_second,
            speed_estimate_period=speed_estimate_period,
            transient=transient,
            redirect_stdout=redirect_stdout,
            redirect_stderr=redirect_stderr,
            get_time=get_time,
            disable=disable,
            expand=expand,
        )

        for column in self.columns:
            if isinstance(column, ProgressColumn):
                # Pass progress reference to column
                column.progress = self  # type: ignore[attr-defined]

    def add_task(
        self,
        description: str,
        start: bool = True,
        total: float | None = 100,
        completed: int = 0,
        visible: bool = True,
        finished_text: str | None = None,
        failure_text: str | None = None,
        **fields: Any,
    ) -> TaskID:
        """Add a new 'task' to the Progress display.

        This version support custom progress system.

        Parameters
        ----------
        description: :class:`str`
            Description of the task.
        start: :class:`bool`, optional
            Whether to start the task immediately. Default to ``True``.
        total: :class:`float` | :class:`None`, optional
            Total number of steps in the task. If ``None``, the task is
            considered to be indeterminate. Default to ``100``.
        completed: :class:`int`, optional
            Number of steps already completed. Default to ``0``.
        visible: :class:`bool`, optional
            Whether the task is visible. Default to ``True``.
        finished_text: :class:`str` | :class:`None`, optional
            Custom finished text to display when the task is completed.
        failure_text: :class:`str` | :class:`None`, optional
            Custom failure text to display when the task is stopped early.
        **fields: Any
            Additional fields for the task.

        Returns
        -------
        :class:`TaskID`
            The ID of the newly created task.
        """

        with self._lock:
            task = NMRichTask(
                id=self._task_index,
                description=description,
                total=total,
                completed=completed,
                visible=visible,
                finished_text=finished_text,
                failure_text=failure_text,
                fields=fields,
                _get_time=self.get_time,
                _lock=self._lock,
            )
            self._tasks[self._task_index] = task
            if start:
                self.start_task(self._task_index)
            new_task_index = self._task_index
            self._task_index = TaskID(int(self._task_index) + 1)
        self.refresh()
        return new_task_index

    def stop(self) -> None:
        """Stop the progress display."""
        # Loop through tasks and mark any indeterminate tasks as finished
        for task in self.tasks:
            if task.total is None and not task.finished:
                task.total = task.completed
                task.finished_time = task.elapsed
            elif task.total is not None and task.finished_time is None:
                task.finished_time = task.elapsed
                if isinstance(task, NMRichTask):
                    task.stop_state = ProgressStopState.EARLY_STOP
        return super().stop()

    def update(
        self,
        task_id: TaskID,
        *,
        total: float | None = None,
        completed: float | None = None,
        advance: float | None = None,
        description: str | None = None,
        visible: bool | None = None,
        refresh: bool = False,
        run_state: ProgressStopState | None = None,
        **fields: Any,
    ) -> None:
        # Update parent
        with self._lock:
            task = self._tasks[task_id]
            completed_start = task.completed

            if total is not None and total != task.total:
                task.total = total
                task._reset()
            if advance is not None:
                task.completed += advance
            if completed is not None:
                task.completed = completed
            if description is not None:
                task.description = description
            if visible is not None:
                task.visible = visible
            task.fields.update(fields)
            update_completed = task.completed - completed_start

            current_time = self.get_time()
            old_sample_time = current_time - self.speed_estimate_period
            progress_deq = task._progress

            popleft = progress_deq.popleft
            while progress_deq and progress_deq[0].timestamp < old_sample_time:
                popleft()
            if update_completed > 0:
                progress_deq.append(ProgressSample(current_time, update_completed))
            if task.total is not None and task.completed >= task.total and task.finished_time is None:
                task.finished_time = task.elapsed
                if isinstance(task, NMRichTask):
                    task.stop_state = ProgressStopState.COMPLETED
            if isinstance(task, NMRichTask) and run_state is not None:
                task.stop_state = run_state  # Override stop state

        if refresh:
            self.refresh()

    @classmethod
    def get_default_columns(cls) -> tuple[ProgressColumn, ...]:
        """
        An override to get the default columns for :class:`NMProgress`.

        This includes formatting like this:

        ```
        ⠸ Simulating work... [######>                          ]   5/100   5% 0:00:00 0:00:06
        ```

        Which utilize:
        - :class:`BetterSpinnerColumn` - spinner at the start with more granular display
        - :class:`BetterDescriptionColumn` - description with finished text support
        - :class:`TrackerBarColumn` - custom progress bar that fills available width
        - :class:`MofNCompleteColumn` - shows completed/total
        - :class:`MaybePercentageColumn` - shows percentage or ??? if indeterminate
        - :class:`TimeElapsedColumn` - time elapsed
        - :class:`TimeRemainingColumn` - estimated time remaining
        """

        return (
            BetterSpinnerColumn(spinner_name="dots", finished_text="[green]✓[/green]", failure_text="[red]✗[/red]"),
            BetterDescriptionColumn(),
            TrackerBarColumn(),
            MofNCompleteColumn(table_column=Column(no_wrap=True, justify="right")),
            MaybePercentageColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

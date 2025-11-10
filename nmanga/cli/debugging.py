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

import logging
import random
from time import sleep

import rich_click as click

from .. import term
from ..common import threaded_worker
from .base import NMangaCommandHandler

console = term.get_console()
logger = logging.getLogger(__name__)


def simulate_thread_progress(log_queue: term.MessageQueue, thread_id: int, task_name: str) -> None:
    console = term.with_thread_queue(log_queue)
    sleep(random.uniform(0.1, 0.5))  # noqa: S311
    console.info(f"Thread {thread_id} starting task: {task_name}")
    logger.info(f"Logger {thread_id} starting task: {task_name}")
    sleep(0.5 + thread_id * 0.2)


def simulate_thread_progress_star(args: tuple[term.MessageQueue, int, str]) -> None:
    return simulate_thread_progress(*args)


def simulate_thread_progress_creation(args: tuple[term.MessageQueue, str]) -> None:
    log_queue, task_name = args
    console = term.with_thread_queue(log_queue)
    sleep(random.uniform(0.1, 0.5))  # noqa: S311

    TOTAL_GEN = 50
    task_name_task = None
    if "-2" in task_name:
        task_name_task = f"{task_name} (done)"
    task_id = console.new_task(task_name, total=TOTAL_GEN, finished_text=task_name_task)
    for _ in range(TOTAL_GEN):
        console.update_progress(task_id, advance=1)
        sleep(0.1)


@click.command(
    name="simulate-progress",
    help="Simulate a progress bar for testing purposes",
    cls=NMangaCommandHandler,
    hidden=True,
)
def simulate_progress():
    """Simulate a progress bar for testing purposes."""

    progress = console.make_progress()
    task = progress.add_task("Simulating work...", total=10)
    for _ in range(10):
        sleep(0.1)
        progress.update(task, advance=1)

    # Try threaded progress
    console.info("Starting threaded progress simulation...")

    tasque = [(i, f"Task-{i}") for i in range(10)]
    task2 = progress.add_task("Threaded work...", total=len(tasque))
    with threaded_worker(console, len(tasque)) as (pool, log_queue):
        for _ in pool.imap_unordered(
            simulate_thread_progress_star,
            [(log_queue, i, task_name) for i, task_name in tasque],
        ):
            progress.update(task2, advance=1)

    console.info("Thread worker progress creation...")
    new_tasque = [(f"New-Task-Thread-{i}") for i in range(5)]
    with threaded_worker(console, len(new_tasque)) as (pool, log_queue):
        for _ in pool.imap_unordered(
            simulate_thread_progress_creation,
            [(log_queue, task_name) for task_name in new_tasque],
        ):
            pass
    console.stop_progress(progress, "Finished simulated progress.")

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

import shutil
import subprocess as sp
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict, Field

from ... import file_handler, term
from ...common import lowest_or, threaded_worker
from ..common import SkipActionKind, perform_skip_action
from ._base import ActionColorMixin, ActionKind, BaseAction, ThreadedResult, ToolsKind, WorkerContext

if TYPE_CHECKING:
    from .. import OrchestratorConfig, VolumeConfig

__all__ = (
    "ActionColorDetect",
    "ActionColorJpegify",
    "ActionMoveColor",
)


class ActionMoveColor(BaseAction, ActionColorMixin):
    """
    Action to move the tagged color images to a separate folder

    Everything would be derived from the main config so they are not included here.
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Move Color Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.MOVE_COLOR] = Field(ActionKind.MOVE_COLOR, title="Move Color Images Action")
    """The kind of action"""
    base_path: str = Field("colors", title="Output Base Path")
    """The base path to save the color images to"""

    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        if context.dry_run:
            context.terminal.info(f"- Base path: {self.base_path}")
            context.terminal.info(f"- Total {len(volume.colors)} would be moved.")
            return

        output_dir = context.root_dir / Path(self.base_path) / Path(volume.path)
        output_dir.mkdir(parents=True, exist_ok=True)

        moved_count = 0
        progress = context.terminal.make_progress()
        task = progress.add_task("Moving color images...", finished_text="Moved color images", total=len(volume.colors))
        for img_file, _, _, _ in file_handler.collect_image_from_folder(context.current_dir):
            pg_num, is_color = self.is_color_page(img_file, context=context, volume=volume, orchestrator=orchestrator)
            if is_color and pg_num is not None:
                dest_path = output_dir / img_file.name
                if dest_path.exists():
                    context.terminal.warning(f"Skipping existing file: {dest_path}")
                    continue
                shutil.move(img_file, dest_path)
                moved_count += 1
            progress.update(task, advance=1)

        context.terminal.stop_progress(progress, f"Moved {moved_count} color images to {output_dir}", skip_total=True)


def _runner_jpegify_threaded(
    log_q: term.MessageOrInterface,
    img_path: Path,
    output_dir: Path,
    cjpegli: str,
    quality: int,
    skip_action: SkipActionKind | None = None,
) -> ThreadedResult:
    cnsl = term.with_thread_queue(log_q)
    if skip_action is not None:
        perform_skip_action(img_path, output_dir, skip_action, cnsl)
        return ThreadedResult.COPIED if skip_action != SkipActionKind.IGNORE else ThreadedResult.IGNORED
    dest_path = output_dir / f"{img_path.stem}.jpg"
    if dest_path.exists():
        cnsl.warning(f"Skipping existing file: {dest_path}")
        return ThreadedResult.COPIED

    cmd = [cjpegli, "-q", str(quality), str(img_path), str(dest_path)]
    sp.run(cmd, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    return ThreadedResult.PROCESSED


def _runner_jpegify_threaded_star(
    args: tuple[term.MessageQueue, Path, Path, str, int, SkipActionKind | None],
) -> ThreadedResult:
    return _runner_jpegify_threaded(*args)


class ActionColorJpegify(BaseAction, ActionColorMixin):
    """
    Action to convert color images to JPEG format with cjpegli
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Jpegify Color Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.COLOR_JPEGIFY] = Field(ActionKind.COLOR_JPEGIFY, title="Jpegify Color Images Action")
    """The kind of action"""
    base_path: str | None = Field(None, title="Output Base Path")
    """The base path to save the JPEG images to, this would use the last used base path if not provided"""
    quality: int = Field(95, ge=1, le=100, title="JPEG Quality")
    """The quality of the output JPEG images"""
    source_path: str = Field("colors", title="Source Path")
    """The source path to look for images to convert"""
    threads: int = Field(default_factory=cpu_count, ge=1, title="Processing Threads")
    """The number of threads to use for processing"""

    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        if context.dry_run:
            context.terminal.info(f"- Source path: {self.source_path}")
            if self.base_path is not None:
                context.terminal.info(f"- Output base path: {self.base_path}")
            context.terminal.info(f"- Total {len(volume.colors)} would be converted to JPEG.")
            return

        output_dir = context.current_dir
        if self.base_path is not None:
            output_dir = context.root_dir / Path(self.base_path) / Path(volume.path)
            output_dir.mkdir(parents=True, exist_ok=True)
        source_dir = context.root_dir / Path(self.source_path) / Path(volume.path)
        if not source_dir.exists():
            context.terminal.warning(f"Source path {source_dir} does not exist or is not a directory, skipping...")
            return
        if not source_dir.is_dir():
            raise RuntimeError("Source path does not exist or is not a directory.")

        cjpegli = context.toolsets.get("cjpegli")
        if cjpegli is None:
            context.terminal.error("cjpegli is required for JPEG conversion, but not found!")
            raise RuntimeError("JPEG conversion failed due to missing cjpegli.")

        image_candidates: list[tuple[Path, SkipActionKind | None]] = []
        for img_path, _, _, _ in file_handler.collect_image_from_folder(source_dir):
            pg_num, is_color = self.is_color_page(img_path, context=context, volume=volume, orchestrator=orchestrator)
            if is_color and pg_num is not None:
                skip_action = None
                if context.skip_action is not None and pg_num in context.skip_action.pages:
                    skip_action = context.skip_action.action
                image_candidates.append((img_path, skip_action))

        if not image_candidates:
            context.terminal.warning(f"No color images found in {source_dir}, skipping...")
            return

        total_images = len(image_candidates)
        quality = max(0, min(100, self.quality))

        progress = context.terminal.make_progress()
        task = progress.add_task("JPEGifying images...", finished_text="JPEGified images", total=total_images)
        results: list[ThreadedResult] = []
        context.terminal.info(f"Using {self.threads} CPU threads for processing.")
        with threaded_worker(context.terminal, lowest_or(self.threads, image_candidates)) as (pool, log_q):
            for result in pool.imap_unordered(
                _runner_jpegify_threaded_star,
                [(log_q, image, output_dir, cjpegli, quality, skip_action) for image, skip_action in image_candidates],
            ):
                results.append(result)
                progress.update(task, advance=1)

        context.terminal.stop_progress(progress, f"Converted {total_images} images to JPEG.")
        jpegified_count = sum(1 for result in results if result == ThreadedResult.PROCESSED)
        copied_count = sum(1 for result in results if result == ThreadedResult.COPIED)
        ignored_count = sum(1 for result in results if result == ThreadedResult.IGNORED)
        if copied_count > 0:
            context.terminal.info(f" Copied {copied_count} colored images without jpeg-ifying.")
        if jpegified_count > 0:
            context.terminal.info(f" JPEG-ified {jpegified_count} color images.")
        if ignored_count > 0:
            context.terminal.info(f" Ignored {ignored_count} colored images.")

    def get_tools(self):
        """
        Get the required tools for the action

        :return: A dictionary of tool names and their kinds
        """

        return {
            "cjpegli": ToolsKind.BINARY,
        }


class ActionColorDetect(BaseAction, ActionColorMixin):
    """
    Action to detect color pages in the volume and store the result in the context for later actions to use
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Detect Color Pages Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.COLOR_DETECT] = Field(ActionKind.COLOR_DETECT, title="Detect Color Pages Action")
    """The kind of action"""

    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        if context.dry_run:
            if volume.colors == "auto":
                context.terminal.info("- Detect color pages automatically")
                context.terminal.info(f"- Color model path: {orchestrator.metafields.color_model}")
            if isinstance(volume.colors, list):
                context.terminal.info(f"- Has color page information from config: {sorted(volume.colors)}")
            return

        if isinstance(volume.colors, list):
            context.push_detected_colors(*volume.colors)
            context.terminal.info(f"Volume has color page information from config: {sorted(volume.colors)}")
            return

        context.terminal.info(f"Detecting color pages in {context.current_dir}...")
        progress = context.terminal.make_progress()
        task = progress.add_task("Detecting color...", finished_text="Detected pages", total=None)
        for img_path, _, total_file, _ in file_handler.collect_image_from_folder(context.current_dir):
            pg_num, is_color = self.is_color_page(img_path, context=context, volume=volume, orchestrator=orchestrator)
            if pg_num is not None and is_color:
                context.push_detected_colors(pg_num)
            progress.update(task, advance=1, total=total_file)

        context.terminal.stop_progress(
            progress, f"Detected {len(context.detected_colors) if context.detected_colors else 0} color pages."
        )

        if context.detected_colors:
            context.terminal.info(f"Detected color pages: {sorted(context.detected_colors)}")

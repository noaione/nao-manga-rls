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

import json
import shutil
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PIL import Image
from pydantic import ConfigDict, Field

from nmanga.utils import secure_filename

from ... import file_handler, term
from ...autolevel import apply_levels, find_local_peak, find_local_peak_legacy, gamma_correction
from ...common import RegexCollection, threaded_worker
from ..common import SkipActionKind, perform_skip_action
from ._base import ActionKind, BaseAction, ThreadedResult, ToolsKind, WorkerContext

if TYPE_CHECKING:
    from .. import OrchestratorConfig, VolumeConfig

__all__ = (
    "ActionAutolevel",
    "ActionLevel",
)


@dataclass
class AutolevelResult:
    image: str
    black_level: int
    white_level: int
    status: ThreadedResult
    force_gray: bool = False

    @staticmethod
    def from_threaded_result(image: str, status: ThreadedResult) -> AutolevelResult:
        return AutolevelResult(image=image, black_level=-1, white_level=-1, status=status)

    def to_json(self) -> dict:
        if self.black_level == -1 and self.white_level == -1:
            return {
                "image": self.image,
                "status": self.status.value,
            }
        return {
            "image": self.image,
            "status": self.status.value,
            "black_level": self.black_level,
            "white_level": self.white_level,
            "force_gray": self.force_gray,
        }


def _runner_autolevel2_threaded(
    log_q: term.MessageOrInterface,
    img_path: Path,
    output_dir: Path,
    action: "ActionAutolevel",
    is_color: bool,
    is_skipped_action: SkipActionKind | None = None,
) -> AutolevelResult:
    cnsl = term.with_thread_queue(log_q)
    if is_skipped_action is not None:
        perform_skip_action(img_path, output_dir, is_skipped_action, cnsl)
        return AutolevelResult.from_threaded_result(
            img_path.name,
            ThreadedResult.COPIED if is_skipped_action != SkipActionKind.IGNORE else ThreadedResult.IGNORED,
        )
    if is_color and action.skip_color:
        perform_skip_action(img_path, output_dir, SkipActionKind.COPY, cnsl)
        return AutolevelResult.from_threaded_result(img_path.name, ThreadedResult.COPIED)

    img = Image.open(img_path)
    if action.legacy_mode:
        black_level, white_level, _ = find_local_peak_legacy(
            img, upper_limit=action.upper_limit, skip_white_peaks=action.skip_white
        )
    else:
        black_level, white_level, _ = find_local_peak(
            img, upper_limit=action.upper_limit, peak_percentage=action.min_peak_pct
        )

    is_black_bad = black_level <= 0
    is_white_bad = white_level >= 255 if not action.skip_white else False

    if (
        (is_black_bad and is_white_bad and not action.skip_white)  # both levels are bad
        or (is_black_bad and action.skip_white)
        or black_level > action.upper_limit
    ):
        dest_path = output_dir / img_path.name
        if not is_color:
            img = img.convert("L")
            img.save(dest_path.with_suffix(".png"), format="PNG")
            img.close()
            return AutolevelResult(img_path.name, black_level, white_level, ThreadedResult.GRAYSCALED)

        img.close()

        if dest_path.exists():
            cnsl.warning(f"Skipping existing file: {dest_path}")
            return AutolevelResult.from_threaded_result(img_path.name, ThreadedResult.COPIED)
        shutil.copy2(img_path, dest_path)
        return AutolevelResult.from_threaded_result(img_path.name, ThreadedResult.COPIED)

    dest_path = output_dir / f"{img_path.stem}.png"
    if dest_path.exists():
        cnsl.warning(f"Skipping existing file: {dest_path}")
        return AutolevelResult.from_threaded_result(img_path.name, ThreadedResult.COPIED)

    if not is_color:
        img = img.convert("L")
    gamma_correct = gamma_correction(black_level)
    adjusted_img = apply_levels(
        img,
        black_point=black_level + action.peak_offset,
        white_point=255 if action.skip_white else white_level,
        gamma=gamma_correct,
    )

    adjusted_img.save(dest_path, format="PNG")
    img.close()
    adjusted_img.close()
    return AutolevelResult(img_path.name, black_level, white_level, ThreadedResult.PROCESSED)


def _runner_autolevel2_threaded_star(
    args: tuple[term.MessageQueue, Path, Path, "ActionAutolevel", bool, SkipActionKind | None],
) -> AutolevelResult:
    return _runner_autolevel2_threaded(*args)


class ActionAutolevel(BaseAction):
    """
    Action to auto level all images in a volume with Pillow

    This version is "smarter" than the CLI version:
    - It would force convert the image to grayscale for half-tones/b&w images before processing
    - All tagged color images will be leveled in keep colorspace mode
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Auto Level Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.AUTOLEVEL] = Field(ActionKind.AUTOLEVEL, title="Auto Level Images Action")
    """The kind of action"""
    base_path: str = Field("leveled", title="Output Base Path")
    """The base path to save the leveled images to"""
    upper_limit: int = Field(60, ge=1, le=255, title="Upper Limit for Peak Finding")
    """The upper limit for finding local peaks in the histogram"""
    peak_offset: int = Field(0, ge=-20, le=20, title="Peak Offset")
    """The offset to add to the found black level peak, can be negative"""
    min_peak_pct: float = Field(0.25, ge=0.0, le=100.0, title="Minimum Pixels Peak Percentage")
    """The minimum percentage of pixels for a peak to be considered valid"""
    skip_white: bool = Field(True, title="Skip White Levels During Peak Finding")
    """Whether to skip white peaks when finding local peaks in the histogram"""
    skip_color: bool = Field(False, title="Skip Color Images")
    """Skip color images when auto leveling, only level half-tones/b&w images"""
    legacy_mode: bool = Field(False, title="Use the initial auto level algorithm")
    """Whether to use the initial auto level algorithm (might or not might be better)"""
    dump_stats: bool = Field(False, title="Dump Leveling Stats")
    """Whether to dump leveling stats to a JSON file"""
    threads: int = Field(default_factory=cpu_count, ge=1, title="Processing Threads")
    """The number of threads to use for processing"""

    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        # Prepare
        output_dir = context.root_dir / Path(self.base_path) / Path(volume.path)

        if context.dry_run:
            context.terminal.info(f"- Output Base Path: {self.base_path}")
            context.terminal.info(f"- Upper Limit for Peak Finding: {self.upper_limit}")
            context.terminal.info(f"- Peak Offset: {self.peak_offset}")
            context.terminal.info(f"- Minimum Pixels Peak Percentage: {self.min_peak_pct}")
            context.terminal.info(f"- Skip White Levels During Peak Finding: {self.skip_white}")
            context.terminal.info(f"- Skip Color Images: {self.skip_color}")
            context.terminal.info(f"- Use Legacy Algorithm: {'Yes' if self.legacy_mode else 'No'}")
            context.terminal.info(f"- Dump Stats: {'Yes' if self.dump_stats else 'No'}")
            context.terminal.info(f"- Processing Threads: {self.threads}")
            context.update_cwd(output_dir)
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        page_re = RegexCollection.page_re()

        context.terminal.info(f"Processing {context.current_dir} with autolevel...")
        all_images = [img for img, _, _, _ in file_handler.collect_image_from_folder(context.current_dir)]
        total_images = len(all_images)
        all_images.sort(key=lambda x: x.stem)

        # Do pre-processing
        images_complete: list[tuple[Path, bool, SkipActionKind | None]] = []
        for image in all_images:
            img_match = page_re.match(image.stem)
            is_color = False
            is_skip_action = None

            if img_match is not None:
                p01 = int(img_match.group("a"))
                is_color = p01 in volume.colors
                if context.skip_action is not None and p01 in context.skip_action.pages:
                    is_skip_action = context.skip_action.action

            images_complete.append((image, is_color, is_skip_action))

        results: list[AutolevelResult] = []
        progress = context.terminal.make_progress()
        task = progress.add_task("Auto-leveling images...", finished_text="Auto-leveled images", total=total_images)
        if self.threads > 1:
            context.terminal.info(f"Using {self.threads} CPU threads for processing.")
            with threaded_worker(context.terminal, self.threads) as (pool, log_q):
                for result in pool.imap_unordered(
                    _runner_autolevel2_threaded_star,
                    [
                        (log_q, image, output_dir, self, is_color, is_skip_action)
                        for image, is_color, is_skip_action in images_complete
                    ],
                ):
                    results.append(result)
                    progress.update(task, advance=1)
        else:
            for image, is_color, is_skip_action in images_complete:
                results.append(
                    _runner_autolevel2_threaded(context.terminal, image, output_dir, self, is_color, is_skip_action)
                )
                progress.update(task, advance=1)

        context.terminal.stop_progress(
            progress, f"Processed {total_images} images with autolevel in {context.current_dir}"
        )
        autolevel_count = sum(1 for res in results if res.status == ThreadedResult.PROCESSED)
        copied_count = sum(1 for res in results if res.status == ThreadedResult.COPIED)
        grayscaled_count = sum(1 for res in results if res.status == ThreadedResult.GRAYSCALED)
        ignored_count = sum(1 for res in results if res.status == ThreadedResult.IGNORED)
        if copied_count > 0:
            context.terminal.info(f" Copied {copied_count} images without autolevel.")
        if autolevel_count > 0:
            context.terminal.info(f" Autoleveled {autolevel_count} images.")
        if grayscaled_count > 0:
            context.terminal.info(f" Grayscaled {grayscaled_count} images.")
        if ignored_count > 0:
            context.terminal.info(f" Ignored {ignored_count} images.")

        # Dump stats if needed
        if self.dump_stats:
            stats_path = output_dir / secure_filename(volume.path + ".autolevel.json")
            with stats_path.open("w", encoding="utf-8") as stats_file:
                json.dump(
                    [res.to_json() for res in results],
                    stats_file,
                    indent=4,
                    ensure_ascii=False,
                )
        # Update CWD
        context.update_cwd(output_dir)

    def get_tools(self):
        """
        Get the required tools for the action

        :return: A dictionary of tool names and their kinds
        """

        return {
            "scipy": ToolsKind.PACKAGE,
            "numpy": ToolsKind.PACKAGE,
        }


def _runner_manuallevel_threaded(
    log_q: term.MessageOrInterface,
    img_path: Path,
    output_dir: Path,
    action: "ActionLevel",
    is_color: bool,
    is_skipped_action: SkipActionKind | None = None,
) -> ThreadedResult:
    cnsl = term.with_thread_queue(log_q)
    if is_skipped_action is not None:
        perform_skip_action(img_path, output_dir, is_skipped_action, cnsl)
        return ThreadedResult.COPIED if is_skipped_action != SkipActionKind.IGNORE else ThreadedResult.IGNORED
    if is_color and action.skip_color:
        perform_skip_action(img_path, output_dir, SkipActionKind.COPY, cnsl)
        return ThreadedResult.COPIED

    img = Image.open(img_path)

    dest_path = output_dir / f"{img_path.stem}.png"
    if dest_path.exists():
        cnsl.warning(f"Skipping existing file: {dest_path}")
        return ThreadedResult.COPIED

    if not is_color:
        img = img.convert("L")
    gamma_correct = gamma_correction(action.black_level)
    adjusted_img = apply_levels(
        img,
        black_point=action.black_level,
        white_point=action.white_level,
        gamma=gamma_correct,
    )

    adjusted_img.save(dest_path, format="PNG")
    img.close()
    adjusted_img.close()
    return ThreadedResult.PROCESSED


def _runner_manualevel_threaded_star(
    args: tuple[term.MessageQueue, Path, Path, "ActionLevel", bool, SkipActionKind | None],
) -> ThreadedResult:
    return _runner_manuallevel_threaded(*args)


class ActionLevel(BaseAction):
    """
    Manually level all images in a volume with Pillow
    """

    kind: Literal[ActionKind.LEVEL] = Field(ActionKind.LEVEL, title="Level Images Action")
    """The kind of action"""
    base_path: str = Field("leveled", title="Output Base Path")
    """The base path to save the leveled images to"""
    black_level: int = Field(ge=0, le=255, title="Black Level")
    """The black level to set for all images"""
    white_level: int = Field(255, ge=0, le=255, title="White Level")
    """The white level to set for all images"""
    skip_color: bool = Field(False, title="Skip Color Images")
    """Skip color images when auto leveling, only level half-tones/b&w images"""
    threads: int = Field(default_factory=cpu_count, ge=1, title="Processing Threads")
    """The number of threads to use for processing"""

    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        # Prepare
        output_dir = context.root_dir / Path(self.base_path) / Path(volume.path)

        if context.dry_run:
            context.terminal.info(f"- Output Base Path: {self.base_path}")
            context.terminal.info(f"- Black Level: {self.black_level}")
            context.terminal.info(f"- White Level: {self.white_level}")
            context.terminal.info(f"- Skip Color Images: {self.skip_color}")
            context.terminal.info(f"- Processing Threads: {self.threads}")
            context.update_cwd(output_dir)
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        page_re = RegexCollection.page_re()

        context.terminal.info(f"Processing {context.current_dir} with level...")
        all_images = [img for img, _, _, _ in file_handler.collect_image_from_folder(context.current_dir)]
        total_images = len(all_images)
        all_images.sort(key=lambda x: x.stem)

        # Do pre-processing
        images_complete: list[tuple[Path, bool, SkipActionKind | None]] = []
        for image in all_images:
            img_match = page_re.match(image.stem)
            is_color = False
            is_skip_action = None

            if img_match is not None:
                p01 = int(img_match.group("a"))
                is_color = p01 in volume.colors
                if context.skip_action is not None and p01 in context.skip_action.pages:
                    is_skip_action = context.skip_action.action

            images_complete.append((image, is_color, is_skip_action))

        results: list[ThreadedResult] = []
        progress = context.terminal.make_progress()
        task = progress.add_task("Leveling images...", finished_text="Leveled images", total=total_images)
        if self.threads > 1:
            context.terminal.info(f"Using {self.threads} CPU threads for processing.")
            with threaded_worker(context.terminal, self.threads) as (pool, log_q):
                for result in pool.imap_unordered(
                    _runner_manualevel_threaded_star,
                    [
                        (log_q, image, output_dir, self, is_color, is_skip_action)
                        for image, is_color, is_skip_action in images_complete
                    ],
                ):
                    results.append(result)
                    progress.update(task, advance=1)
        else:
            for image, is_color, is_skip_action in images_complete:
                results.append(
                    _runner_manuallevel_threaded(context.terminal, image, output_dir, self, is_color, is_skip_action)
                )
                progress.update(task, advance=1)

        context.terminal.stop_progress(progress, f"Processed {total_images} images with level in {context.current_dir}")
        autolevel_count = sum(1 for res in results if res == ThreadedResult.PROCESSED)
        copied_count = sum(1 for res in results if res == ThreadedResult.COPIED)
        grayscaled_count = sum(1 for res in results if res == ThreadedResult.GRAYSCALED)
        ignored_count = sum(1 for res in results if res == ThreadedResult.IGNORED)
        if copied_count > 0:
            context.terminal.info(f" Copied {copied_count} images without level.")
        if autolevel_count > 0:
            context.terminal.info(f" Leveled {autolevel_count} images.")
        if grayscaled_count > 0:
            context.terminal.info(f" Grayscaled {grayscaled_count} images.")
        if ignored_count > 0:
            context.terminal.info(f" Ignored {ignored_count} images.")

        # Update CWD
        context.update_cwd(output_dir)

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
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PIL import Image
from pydantic import ConfigDict, Field

from ... import file_handler
from ...autolevel import apply_levels, find_local_peak, gamma_correction
from ...common import RegexCollection
from ..common import SkipActionKind, perform_skip_action
from ._base import ActionKind, BaseAction, ThreadedResult, ToolsKind, WorkerContext, threaded_worker

if TYPE_CHECKING:
    from ...term import Console
    from .. import OrchestratorConfig, VolumeConfig

__all__ = ("ActionAutolevel",)


def _runner_autolevel2_threaded(
    img_path: Path,
    output_dir: Path,
    action: "ActionAutolevel",
    console: "Console",
    is_color: bool,
    is_skipped_action: SkipActionKind | None = None,
) -> ThreadedResult:
    if is_skipped_action is not None:
        perform_skip_action(img_path, output_dir, is_skipped_action)
        return ThreadedResult.COPIED if is_skipped_action != SkipActionKind.IGNORE else ThreadedResult.IGNORED
    if is_color and action.skip_color:
        perform_skip_action(img_path, output_dir, SkipActionKind.COPY)
        return ThreadedResult.COPIED

    img = Image.open(img_path)
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
            return ThreadedResult.GRAYSCALED

        img.close()

        if dest_path.exists():
            console.warning(f"Skipping existing file: {dest_path}")
            return ThreadedResult.COPIED
        shutil.copy2(img_path, dest_path)
        return ThreadedResult.COPIED

    dest_path = output_dir / f"{img_path.stem}.png"
    if dest_path.exists():
        console.warning(f"Skipping existing file: {dest_path}")
        return ThreadedResult.COPIED

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
    return ThreadedResult.PROCESSED


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
    base_path: Path = Field("leveled", title="Output Base Path")
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
        output_dir = context.root_dir / self.base_path / volume.path
        output_dir.mkdir(parents=True, exist_ok=True)

        page_re = RegexCollection.page_re()
        context.terminal.status("Processing images with autolevel...")

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
        context.terminal.status(f"Auto-leveling {total_images} images...")
        if self.threads > 1:
            context.terminal.info(f"Using {self.threads} CPU threads for processing.")
            with threaded_worker(context.terminal, self.threads) as pool:
                for idx, (image, is_color, is_skip_action) in enumerate(images_complete):
                    # We need to also get the return value here
                    pool.apply_async(
                        _runner_autolevel2_threaded,
                        args=(image, output_dir, self, context.terminal, is_color, is_skip_action),
                        callback=results.append,
                    )
                pool.close()
                pool.join()
        else:
            for idx, (image, is_color, is_skip_action) in enumerate(images_complete):
                context.terminal.status(f"Auto-leveling images... [{idx + 1}/{total_images}]")
                results.append(
                    _runner_autolevel2_threaded(image, output_dir, self, context.terminal, is_color, is_skip_action)
                )

        autolevel_count = sum(1 for result in results if result == ThreadedResult.PROCESSED)
        copied_count = sum(1 for result in results if result == ThreadedResult.COPIED)
        grayscaled_count = sum(1 for result in results if result == ThreadedResult.GRAYSCALED)
        ignored_count = sum(1 for result in results if result == ThreadedResult.IGNORED)
        context.terminal.stop_status(f"Auto-leveled {total_images} images.")
        if copied_count > 0:
            context.terminal.info(f"Copied {copied_count} images without autolevel.")
        if autolevel_count > 0:
            context.terminal.info(f"Autoleveled {autolevel_count} images.")
        if grayscaled_count > 0:
            context.terminal.info(f"Grayscaled {grayscaled_count} images.")
        if ignored_count > 0:
            context.terminal.info(f"Ignored {ignored_count} images.")

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

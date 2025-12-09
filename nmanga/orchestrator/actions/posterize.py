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
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from PIL import Image
from pydantic import ConfigDict, Field

from ... import file_handler, term
from ...autolevel import (
    analyze_gray_shades,
    npow2,
    posterize_image_by_bits,
    posterize_image_with_imagemagick,
)
from ...common import RegexCollection, lowest_or, threaded_worker
from ..common import SkipActionKind, perform_skip_action
from ._base import ActionKind, BaseAction, ThreadedResult, ToolsKind, WorkerContext

if TYPE_CHECKING:
    from .. import OrchestratorConfig, VolumeConfig

__all__ = ("ActionPosterize",)


def _detect_auto_bpc(img: Image.Image, threshold: float) -> int:
    """
    Detect the best bits per channel for posterization automatically.

    :param img: The image to analyze
    :return: The detected bits per channel
    """

    image = img.convert("L")  # Convert to grayscale for analysis
    results = analyze_gray_shades(image, threshold=threshold)

    num_shades = len(results)
    if len(results) <= 1:
        return 1  # If only 1 shade, return 1 bpc

    raw_bpc = math.ceil(math.log2(num_shades))
    bpc = npow2(raw_bpc)

    return bpc


def _runner_posterize_threaded(
    log_q: term.MessageOrInterface,
    img_path: Path,
    output_dir: Path,
    action: "ActionPosterize",
    imagick: str | None = None,
    is_color: bool = False,
    skip_action: SkipActionKind | None = None,
) -> ThreadedResult:
    cnsl = term.with_thread_queue(log_q)
    if skip_action is not None:
        perform_skip_action(img_path, output_dir, skip_action, cnsl)
        return ThreadedResult.COPIED if skip_action != SkipActionKind.IGNORE else ThreadedResult.IGNORED
    if is_color:
        perform_skip_action(img_path, output_dir, SkipActionKind.COPY, cnsl)
        return ThreadedResult.COPIED

    dest_path = output_dir / f"{img_path.stem}.png"
    if dest_path.exists():
        cnsl.warning(f"Skipping existing file: {dest_path}")
        return ThreadedResult.COPIED

    if imagick is not None:
        real_bpc = action.bpc
        if action.bpc == "auto":
            img = Image.open(img_path)
            real_bpc = _detect_auto_bpc(img, threshold=action.threshold)
            img.close()

        if real_bpc == 8:  # If 8 bpc, no need to posterize, kinda useless
            perform_skip_action(img_path, output_dir, SkipActionKind.COPY, cnsl)
            return ThreadedResult.COPIED

        posterize_image_with_imagemagick(
            img_path, output_dir=output_dir, num_bits=cast(int, real_bpc), magick_path=imagick
        )
        return ThreadedResult.PROCESSED
    else:
        img = Image.open(img_path)
        real_bpc = action.bpc
        if action.bpc == "auto":
            real_bpc = _detect_auto_bpc(img, threshold=action.threshold)

        if real_bpc == 8:  # If 8 bpc, no need to posterize, kinda useless
            perform_skip_action(img_path, output_dir, SkipActionKind.COPY, cnsl)
            return ThreadedResult.COPIED

        quant = posterize_image_by_bits(img, num_bits=cast(int, real_bpc))
        quant.save(dest_path, format="PNG")
        img.close()
        quant.close()
        return ThreadedResult.PROCESSED


def _runner_posterize_threaded_star(
    args: tuple[term.MessageQueue, Path, Path, "ActionPosterize", str | None, bool, SkipActionKind | None],
) -> ThreadedResult:
    return _runner_posterize_threaded(*args)


class ActionPosterize(BaseAction):
    """
    Action to posterize all images in a volume with imagemagick or Pillow
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Posterize Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.POSTERIZE] = Field(ActionKind.POSTERIZE, title="Posterize Images Action")
    """The kind of action"""
    base_path: str = Field("posterized", title="Output Base Path")
    """The base path to save the posterized images to"""
    bpc: int | Literal["auto"] = Field(4, ge=1, le=8, title="Bits Per Channel", examples=[1, 2, 4, 8, "auto"])
    """The number of bitdepth to reduce the image to"""
    threshold: float = Field(0.01, ge=0.0, le=1.0, title="Threshold for Auto bitdepth")
    """The threshold to use when detecting bitdepth automatically"""
    pillow: bool = Field(False, title="Use Python Pillow")
    """Whether to use Pillow for posterizing instead of ImageMagick"""
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
            context.terminal.info(f"- Bits Per Channel: {self.bpc}")
            context.terminal.info(f"- Use Pillow: {'Yes' if self.pillow else 'No'}")
            context.terminal.info(f"- Processing Threads: {self.threads}")
            context.update_cwd(output_dir)
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        if not context.current_dir.exists():
            context.terminal.warning(f"Current directory {context.current_dir} does not exist, skipping posterize.")
            context.update_cwd(output_dir)  # We still need to update CWD
            return

        imagick = context.toolsets.get("magick")
        if imagick is None and not self.pillow:
            context.terminal.error("ImageMagick is required for posterizing, but not found!")
            raise RuntimeError("Spreads action failed due to missing ImageMagick.")

        page_re = RegexCollection.page_re()

        context.terminal.info(f"Processing {context.current_dir} with posterizer...")
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
        task = progress.add_task("Posterizing images...", finished_text="Posterized images", total=total_images)
        context.terminal.info(f"Using {self.threads} CPU threads for processing.")
        with threaded_worker(context.terminal, lowest_or(self.threads, images_complete)) as (pool, log_q):
            for result in pool.imap_unordered(
                _runner_posterize_threaded_star,
                [
                    (log_q, image, output_dir, self, imagick, is_color, is_skip_action)
                    for image, is_color, is_skip_action in images_complete
                ],
            ):
                results.append(result)
                progress.update(task, advance=1)

        context.terminal.stop_progress(progress, f"Posterized {total_images} images in {context.current_dir}")
        posterized_count = sum(1 for result in results if result == ThreadedResult.PROCESSED)
        copied_count = sum(1 for result in results if result == ThreadedResult.COPIED)
        ignored_count = sum(1 for result in results if result == ThreadedResult.IGNORED)
        if copied_count > 0:
            context.terminal.info(f" Copied {copied_count} images without posterizing.")
        if posterized_count > 0:
            context.terminal.info(f" Posterized {posterized_count} images.")
        if ignored_count > 0:
            context.terminal.info(f" Ignored {ignored_count} images.")

        # Update CWD
        context.update_cwd(output_dir)

    def get_tools(self):
        """
        Get the required tools for the action

        :return: A dictionary of tool names and their kinds
        """

        if self.pillow:
            return {}

        return {
            "magick": ToolsKind.BINARY,
        }

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
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from PIL import Image
from pydantic import ConfigDict, Field

from ... import file_handler, term
from ...common import RegexCollection, lowest_or, threaded_worker
from ...spreads import SpreadDirection, join_spreads, join_spreads_imagemagick, select_exts
from ._base import ActionKind, BaseAction, ToolsKind, WorkerContext

if TYPE_CHECKING:
    from .. import OrchestratorConfig, VolumeConfig

__all__ = ("ActionSpreads",)


class _SpreadStuff(TypedDict):
    images: list[Path]
    prefix: str
    postfix: str


def _runner_image_spreads_threaded(
    log_q: term.MessageOrInterface,
    current_dir: Path,
    spread_key: str,
    images: _SpreadStuff,
    imagick: str,
    action: "ActionSpreads",
) -> None:
    cnsl = term.with_thread_queue(log_q)
    final_filename = f"{images['prefix']}p{spread_key}{images['postfix']}"
    if len(images["images"]) < 2:
        cnsl.warning(f"Spread {spread_key} has less than 2 images, skipping...")
        return

    if action.pillow:
        # Load all images
        all_img_paths = [pp for pp in images["images"]]
        loaded_images = [Image.open(p) for p in all_img_paths]

        joined_image = join_spreads(loaded_images, action.direction)
        extension = select_exts(all_img_paths)
        if action.output_fmt != "auto":
            extension = f".{action.output_fmt.lower()}"
        final_filename += extension

        joined_image.save(current_dir / final_filename, quality=int(action.quality))

        # Close all images
        for im in loaded_images:
            im.close()
        joined_image.close()
    else:
        temp_filename = join_spreads_imagemagick(
            images=images["images"],
            output_directory=current_dir,
            quality=action.quality,
            direction=action.direction,
            output_format=action.output_fmt,
            magick_path=imagick,
        )

        input_name = current_dir / temp_filename
        input_name.rename(current_dir / final_filename)


def _runner_image_spreads_threaded_star(
    args: tuple[term.MessageQueue, Path, str, _SpreadStuff, str, "ActionSpreads"],
) -> None:
    return _runner_image_spreads_threaded(*args)


class ActionSpreads(BaseAction):
    """
    Action to join spreads of a volume
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Image Spreads Joiner Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.SPREADS] = Field(ActionKind.SPREADS, title="Image Spreads Joiner Action")
    """The kind of action"""
    direction: SpreadDirection = Field(SpreadDirection.LTR, title="Spread Direction")
    """Whether to use reverse mode when joining spreads"""
    quality: float = Field(100.0, ge=1.0, le=100.0, title="Output Quality")
    """The quality of the output images"""
    output_fmt: Literal["auto", "jpg", "png"] = Field("auto", title="Output Format")
    """The output format of the joined images, auto will use the parent extensions"""
    pillow: bool = Field(False, title="Use Python Pillow")
    """Whether to use Pillow for joining spreads instead of ImageMagick"""
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
            context.terminal.info(f"- Spread Direction: {self.direction.value}")
            context.terminal.info(f"- Output Quality: {self.quality}")
            context.terminal.info(f"- Output Format: {self.output_fmt}")
            context.terminal.info(f"- Use Pillow: {'Yes' if self.pillow else 'No'}")
            context.terminal.info(f"- Processing Threads: {self.threads}")
            return

        if not volume.spreads:
            context.terminal.warning(f"No spreads found for volume {volume.number}, skipping spreads action.")
            return

        if not context.current_dir.exists():
            context.terminal.warning(f"Current directory {context.current_dir} does not exist, skipping spread join.")
            return

        imagick = context.toolsets.get("magick")
        if imagick is None and not self.pillow:
            context.terminal.error(
                "ImageMagick 'magick' tool not found in toolsets, cannot proceed with spreads action."
            )
            raise RuntimeError("ImageMagick 'magick' tool not found in toolsets.")

        exported_images: dict[str, _SpreadStuff] = {}
        page_re = RegexCollection.page_re()
        for image_file, _, _, _ in file_handler.collect_image_from_folder(context.current_dir):
            title_match = page_re.match(image_file.stem)
            if title_match is None:
                context.terminal.error(f"Image {image_file} does not match page regex, aborting...")
                raise RuntimeError("Spreads action failed due to regex mismatch.")

            first_part = title_match.group("a")
            second_part = title_match.group("b")
            prefix_text = title_match.group("any")
            postfix_text = title_match.group("anyback")
            if second_part:
                # Already joined spread
                continue
            first_part = int(first_part)
            for spread_start, spread_end in volume.spreads:
                if spread_start <= first_part <= spread_end:
                    spread_key = f"{spread_start:03d}-{spread_end:03d}"
                    if spread_key not in exported_images:
                        exported_images[spread_key] = {
                            "images": [],
                            "prefix": prefix_text or "",
                            "postfix": postfix_text or "",
                        }
                    exported_images[spread_key]["images"].append(image_file)
                    break

        if not exported_images:
            context.terminal.warning(f"No spreads found in {context.current_dir}, skipping...")
            return

        total_match_spreads = len(list(exported_images.keys()))
        context.terminal.info(f"Joining {total_match_spreads} spreads from {context.current_dir}...")

        progress = context.terminal.make_progress()
        task = progress.add_task("Joining spreads...", finished_text="Joined spreads", total=total_match_spreads)
        context.terminal.info(f"Using {self.threads} CPU threads for processing.")
        with threaded_worker(context.terminal, lowest_or(self.threads, exported_images)) as (pool, log_q):
            for _ in pool.imap_unordered(
                _runner_image_spreads_threaded_star,
                [
                    (log_q, context.current_dir, spread, images, cast(str, imagick), self)
                    for spread, images in exported_images.items()
                ],
            ):
                progress.update(task, advance=1)

        task = progress.tasks[task]
        context.terminal.stop_progress(
            progress, f"Joined {task.completed} spreads from {context.current_dir}", skip_total=True
        )

        # Make backup folder here
        backup_dir = context.root_dir / "backup" / context.current_dir.name
        backup_dir.mkdir(exist_ok=True, parents=True)

        context.terminal.info(f"Backing up original images to {backup_dir}...")
        for image_data in exported_images.values():
            for img_path in image_data["images"]:
                shutil.move(img_path, backup_dir / img_path.name)

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

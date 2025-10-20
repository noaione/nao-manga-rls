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

from ... import file_handler
from ...common import RegexCollection, threaded_worker
from ...term import get_console
from ._base import ActionKind, BaseAction, ToolsKind, WorkerContext

if TYPE_CHECKING:
    from .. import OrchestratorConfig, VolumeConfig

__all__ = (
    "ActionColorJpegify",
    "ActionMoveColor",
)


class ActionMoveColor(BaseAction):
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

        cmx_re = RegexCollection.cmx_re()
        output_dir = context.root_dir / Path(self.base_path) / Path(volume.path)
        output_dir.mkdir(parents=True, exist_ok=True)

        moved_count = 0
        for img_file, _, total_img, _ in file_handler.collect_image_from_folder(context.current_dir):
            title_match = cmx_re.match(img_file.stem)
            if title_match is None:
                context.terminal.error(f"Image {img_file} does not match page regex, aborting...")
                continue

            context.terminal.status(f"Moving color images... [{moved_count}/{total_img}]")
            p01 = int(title_match.group("a"))
            if p01 in volume.colors:
                dest_path = output_dir / img_file.name
                if dest_path.exists():
                    context.terminal.warning(f"Skipping existing file: {dest_path}")
                    continue
                shutil.move(img_file, dest_path)
                moved_count += 1

        context.terminal.stop_status(f"Moved {moved_count} color images to {output_dir}")


def _runner_jpegify_threaded(
    img_path: Path,
    output_dir: Path,
    cjpegli: str,
    quality: int,
) -> None:
    console = get_console()
    dest_path = output_dir / f"{img_path.stem}.jpg"
    if dest_path.exists():
        console.warning(f"Skipping existing file: {dest_path}")
        return

    cmd = [cjpegli, "-q", str(quality), str(img_path), str(dest_path)]
    sp.run(cmd, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)


class ActionColorJpegify(BaseAction):
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
        if not source_dir.exists() or not source_dir.is_dir():
            context.terminal.warning(f"Source path {source_dir} does not exist or is not a directory, skipping...")
            raise RuntimeError("Source path does not exist or is not a directory.")

        cjpegli = context.toolsets.get("cjpegli")
        if cjpegli is None:
            context.terminal.error("cjpegli is required for JPEG conversion, but not found!")
            raise RuntimeError("JPEG conversion failed due to missing cjpegli.")

        page_re = RegexCollection.page_re()

        image_candidates: list[Path] = []
        for img_path, _, _, _ in file_handler.collect_image_from_folder(source_dir):
            title_match = page_re.match(img_path.stem)
            if title_match is None:
                context.terminal.warn(f"Image {img_path} does not match page regex, ignoring...")
                continue
            p01 = int(title_match.group("a"))
            if p01 in volume.colors:
                image_candidates.append(img_path)

        if not image_candidates:
            context.terminal.warning(f"No color images found in {source_dir}, skipping...")
            return

        total_images = len(image_candidates)
        quality = max(0, min(100, self.quality))

        context.terminal.status(f"Converting {total_images} images to JPEG with cjpegli...")
        if self.threads > 1:
            context.terminal.info(f"Using {self.threads} CPU threads for processing.")
            with threaded_worker(context.terminal, self.threads) as pool:
                pool.starmap(
                    _runner_jpegify_threaded,
                    [(image, output_dir, cjpegli, quality) for image in image_candidates],
                )
        else:
            for idx, image in enumerate(image_candidates):
                context.terminal.status(f"Converting images to JPEG... [{idx + 1}/{total_images}]")
                _runner_jpegify_threaded(image, output_dir, cjpegli, quality)

        context.terminal.stop_status(f"Converted {total_images} images to JPEG in {output_dir}")

    def get_tools(self):
        """
        Get the required tools for the action

        :return: A dictionary of tool names and their kinds
        """

        return {
            "cjpegli": ToolsKind.BINARY,
        }

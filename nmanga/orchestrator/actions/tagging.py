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

import subprocess as sp
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict, Field

from nmanga import file_handler

from ... import term
from ...common import (
    ALLOWED_TAG_EXTENSIONS,
    format_archive_filename,
    format_volume_text,
    make_metadata_command,
    threaded_worker,
)
from ._base import ActionKind, BaseAction, ToolsKind, WorkerContext

if TYPE_CHECKING:
    from .. import OrchestratorConfig, VolumeConfig

__all__ = ("ActionTagging",)


def _runner_tagging_threaded(
    exiftool_exe: str,
    image_path: Path,
    archive_filename: str,
    rls_email: str,
) -> None:
    """Threaded helper for tagging images"""
    ext = image_path.suffix.lower().lstrip(".")
    if ext not in ALLOWED_TAG_EXTENSIONS:
        cnsl = term.get_console()
        cnsl.warning(f"Skipping unsupported image format for tagging: {image_path.name}")
        return
    base_cmd = make_metadata_command(exiftool_exe, archive_filename, rls_email)
    proc = sp.Popen(
        [*base_cmd, str(image_path)],
        stdout=sp.PIPE,
        stderr=sp.PIPE,
    )
    proc.wait()


def _runner_tagging_threaded_star(args: tuple[str, Path, str, str]) -> None:
    return _runner_tagging_threaded(*args)


class ActionTagging(BaseAction):
    """
    Action to add metadata tags to the images with exiftool

    Everything would be derived from the main config so they are not included here.
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Image Tagging Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.TAGGING] = Field(ActionKind.TAGGING, title="Image Tagging Action")
    """The kind of action"""
    title: str | None = Field(None, title="Override Title")
    """Override the title metadata"""
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
            context.terminal.info(f"- Title Override: {self.title if self.title else 'None'}")
            return

        exiftool = context.toolsets.get("exiftool")
        if exiftool is None:
            context.terminal.error("exiftool is required for image tagging, but not found!")
            raise RuntimeError("exiftool not found in toolsets")

        volume_text = format_volume_text(manga_volume=volume.number)

        archive_filename = format_archive_filename(
            manga_title=self.title or orchestrator.title,
            manga_year=volume.year,
            publication_type=volume.publication,
            ripper_credit=orchestrator.credit,
            bracket_type=orchestrator.bracket_type,
            manga_volume_text=volume_text if not volume.oneshot else None,
            extra_metadata=volume.extra_text,
            rls_revision=volume.revision,
        )

        all_images = [img for img, _, _, _ in file_handler.collect_image_from_folder(context.current_dir)]
        total_images = len(all_images)
        all_images.sort(key=lambda x: x.stem)

        context.terminal.info(f"Tagging images in {context.current_dir} with exif metadata...")

        progress = context.terminal.make_progress()
        task = progress.add_task("Tagging images...", total=total_images)
        if self.threads > 1:
            context.terminal.info(f"Using {self.threads} CPU threads for processing.")
            with threaded_worker(context.terminal, self.threads) as pool:
                for _ in pool.imap_unordered(
                    _runner_tagging_threaded_star,
                    [(exiftool, image, archive_filename, orchestrator.email) for image in all_images],
                ):
                    progress.update(task, advance=1)
        else:
            for image in all_images:
                _runner_tagging_threaded(exiftool, image, archive_filename, orchestrator.email)
                progress.update(task, advance=1)

        context.terminal.stop_progress(progress, f"Finished tagging images in {context.current_dir}")

    def get_tools(self):
        """
        Get the required tools for the action

        :return: A dictionary of tool names and their kinds
        """

        return {
            "exiftool": ToolsKind.BINARY,
        }

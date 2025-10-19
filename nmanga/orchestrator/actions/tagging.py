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

from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict, Field

from ...common import format_archive_filename, format_volume_text, inject_metadata
from ._base import ActionKind, BaseAction, ToolsKind, WorkerContext

if TYPE_CHECKING:
    from .. import OrchestratorConfig, VolumeConfig

__all__ = ("ActionTagging",)


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
        context.terminal.status(f"Tagging images in {context.current_dir} with exif metadata...")
        inject_metadata(exiftool, context.current_dir, archive_filename, orchestrator.email)

    def get_tools(self):
        """
        Get the required tools for the action

        :return: A dictionary of tool names and their kinds
        """

        return {
            "exiftool": ToolsKind.BINARY,
        }

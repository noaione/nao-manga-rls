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

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict, Field

from ... import exporter, file_handler
from ...common import format_archive_filename, format_volume_text
from ._base import ActionKind, BaseAction, WorkerContext

if TYPE_CHECKING:
    from .. import OrchestratorConfig, VolumeConfig

__all__ = ("ActionPack",)


class ActionPack(BaseAction):
    """
    Action to pack the volume into an archive
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Pack Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.PACK] = Field(ActionKind.PACK, title="Pack Action")
    """The kind of action"""
    output_mode: exporter.ExporterType = Field(exporter.ExporterType.cbz, title="Output Mode")
    """The output mode to use for packing the archive"""
    source_dir: Path | None = Field(None, title="Source Directory")
    """The source directory to pack, this would use the last used base path if not provided"""
    compress_level: int = Field(7, title="Compression Level", ge=0, le=9)
    """The compression level to use for packing the archive, from 0 (no compression) to 9 (maximum compression)"""

    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        if context.dry_run:
            context.terminal.info(f"- Output Mode: {self.output_mode.value}")
            context.terminal.info(
                f"- Source Directory: {self.source_dir if self.source_dir else 'Last used base path'}"
            )
            context.terminal.info(f"- Compression Level: {self.compress_level}")
            return

        source_dir = (self.source_dir or context.current_dir).resolve()
        volume_text = format_volume_text(manga_volume=volume.number)

        context.terminal.info(f"Packing volume {volume.number} in {source_dir}...")
        archive_filename = format_archive_filename(
            manga_title=orchestrator.title,
            manga_year=volume.year,
            publication_type=volume.publication,
            ripper_credit=orchestrator.credit,
            bracket_type=orchestrator.bracket_type,
            manga_volume_text=volume_text if not volume.oneshot else None,
            extra_metadata=volume.extra_text,
            rls_revision=volume.revision,
        )
        parent_dir = source_dir.parent
        arc_target = exporter.exporter_factory(
            archive_filename,
            parent_dir,
            mode=self.output_mode,
            compression_level=self.compress_level,
            manga_title=orchestrator.title,
        )
        if self.output_mode == exporter.ExporterType.epub:
            context.terminal.warning("Packing as EPUB, this will be a slower operation because of size checking!")

        precollected_images = [file for file, _, _, _ in file_handler.collect_image_from_folder(source_dir)]
        precollected_images.sort(key=lambda x: x.name)

        arc_target.set_comment(orchestrator.email)
        progress = context.terminal.make_progress()
        task = progress.add_task("Packing...", finished_text="Packed", total=len(precollected_images))
        for img_file in precollected_images:
            arc_target.add_image(img_file.name, img_file)
            progress.update(task, advance=1)
        context.terminal.stop_progress(progress, f"Finished packing volume {volume.number} to {archive_filename}")
        arc_target.close()

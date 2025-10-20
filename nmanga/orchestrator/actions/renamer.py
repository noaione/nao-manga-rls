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

from ... import file_handler
from ...common import ChapterRange, RegexCollection, format_daiz_like_filename, format_volume_text
from ._base import ActionKind, BaseAction, WorkerContext

if TYPE_CHECKING:
    from .. import OrchestratorConfig, VolumeConfig


__all__ = (
    "ActionRename",
    "ActionShiftName",
)


class ActionShiftName(BaseAction):
    """
    Action to shift rename of a volume

    Title and volume are derived from the main config so they are not included here.
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Simple Shift Renamer Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.SHIFT_RENAME] = Field(ActionKind.SHIFT_RENAME, title="Simple Shift Renamer Action")
    """The kind of action"""
    start: int = Field(0, ge=0, title="Starting Index")
    """The starting index to rename the files to"""
    title: str | None = Field(None, title="Title of the Series")
    """Optional override title for the shift rename action"""

    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        if context.dry_run:
            context.terminal.info(f"- Starting Index: {self.start}")
            context.terminal.info(f"- Title Override: {self.title if self.title else 'None'}")
            return

        all_images: list[Path] = []
        for image_file, _, _, _ in file_handler.collect_image_from_folder(context.current_dir):
            all_images.append(image_file.resolve())

        all_images.sort(key=lambda x: x.stem)

        volume_text = format_volume_text(manga_volume=volume.number, manga_chapter=None)
        total_files = len(all_images) + self.start
        padding = max(3, len(str(total_files + self.start - 1)))
        context.terminal.status(f"Renaming {len(all_images)} images in {context.current_dir}...")
        remapped_names = set()
        should_revert = False
        total_rename = 0

        manga_title = self.title or orchestrator.title
        for idx, image_path in enumerate(all_images):
            context.terminal.status(f"Renaming images [{idx + 1}/{total_files}]...")
            new_name = f"{manga_title} - {volume_text} - p{str(self.start + idx).zfill(padding)}"

            img_suffix = image_path.suffix.lower()
            new_path = image_path.with_name(new_name).with_suffix(img_suffix)
            if new_path.name == image_path.name:
                continue
            remapped_names.add((image_path, new_path))
            if new_path.exists():
                context.terminal.warning("Conflict detected, reverting all changes...")
                context.terminal.log(f"Conflicting file: {new_path}")
                should_revert = True
                break
            image_path.rename(new_path)
            total_rename += 1

        if should_revert:
            context.terminal.stop_status(f"Renamed {total_rename} images, reverting...")
            context.terminal.enter()
            context.terminal.status("Reverting all changes...")
            for old_path, new_path in remapped_names:
                if new_path.exists():
                    new_path.rename(old_path)
            context.terminal.stop_status("Reverted all changes.")
            raise RuntimeError("Shift rename action failed due to filename conflicts.")


class ActionRename(BaseAction):
    """
    Action to rename all images in a volume

    This use the daiz-like renaming scheme.

    All the other options are derived from the main config so they are not included here.
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Daiz-like Rename Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.RENAME] = Field(ActionKind.RENAME, title="Daiz-like Rename Images Action")
    """The kind of action"""
    title: str | None = Field(None, title="Title of the Series")
    """Override title for the rename action"""
    volume_filename: bool = Field(False, title="Use Volume in Filename")
    """Use volume in filename, ignoring the volume number from the volume config"""

    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        if context.dry_run:
            context.terminal.info(f"- Title Override: {self.title if self.title else 'None'}")
            context.terminal.info(f"- Total {len(volume.chapters)} chapters would be used for renaming.")
            return

        cmx_re = RegexCollection.cmx_re()

        packing_extra: dict[int, list[ChapterRange]] = {}
        all_chapters = volume.to_chapter_ranges()
        for chapter in all_chapters:
            if chapter.base not in packing_extra:
                packing_extra[chapter.base] = []
            packing_extra[chapter.base].append(chapter)

        # Make into chapter range
        meta_namings = volume.meta_name_maps
        renaming_maps: dict[str, Path] = {}  # This is new name -> old path (yeah)
        for image, _, _, _ in file_handler.collect_image_from_folder(context.current_dir):
            # console.status(f"Renaming with daiz-like format: {image.name}/{total_img}...")
            title_match = cmx_re.match(image.name)
            if title_match is None:
                context.terminal.error(f"Image {image} does not match regex, aborting...")
                raise RuntimeError("Rename action failed due to regex mismatch.")

            page_num = title_match.group("a")
            page_num_int = int(title_match.group("a"))
            page_back = title_match.group("b")
            if page_back is not None:
                page_num = f"{page_num}-{page_back}"

            vol_str = title_match.group("vol")
            vol_str_ex = title_match.group("volex")
            vol_actual = volume.number if not volume.oneshot else None
            if not volume.oneshot:
                if vol_str is not None:
                    if vol_str.startswith("v"):
                        vol_str = vol_str[1:]
                    vol_actual = int(vol_str)

            if vol_actual is not None and vol_str_ex is not None:
                if vol_str_ex.startswith("."):
                    vol_str_ex = vol_str_ex[1:]
                vol_actual = float(f"{vol_actual}.{int(vol_str_ex)}")

            selected_range: ChapterRange | None = None
            for chapter in all_chapters:
                if chapter.is_single:
                    if page_num_int >= chapter.range[0]:
                        selected_range = chapter
                        break
                else:
                    if page_num_int in chapter.range:
                        selected_range = chapter
                        break

            if selected_range is None:
                context.terminal.error(f"Image {image} does not match any chapter range, aborting...")
                raise RuntimeError("Rename action failed due to chapter range mismatch.")

            extra_name: str | None = None
            if page_num_int in meta_namings:
                extra_name = meta_namings[page_num_int]

            image_filename, _ = format_daiz_like_filename(
                manga_title=self.title or orchestrator.title,
                manga_publisher=orchestrator.publisher,
                manga_year=volume.year,
                chapter_info=selected_range,
                page_number=page_num,
                publication_type=volume.publication,
                ripper_credit=orchestrator.credit,
                bracket_type=orchestrator.bracket_type,
                manga_volume=vol_actual,
                extra_metadata=extra_name,
                image_quality=volume.quality,
                rls_revision=volume.revision,
                chapter_extra_maps=packing_extra,
                extra_archive_metadata=volume.extra_text,
                fallback_volume_name="OShot",
            )
            renaming_maps[image_filename] = image.resolve()

        if not renaming_maps:
            context.terminal.warning(f"No images found in {context.current_dir}, skipping...")
            return

        # Check for conflict
        unique_names = set()
        for new_name in renaming_maps.keys():
            if new_name in unique_names:
                context.terminal.error(f"Conflict detected: {new_name} would be duplicated, aborting...")
                raise RuntimeError("Rename action failed due to filename conflicts.")
            unique_names.add(new_name)

        total_img = len(renaming_maps)
        context.terminal.status(f"Renaming {total_img} images in {context.current_dir}...")
        for idx, (new_name, old_path) in enumerate(renaming_maps.items()):
            context.terminal.status(f"Renaming images: {idx + 1}/{total_img} ({new_name})...")
            new_path = old_path.with_stem(new_name)
            if new_path.name == old_path.name:
                continue
            old_path.rename(new_path)
        context.terminal.stop_status(f"Renamed {total_img} images.")

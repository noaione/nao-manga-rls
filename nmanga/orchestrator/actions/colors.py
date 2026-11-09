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

import re
import shutil
import subprocess as sp
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict, Field

from ... import file_handler, term
from ...common import default_cpu_threads, lowest_or, threaded_worker
from ...lazy import get_vapoursynth
from ...ogsov import DetectedColor
from ...vapour import vs_analyze_ogsov_frame, vs_attach_logger, vs_iter_frames_async, vs_prepare_image_bulk_raw
from ..common import ColorDetectionKind, SkipActionKind, perform_skip_action
from ._base import ActionColorMixin, ActionKind, BaseAction, ThreadedResult, ToolsKind, WorkerContext

if TYPE_CHECKING:
    from ..models import OrchestratorConfig, VolumeConfig

__all__ = (
    "ActionColorDetect",
    "ActionColorJpegify",
    "ActionMoveColor",
)


# A chapter of raw frames is tens of megabytes each, so the number in flight stays bounded.
_VS_WINDOW_MAX = 16


def _vapoursynth_method(orchestrator: "OrchestratorConfig") -> Literal["ogsov", "ogsox"]:
    """
    Pick the `mangacolor` node that classifies the pages.

    The plugin loads its own model, so the configured color model only names the kernel:
    `ogsox` needs the classifier the config already validates, everything else is `ogsov`.

    :param orchestrator: The orchestrator configuration
    :return: The plugin method name
    """

    color_model = orchestrator.metafields.color_model
    if color_model is not None and color_model.kind == ColorDetectionKind.OGSOX:
        return "ogsox"
    return "ogsov"


def _page_images(directory: Path, page_re: re.Pattern[str]) -> list[tuple[Path, int]]:
    """
    Collect the images whose stem carries a page number, with that number.

    :param directory: The directory to read
    :param page_re: The page number regex
    :return: A list of image paths and their page numbers
    """

    pages: list[tuple[Path, int]] = []
    for img_path, _, _, _ in file_handler.collect_image_from_folder(directory):
        title_match = page_re.match(img_path.stem)
        if title_match is not None:
            pages.append((img_path, int(title_match.group("a"))))
    return pages


def _iter_vapoursynth_color_pages(
    pages: Sequence[tuple[Path, int]],
    *,
    method: Literal["ogsov", "ogsox"],
    prefetch: int,
    window: int,
    debug: bool,
) -> Iterator[tuple[Path, int, DetectedColor]]:
    """
    Classify `pages` with the `mangacolor` plugin, one bulk clip for the whole list.

    Frames come back in completion order and carry their own source path, so the page
    number is looked up by name instead of by position.

    :param pages: The images to classify and their page numbers
    :param method: The `mangacolor` method to run
    :param prefetch: How many frames the decoder reads ahead
    :param window: How many frames stay in flight
    :param debug: Whether to log the plugin's messages
    :return: An iterator of the image path, page number and detection result
    """

    vs = get_vapoursynth()
    page_by_name = {path.name: page for path, page in pages}
    clip = vs_prepare_image_bulk_raw([path for path, _ in pages], debug=debug, prefetch=prefetch)
    if method == "ogsox":
        analyzed = vs.core.mangacolor.OGSOX(clip, debug=debug)
    else:
        analyzed = vs.core.mangacolor.OGSOV(clip, debug=debug)

    for _, frame in vs_iter_frames_async(analyzed, window=window):
        detected, img_path = vs_analyze_ogsov_frame(frame)
        yield img_path, page_by_name[img_path.name], detected


def _known_color_pages(volume: "VolumeConfig", context: WorkerContext) -> set[int] | None:
    """
    The color pages already decided for a volume, or `None` when they need detecting.

    :param volume: The volume configuration
    :param context: The worker context
    :return: The page numbers, or `None` when no earlier step decided them
    """

    if isinstance(volume.colors, list):
        return set(volume.colors)
    if isinstance(context.detected_colors, list):
        return set(context.detected_colors)
    return None


def _move_color_page(img_file: Path, output_dir: Path, context: WorkerContext) -> bool:
    """
    Move one page into `output_dir`, refusing to overwrite it.

    :param img_file: The page to move
    :param output_dir: The directory to move it into
    :param context: The worker context
    :return: Whether the page was moved
    """

    dest_path = output_dir / img_file.name
    if dest_path.exists():
        context.terminal.warning(f"Skipping existing file: {dest_path}")
        return False
    shutil.move(img_file, dest_path)
    return True


def _color_kernel_tools(orchestrator: "OrchestratorConfig") -> dict[str, ToolsKind]:
    """
    The packages the configured color detection kernel imports.

    Only the ML kernels need anything beyond Pillow: OGSOV runs torch and scipy, OGSOX
    runs ONNX Runtime. A volume that lists its color pages needs neither.

    :param orchestrator: The orchestrator configuration
    :return: A dictionary of package names and their kinds
    """

    color_model = orchestrator.metafields.color_model
    if color_model is None:
        return {}
    if color_model.kind == ColorDetectionKind.OGSOX:
        return {
            "onnxruntime": ToolsKind.PACKAGE,
            "numpy": ToolsKind.PACKAGE,
        }
    return {
        "numpy": ToolsKind.PACKAGE,
        "scipy": ToolsKind.PACKAGE,
        "torch": ToolsKind.PACKAGE,
    }


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
    mode: Literal["auto", "vapoursynth"] = Field("auto", title="Detection Method")
    """
    How the color pages are found:
    - auto: ask the configured color model per page, or Pillow when there is none
    - vapoursynth: classify the whole volume with the `mangacolor` plugin in one pass
    """
    prefetch: int = Field(
        16,
        ge=0,
        title="VapourSynth Decoder Prefetch",
        description="How many VapourSynth frames to decode ahead of the one being processed, 0 disables it",
    )
    """How many VapourSynth frames to decode ahead of the one being processed, only used by the `vapoursynth` mode"""
    threads: int = Field(default_factory=default_cpu_threads, ge=1, title="Processing Threads")
    """How many VapourSynth frames to keep in flight, only used by the `vapoursynth` mode"""

    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        if context.dry_run:
            context.terminal.info(f"- Base path: {self.base_path}")
            context.terminal.info(f"- Detection Method: {self.mode}")
            if self.mode == "vapoursynth":
                context.terminal.info(f"- VapourSynth Decoder Prefetch: {self.prefetch}")
                context.terminal.info(f"- Processing Threads: {self.threads}")
            if isinstance(volume.colors, list):
                context.terminal.info(f"- Total {len(volume.colors)} would be moved.")
            else:
                context.terminal.info("- Color pages would be detected on the fly")
            return

        output_dir = context.root_dir / Path(self.base_path) / Path(volume.path)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.mode == "vapoursynth":
            self._run_vapoursynth(context, volume, orchestrator, output_dir)
            return

        moved_count = 0
        progress = context.terminal.make_progress()
        task = progress.add_task("Moving color images...", finished_text="Moved color images", total=None)
        for img_file, _, total_img, _ in file_handler.collect_image_from_folder(context.current_dir):
            pg_num, is_color = self.is_color_page(img_file, context=context, volume=volume, orchestrator=orchestrator)
            if is_color and pg_num is not None and _move_color_page(img_file, output_dir, context):
                moved_count += 1
            progress.update(task, advance=1, total=total_img)

        context.terminal.stop_progress(progress, f"Moved {moved_count} color images to {output_dir}", skip_total=True)

    def _run_vapoursynth(
        self,
        context: WorkerContext,
        volume: "VolumeConfig",
        orchestrator: "OrchestratorConfig",
        output_dir: Path,
    ) -> None:
        """
        Move the color pages found by one `mangacolor` pass over the volume.

        A chain that already ran `color_detect` hands the verdict over in the context, and
        an explicit page list in the volume wins over both, so the folder is classified at
        most once.

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        :param output_dir: The directory the color images are moved to
        """

        color_pages = _known_color_pages(volume, context)
        progress = context.terminal.make_progress()
        task = progress.add_task("Moving color images...", finished_text="Moved color images", total=None)

        if color_pages is None:
            pages = _page_images(context.current_dir, self._page_re)
            if not pages:
                context.terminal.warning(
                    f"No page-numbered images found in {context.current_dir}, skipping color detection."
                )
                context.terminal.stop_progress(progress)
                return

            vs_attach_logger(context.terminal)
            context.terminal.info(f"Detecting color pages in {context.current_dir} (VapourSynth)...")
            color_pages = set()
            for _, page, detected in _iter_vapoursynth_color_pages(
                pages,
                method=_vapoursynth_method(orchestrator),
                prefetch=self.prefetch,
                window=max(2, min(self.threads, _VS_WINDOW_MAX)),
                debug=context.terminal.debugged,
            ):
                if detected.is_color:
                    color_pages.add(page)
                    context.push_detected_colors(page)
            context.terminal.info(f"Detected {len(color_pages)} color pages.")

        moved_count = 0
        for img_file, _, total_img, _ in file_handler.collect_image_from_folder(context.current_dir):
            title_match = self._page_re.match(img_file.stem)
            if title_match is not None and int(title_match.group("a")) in color_pages:
                if _move_color_page(img_file, output_dir, context):
                    moved_count += 1
            progress.update(task, advance=1, total=total_img)

        context.terminal.stop_progress(progress, f"Moved {moved_count} color images to {output_dir}", skip_total=True)

    def get_tools(self, orchestrator: "OrchestratorConfig"):
        """
        Get the required tools for the action

        :param orchestrator: The orchestrator configuration
        :return: A dictionary of tool names and their kinds
        """

        if self.mode == "vapoursynth":
            return {
                "vapoursynth": ToolsKind.PACKAGE,
                "xyz.n4o.mangacolor": ToolsKind.VAPOURSYNTH,
                "xyz.n4o.imgseqs": ToolsKind.VAPOURSYNTH,
            }
        return _color_kernel_tools(orchestrator)


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
    threads: int = Field(default_factory=default_cpu_threads, ge=1, title="Processing Threads")
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
            if isinstance(volume.colors, list):
                context.terminal.info(f"- Total {len(volume.colors)} would be converted to JPEG.")
            else:
                context.terminal.info("- Color pages would be detected on the fly")
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

    def get_tools(self, orchestrator: "OrchestratorConfig"):
        """
        Get the required tools for the action

        :param orchestrator: The orchestrator configuration
        :return: A dictionary of tool names and their kinds
        """

        return {
            "cjpegli": ToolsKind.BINARY,
            **_color_kernel_tools(orchestrator),
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
    mode: Literal["auto", "vapoursynth"] = Field("auto", title="Detection Method")
    """
    How the color pages are found:
    - auto: ask the configured color model per page, or Pillow when there is none
    - vapoursynth: classify the whole volume with the `mangacolor` plugin in one pass
    """
    prefetch: int = Field(
        16,
        ge=0,
        title="VapourSynth Decoder Prefetch",
        description="How many VapourSynth frames to decode ahead of the one being processed, 0 disables it",
    )
    """How many VapourSynth frames to decode ahead of the one being processed, only used by the `vapoursynth` mode"""
    threads: int = Field(default_factory=default_cpu_threads, ge=1, title="Processing Threads")
    """How many VapourSynth frames to keep in flight, only used by the `vapoursynth` mode"""

    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        if context.dry_run:
            context.terminal.info(f"- Detection Method: {self.mode}")
            if self.mode == "vapoursynth":
                context.terminal.info(f"- VapourSynth Decoder Prefetch: {self.prefetch}")
                context.terminal.info(f"- Processing Threads: {self.threads}")
            if volume.colors == "auto":
                context.terminal.info("- Detect color pages automatically")
                context.terminal.info(f"- Color model path: {orchestrator.metafields.color_model}")
            elif isinstance(volume.colors, list) and volume.colors:
                context.terminal.info(f"- Has color page information from config: {sorted(volume.colors)}")
            else:
                context.terminal.info("- No color page information from config, would detect automatically")
            return

        if isinstance(volume.colors, list):
            context.push_detected_colors(*volume.colors)
            context.terminal.info(f"Volume has color page information from config: {sorted(volume.colors)}")
            return

        if self.mode == "vapoursynth":
            self._run_vapoursynth(context, orchestrator)
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

    def _run_vapoursynth(self, context: WorkerContext, orchestrator: "OrchestratorConfig") -> None:
        """
        Classify the whole volume with the `mangacolor` plugin and cache the result.

        The pages are pushed into the context, so the move and jpegify actions read the
        verdict instead of detecting every page again.

        :param context: The worker context
        :param orchestrator: The orchestrator configuration
        """

        pages = _page_images(context.current_dir, self._page_re)
        if not pages:
            context.terminal.warning(
                f"No page-numbered images found in {context.current_dir}, skipping color detection."
            )
            return

        vs_attach_logger(context.terminal)
        context.terminal.info(f"Detecting color pages in {context.current_dir} (VapourSynth)...")
        progress = context.terminal.make_progress()
        task = progress.add_task("Detecting color...", finished_text="Detected pages", total=len(pages))
        for _, page, detected in _iter_vapoursynth_color_pages(
            pages,
            method=_vapoursynth_method(orchestrator),
            prefetch=self.prefetch,
            window=max(2, min(self.threads, _VS_WINDOW_MAX)),
            debug=context.terminal.debugged,
        ):
            if detected.is_color:
                context.push_detected_colors(page)
            progress.update(task, advance=1)

        context.terminal.stop_progress(
            progress, f"Detected {len(context.detected_colors) if context.detected_colors else 0} color pages."
        )

        if context.detected_colors:
            context.terminal.info(f"Detected color pages: {sorted(context.detected_colors)}")

    def get_tools(self, orchestrator: "OrchestratorConfig"):
        """
        Get the required tools for the action

        :param orchestrator: The orchestrator configuration
        :return: A dictionary of tool names and their kinds
        """

        if self.mode == "vapoursynth":
            return {
                "vapoursynth": ToolsKind.PACKAGE,
                "xyz.n4o.mangacolor": ToolsKind.VAPOURSYNTH,
                "xyz.n4o.imgseqs": ToolsKind.VAPOURSYNTH,
            }
        return _color_kernel_tools(orchestrator)

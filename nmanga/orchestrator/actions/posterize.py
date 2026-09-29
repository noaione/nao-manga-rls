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
import shutil
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from PIL import Image
from pydantic import ConfigDict, Field, field_validator

from ... import file_handler, term
from ...autolevel import (
    PosterizePointMethod,
    analyze_gray_shades,
    npow2,
    posterize_image_by_bits,
    posterize_image_with_imagemagick,
)
from ...common import BoundedWritePool, lowest_or, threaded_worker
from ...lazy import get_vapoursynth
from ...vapour import (
    vs_attach_logger,
    vs_frame_to_grays,
    vs_frame_to_image,
    vs_posterize_clip,
    vs_prepare_image,
    vs_ssimulacra2,
)
from ..common import SkipActionKind, SSIMULACRA2CheckConfig, perform_skip_action
from ._base import ActionColorMixin, ActionKind, BaseAction, ThreadedResult, ToolsKind, WorkerContext

if TYPE_CHECKING:
    from ...term import ConsoleInterface
    from ..models import OrchestratorConfig, VolumeConfig

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
    ssimulacra2: SSIMULACRA2CheckConfig | None = None,
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

        quant = posterize_image_by_bits(img, num_bits=cast(int, real_bpc), method=action.method)

        if ssimulacra2 is not None and ssimulacra2.enabled:
            cnsl = term.with_thread_queue(log_q)
            is_diff = False
            if img.mode != quant.mode:
                is_diff = True
                distorted_test = quant.convert(img.mode)  # convert to same mode
            else:
                distorted_test = quant
            img_reference = vs_prepare_image(img)
            img_distorted = vs_prepare_image(distorted_test)
            if is_diff:
                distorted_test.close()
            cnsl.log(f"Reference ({img.mode}): {img_reference!r}")
            cnsl.log(f"Distorted ({quant.mode}): {img_distorted!r}")
            score = vs_ssimulacra2(img_reference, img_distorted)
            cnsl.log(f"SSIM score for {img_path.name}: {score}")
            del img_reference, img_distorted

            # if score is lower than minimum, copy file
            if score < ssimulacra2.minimum:
                img.close()
                quant.close()
                perform_skip_action(img_path, output_dir, SkipActionKind.COPY, cnsl)
                return ThreadedResult.COPIED

        quant.save(dest_path, format="PNG")
        img.close()
        quant.close()
        return ThreadedResult.PROCESSED


def _runner_posterize_threaded_star(
    args: tuple[
        term.MessageQueue,
        Path,
        Path,
        "ActionPosterize",
        str | None,
        bool,
        SkipActionKind | None,
        SSIMULACRA2CheckConfig | None,
    ],
) -> ThreadedResult:
    return _runner_posterize_threaded(*args)


# The VapourSynth page writes: both return whether the page was written, and the caller maps
# that to a `ThreadedResult`. They are module level so the write pool can call them directly.
def _posterize_write_png(image: Image.Image, dest_path: Path) -> bool:
    """Encode and write one finished page. Runs on the write pool."""
    image.save(dest_path, format="PNG")
    image.close()
    return True


def _posterize_copy_page(img_path: Path, dest_path: Path, cnsl: "ConsoleInterface") -> bool:
    """Copy the source page over the output. Runs on the write pool."""
    if dest_path.exists():
        cnsl.warning(f"Skipping existing file: {dest_path}")
        return False
    shutil.copy2(img_path, dest_path)
    return False


def _posterize_result(written: bool) -> ThreadedResult:
    """Map a write pool result to the action's own result."""

    return ThreadedResult.PROCESSED if written else ThreadedResult.COPIED


class ActionPosterize(BaseAction, ActionColorMixin):
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
    bpc: int | Literal["auto"] = Field(
        4,
        title="Bits Per Channel",
        examples=[1, 2, 4, 8, "auto"],
        description="The number of bitdepth to reduce the image to, or `auto` to detect it per image",
    )
    """The number of bitdepth to reduce the image to, or `auto` to detect it per image"""
    threshold: float = Field(0.01, ge=0.0, le=1.0, title="Threshold for Auto bitdepth")
    """The threshold to use when detecting bitdepth automatically"""
    ssimulacra2: SSIMULACRA2CheckConfig | None = Field(None, title="SSIMULACRA2 Check")
    """Do a SSIMULACRA2 check for the posterized images, ensuring they are not lossy"""
    mode: Literal["pillow", "magick", "vapoursynth"] = Field("pillow", title="The processor to use for posterizing")
    """
    The processor to use for posterizing:
    - magick: Use ImageMagick for posterizing
    - pillow: Use Pillow for posterizing (fast)
    - vapoursynth: Use VapourSynth for posterizing (faster)
    """
    method: PosterizePointMethod = Field(PosterizePointMethod.LLOYD, title="Posterizing method to be used")
    """
    The method to use for posterizing
    - even: Evenly distribute the shades
    - lloyd: Use Lloyd-Max quantization to distribute the shades
    """
    cache_mb: int = Field(512, ge=64, le=8192, title="VapourSynth Frame Cache in MiB")
    """VapourSynth frame cache in MiB, only used by the `vapoursynth` mode"""
    prefetch: int = Field(
        16,
        ge=0,
        title="VapourSynth Decoder Prefetch",
        description="How many VapourSynth frames to decode ahead of the one being processed, 0 disables it",
    )
    """How many VapourSynth frames to decode ahead of the one being processed, only used by the `vapoursynth` mode"""
    threads: int = Field(default_factory=cpu_count, ge=1, title="Processing Threads")
    """The number of threads to use for processing"""

    @field_validator("bpc", mode="after")
    @classmethod
    def _check_bpc(cls, value: int | Literal["auto"]) -> int | Literal["auto"]:
        """
        Range check the numeric bit depth, and leave ``auto`` alone.

        The range cannot be expressed as `ge`/`le` on the field: those constraints would be
        applied to the ``"auto"`` literal as well, which raises a `TypeError` from inside
        pydantic and makes ``bpc: auto`` impossible to construct.
        """

        if value == "auto":
            return value
        if not 1 <= value <= 8:
            raise ValueError("bpc must be between 1 and 8, or 'auto'")
        return value

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
            context.terminal.info(f"- Mode: {self.mode}")
            context.terminal.info(f"- Method: {self.method}")
            if self.mode == "vapoursynth":
                context.terminal.info(f"- VapourSynth Frame Cache: {self.cache_mb} MiB")
                context.terminal.info(f"- VapourSynth Decoder Prefetch: {self.prefetch}")
            context.terminal.info(f"- Processing Threads: {self.threads}")
            context.update_cwd(output_dir)
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        if not context.current_dir.exists():
            context.terminal.warning(f"Current directory {context.current_dir} does not exist, skipping posterize.")
            context.update_cwd(output_dir)  # We still need to update CWD
            return

        if self.mode == "vapoursynth":
            self._run_vapoursynth(context, volume, orchestrator, output_dir)
            return

        imagick = context.toolsets.get("magick")
        if imagick is None and self.mode == "magick":
            context.terminal.error("ImageMagick is required for posterizing, but not found!")
            raise RuntimeError("Spreads action failed due to missing ImageMagick.")

        context.terminal.info(f"Processing {context.current_dir} with posterizer...")
        all_images = [img for img, _, _, _ in file_handler.collect_image_from_folder(context.current_dir)]
        total_images = len(all_images)
        all_images.sort(key=lambda x: x.stem)

        # Do pre-processing
        images_complete: list[tuple[Path, bool, SkipActionKind | None]] = []
        for image in all_images:
            is_skip_action = None

            pg_num, is_color = self.is_color_page(image, context=context, volume=volume, orchestrator=orchestrator)
            if pg_num is not None and context.skip_action is not None and pg_num in context.skip_action.pages:
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
                    (log_q, image, output_dir, self, imagick, is_color, is_skip_action, self.ssimulacra2)
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

    def _run_vapoursynth(
        self,
        context: WorkerContext,
        volume: "VolumeConfig",
        orchestrator: "OrchestratorConfig",
        output_dir: Path,
    ) -> None:
        """
        Posterize a volume with the `nimages` VapourSynth plugin, mirroring `nmanga posterize2`.

        One clip is built for the whole volume and the pages are pulled from it in order, so
        the decoder readahead stays useful. The gate decision is made on this thread, from the
        gate's own clips; only the save or the copy goes to the write pool, which is the
        expensive half.

        The frames are pulled on this thread, so this cannot go through `threaded_worker`: a
        VapourSynth clip holds native state and cannot be handed to a worker process.
        """

        vs = get_vapoursynth()
        vs_attach_logger(context.terminal)

        all_images = [img for img, _, _, _ in file_handler.collect_image_from_folder(context.current_dir)]
        all_images.sort(key=lambda x: x.stem)
        total_images = len(all_images)

        to_process: list[Path] = []
        skip_actions: list[tuple[Path, SkipActionKind]] = []
        for image in all_images:
            pg_num, is_color = self.is_color_page(image, context=context, volume=volume, orchestrator=orchestrator)
            if is_color:
                # Colour pages are never posterized, exactly as the Pillow path does.
                skip_actions.append((image, SkipActionKind.COPY))
                continue
            if pg_num is not None and context.skip_action is not None and pg_num in context.skip_action.pages:
                requested = context.skip_action.action
                if requested != SkipActionKind.IGNORE:
                    skip_actions.append((image, requested))
                    continue
            to_process.append(image)

        context.terminal.info(f"Processing {context.current_dir} with posterizer (VapourSynth)...")

        progress = context.terminal.make_progress()
        task = progress.add_task("Posterizing images...", finished_text="Posterized images", total=total_images)

        results: list[ThreadedResult] = []
        for image, skip_action in skip_actions:
            perform_skip_action(image, output_dir, skip_action, context.terminal)
            results.append(ThreadedResult.COPIED)
            progress.update(task, advance=1)

        if to_process:
            # `auto` has no VapourSynth equivalent: the plugin takes a fixed bit depth and the
            # shade analysis that picks one is Pillow work. Resolved once, up front, and only
            # when a page is actually the subject of it.
            bits = self._resolve_bpc(to_process, context)
            gate = self.ssimulacra2

            chain = vs_posterize_clip(
                to_process,
                bits=bits,
                method=self.method.to_nimages(),
                prefetch=self.prefetch,
                debug=context.terminal.debugged,
                cache_mb=self.cache_mb,
                core=vs.core,
            )

            with BoundedWritePool(self.threads) as pool:
                for index, img_path in enumerate(to_process):
                    dest_path = output_dir / f"{img_path.stem}.png"
                    with chain.posterized.get_frame(index) as frame:
                        image = vs_frame_to_image(frame)
                        copy_instead = False
                        if gate is not None and gate.enabled:
                            # The reference is the gray clip the posterize chain already
                            # decoded, which VapourSynth serves from its frame cache, so the
                            # second pull costs no extra decode and the score is about the
                            # posterization alone rather than the decoder as well.
                            with chain.gray.get_frame(index) as source_frame:
                                score = vs_ssimulacra2(
                                    vs_frame_to_grays(source_frame, core=vs.core),
                                    vs_frame_to_grays(frame, core=vs.core),
                                )
                            context.terminal.log(f"SSIM score for {img_path.name}: {score}")
                            copy_instead = score < gate.minimum

                    if copy_instead:
                        image.close()
                        oldest = pool.submit(_posterize_copy_page, img_path, dest_path, context.terminal)
                    else:
                        oldest = pool.submit(_posterize_write_png, image, dest_path)

                    if oldest is not None:
                        results.append(_posterize_result(oldest.result()))
                        progress.update(task, advance=1)

                for future in pool.pending():
                    results.append(_posterize_result(future.result()))
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

        context.update_cwd(output_dir)

    def _resolve_bpc(self, images: list[Path], context: WorkerContext) -> int:
        """
        Turn :attr:`bpc` into a concrete bit depth for the VapourSynth clip.

        ``auto`` is decided from the first page, because the plugin takes one bit depth for
        the whole clip. The Pillow path decides per page, so an ``auto`` volume can differ
        between the two modes; the CLI has the same property.
        """

        if self.bpc != "auto":
            return self.bpc
        if not images:
            return 8

        first = images[0]
        with Image.open(first) as img:
            detected = _detect_auto_bpc(img, threshold=self.threshold)
        context.terminal.info(f"Auto detected bit depth {detected} from {first.name}")
        return detected

    def get_tools(self):
        """
        Get the required tools for the action

        :return: A dictionary of tool names and their kinds
        """

        ssim = {}
        if self.ssimulacra2 is not None and self.ssimulacra2.enabled:
            ssim = {"vapoursynth": ToolsKind.PACKAGE, "com.lumen.vship": ToolsKind.VAPOURSYNTH}

        if self.mode == "magick":
            return {
                "magick": ToolsKind.BINARY,
            }
        elif self.mode == "vapoursynth":
            base = {
                "vapoursynth": ToolsKind.PACKAGE,
                "xyz.n4o.nimages": ToolsKind.VAPOURSYNTH,
                "xyz.n4o.imgseqs": ToolsKind.VAPOURSYNTH,
            }
            return {**base, **ssim}

        return ssim

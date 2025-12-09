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

import abc
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from ... import file_handler, term
from ...common import RegexCollection, lowest_or, threaded_worker
from ...resizer import ResizeKernel, ResizeMode, ResizeTarget, rescale_image
from ..common import SkipActionKind, perform_skip_action
from ._base import ActionKind, BaseAction, ThreadedResult, WorkerContext

if TYPE_CHECKING:
    from .. import OrchestratorConfig, VolumeConfig

__all__ = (
    "ActionRescale",
    "RescaleTarget",
)


class RescaleTargetBase(BaseModel, abc.ABC):
    """
    Base class for rescale target configurations.
    """

    @abc.abstractmethod
    def to_params(self) -> ResizeTarget:
        """
        Convert to `ResizeTarget` parameters.
        """
        raise NotImplementedError()


class RescaleTargetExact(BaseModel):
    """
    Rescale target specifying exact width and height.
    """

    mode: Literal[ResizeMode.Exact] = Field(ResizeMode.Exact, title="Resize Mode")
    """The resize mode"""
    width: int = Field(..., title="Target Width", ge=1)
    """The target width in pixels."""
    height: int = Field(..., title="Target Height", ge=1)
    """The target height in pixels."""

    def to_params(self) -> ResizeTarget:
        return ResizeTarget(self.mode, width=self.width, height=self.height)


class RescaleTargetWidth(BaseModel):
    """
    Rescale target specifying width only, height is computed to maintain aspect ratio.
    """

    mode: Literal[ResizeMode.Width] = Field(ResizeMode.Width, title="Resize Mode")
    """The resize mode"""
    width: int = Field(..., title="Target Width", ge=1)
    """The target width in pixels."""

    def to_params(self) -> ResizeTarget:
        return ResizeTarget(self.mode, width=self.width)


class RescaleTargetHeight(BaseModel):
    """
    Rescale target specifying height only, width is computed to maintain aspect ratio.
    """

    mode: Literal[ResizeMode.Height] = Field(ResizeMode.Height, title="Resize Mode")
    """The resize mode"""
    height: int = Field(..., title="Target Height", ge=1)
    """The target height in pixels."""

    def to_params(self) -> ResizeTarget:
        return ResizeTarget(self.mode, height=self.height)


class RescaleTargetFit(BaseModel):
    """
    Rescale target specifying maximum width and height, maintaining aspect ratio.
    """

    mode: Literal[ResizeMode.Fit] = Field(ResizeMode.Fit, title="Resize Mode")
    """The resize mode"""
    width: int = Field(..., title="Maximum Width", ge=1)
    """The maximum width in pixels."""
    height: int = Field(..., title="Maximum Height", ge=1)
    """The maximum height in pixels."""

    def to_params(self) -> ResizeTarget:
        return ResizeTarget(self.mode, width=self.width, height=self.height)


class RescaleTargetMultiply(BaseModel):
    """
    Rescale target specifying a scaling factor.
    """

    mode: Literal[ResizeMode.Multiply] = Field(ResizeMode.Multiply, title="Resize Mode")
    """The resize mode"""
    factor: int | float = Field(..., title="Scaling Factor", gt=0)
    """The scaling factor (e.g., 0.5 for half size, 2.0 for double size)."""

    def to_params(self) -> ResizeTarget:
        return ResizeTarget(self.mode, factor=self.factor)


RescaleTarget: TypeAlias = Annotated[
    RescaleTargetExact | RescaleTargetWidth | RescaleTargetHeight | RescaleTargetFit | RescaleTargetMultiply,
    Field(discriminator="mode", title="Rescale Target"),
]
"""
The rescale target configuration.
"""


def _runner_rescale_threaded(
    log_q: term.MessageOrInterface,
    img_path: Path,
    output_dir: Path,
    rescale: RescaleTarget,
    kernel: ResizeKernel,
    skip_action: SkipActionKind | None = None,
) -> ThreadedResult:
    cnsl = term.with_thread_queue(log_q)
    if skip_action is not None:
        perform_skip_action(img_path, output_dir, skip_action, cnsl)
        return ThreadedResult.COPIED if skip_action != SkipActionKind.IGNORE else ThreadedResult.IGNORED

    dest_path = output_dir / f"{img_path.stem}.png"
    if dest_path.exists():
        cnsl.warning(f"Skipping existing file: {dest_path}")
        return ThreadedResult.COPIED

    img = Image.open(img_path)
    res_target = rescale.to_params()
    rescaled_img = rescale_image(
        img,
        target=res_target,
        kernel=kernel,
    )

    rescaled_img.save(dest_path, format="PNG")
    img.close()
    rescaled_img.close()
    return ThreadedResult.PROCESSED


def _runner_rescale_threaded_star(
    args: tuple[term.MessageQueue, Path, Path, RescaleTarget, ResizeKernel, SkipActionKind | None],
) -> ThreadedResult:
    return _runner_rescale_threaded(*args)


class ActionRescale(BaseAction):
    """
    Action to rescale images to a specified size.
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Rescale Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.RESCALE] = Field(ActionKind.RESCALE, title="Rescale Images Action")
    """The kind of action"""
    base_path: str = Field("rescaled", title="Output Base Path")
    """The base path to save the rescaled images to"""
    rescale: RescaleTarget = Field(title="Rescale Target")
    """The rescale target configuration"""
    kernel: ResizeKernel = Field(..., title="Rescale Kernel")
    """The rescale kernel to use"""
    threads: int = Field(default_factory=cpu_count, ge=1, title="Processing Threads")
    """The number of threads to use for processing"""

    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the rescale action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        output_dir = context.root_dir / Path(self.base_path) / Path(volume.path)

        if context.dry_run:
            context.terminal.info(f"- Output Base Path: {self.base_path}")
            as_params = self.rescale.to_params()
            context.terminal.info(f"- Rescale Target: {as_params!s}")
            context.terminal.info(f"- Rescale Kernel: {self.kernel.name}")
            context.terminal.info(f"- Processing Threads: {self.threads}")
            context.update_cwd(output_dir)
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        page_re = RegexCollection.page_re()

        context.terminal.info(f"Processing {context.current_dir} with posterizer...")
        all_images = [img for img, _, _, _ in file_handler.collect_image_from_folder(context.current_dir)]
        total_images = len(all_images)
        all_images.sort(key=lambda x: x.stem)

        # Do pre-processing
        images_complete: list[tuple[Path, SkipActionKind | None]] = []
        for image in all_images:
            img_match = page_re.match(image.stem)
            is_skip_action = None

            if img_match is not None:
                p01 = int(img_match.group("a"))
                if context.skip_action is not None and p01 in context.skip_action.pages:
                    is_skip_action = context.skip_action.action

            images_complete.append((image, is_skip_action))

        rescaler = self.rescale
        if metafield := volume.metafields:
            if metafield.rescale is not None:
                rescaler = metafield.rescale

        results: list[ThreadedResult] = []
        progress = context.terminal.make_progress()
        task = progress.add_task("Rescaling images...", finished_text="Rescaled images", total=total_images)
        context.terminal.info(f"Using {self.threads} CPU threads for processing.")
        with threaded_worker(context.terminal, lowest_or(self.threads, images_complete)) as (pool, log_q):
            for result in pool.imap_unordered(
                _runner_rescale_threaded_star,
                [
                    (log_q, image, output_dir, rescaler, self.kernel, is_skip_action)
                    for image, is_skip_action in images_complete
                ],
            ):
                results.append(result)
                progress.update(task, advance=1)

        context.terminal.stop_progress(progress, f"Rescaled {total_images} images in {context.current_dir}")
        rescaled_count = sum(1 for result in results if result == ThreadedResult.PROCESSED)
        copied_count = sum(1 for result in results if result == ThreadedResult.COPIED)
        ignored_count = sum(1 for result in results if result == ThreadedResult.IGNORED)
        if copied_count > 0:
            context.terminal.info(f" Copied {copied_count} images without rescaling.")
        if rescaled_count > 0:
            context.terminal.info(f" Rescaled {rescaled_count} images.")
        if ignored_count > 0:
            context.terminal.info(f" Ignored {ignored_count} images.")

        # Update CWD
        context.update_cwd(output_dir)

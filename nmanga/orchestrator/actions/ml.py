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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, FilePath, model_validator
from pydantic_core import PydanticCustomError
from typing_extensions import Self

from ... import file_handler
from ...common import RegexCollection
from ...denoiser import MLDataType, denoise_single_image, prepare_model_runtime, prepare_model_runtime_builders
from ...resizer import ResizeKernel, rescale_image
from ..common import perform_skip_action
from ._base import ActionKind, BaseAction, ToolsKind, WorkerContext
from .rescale import RescaleTarget

if TYPE_CHECKING:
    from .. import OrchestratorConfig, VolumeConfig

__all__ = (
    "ActionDenoise",
    "ActionUpscale",
)


@dataclass
class MLActionName:
    load: str
    run: str
    finish: str


class ActionML(BaseAction):
    """
    Base class for ML-based actions, mainly denoising and upscaling.
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - ML-based Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    model: FilePath = Field(..., title="The ONNX Model Path")
    """The path to the ONNX model"""
    device_id: int = Field(0, ge=0, title="Device ID")
    """The device ID to use for denoising"""
    batch_size: int = Field(64, ge=1, title="Batch Size")
    """The batch size to use for denoising"""
    tile_size: int = Field(128, ge=64, title="Tile Size")
    """The tile size to use for denoising"""
    background: Literal["white", "black"] = Field("black", title="Padding Background Color")
    """The background color to use for padding"""
    contrast_strectch: bool = Field(False, title="Contrast Stretch After Denoising")
    """Whether to apply contrast stretch after denoising"""
    precompiled: bool = Field(False, title="Is precompiled TensorRT model")
    """Whether the model is a precompiled TensorRT model, so we do not need to build it"""
    data_type: MLDataType | None = Field(None, title="The data type to use for TensorRT model")
    """The data type to use for TensorRT model"""
    base_path: str = Field("output", title="Output Base Path")
    """The base path to save the resulting images to"""

    @model_validator(mode="after")
    def validate_precompiled_and_data_type(self) -> Self:
        if self.precompiled and self.data_type is None:
            raise PydanticCustomError(
                "precompiled_data_type_missing",
                "data_type must be specified when precompiled is True",
            )
        return self

    @property
    def action_name(self) -> MLActionName:
        return MLActionName(
            load="generic",
            run="Processing",
            finish="Processed",
        )

    def post_process(
        self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig", image: Image.Image
    ) -> Image.Image:
        """
        Post-process the image after denoising/upscaling

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        :param image: The image to post-process
        :return: The post-processed image
        """

        return image

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
            context.terminal.info(f"- Model Path: {self.model}")
            context.terminal.info(f"- Output Base Path: {self.base_path}")
            context.terminal.info(f"- Device ID: {self.device_id}")
            context.terminal.info(f"- Batch Size: {self.batch_size}")
            context.terminal.info(f"- Tile Size: {self.tile_size}")
            context.terminal.info(f"- Background Color: {self.background}")
            context.terminal.info(f"- Contrast Stretch: {self.contrast_strectch}")
            context.terminal.info(f"- Precompiled Model: {'Yes' if self.precompiled else 'No'}")
            if self.data_type is not None:
                context.terminal.info(f"- Data Type: {self.data_type.value}")
            context.update_cwd(output_dir)
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        context.terminal.info(f"Loading {self.action_name.load} model from {self.model.name}...")

        page_re = RegexCollection.page_re()
        if self.precompiled:
            session = prepare_model_runtime(
                self.model,
                device_id=self.device_id,
                is_verbose=context.terminal.debugged,
            )
        else:
            context.terminal.info(
                f"Building/loading TensorRT Engine for {self.model.name}. "
                + "If it is building, you may need to wait up to 20-25 minutes for the first time."
            )
            session = prepare_model_runtime_builders(
                self.model,
                device_id=self.device_id,
                is_verbose=context.terminal.debugged,
                tile_size=self.tile_size,
                batch_size=self.batch_size,
                data_type=cast(MLDataType, self.data_type),
            )

        current_index = 1
        total_image = 0
        context.terminal.info(f"{self.action_name.run} images in {context.current_dir}...")
        progress = context.terminal.make_progress()
        task = context.terminal.make_task(
            progress,
            f"{self.action_name.run} images...",
            finished_text=f"{self.action_name.finish} images",
            total=None,
        )
        for file_path, _, total_image, _ in file_handler.collect_image_from_folder(context.current_dir):
            progress.update(task, total=total_image, completed=current_index - 1)

            if title_match := page_re.match(file_path.stem):
                first_part = int(title_match.group("a"))  # We only care about this
                if context.skip_action is not None and first_part in context.skip_action.pages:
                    perform_skip_action(file_path, output_dir, context.skip_action.action, context.terminal)
                    current_index += 1
                    continue

            img_file = Image.open(file_path)
            output_image = denoise_single_image(
                img_file,
                session,
                batch_size=self.batch_size,
                tile_size=self.tile_size,
                contrast_stretch=self.contrast_strectch,
                background=self.background,
                use_fp32=not self.precompiled,
            )
            output_image = self.post_process(context, volume, orchestrator, output_image)

            output_path = output_dir / f"{file_path.stem}.png"
            output_image.save(output_path, format="PNG")
            img_file.close()
            output_image.close()
            current_index += 1
        progress.update(task, advance=1)
        context.terminal.stop_progress(progress, f"{self.action_name.finish} {total_image} images.")

        # Update CWD
        context.update_cwd(output_dir)

    def get_tools(self):
        """
        Get the required tools for the action

        :return: A dictionary of tool names and their kinds
        """

        base_tools = {
            "onnxruntime": ToolsKind.PACKAGE,
            "einops": ToolsKind.PACKAGE,
            "numpy": ToolsKind.PACKAGE,
        }

        if not self.precompiled:
            combined_tools = {
                "tensorrt": ToolsKind.PACKAGE,
                "torch": ToolsKind.PACKAGE,
                "onnx": ToolsKind.PACKAGE,
                **base_tools,
            }
            return combined_tools

        return base_tools


class ActionDenoise(ActionML):
    """
    Action to denoise all images in a volume with denoise-trt
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Denoise Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.DENOISE] = Field(ActionKind.DENOISE, title="Denoise Images Action")
    """The kind of action"""
    base_path: str = Field("denoised", title="Output Base Path")
    """The base path to save the denoised images to"""

    @property
    def action_name(self) -> MLActionName:
        return MLActionName(
            load="denoise",
            run="Denoising",
            finish="Denoised",
        )


class UpscaleRescaleConfig(BaseModel):
    """
    Configuration for rescaling after upscaling.
    """

    target: RescaleTarget = Field(..., title="Rescale Target")
    """The rescale target configuration"""
    kernel: ResizeKernel = Field(..., title="Rescale Kernel")
    """The rescale kernel to use"""


class ActionUpscale(ActionML):
    """
    Action to upscale all images in a volume with upscale-trt
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Upscale Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.UPSCALE] = Field(ActionKind.UPSCALE, title="Upscale Images Action")
    """The kind of action"""
    base_path: str = Field("upscaled", title="Output Base Path")
    """The base path to save the upscaled images to"""

    rescale: UpscaleRescaleConfig | None = Field(None, title="Rescale Configuration")
    """The rescale configuration after upscaling"""

    @property
    def action_name(self) -> MLActionName:
        return MLActionName(
            load="upscale",
            run="Upscaling",
            finish="Upscaled",
        )

    def post_process(
        self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig", image: Image.Image
    ) -> Image.Image:
        """
        Post-process the image after denoising/upscaling

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        :param image: The image to post-process
        :return: The post-processed image
        """

        if self.rescale is None:
            return image

        rescaler = self.rescale.target
        if metafield := volume.metafields:
            if metafield.rescale is not None:
                rescaler = metafield.rescale

        target = rescaler.to_params()
        return rescale_image(
            image,
            target=target,
            kernel=self.rescale.kernel,
        )

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

from PIL import Image
from pydantic import ConfigDict, Field, FilePath

from ... import file_handler
from ...common import RegexCollection
from ...denoiser import denoise_single_image, prepare_model_runtime
from ..common import perform_skip_action
from ._base import ActionKind, BaseAction, ToolsKind, WorkerContext

if TYPE_CHECKING:
    from .. import OrchestratorConfig, VolumeConfig

__all__ = ("ActionDenoise",)


class ActionDenoise(BaseAction):
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
    model: FilePath = Field(..., title="The ONNX Model Path")
    """The path to the ONNX model"""
    base_path: str = Field("denoised", title="Output Base Path")
    """The base path to save the denoised images to"""
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
            context.update_cwd(output_dir)
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        context.terminal.info(f"Loading denoising model from {self.model.name}...")

        page_re = RegexCollection.page_re()
        session = prepare_model_runtime(
            self.model,
            device_id=self.device_id,
            is_verbose=context.terminal.debugged,
        )

        current_index = 1
        total_image = 0
        context.terminal.info(f"Denoising images in {context.current_dir}...")
        for file_path, _, total_image, _ in file_handler.collect_image_from_folder(context.current_dir):
            context.terminal.status(f"Denoising images... [{current_index}/{total_image}]")

            if title_match := page_re.match(file_path.stem):
                first_part = int(title_match.group("a"))  # We only care about this
                if context.skip_action is not None and first_part in context.skip_action.pages:
                    perform_skip_action(file_path, output_dir, context.skip_action.action, context.terminal)
                    continue

            img_file = Image.open(file_path)
            output_image = denoise_single_image(
                img_file,
                session,
                batch_size=self.batch_size,
                tile_size=self.tile_size,
                contrast_stretch=self.contrast_strectch,
                background=self.background,
            )

            output_path = output_dir / f"{file_path.stem}.png"
            output_image.save(output_path, format="PNG")
            img_file.close()
            output_image.close()
            current_index += 1
        context.terminal.stop_status(f"Denoised {total_image} images.")

        # Update CWD
        context.update_cwd(output_dir)

    def get_tools(self):
        """
        Get the required tools for the action

        :return: A dictionary of tool names and their kinds
        """

        return {
            "onnxruntime": ToolsKind.PACKAGE,
            "einops": ToolsKind.PACKAGE,
            "numpy": ToolsKind.PACKAGE,
        }

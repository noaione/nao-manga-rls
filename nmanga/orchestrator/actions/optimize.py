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

from ... import file_handler
from ...common import is_pingo_alpha, run_pingo_and_verify
from ._base import ActionKind, BaseAction, ToolsKind, WorkerContext

if TYPE_CHECKING:
    from .. import OrchestratorConfig, VolumeConfig

__all__ = ("ActionOptimize",)


class ActionOptimize(BaseAction):
    """
    Optimize all images in a volume with pingo
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Optimize Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.OPTIMIZE] = Field(ActionKind.OPTIMIZE, title="Optimize Images Action")
    """The kind of action"""
    aggresive: bool = Field(False, title="Aggressive Mode")
    """Whether to use the aggressive mode of pingo, which would force grayscale conversion for all images"""
    limiter: str | None = Field(None, title="Limiter", examples=[".png", ".jpg"])
    """Limit the optimization to certain image types, e.g. .png"""

    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        if context.dry_run:
            context.terminal.info(f"- Aggressive Mode: {'Yes' if self.aggresive else 'No'}")
            context.terminal.info(f"- Limiter: {self.limiter if self.limiter else 'None'}")
            return

        pingo = context.toolsets.get("pingo")
        if pingo is None:
            context.terminal.error("Pingo is required for image optimization, but not found!")
            raise RuntimeError("Pingo not found in toolsets")

        all_images = [img for img, _, _, _ in file_handler.collect_image_from_folder(context.current_dir)]
        total_images = len(all_images)
        context.terminal.status(f"Optimizing {total_images} images with pingo...")

        alpha_ver = is_pingo_alpha(pingo)
        base_cmd = [pingo, "-strip", "-sb"] if alpha_ver else [pingo, "-notime", "-lossless", "-notrans", "-s4"]
        full_dir = context.current_dir.resolve()
        folder_text = str(full_dir / "*")
        if self.limiter:
            folder_text += self.limiter
        if self.aggresive and not alpha_ver:
            base_cmd.append("-grayscale")

        cmd = [*base_cmd, folder_text]
        context.terminal.status(f"Optimizing images in: {folder_text}...")
        proc = run_pingo_and_verify(cmd)
        end_msg = "Optimized images files!"
        if proc is not None:
            end_msg += f" [{proc}]"
        context.terminal.stop_status(end_msg)

    def get_tools(self):
        """
        Get the required tools for the action

        :return: A dictionary of tool names and their kinds
        """

        return {
            "pingo": ToolsKind.BINARY,
        }

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
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from PIL import Image
from pydantic import BaseModel, Field

from ...common import RegexCollection
from ...ogsov import detect_image_color, detect_image_color_ogsov
from ..dsl import Context
from ..rules import RuleModel

if TYPE_CHECKING:
    from ...denoiser import InferenceSessionWithScale
    from ...term import Console
    from .. import OrchestratorConfig, SkipActionConfig, VolumeConfig


__all__ = (
    "ActionColorMixin",
    "ActionKind",
    "BaseAction",
    "ThreadedResult",
    "ToolsKind",
    "WorkerContext",
)


class ThreadedResult(int, Enum):
    """
    Result of a threaded worker action.
    """

    IGNORED = 0
    """The image was ignored and not processed."""
    COPIED = 1
    """The image was copied without processing."""
    GRAYSCALED = 2
    """The image was converted to grayscale only."""
    PROCESSED = 3
    """The image was processed."""


class WorkerContext(Context):
    """The correct worker context for actions."""

    root_dir: Path
    """The root directory for the orchestrator."""
    current_dir: Path
    """The current working directory for the action."""
    terminal: "Console"
    """The terminal console for logging."""
    toolsets: dict[str, str]
    """The toolsets available for the action."""
    skip_action: SkipActionConfig | None = None
    """The skip action configuration, if any."""
    dry_run: bool = False
    """Run the action in dry run mode."""
    ml_model_session: ClassVar[dict[str, "InferenceSessionWithScale"]] = {}
    """The current running ML model sessions, stored as a class variable"""
    detected_colors: list[int] | None = None
    """The detected color pages in the volume, used for passing from color detection to other actions

    Only affected if the color detection action is in the action chain, otherwise it will be None
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        current_dir: Path,
        terminal: "Console",
        toolsets: dict[str, str],
        skip_action: SkipActionConfig | None = None,
        dry_run: bool = False,
        **kwargs,
    ):
        kwargs.pop("ml_model_session", None)  # prevent passing ml_model_session in init
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.current_dir = current_dir
        self.terminal = terminal
        self.toolsets = toolsets
        self.skip_action = skip_action
        self.dry_run = dry_run

    def update_cwd(self, new_dir: Path) -> None:
        """Update the current working directory."""
        self.current_dir = new_dir

    def set_skip_action(self, skip_action: SkipActionConfig | None) -> None:
        """Set the skip action configuration."""
        self.skip_action = skip_action

    def push_detected_colors(self, page: int) -> None:
        """Push the detected color pages to the context."""
        if self.detected_colors is None:
            self.detected_colors = []
        if page not in self.detected_colors:
            self.detected_colors.append(page)


class ActionKind(str, Enum):
    """
    All supported action kinds.

    This list is incomplete and might expand in the future.
    """

    SHIFT_RENAME = "shift_rename"
    """Shift rename of a volume"""
    SPREADS = "spreads"
    """Spreads join a volume"""
    RENAME = "rename"
    """Rename all images in a volume, similar to `nmanga releases` command rename"""
    DENOISE = "denoise"
    """Denoise all images in a volume with denoise-trt"""
    UPSCALE = "upscale"
    """Upscale all images in a volume with upscale-trt"""
    RESCALE = "rescale"
    """Rescale all images in a volume with Pillow and cykooz.resizer"""
    AUTOLEVEL = "autolevel"
    """Auto level all images with Pillow"""
    LEVEL = "level"
    """Level all images with Pillow, use this if you want to provide single level values"""
    POSTERIZE = "posterize"
    """Posterize all images with imagemagick or Pillow"""
    OPTIMIZE = "optimize"
    """Optimize all images with pingo"""
    TAGGING = "tagging"
    """Add metadata tags to the images with exiftool"""
    PACK = "pack"
    """Pack the volume into an archive"""
    MOVE_COLOR = "move_color"
    """Move the tagged color images to a separate folder"""
    COLOR_JPEGIFY = "color_jpegify"
    """Convert color images to JPEG format with cjpegli"""
    COLOR_DETECT = "color_detect"
    """Detect color pages with ML-based model or Pillow"""
    INTERRUPT = "interrupt"
    """Interrupt the action chain"""
    CHANGE_CWD = "change_cwd"
    """Change the current working directory"""


class ToolsKind(str, Enum):
    """
    All supported tool kinds for actions.
    """

    BINARY = "binary"
    """The tool is an executable binary"""
    PACKAGE = "package"
    """The tool is a python package"""


class BaseAction(BaseModel, abc.ABC):
    """
    The base action model
    """

    conditions: RuleModel | None = Field(None, title="Conditions to run the action")
    """Run the action only if a certain condition is met"""

    @abc.abstractmethod
    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        pass

    def get_tools(self) -> dict[str, ToolsKind]:
        """
        Get the required tools for the action

        :return: A dictionary of tool names and their kinds
        """

        return {}


class ActionColorMixin:
    """
    Mixin for actions that operate on color images,
    providing common utilities and context handling for color-related actions.
    """

    _page_re = RegexCollection.page_re()

    def is_color_page(
        self, path: Path, *, context: WorkerContext, volume: VolumeConfig, orchestrator: "OrchestratorConfig"
    ) -> tuple[int | None, bool]:
        """
        Determine if a given image file is a color page based on the volume's detected colors

        :param path: The path to the image file
        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        :return: tuple of page number (p01) and whether it is a color page, or (None, False)
                 if the page number cannot be determined
        """

        title_match = self._page_re.match(path.stem)
        if title_match is None:
            return None, False

        p01 = int(title_match.group("a"))
        if isinstance(volume.colors, list):
            return p01, p01 in volume.colors
        if isinstance(context.detected_colors, list):
            return p01, p01 in context.detected_colors

        # Use ML-based color detection when possible
        if orchestrator.metafields.color_model is not None:
            # Read bytes
            color_model_path = Path(orchestrator.metafields.color_model).resolve()
            img_bytes = path.read_bytes()
            is_color = detect_image_color_ogsov(img_bytes, weights_file=color_model_path).is_color
            if is_color:
                context.push_detected_colors(p01)  # push for caching
            return p01, is_color

        # Fall back to normal
        img_data = Image.open(path)
        detected = detect_image_color(img_data)
        img_data.close()
        if detected.is_color:
            context.push_detected_colors(p01)  # push for caching
        return p01, detected.is_color

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
import shutil
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FilePath, model_validator
from pydantic_core import PydanticCustomError

from ..ogsov import MangaColorInference, register_ogsov_instance, register_ogsox_instance

if TYPE_CHECKING:
    from ..term import ConsoleInterface

__all__ = (
    "ColorDetectionConfig",
    "ColorDetectionKind",
    "ColorDetectionOGSOVConfig",
    "ColorDetectionOGSOXConfig",
    "SSIMULACRA2CheckConfig",
    "SkipActionConfig",
    "SkipActionKind",
    "perform_skip_action",
)


class SkipActionKind(str, Enum):
    """
    What do you want to do when skipping an action
    """

    IGNORE = "ignore"
    """Ignore the skip and perform the action as normal"""
    COPY = "copy"
    """Copy the original file to the output folder without performing the action"""
    MOVE = "move"
    """Move the original file to the output folder without performing the action"""


class SkipActionConfig(BaseModel):
    """
    What actions to skip for a volume

    Note: Not all actions support skipping, those that do not will be ignored
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    action: SkipActionKind = Field(..., title="Skip Action")
    """The action to take when skipping"""
    step: str = Field(..., title="Action Step Name", examples=["autolevel-1", "optimize-1", "optimize-2"])
    """The action step name to skip, e.g. autolevel-1, optimize-2, etc."""
    pages: list[int] = Field(..., min_length=1, title="Pages to Skip On")
    """The list of page numbers to skip the action on"""


class SSIMULACRA2CheckConfig(BaseModel):
    """
    Do a SSIMULACRA2 check for an image
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    enabled: bool = Field(False, title="Enable SSIMULACRA2 Check")
    """Enable/disable the SSIMULACRA2 check"""
    minimum: float = Field(80.0, title="Minimum SSIMULACRA2 Score", ge=0.0, le=100.0)
    """Minimum SSIMULACRA2 score to keep the image"""


class ColorDetectionKind(str, Enum):
    """
    All supported action kinds.

    This list is incomplete and might expand in the future.
    """

    OGSOV = "ogsov"
    """
    Color detection with OGSOV

    Colourfulness histogram + five MLPs
    """

    OGSOX = "ogsox"
    """
    Color detection with OGSOX

    512² tiling + five attention heads
    """


class BaseColorDetectionConfig(BaseModel, abc.ABC):
    """
    Base metafields config for color detection
    """

    fast: bool = Field(False, title="Fast Mode")
    """Utilize fast-path calculation to speed-up processing"""

    @abc.abstractmethod
    def get_kernel(self) -> MangaColorInference:
        """
        Get the inference kernel
        """

        raise NotImplementedError


class ColorDetectionOGSOVConfig(BaseColorDetectionConfig):
    """
    Extra metafields information for ogsov color detection kernel
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Color Detection - OGSOV kernel",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ColorDetectionKind.OGSOV] = Field(ColorDetectionKind.OGSOV, title="Color Detection Kind")
    """The kind of the color model"""
    weights_table: FilePath = Field(..., title="Weights Table Path")
    """Path to the npz for the weights table"""

    @model_validator(mode="after")
    def ensure_lookup_table_provided(self) -> Self:
        resolved_weights = Path(self.weights_table).resolve()
        if not resolved_weights.is_file():
            raise PydanticCustomError(
                "weights_table_not_found", "Weights table file not found: {path}", {"path": resolved_weights}
            )
        return self

    def get_kernel(self) -> MangaColorInference:
        return register_ogsov_instance(self.weights_table)


class ColorDetectionOGSOXConfig(BaseColorDetectionConfig):
    """
    Extra metafields information for ogsox color detection kernel
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Color Detection - OGSOX kernel",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ColorDetectionKind.OGSOX] = Field(ColorDetectionKind.OGSOX, title="Color Detection Kind")
    """The kind of the color model"""
    lookup_table: FilePath = Field(..., title="Lookup Table Path")
    """Path to the npz file for mask lookup table"""
    classifier: FilePath = Field(..., title="Classifier Path")
    """Path to the ONNX model for the classifier"""

    @model_validator(mode="after")
    def ensure_lookup_table_provided(self) -> Self:
        resolved_lookup = Path(self.lookup_table).resolve()
        if not resolved_lookup.is_file():
            raise PydanticCustomError(
                "lookup_table_not_found", "Lookup table file not found: {path}", {"path": resolved_lookup}
            )
        resolved_classifier = Path(self.classifier).resolve()
        if not resolved_classifier.is_file():
            raise PydanticCustomError(
                "classifier_not_found", "Classifier file not found: {path}", {"path": resolved_classifier}
            )
        return self

    def get_kernel(self) -> MangaColorInference:
        return register_ogsox_instance(self.lookup_table, self.classifier)


ColorDetectionConfig = Annotated[
    ColorDetectionOGSOVConfig | ColorDetectionOGSOXConfig,
    Field(discriminator="kind", description="The collection of all supported color detection fields."),
]
"""Color detection metafields"""


def perform_skip_action(
    img_path: Path,
    output_dir: Path,
    action: SkipActionKind,
    console: "ConsoleInterface",
) -> None:
    dest_path = output_dir / img_path.name
    match action:
        case SkipActionKind.IGNORE:
            return
        case SkipActionKind.COPY:
            if dest_path.exists():
                console.warning(f"Skipping existing file: {dest_path}")
                return
            shutil.copy2(img_path, dest_path)
        case SkipActionKind.MOVE:
            # Just force move
            shutil.move(img_path, dest_path)

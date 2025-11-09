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

import shutil
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ..term import ConsoleInterface

__all__ = (
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

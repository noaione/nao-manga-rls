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

from ._base import ActionKind, BaseAction, WorkerContext

if TYPE_CHECKING:
    from .. import OrchestratorConfig, VolumeConfig

__all__ = (
    "ActionChangeCwd",
    "ActionInterrupt",
)


class ActionChangeCwd(BaseAction):
    """
    Action to change the current working directory
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Change Working Directory Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.CHANGE_CWD] = Field(ActionKind.CHANGE_CWD, title="Change Working Directory Action")
    """The kind of action"""
    directory: str = Field(title="Target Path")
    """The target directory to change to"""

    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        output_dir = context.root_dir / Path(self.directory) / Path(volume.path)
        context.update_cwd(output_dir)

        if context.dry_run:
            context.terminal.info(f"- Target: {output_dir}")
            return


class ActionInterrupt(BaseAction):
    """
    Action to interrupt the action chain

    This will "pause" the action chain in-place, this would also allow you to quit the action chain
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Interrupt Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.INTERRUPT] = Field(ActionKind.INTERRUPT)
    """The kind of action"""
    whole_chain: bool = Field(True, title="Interrupt Whole Chain")
    """When quitting, just don't stop at the current volume but the whole volume"""

    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        # TODO: Implement interrupt logic, in here
        context.terminal.warning("Not implemented")
        if context.dry_run:
            context.terminal.info(f"- Whole Chain: {'Yes' if self.whole_chain else 'No'}")
            return

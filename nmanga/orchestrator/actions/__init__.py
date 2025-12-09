"""
nmanga.orchestrator.actions
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module contains all action implementations for the nmanga orchestrator.

:copyright: (c) 2022-present noaione
:license: MIT, see LICENSE for more details.
"""

from typing import Annotated, TypeAlias

from pydantic import Field

from ._base import *
from .colors import *
from .leveling import *
from .ml import *
from .optimize import *
from .pack import *
from .posterize import *
from .renamer import *
from .rescale import *
from .spreads import *
from .tagging import *
from .utils import *

__all__ = (
    "ActionAutolevel",
    "ActionChangeCwd",
    "ActionColorJpegify",
    "ActionDenoise",
    "ActionInterrupt",
    "ActionKind",
    "ActionLevel",
    "ActionMoveColor",
    "ActionOptimize",
    "ActionPack",
    "ActionPosterize",
    "ActionRename",
    "ActionRescale",
    "ActionShiftName",
    "ActionSpreads",
    "ActionTagging",
    "ActionUpscale",
    "Actions",
    "BaseAction",
    "ThreadedResult",
    "ToolsKind",
    "WorkerContext",
)

ActionType: TypeAlias = (
    ActionShiftName
    | ActionSpreads
    | ActionRename
    | ActionDenoise
    | ActionUpscale
    | ActionRescale
    | ActionAutolevel
    | ActionLevel
    | ActionPosterize
    | ActionOptimize
    | ActionTagging
    | ActionMoveColor
    | ActionColorJpegify
    | ActionPack
    | ActionInterrupt
    | ActionChangeCwd
)
Actions = Annotated[ActionType, Field(discriminator="kind", description="The collection of all supported actions.")]
"""
The list of all supported actions.
"""

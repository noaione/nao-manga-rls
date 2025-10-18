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
from .denoise import *
from .leveling import *
from .optimize import *
from .others import *
from .pack import *
from .posterize import *
from .renamer import *
from .spreads import *
from .tagging import *

__all__ = (
    *_base.__all__,
    *colors.__all__,
    *denoise.__all__,
    *leveling.__all__,
    *optimize.__all__,
    *others.__all__,
    *pack.__all__,
    *posterize.__all__,
    *renamer.__all__,
    *spreads.__all__,
    *tagging.__all__,
    "Actions",
)

ActionType: TypeAlias = (
    ActionShiftName
    | ActionSpreads
    | ActionRename
    | ActionDenoise
    | ActionAutolevel
    | ActionPosterize
    | ActionOptimize
    | ActionTagging
    | ActionMoveColor
    | ActionColorJpegify
    | ActionPack
    | ActionInterrupt
)
Actions = Annotated[ActionType, Field(discriminator="kind", description="The collection of all supported actions.")]
"""
The list of all supported actions.
"""

"""
nmanga.orchestrator
~~~~~~~~~~~~~~~~~~~
The orchestrator module for nmanga, handling complex workflows and actions.

:copyright: (c) 2022-present noaione
:license: MIT, see LICENSE for more details.
"""

import json
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError
from typing_extensions import Self

from ..common import ChapterRange
from ..constants import MANGA_PUBLICATION_TYPES, MangaPublication
from .actions import *
from .common import SkipActionConfig, SkipActionKind

__all__ = (
    "ActionAutolevel",
    "ActionColorJpegify",
    "ActionDenoise",
    "ActionInterrupt",
    "ActionKind",
    "ActionMoveColor",
    "ActionOptimize",
    "ActionPack",
    "ActionPosterize",
    "ActionRename",
    "ActionShiftName",
    "ActionSpreads",
    "ActionTagging",
    "Actions",
    "BaseAction",
    "ChapterConfig",
    "CustomJSONEncoder",
    "MetadataNamingConfig",
    "OrchestratorConfig",
    "SkipActionConfig",
    "SkipActionKind",
    "ThreadedResult",
    "ToolsKind",
    "VolumeConfig",
    "WorkerContext",
)


def current_year() -> int:
    # jst tz
    jst_tz = timezone(offset=timedelta(hours=9), name="JST")
    return datetime.now(tz=jst_tz).year


def is_publication_type(pub_type: str) -> str:
    if pub_type not in MANGA_PUBLICATION_TYPES:
        raise ValueError(f"Invalid publication type: {pub_type}")
    return pub_type


class ChapterConfig(BaseModel):
    """
    The configuration for a chapter

    The ending page is implied to of the next chapter's starting page - 1 if not specified.
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator Chapter Configuration",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    number: int | float = Field(..., ge=0.0, title="Chapter Number")
    """The chapter number"""
    title: str | None = Field(None, title="Chapter Title")
    """The title of the chapter, this is optional"""
    start: int = Field(..., ge=0, title="Starting Page Number")
    """The starting page number of the chapter"""


class MetadataNamingConfig(BaseModel):
    """
    The configuration for metadata naming

    This is used to add extra metadata to the filename, e.g. [Cover], [ToC], etc.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    tag: str = Field(..., title="Metadata Tag")
    """The tag to add to the filename"""
    page: int | list[int] = Field(..., title="Page Number")
    """The page number to add the tag to, can be a list for multiple pages"""


class VolumeConfig(BaseModel):
    """
    The configuration for a volume
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator Volume Configuration",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    path: Path = Field(..., title="Volume Directory")
    """The directory of the volume, relative to the base_path"""
    number: int | float = Field(..., title="Volume Number")
    """The volume number, can be float for decimals"""
    year: int = Field(default_factory=current_year, ge=1000, le=9999, title="Volume Year")
    """The year of the volume release, this is used for tagging"""
    oneshot: bool = Field(False, title="Is Oneshot")
    """Whether the volume is a oneshot"""
    spreads: list[tuple[int, int]] = Field(
        default_factory=list, title="Volume Spreads", examples=[[(1, 2), (5, 6), (13, 15)]]
    )
    """
    The list of spreads in the volume, each tuple is a pair of page numbers

    This would be made into a range from the first to the second number inclusive.
    """
    colors: list[int] = Field(default_factory=list, title="Volume Color Pages", examples=[[0]])
    """The list of color tagged pages in the volume"""
    meta_naming: list[MetadataNamingConfig] = Field(default_factory=list, title="Metadata Naming Configurations")
    """The list of metadata naming configurations"""
    chapters: list[ChapterConfig] = Field(default_factory=list, title="Volume Chapters")
    """The list of chapters in the volume"""
    extra_text: str | None = Field(None, title="Extra Text", examples=["Omnibus", "2-in-1 Edition", "PNG4"])
    """Extra text to add to the filename"""
    revision: int = Field(1, ge=1, title="Volume Revision Number")
    """The revision number of the volume, this is used for tagging

    Only applied if greater than 1
    """
    quality: Literal["LQ", "HQ"] | None = Field(None, title="Volume Quality")
    """The quality of the volume, this is used for tagging"""
    pub_type: Annotated[str, AfterValidator(is_publication_type)] = Field(
        "digital",
        title="Publication Type",
        examples=["digital", "magazine", "scan", "web", "digital-raw", "magazine-raw", "mix", "none"],
    )
    """The publication type of the volume, this is used for tagging"""
    skip_actions: list[SkipActionConfig] = Field(default_factory=list, title="Skip Actions Configurations")
    """The list of actions to skip for this volume"""

    @cached_property
    def publication(self) -> MangaPublication:
        """
        A cached property to get the MangaPublication for the volume
        """
        return MANGA_PUBLICATION_TYPES[self.pub_type]

    @cached_property
    def meta_name_maps(self) -> dict[int, str]:
        """
        A cached property mapping page number to metadata tag
        """

        mappings: dict[int, str] = {}
        for meta in self.meta_naming:
            pages = [meta.page] if isinstance(meta.page, int) else meta.page
            for page in pages:
                mappings[page] = meta.tag
        return mappings

    @model_validator(mode="after")
    def check_duplicate_meta_naming(self) -> Self:
        existing_pages = set()
        for meta in self.meta_naming:
            pages = [meta.page] if isinstance(meta.page, int) else meta.page
            for page in pages:
                if page in existing_pages:
                    raise PydanticCustomError(
                        "duplicate_meta_naming",
                        "Duplicate metadata naming page found: {pg}",
                        {"pg": page},
                    )
                existing_pages.add(page)
        return self

    @model_validator(mode="after")
    def check_duplicate_chapters(self) -> Self:
        existing_chapters = set()
        for ch in self.chapters:
            if ch.number in existing_chapters:
                raise PydanticCustomError(
                    "duplicate_chapter",
                    "Duplicate chapter number found: {chapter}",
                    {"chapter": ch.number},
                )
            existing_chapters.add(ch.number)
        return self

    @model_validator(mode="after")
    def check_chapters_pages(self) -> Self:
        # Renamer only require the start page and not end page
        # The `end` page is implied to be the next chapter's start - 1
        chapter_starts = [ch.start for ch in self.chapters]
        if len(chapter_starts) != len(set(chapter_starts)):
            raise PydanticCustomError(
                "duplicate_chapter_start_page",
                "Duplicate chapter start page found in chapters",
            )
        # Check chapter start pages are in ascending order
        if chapter_starts != sorted(chapter_starts):
            raise PydanticCustomError(
                "unsorted_chapter_start_pages",
                "Chapter start pages must be in ascending order",
            )
        return self

    def to_chapter_ranges(self) -> list[ChapterRange]:
        """
        Convert to a list of ChapterRange

        This would be guaranted to have non-overlapping and sorted page ranges.
        """

        ranges: list[ChapterRange] = []
        # We do this reversed to easily get the end page of each chapter
        for idx in range(len(self.chapters) - 1, -1, -1):
            ch = self.chapters[idx]
            start_page = ch.start
            end_page = None
            if idx + 1 < len(self.chapters):
                end_page = self.chapters[idx + 1].start - 1
            if end_page is not None:
                page_range = list(range(start_page, end_page + 1))
            else:
                page_range = [start_page]
            ranges.insert(
                0,
                ChapterRange(
                    number=ch.number,
                    name=ch.title,
                    range=page_range,
                    is_single=end_page is None,
                ),
            )
        return ranges


class OrchestratorConfig(BaseModel):
    """
    The configuration for the orchestrator
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator Configuration",
        strict=True,
        extra="ignore",
        validate_default=True,
    )

    title: str = Field(..., title="Manga Title", examples=["I Want to Love You Till Your Dying Day"])
    """The title of the manga"""
    publisher: str = Field(..., title="Manga Publisher", examples=["Kodansha Comics"])
    """The publisher of the manga"""
    credit: str = Field(..., title="Ripper Credit", examples=["Ripper"])
    """The ripper credit for the manga"""
    email: str = Field(..., title="Ripper Email", examples=["ripper-mail@example.com"])
    """The ripper email for the manga"""
    bracket_type: Literal["square", "round", "curly"] = Field("round", title="Bracket Type in Filename")
    """The bracket type to use for the ripper credit

    This is used in the filename, e.g. [Ripper], (Ripper), {Ripper}
    """
    base_path: Path = Field(Path("source"), title="Base Path for Volumes")
    """The first path to look for volumes, relative to the orchestrator config file"""
    volumes: list[VolumeConfig] = Field(default_factory=list, min_length=1)
    """The list of volumes to process"""
    actions: list[Actions] = Field(default_factory=list, min_length=1)
    """The list of actions to perform on each volume"""

    @model_validator(mode="after")
    def check_volumes_duplicates(self) -> Self:
        existing_volumes = set()
        for vol in self.volumes:
            if vol.number in existing_volumes:
                raise PydanticCustomError(
                    "duplicate_volume",
                    "Duplicate volume number found: {volume}",
                    {"volume": vol.number},
                )
            existing_volumes.add(vol.number)
        return self

    @cached_property
    def action_names(self) -> list[str]:
        """
        A cached property to get the list of action names
        """

        actions_counter: dict[ActionKind, int] = {}
        actions_names = []
        for action in self.actions:
            step_number = actions_counter.get(action.kind, 0) + 1
            actions_counter[action.kind] = step_number
            actions_names.append(f"{action.kind.value}-{step_number}")
        return actions_names

    @property
    def actions_maps(self) -> dict[str, Actions]:
        """
        Get the list of actions with their names
        """

        action_names = self.action_names
        if len(action_names) != len(self.actions):
            raise ValueError("Action names and actions length mismatch")
        return dict(zip(action_names, self.actions, strict=True))

    @model_validator(mode="after")
    def check_no_actions(self) -> Self:
        for idx, action in enumerate(self.actions):
            if not action:
                raise PydanticCustomError(
                    "empty_action",
                    "Action {index} cannot be empty",
                    {
                        "index": idx,
                    },
                )
        return self

    # Validate that skip actions refer to valid action steps
    @model_validator(mode="after")
    def check_skip_actions(self) -> Self:
        valid_steps = set(self.action_names)
        for vol in self.volumes:
            for skip in vol.skip_actions:
                if skip.step not in valid_steps:
                    raise PydanticCustomError(
                        "invalid_skip_step",
                        "Invalid skip action step: {step}. Valid steps are: {valid}",
                        {"step": skip.step, "valid": ", ".join(valid_steps)},
                    )
        return self

    @model_validator(mode="after")
    def check_actions_order(self) -> Self:
        if len(self.actions) != len(self.action_names):
            raise PydanticCustomError(
                "actions_order_mismatch",
                "Actions and action names length mismatch",
            )
        return self


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Enum):
            return o.value
        elif isinstance(o, Path):
            return str(o)
        return super().default(o)

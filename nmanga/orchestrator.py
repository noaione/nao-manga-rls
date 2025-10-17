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

from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import cached_property
from multiprocessing import cpu_count
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError
from typing_extensions import Self

from .common import ChapterRange
from .constants import MANGA_PUBLICATION_TYPES, MangaPublication
from .exporter import ExporterType
from .spreads import SpreadDirection

__all__ = (
    "ActionAutolevel",
    "ActionColorJpegify",
    "ActionDenoise",
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
    "ChapterConfig",
    "MetadataNamingConfig",
    "OrchestratorConfig",
    "SkipActionConfig",
    "SkipActionKind",
    "VolumeConfig",
)


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
    AUTOLEVEL = "autolevel"
    """Auto level all images with Pillow"""
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
    INTERRUPT = "interrupt"
    """Interrupt the action chain"""


class BaseAction(BaseModel):
    """
    The base action model
    """

    conditional: bool = Field(False)
    """Run the action only if a certain condition is met"""


class ActionShiftName(BaseAction):
    """
    Action to shift rename of a volume

    Title and volume are derived from the main config so they are not included here.
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Simple Shift Renamer Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.SHIFT_RENAME] = Field(ActionKind.SHIFT_RENAME, title="Simple Shift Renamer Action")
    """The kind of action"""
    start: int = Field(0, ge=0, title="Starting Index")
    """The starting index to rename the files to"""
    title: str | None = Field(None, title="Title of the Series")
    """Optional override title for the shift rename action"""


class ActionSpreads(BaseAction):
    """
    Action to join spreads of a volume
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Image Spreads Joiner Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.SPREADS] = Field(ActionKind.SPREADS, title="Image Spreads Joiner Action")
    """The kind of action"""
    direction: SpreadDirection = Field(SpreadDirection.LTR, title="Spread Direction")
    """Whether to use reverse mode when joining spreads"""
    quality: float = Field(100.0, ge=1.0, le=100.0, title="Output Quality")
    """The quality of the output images"""
    output_fmt: Literal["auto", "jpg", "png"] = Field("auto", title="Output Format")
    """The output format of the joined images, auto will use the parent extensions"""
    pillow: bool = Field(False, title="Use Python Pillow")
    """Whether to use Pillow for joining spreads instead of ImageMagick"""
    threads: int = Field(default_factory=cpu_count, ge=1, title="Processing Threads")
    """The number of threads to use for processing"""


class ActionRename(BaseAction):
    """
    Action to rename all images in a volume

    This use the daiz-like renaming scheme.

    All the other options are derived from the main config so they are not included here.
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Daiz-like Rename Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.RENAME] = Field(ActionKind.RENAME, title="Daiz-like Rename Images Action")
    """The kind of action"""


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
    model: Path = Field(..., title="The ONNX Model Path")
    """The path to the ONNX model"""
    base_path: Path = Field("denoised", title="Output Base Path")
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


class ActionAutolevel(BaseAction):
    """
    Action to auto level all images in a volume with Pillow

    This version is "smarter" than the CLI version:
    - It would force convert the image to grayscale for half-tones/b&w images before processing
    - All tagged color images will be leveled in keep colorspace mode
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Auto Level Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.AUTOLEVEL] = Field(ActionKind.AUTOLEVEL, title="Auto Level Images Action")
    """The kind of action"""
    base_path: Path = Field("leveled", title="Output Base Path")
    """The base path to save the leveled images to"""
    upper_limit: int = Field(60, ge=1, le=255, title="Upper Limit for Peak Finding")
    """The upper limit for finding local peaks in the histogram"""
    peak_offset: int = Field(0, ge=-20, le=20, title="Peak Offset")
    """The offset to add to the found black level peak, can be negative"""
    min_peak_pct: float = Field(0.25, ge=0.0, le=100.0, title="Minimum Pixels Peak Percentage")
    """The minimum percentage of pixels for a peak to be considered valid"""
    skip_white: bool = Field(True, title="Skip White Levels During Peak Finding")
    """Whether to skip white peaks when finding local peaks in the histogram"""
    skip_color: bool = Field(False, title="Skip Color Images")
    """Skip color images when auto leveling, only level half-tones/b&w images"""
    threads: int = Field(default_factory=cpu_count, ge=1, title="Processing Threads")
    """The number of threads to use for processing"""


class ActionPosterize(BaseAction):
    """
    Action to posterize all images in a volume with imagemagick or Pillow
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Posterize Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.POSTERIZE] = Field(ActionKind.POSTERIZE, title="Posterize Images Action")
    """The kind of action"""
    base_path: Path = Field("posterized", title="Output Base Path")
    """The base path to save the posterized images to"""
    bpc: int = Field(4, ge=1, le=8, title="Bits Per Channel", examples=[1, 2, 4, 8])
    """The number of bitdepth to reduce the image to"""
    pillow: bool = Field(False, title="Use Python Pillow")
    """Whether to use Pillow for posterizing instead of ImageMagick"""
    threads: int = Field(default_factory=cpu_count, ge=1, title="Processing Threads")
    """The number of threads to use for processing"""

    @model_validator(mode="after")
    def check_bpc(self) -> Self:
        if self.bpc not in (1, 2, 4, 8):
            raise PydanticCustomError(
                "invalid_bpc",
                "Bits per channel (bpc) must be one of 1, 2, 4, or 8",
                {"bpc": self.bpc},
            )
        return self


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

    @model_validator(mode="after")
    def check_limiter(self) -> Self:
        if self.limiter is not None and not self.limiter.startswith("."):
            raise PydanticCustomError(
                "invalid_limiter",
                "Limiter must start with a dot, e.g. .png,.jpg",
                {"limiter": self.limiter},
            )
        return self


class ActionTagging(BaseAction):
    """
    Action to add metadata tags to the images with exiftool

    Everything would be derived from the main config so they are not included here.
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Image Tagging Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.TAGGING] = Field(ActionKind.TAGGING, title="Image Tagging Action")
    """The kind of action"""


class ActionMoveColor(BaseAction):
    """
    Action to move the tagged color images to a separate folder

    Everything would be derived from the main config so they are not included here.
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Move Color Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.MOVE_COLOR] = Field(ActionKind.MOVE_COLOR, title="Move Color Images Action")
    """The kind of action"""
    base_path: Path = Field("colors", title="Output Base Path")
    """The base path to save the color images to"""


class ActionColorJpegify(BaseAction):
    """
    Action to convert color images to JPEG format with cjpegli
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Jpegify Color Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.COLOR_JPEGIFY] = Field(ActionKind.COLOR_JPEGIFY, title="Jpegify Color Images Action")
    """The kind of action"""
    base_path: Path | None = Field(None, title="Output Base Path")
    """The base path to save the JPEG images to, this would use the last used base path if not provided"""
    quality: int = Field(95, ge=1, le=100, title="JPEG Quality")
    """The quality of the output JPEG images"""
    source_path: Path = Field("colors", title="Source Path")
    """The source path to look for images to convert"""
    threads: int = Field(default_factory=cpu_count, ge=1, title="Processing Threads")
    """The number of threads to use for processing"""


class ActionPack(BaseAction):
    """
    Action to pack the volume into an archive
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Pack Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.PACK] = Field(ActionKind.PACK, title="Pack Action")
    """The kind of action"""
    output_mode: ExporterType = Field(ExporterType.cbz, title="Output Mode")
    """The output mode to use for packing the archive"""
    source_dir: Path | None = Field(None, title="Source Directory")
    """The source directory to pack, this would use the last used base path if not provided"""


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
    end: int | None = Field(None, ge=1, title="Ending Page Number")
    """
    The ending page number of the chapter, this is optional.
    If not provided it would be the same as start till the end of the volume
    """

    @model_validator(mode="after")
    def check_end(self) -> Self:
        if self.end is not None and self.end < self.start:
            raise PydanticCustomError(
                "end_before_start",
                "Chapter end page {end} cannot be before start page {start}",
                {
                    "end": self.end,
                    "start": self.start,
                },
            )
        return self

    def to_chapter_range(self) -> ChapterRange:
        """
        Convert to ChapterRange
        """

        ranges = [int(self.start)]
        if self.end is not None:
            ranges = list(range(self.start, self.end + 1))
        return ChapterRange(number=self.number, name=self.title, range=ranges, is_single=self.end is None)


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
                    "Duplicate chapter number found: {ch}",  # noqa: RUF027
                    {"ch": ch.number},
                )
            existing_chapters.add(ch.number)
        return self

    @model_validator(mode="after")
    def check_chapters_pages(self) -> Self:
        # Check if there is more than one chapter with no `end` page
        no_end_count = sum(1 for ch in self.chapters if ch.end is None)
        if no_end_count > 1:
            raise PydanticCustomError(
                "multiple_no_end_chapters",
                "There can only be one chapter with no end page, found {cnt}",
                {"cnt": no_end_count},
            )

        # Check if chapters have overlapping pages
        page_ranges = []
        for ch in self.chapters:
            end_page = ch.end if ch.end is not None else float("inf")
            page_ranges.append((ch.start, end_page))
        page_ranges.sort()
        for i in range(1, len(page_ranges)):
            if page_ranges[i][0] <= page_ranges[i - 1][1]:
                raise PydanticCustomError(
                    "overlapping_chapter_pages",
                    "Chapters have overlapping pages: {r1} and {r2}",
                    {"r1": page_ranges[i - 1], "r2": page_ranges[i]},
                )

        # Check for possibly missing pages between chapters
        for i in range(1, len(page_ranges)):
            if page_ranges[i][0] > page_ranges[i - 1][1] + 1:
                raise PydanticCustomError(
                    "missing_chapter_pages",
                    "There are possibly missing pages between chapters: {r1} and {r2}",
                    {"r1": page_ranges[i - 1], "r2": page_ranges[i]},
                )
        return self


class OrchestratorConfig(BaseModel):
    """
    The configuration for the orchestrator
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator Configuration",
        strict=True,
        extra="forbid",
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
    base_path: Path = Field("source")
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
        return dict(zip(action_names, self.actions))

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

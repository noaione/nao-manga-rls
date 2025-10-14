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
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, model_validator
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
    "ActionRename",
    "ActionShiftName",
    "ActionSpreads",
    "ActionTagging",
    "Actions",
    "ChapterConfig",
    "MetadataNamingConfig",
    "OrchestratorConfig",
    "SkipActionConfig",
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


class ActionShiftName(BaseModel):
    """
    Action to shift rename of a volume

    Title and volume are derived from the main config so they are not included here.
    """

    kind: Literal[ActionKind.SHIFT_RENAME] = Field(ActionKind.SHIFT_RENAME)
    """The kind of action"""
    start: int = Field(0, ge=0)
    """The starting index to rename the files to"""
    title: str | None = Field(None)
    """Optional override title for the shift rename action"""


class ActionSpreads(BaseModel):
    """
    Action to join spreads of a volume
    """

    kind: Literal[ActionKind.SPREADS] = Field(ActionKind.SPREADS)
    """The kind of action"""
    direction: SpreadDirection = Field(SpreadDirection.LTR)
    """Whether to use reverse mode when joining spreads"""
    quality: float = Field(100.0, ge=1.0, le=100.0)
    """The quality of the output images"""
    output_fmt: Literal["auto", "jpg", "png"] = Field("auto")
    """The output format of the joined images, auto will use the parent extensions"""
    pillow: bool = Field(False)
    """Whether to use Pillow for joining spreads instead of ImageMagick"""
    threads: int = Field(default_factory=cpu_count, ge=1)
    """The number of threads to use for processing"""


class ActionRename(BaseModel):
    """
    Action to rename all images in a volume

    This use the daiz-like renaming scheme.

    All the other options are derived from the main config so they are not included here.
    """

    kind: Literal[ActionKind.RENAME] = Field(ActionKind.RENAME)
    """The kind of action"""


class ActionDenoise(BaseModel):
    """
    Action to denoise all images in a volume with denoise-trt
    """

    kind: Literal[ActionKind.DENOISE] = Field(ActionKind.DENOISE)
    """The kind of action"""
    model: Path
    """The path to the ONNX model"""
    base_path: Path = Field("denoised")
    """The base path to save the denoised images to"""
    device_id: int = Field(0, ge=0)
    """The device ID to use for denoising"""
    batch_size: int = Field(64, ge=1)
    """The batch size to use for denoising"""
    tile_size: int = Field(128, ge=64)
    """The tile size to use for denoising"""
    background: Literal["white", "black"] = Field("black")
    """The background color to use for padding"""
    contrast_strectch: bool = Field(False)
    """Whether to apply contrast stretch after denoising"""


class ActionAutolevel(BaseModel):
    """
    Action to auto level all images in a volume with Pillow

    This version is "smarter" than the CLI version:
    - It would force convert the image to grayscale for half-tones/b&w images before processing
    - All tagged color images will be leveled in keep colorspace mode
    """

    kind: Literal[ActionKind.AUTOLEVEL] = Field(ActionKind.AUTOLEVEL)
    """The kind of action"""
    base_path: Path = Field("leveled")
    """The base path to save the leveled images to"""
    upper_limit: int = Field(60, ge=1, le=255)
    """The upper limit for finding local peaks in the histogram"""
    peak_offset: int = Field(0, ge=-20, le=20)
    """The offset to add to the found black level peak, can be negative"""
    skip_white: bool = Field(True)
    """Whether to skip white peaks when finding local peaks in the histogram"""
    skip_first: bool = Field(True)
    """Always skip the first image in the volume, useful for omitting cover pages (and just copying)"""
    threads: int = Field(default_factory=cpu_count, ge=1)
    """The number of threads to use for processing"""


class ActionOptimize(BaseModel):
    """
    Optimize all images in a volume with pingo
    """

    kind: Literal[ActionKind.OPTIMIZE] = Field(ActionKind.OPTIMIZE)
    """The kind of action"""
    aggresive: bool = Field(False)
    """Whether to use the aggressive mode of pingo, which would force grayscale conversion for all images"""
    limiter: str | None = Field(None)
    """Limit the optimization to certain image types, e.g. png,jpg"""


class ActionTagging(BaseModel):
    """
    Action to add metadata tags to the images with exiftool

    Everything would be derived from the main config so they are not included here.
    """

    kind: Literal[ActionKind.TAGGING] = Field(ActionKind.TAGGING)
    """The kind of action"""


class ActionMoveColor(BaseModel):
    """
    Action to move the tagged color images to a separate folder

    Everything would be derived from the main config so they are not included here.
    """

    kind: Literal[ActionKind.MOVE_COLOR] = Field(ActionKind.MOVE_COLOR)
    """The kind of action"""
    base_path: Path = Field("colors")
    """The base path to save the color images to"""


class ActionColorJpegify(BaseModel):
    """
    Action to convert color images to JPEG format with cjpegli
    """

    kind: Literal[ActionKind.COLOR_JPEGIFY] = Field(ActionKind.COLOR_JPEGIFY)
    """The kind of action"""
    base_path: Path | None = Field(None)
    """The base path to save the JPEG images to, this would use the last used base path if not provided"""
    quality: int = Field(95, ge=1, le=100)
    """The quality of the output JPEG images"""
    source_path: Path = Field("colors")
    """The source path to look for images to convert"""
    threads: int = Field(default_factory=cpu_count, ge=1)
    """The number of threads to use for processing"""


class ActionPack(BaseModel):
    """
    Action to pack the volume into an archive
    """

    kind: Literal[ActionKind.PACK]
    """The kind of action"""
    output_mode: ExporterType = Field(ExporterType.cbz)
    """The output mode to use for packing the archive"""
    source_dir: Path | None = Field(None)
    """The source directory to pack, this would use the last used base path if not provided"""


Actions = Annotated[
    ActionShiftName
    | ActionSpreads
    | ActionRename
    | ActionDenoise
    | ActionAutolevel
    | ActionOptimize
    | ActionTagging
    | ActionMoveColor
    | ActionColorJpegify
    | ActionPack,
    Field(discriminator="kind"),
]
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

    number: int | float = Field(..., ge=0.0)
    """The chapter number"""
    title: str | None = Field(None)
    """The title of the chapter, this is optional"""
    start: int = Field(..., ge=0)
    """The starting page number of the chapter"""
    end: int | None = Field(None, ge=1)
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


class SkipActionConfig(BaseModel):
    """
    What actions to skip for a volume

    Note: Not all actions support skipping, those that do not will be ignored
    """

    step: int = Field(..., ge=1)
    """The step number of the action to skip, 1-based index"""
    pages: list[int] = Field(..., min_length=1)
    """The list of page numbers to skip the action on"""


class MetadataNamingConfig(BaseModel):
    """
    The configuration for metadata naming

    This is used to add extra metadata to the filename, e.g. [Color], [LQ], etc.
    """

    tag: str
    """The tag to add to the filename"""
    page: int | list[int]
    """The page number to add the tag to, can be a list for multiple pages"""


class VolumeConfig(BaseModel):
    """
    The configuration for a volume
    """

    path: Path
    """The directory of the volume, relative to the base_path"""
    number: int | float
    """The volume number"""
    year: int = Field(default_factory=current_year, ge=1000, le=9999)
    """The year of the volume, this is used for tagging"""
    oneshot: bool = Field(False)
    """Whether the volume is a oneshot"""
    spreads: list[tuple[int, int]] = Field(default_factory=list)
    """
    The list of spreads in the volume, each tuple is a pair of page numbers

    This would be made into a range from the first to the second number inclusive.
    """
    colors: list[int] = Field(default_factory=list)
    """The list of color tagged pages in the volume"""
    meta_naming: list[MetadataNamingConfig] = Field(default_factory=list)
    """The list of metadata naming configurations"""
    chapters: list[ChapterConfig] = Field(..., min_length=1)
    """The list of chapters in the volume"""
    extra_text: str | None = Field(None)
    """Extra text to add to the filename"""
    revision: int = Field(1, ge=1)
    """The revision number of the volume, this is used for tagging"""
    quality: Literal["LQ", "HQ"] | None = Field(None)
    """The quality of the volume, this is used for tagging"""
    pub_type: Annotated[str, AfterValidator(is_publication_type)] = Field("digital")
    """The publication type of the volume, this is used for tagging"""
    skip_actions: list[SkipActionConfig] = Field(default_factory=list)
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

    title: str
    """The title of the manga"""
    publisher: str
    """The publisher of the manga"""
    credit: str
    """The ripper credit for the manga"""
    email: str
    """The ripper email for the manga"""
    bracket_type: Literal["square", "round", "curly"] = Field("round")
    """The bracket type to use for the ripper credit"""
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

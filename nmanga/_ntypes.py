from __future__ import annotations

from typing import Literal, TypedDict

__all__ = (
    "BracketTypeT",
    "ConfigT",
)


BracketTypeT = Literal["square", "round", "curly"]


class _ConfigDefaultsT(TypedDict, total=False):
    bracket_type: BracketTypeT
    ripper_credit: str
    ripper_email: str
    chapter_add_c_prefix: bool
    chapter_special_tag: str
    release_publication_type: str
    release_chapter_publication_type: str


class _ConfigExecutableT(TypedDict, total=False):
    magick_path: str
    pingo_path: str
    exiftool_path: str
    cjpegli_path: str
    w2x_trt_path: str | None


class _ConfigExperimentsT(TypedDict, total=False):
    png_tag: bool


class ConfigT(TypedDict, total=False):
    _is_first_time: bool
    defaults: _ConfigDefaultsT
    executables: _ConfigExecutableT
    experimentals: _ConfigExperimentsT

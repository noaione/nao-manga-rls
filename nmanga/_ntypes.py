from __future__ import annotations

from typing import Literal, TypedDict

__all__ = (
    "STOP_SIGNAL",
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
    lq_threshold: int
    hq_threshold: int


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


class _Sentinel:
    """
    A singleton sentinel.
    You can safely use `is` to check for it, even across processes.
    """

    __slots__ = ()

    def __reduce__(self):
        # This tells pickle to rebuild the object by
        # looking up the global variable "STOP_SIGNAL"
        return "STOP_SIGNAL"

    def __eq__(self, other) -> bool:
        # `is` is preferred, but this makes `==` work too
        return self is other

    def __bool__(self) -> bool:
        return False

    def __hash__(self) -> int:
        return 0

    def __repr__(self):
        return "STOP_SIGNAL"


# Create the one-and-only instance
STOP_SIGNAL = _Sentinel()

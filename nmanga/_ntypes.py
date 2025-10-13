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


class _ConfigExecutableT(TypedDict, total=False):
    magick_path: str
    pingo_path: str
    exiftool_path: str
    w2x_trt_path: str | None


class ConfigT(TypedDict, total=False):
    _is_first_time: bool
    defaults: _ConfigDefaultsT
    executables: _ConfigExecutableT

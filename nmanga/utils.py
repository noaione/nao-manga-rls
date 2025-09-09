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

import re
import sys
from typing import Optional, Union

__all__ = (
    "clean_title",
    "decode_or",
    "encode_or",
    "is_not_volume_number",
    "is_oneshot",
    "secure_filename",
    "unsecure_filename",
)


def secure_filename(fn: str):
    replacement = {
        "/": "／",  # noqa: RUF001
        ":": "：",  # noqa: RUF001
        "<": "＜",  # noqa: RUF001
        ">": "＞",  # noqa: RUF001
        '"': "”",
        # "'": "’",  # noqa: RUF003
        "\\": "＼",  # noqa: RUF001
        "?": "？",  # noqa: RUF001
        "*": "⋆",
        "|": "｜",  # noqa: RUF001
        "#": "",
    }
    for k, v in replacement.items():
        fn = fn.replace(k, v)
    EMOJI_PATTERN = re.compile(
        "([" + "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f700-\U0001f77f"  # alchemical symbols
        "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
        "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
        "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
        "\U0001fa00-\U0001fa6f"  # Chess Symbols
        "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027b0"  # Dingbats
        "])"
    )
    fn = re.sub(EMOJI_PATTERN, "_", fn)
    return fn


def unsecure_filename(fn: str):
    # Remap back to original.
    # Only works on Linux-based OS.
    replacement = {
        "：": ":",  # noqa: RUF001
        "＜": "<",  # noqa: RUF001
        "＞": ">",  # noqa: RUF001
        "”": '"',
        "“": '"',
        "’": "'",  # noqa: RUF001
        "‘": "'",  # noqa: RUF001
        "＼": "\\",  # noqa: RUF001
        "？": "?",  # noqa: RUF001
        "⋆": "*",
        "｜": "|",  # noqa: RUF001
    }

    if sys.platform == "win32":
        return fn

    for k, v in replacement.items():
        fn = fn.replace(k, v)
    return fn


def clean_title(title: str):
    if not title:
        return title
    if title.endswith("]"):
        title = title[:-1]
    return title


def is_oneshot(title: str):
    title = title.lower()
    valid_oshot = [
        "na",
        "oshot",
        "oneshot",
        "one-shot",
        "one shot",
    ]
    return title in valid_oshot


def is_not_volume_number(volume: str):
    if not volume:
        return True
    if is_oneshot(volume):
        return True

    # volume number is formatted like v01, v02, v003, etc.
    if re.match(r"^v[\d]+$", volume):
        return False
    return True


def decode_or(any: Optional[Union[str, bytes]]) -> Optional[str]:
    if any is None:
        return None
    if isinstance(any, bytes):
        return any.decode("utf-8")
    return any


def encode_or(any: Optional[Union[str, bytes]]) -> Optional[bytes]:
    if any is None:
        return None
    if isinstance(any, str):
        return any.encode("utf-8")
    return any

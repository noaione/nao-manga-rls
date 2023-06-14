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
    "secure_filename",
    "unsecure_filename",
    "clean_title",
    "is_oneshot",
    "decode_or",
    "encode_or",
)


def secure_filename(fn: str):
    replacement = {
        "/": "／",
        ":": "：",
        "<": "＜",
        ">": "＞",
        '"': "”",
        # "'": "’",
        "\\": "＼",
        "?": "？",
        "*": "⋆",
        "|": "｜",
        "#": "",
    }
    for k, v in replacement.items():
        fn = fn.replace(k, v)
    EMOJI_PATTERN = re.compile(
        "(["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "])"
    )
    fn = re.sub(EMOJI_PATTERN, "_", fn)
    return fn


def unsecure_filename(fn: str):
    # Remap back to original.
    # Only works on Linux-based OS.
    replacement = {
        "：": ":",
        "＜": "<",
        "＞": ">",
        "”": '"',
        "“": '"',
        "’": "'",
        "‘": "'",
        "＼": "\\",
        "？": "?",
        "⋆": "*",
        "｜": "|",
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

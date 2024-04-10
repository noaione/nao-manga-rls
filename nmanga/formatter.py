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

from string import Formatter
from typing import Any, Dict

__all__ = (
    "OptionalFormatter",
    "format_with_optional",
)


class _OptinalDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class OptionalFormatter:
    def __init__(self, data: Dict[str, Any]) -> None:
        self.fmt = Formatter()
        self.data = _OptinalDict(data)

    @classmethod
    def format(cls, text: str, **kwargs: Any):
        formatter = cls(kwargs)
        return formatter.fmt.vformat(text, (), formatter.data)


def format_with_optional(text: str, **kwargs: Any) -> str:
    return OptionalFormatter.format(text, **kwargs)

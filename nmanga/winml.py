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

# Custom solution to load WinML, compatible with Python 3.14

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ("get_winml_ep_libraries",)


def win():
    return sys.platform == "win32"


def get_winml_ep_libraries(exclusion_list: list[str] | None = None) -> list[tuple[str, Path]]:
    if not win():
        return []

    excluded = set(exclusion_list or [])

    from windowsml import EpCatalog, EpReadyState

    new_sets: list[tuple[str, Path]] = []
    with EpCatalog() as catalog:
        providers = catalog.find_all_providers()
        for provider in providers:
            if provider.name in excluded:
                continue

            # if not, let's play
            provider.ensure_ready()
            if provider.ready_state == EpReadyState.Ready:
                new_sets.append((provider.name, Path(provider.library_path)))
    return new_sets

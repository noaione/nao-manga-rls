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
from __future__ import annotations

from dataclasses import dataclass

__all__ = (
    "MANGA_PUBLICATION_TYPES",
    "TARGET_FORMAT",
    "TARGET_FORMAT_ALT",
    "TARGET_TITLE",
    "MangaPublication",
)


@dataclass
class MangaPublication:
    image: str
    """Used in the image filename"""
    archive: str
    """Used in the archive filename"""
    description: str
    """Description, this will be used in the config command for what it means."""


MANGA_PUBLICATION_TYPES = {
    "digital": MangaPublication("dig", "Digital", "Digital manga release from Amazon, etc."),
    "magazine": MangaPublication("mag", "c2c", "Magazine release, can be used for scanlation batch."),
    "scan": MangaPublication("c2c", "c2c", "Scanned manga release, physical turned to digital."),
    "web": MangaPublication(
        "web", "Digital", "Webcomic release, from many kind of website that are not using volume format."
    ),
    "digital-raw": MangaPublication("raw-d", "Digital", "Digital raw release, from Amazon.co.jp, etc."),
    "magazine-raw": MangaPublication("raw-m", "c2c", "Magazine raw release, can be used for scanlation batch."),
    "mix": MangaPublication("mix", "Digital", "Mixed release format"),
    "none": MangaPublication("", "Digital", "No publication type used in image filename, Digital in archive filename"),
}

TARGET_FORMAT = "{mt} - c{ch}{chex} ({vol}) - p{pg}{ex}{pt} [{t}] [{pb}] [{c}]"
TARGET_FORMAT_ALT = "{mt} - c{ch}{chex} ({vol}) - p{pg}{ex}{pt} [{pb}] [{c}]"
TARGET_TITLE = "{mt}{vol} ({year}) {ex}({pt}) {cpa}{c}{cpb}"

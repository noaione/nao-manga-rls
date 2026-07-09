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
from decimal import Decimal
from enum import Enum
from typing import TypeAlias

import pikepdf

__all__ = ("PdfBoxExpansion", "expand_page_cropping", "get_pdf_box")

BoxArea: TypeAlias = tuple[float, float, float, float]


class PdfCropKind(str, Enum):
    Crop = "/CropBox"
    Media = "/MediaBox"
    Bleed = "/BleedBox"
    Trim = "/TrimBox"
    Art = "/ArtBox"


@dataclass(kw_only=True)
class PdfBoxExpansion:
    top: float
    right: float
    bottom: float
    left: float


def assume_box_area(numbers: list[float]) -> tuple[float, float, float, float]:
    if len(numbers) != 4:
        raise ValueError(f"Expected 4 numbers, got {len(numbers)}")
    return (numbers[0], numbers[1], numbers[2], numbers[3])


def get_pdf_box(page: pikepdf.Page, box_type: PdfCropKind, *, default_box: PdfCropKind = PdfCropKind.Media) -> BoxArea:
    node = page
    while node is not None:
        if box_type.value in node:
            box = node[box_type.value].as_list()
            return assume_box_area([float(v) for v in box])
        node = node.get("/Parent")

    box = page[default_box.value].as_list()
    return assume_box_area([float(v) for v in box])


def _float_dec(num: float) -> Decimal:
    return Decimal(str(round(num, 4)))


def expand_page_cropping(page: pikepdf.Page, box: PdfBoxExpansion, *, box_type: PdfCropKind = PdfCropKind.Crop) -> None:
    x0, y0, x1, y1 = get_pdf_box(page, box_type)
    new_crop_box = [
        _float_dec(max(0.0, x0 - box.left)),
        _float_dec(max(0.0, y0 - box.top)),
        _float_dec(x1 + box.right),
        _float_dec(y1 + box.bottom),
    ]

    page[box_type.value] = pikepdf.Array(new_crop_box)

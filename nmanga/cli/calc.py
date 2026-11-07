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

import re

import rich_click as click

_DIMENSION_PATTERN = re.compile(r"^\s*(\d+)\s*[x×,]\s*(\d+)\s*$")  # ruff: ignore[ambiguous-unicode-character-string]
_DPI_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d*)?|\.\d+)(?:\s*[x×,]\s*(\d+(?:\.\d*)?|\.\d+))?\s*$",  # ruff: ignore[ambiguous-unicode-character-string]
)
_SIZE_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d*)?|\.\d+)\s*[x×,]\s*(\d+(?:\.\d*)?|\.\d+)\s*(in(?:ch(?:es)?)?|cm|mm)\s*$",  # ruff: ignore[ambiguous-unicode-character-string]
    re.IGNORECASE,
)
_UNIT_TO_INCHES = {"in": 1.0, "inch": 1.0, "inches": 1.0, "cm": 1 / 2.54, "mm": 1 / 25.4}


@click.group(name="calc", help="Calculate image and book measurements")
def calc_group() -> None:
    """Calculate image and book measurements."""


@calc_group.command(
    name="dpi",
    help="Calculate DPI from pixel resolution and physical page size",
)
@click.argument("resolution", help="Pixel width x height, for example 2400x3300")
@click.argument("page_size", help="Physical width x height followed by in, cm, or mm, for example 5.83x8.27in")
def calc_dpi(resolution: str, page_size: str) -> None:
    """Calculate horizontal and vertical DPI.

    RESOLUTION is pixel width x height, for example 2400x3300.
    PAGE_SIZE is physical width x height followed by in, cm, or mm,
    for example 5.83x8.27in or 148x210mm.
    """
    resolution_match = _DIMENSION_PATTERN.fullmatch(resolution)
    if resolution_match is None:
        raise click.BadParameter(
            "use pixel width x height, for example 2400x3300",
            param_hint="resolution",
        )

    size_match = _SIZE_PATTERN.fullmatch(page_size)
    if size_match is None:
        raise click.BadParameter(
            "use physical width x height followed by in, cm, or mm, for example 5.83x8.27in",
            param_hint="page_size",
        )

    pixel_width, pixel_height = map(int, resolution_match.groups())
    physical_width, physical_height = map(float, size_match.groups()[:2])
    unit = size_match.group(3).lower()

    if pixel_width <= 0 or pixel_height <= 0:
        raise click.BadParameter("pixel dimensions must be greater than zero", param_hint="resolution")
    if physical_width <= 0 or physical_height <= 0:
        raise click.BadParameter("physical dimensions must be greater than zero", param_hint="page_size")

    inches_per_unit = _UNIT_TO_INCHES[unit]
    horizontal_dpi = pixel_width / (physical_width * inches_per_unit)
    vertical_dpi = pixel_height / (physical_height * inches_per_unit)

    click.echo(f"Horizontal DPI: {horizontal_dpi:.2f}")
    click.echo(f"Vertical DPI:   {vertical_dpi:.2f}")


@calc_group.command(
    name="size",
    help="Calculate pixel dimensions from DPI and physical page size",
)
@click.argument("dpi", help="DPI as one value or width x height, for example 300 or 300x300")
@click.argument("page_size", help="Physical width x height followed by in, cm, or mm, for example 5.83x8.27in")
def calc_size(dpi: str, page_size: str) -> None:
    """Calculate image pixel dimensions from DPI and physical page size.

    DPI can be one value for both dimensions, or width x height, for example
    300 or 300x300. PAGE_SIZE is physical width x height followed by in, cm,
    or mm, for example 5.83x8.27in or 148x210mm.
    """
    dpi_match = _DPI_PATTERN.fullmatch(dpi)
    if dpi_match is None:
        raise click.BadParameter(
            "use one DPI value or width x height, for example 300 or 300x300",
            param_hint="dpi",
        )

    size_match = _SIZE_PATTERN.fullmatch(page_size)
    if size_match is None:
        raise click.BadParameter(
            "use physical width x height followed by in, cm, or mm, for example 5.83x8.27in",
            param_hint="page_size",
        )

    horizontal_dpi = float(dpi_match.group(1))
    vertical_dpi = float(dpi_match.group(2) or dpi_match.group(1))
    physical_width, physical_height = map(float, size_match.groups()[:2])
    unit = size_match.group(3).lower()

    if horizontal_dpi <= 0 or vertical_dpi <= 0:
        raise click.BadParameter("DPI values must be greater than zero", param_hint="dpi")
    if physical_width <= 0 or physical_height <= 0:
        raise click.BadParameter("physical dimensions must be greater than zero", param_hint="page_size")

    inches_per_unit = _UNIT_TO_INCHES[unit]
    pixel_width = round(physical_width * inches_per_unit * horizontal_dpi)
    pixel_height = round(physical_height * inches_per_unit * vertical_dpi)

    click.echo(f"Image size: {pixel_width}x{pixel_height} px")

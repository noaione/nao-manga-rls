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

from pathlib import Path
from typing import Literal

import pikepdf
import rich_click as click

from .. import term
from ..pdfs import PdfBoxExpansion, PdfCropKind, expand_page_cropping, expand_wide_page, is_wide_spread_page
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler

__all__ = ("pdf_group",)

console = term.get_console()

BOX_KIND_NAMES: dict[str, PdfCropKind] = {
    "crop": PdfCropKind.Crop,
    "media": PdfCropKind.Media,
    "bleed": PdfCropKind.Bleed,
    "trim": PdfCropKind.Trim,
    "art": PdfCropKind.Art,
}


@click.group(
    name="pdf",
    help="PDF operation toolsets",
)
def pdf_group():
    """PDF operation toolsets"""
    pass


@pdf_group.command(
    name="expand",
    help="Expand a PDF page bounding box by a fixed amount and write a new PDF",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=False, disable_folder=True)
@options.dest_output(file_okay=True, dir_okay=False, optional=False)
@click.option(
    "-e",
    "--expand",
    "box_expansion",
    type=options.PDF_BOX,
    required=True,
    help=(
        "Amount to expand the box by, in PDF points (72pt = 1 inch). Accepts 1 to 4 numbers as a CSS-like "
        "shorthand: `10` (all sides), `10/20` (top/bottom, left/right), `10/20/30` (top, left/right, bottom), "
        "or `10/20/30/40` (top, right, bottom, left)"
    ),
    panel="Expansion Options",
)
@click.option(
    "-p",
    "--pages",
    "pages",
    type=options.PAGES_RANGE,
    default=None,
    help="Pages to expand, e.g. `1`, `1,3`, `2-5`, `odd`, `even`, `odd,10-12` (default: all pages)",
    panel="Input Options",
)
@click.option(
    "-b",
    "--box",
    "box_type",
    type=click.Choice(list(BOX_KIND_NAMES.keys())),
    default="crop",
    show_default=True,
    help="Which PDF box to expand",
    panel="Input Options",
)
@options.force
@time_program
def pdf_expand(
    path_or_archive: Path,
    dest_output: Path,
    box_expansion: PdfBoxExpansion,
    pages: options.AnyPageRange | None,
    box_type: str,
    force: bool,
) -> None:
    """
    Expand a PDF page bounding box by a fixed amount and write a new PDF.
    """

    if not path_or_archive.is_file():
        raise click.BadParameter(
            f"{path_or_archive} is not a valid PDF file.",
            param_hint="path_or_archive",
        )
    if path_or_archive.suffix.lower() != ".pdf":
        raise click.BadParameter(
            f"{path_or_archive} is not a PDF file (expected .pdf extension).",
            param_hint="path_or_archive",
        )
    if dest_output.exists() and not force:
        raise click.BadParameter(
            f"{dest_output} already exists, use --force to overwrite it.",
            param_hint="dest_output",
        )

    kind = BOX_KIND_NAMES[box_type]

    console.info(f"Opening PDF file: {path_or_archive}")
    with pikepdf.open(path_or_archive, allow_overwriting_input=force) as pdf:
        total = len(pdf.pages)
        target_pages = set(pages.iterate(total)) if pages is not None else set(range(total))

        modified = 0
        for i, page in enumerate(pdf.pages):
            if i in target_pages:
                expand_page_cropping(page, box_expansion, box_type=kind)
                modified += 1

        dest_output.parent.mkdir(parents=True, exist_ok=True)
        pdf.save(dest_output)

    skipped = total - modified
    console.info(f"Processed {total} page(s): {modified} expanded, {skipped} unchanged.")
    console.info(
        f"Expansion — top: {box_expansion.top}pt  right: {box_expansion.right}pt  "
        f"bottom: {box_expansion.bottom}pt  left: {box_expansion.left}pt"
    )
    console.info(f"Box type  : {kind.name}")
    console.info(f"Output    : {dest_output}")


@pdf_group.command(
    name="wide",
    help="Detect spread pages cropped down to a fraction of the MediaBox and expand the box back out",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=False, disable_folder=True)
@options.dest_output(file_okay=True, dir_okay=False, optional=False)
@click.option(
    "-s",
    "--side",
    "side",
    type=click.Choice(["left", "right"]),
    default="right",
    show_default=True,
    help="Which side to expand towards on detected wide pages",
    panel="Expansion Options",
)
@click.option(
    "-fc",
    "--factor",
    "factor",
    type=click.FloatRange(min=1.0, min_open=True),
    default=2.0,
    show_default=True,
    help="A page is considered wide if its MediaBox is at least this many times wider than its box",
    panel="Expansion Options",
)
@click.option(
    "-p",
    "--pages",
    "pages",
    type=options.PAGES_RANGE,
    default=None,
    help="Pages to check, e.g. `1`, `1,3`, `2-5`, `odd`, `even`, `odd,10-12` (default: all pages)",
    panel="Input Options",
)
@click.option(
    "-b",
    "--box",
    "box_type",
    type=click.Choice(list(BOX_KIND_NAMES.keys())),
    default="crop",
    show_default=True,
    help="Which PDF box to check and expand",
    panel="Input Options",
)
@options.force
@time_program
def pdf_wide(
    path_or_archive: Path,
    dest_output: Path,
    side: Literal["left", "right"],
    factor: float,
    pages: options.AnyPageRange | None,
    box_type: str,
    force: bool,
) -> None:
    """
    Detect spread pages cropped down to a fraction of the MediaBox and expand the box back out.
    """

    if not path_or_archive.is_file():
        raise click.BadParameter(
            f"{path_or_archive} is not a valid PDF file.",
            param_hint="path_or_archive",
        )
    if path_or_archive.suffix.lower() != ".pdf":
        raise click.BadParameter(
            f"{path_or_archive} is not a PDF file (expected .pdf extension).",
            param_hint="path_or_archive",
        )
    if dest_output.exists() and not force:
        raise click.BadParameter(
            f"{dest_output} already exists, use --force to overwrite it.",
            param_hint="dest_output",
        )

    kind = BOX_KIND_NAMES[box_type]

    console.info(f"Opening PDF file: {path_or_archive}")
    with pikepdf.open(path_or_archive, allow_overwriting_input=force) as pdf:
        total = len(pdf.pages)
        target_pages = set(pages.iterate(total)) if pages is not None else set(range(total))

        wide_pages = [
            i
            for i, page in enumerate(pdf.pages)
            if i in target_pages and is_wide_spread_page(page, box_type=kind, factor=factor)
        ]

        if not wide_pages:
            console.warning("No wide pages detected, output will be identical to input.")
        else:
            page_list = ", ".join(str(i + 1) for i in wide_pages)
            console.info(f"Found {len(wide_pages)} wide page(s): {page_list}")

        for i in wide_pages:
            expand_wide_page(pdf.pages[i], box_type=kind, side=side)

        dest_output.parent.mkdir(parents=True, exist_ok=True)
        pdf.save(dest_output)

    skipped = total - len(wide_pages)
    console.info(f"Processed {total} page(s): {len(wide_pages)} widened, {skipped} unchanged.")
    console.info(f"Side      : {side}")
    console.info(f"Factor    : {factor}")
    console.info(f"Box type  : {kind.name}")
    console.info(f"Output    : {dest_output}")

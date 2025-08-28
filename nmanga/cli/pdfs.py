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

# PDF management utilities

from pathlib import Path
from typing import Optional

import click
import pymupdf
from pymupdf.utils import get_image_info

from .. import term
from ..pdfs import PDFColorspaceGenerate, determine_dpi_from_width, generate_image_from_page
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler

console = term.get_console()


@click.group(name="pdf", help="PDF management utilities")
def pdf_manager():
    pass


pdf_file = click.argument(
    "pdf_file",
    metavar="PDF_FILE",
    required=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False, path_type=Path),
)


@pdf_manager.command(
    name="identify",
    help="List all the images DPI in the PDF to help when exporting",
    cls=NMangaCommandHandler,
)
@pdf_file
@time_program
def identify_dpi(pdf_file: Path):
    console.info(f"Identifying images in {pdf_file}")

    doc = pymupdf.Document(str(pdf_file))
    page_count = doc.page_count
    console.info(f"Calculating DPI for {page_count} pages...")
    for page_num in range(page_count):
        console.status(f"Calculating DPI... (Page {page_num + 1}/{page_count})")
        page = doc.load_page(page_num)
        images = get_image_info(page, hashes=False, xrefs=True)
        if not images:
            console.info(f"- Page {(page_num + 1):03d}: No images found")
            continue

        maximum_width: int = max(img.get("width", 0) for img in images)

        dpi_calc = determine_dpi_from_width(page, maximum_width)
        console.info(f"- Page {(page_num + 1):03d}: {dpi_calc} DPI")

    console.stop_status(f"Calculated DPI for {page_count} pages.")
    doc.close()


@pdf_manager.command(
    name="export",
    help="Export PDF pages as images",
    cls=NMangaCommandHandler,
)
@pdf_file
@options.dest_output()
@click.option(
    "-d",
    "--dpi",
    "dpi",
    required=False,
    default=300,
    type=int,
    help="The DPI to use when exporting the PDF pages as images",
)
@click.option(
    "-c",
    "--colorspace",
    "colorspace",
    required=False,
    type=click.Choice(PDFColorspaceGenerate, case_sensitive=False),
)
@click.option(
    "--alpha/--no-alpha",
    "with_alpha",
    default=False,
    help="Whether to include alpha channel when exporting images",
)
@click.option(
    "--force-cmyk/--no-force-cmyk",
    "force_cmyk",
    default=False,
    help="Force CMYK colorspace when exporting images, this would use PAM instead of PNG",
)
@time_program
def export_pdf(
    pdf_file: Path,
    dest_output: Path,
    dpi: int,
    colorspace: Optional[PDFColorspaceGenerate],
    with_alpha: bool,
    force_cmyk: bool,
):
    """
    Export PDF pages as images.

    This will export each page of the PDF as a PNG image in the specified output directory.
    """

    console.info(f"Exporting images from {pdf_file} to {dest_output} at {dpi} DPI")
    doc = pymupdf.Document(str(pdf_file))
    page_count = doc.page_count
    dest_output.mkdir(parents=True, exist_ok=True)

    console.status("Exporting images...")
    for page_num in range(page_count):
        console.status(f"Exporting images... (Page {page_num + 1}/{page_count})")

        page = doc.load_page(page_num)
        image, fmt_ext = generate_image_from_page(
            page=page, dpi=dpi, force_colorspace=colorspace, with_alpha=with_alpha, coerce_cmyk=not force_cmyk
        )
        output_name = f"page{(page_num + 1):06d}.{fmt_ext}"
        if isinstance(image, bytes):
            (dest_output / output_name).write_bytes(image)
        else:
            image.save(dest_output / output_name, format="PNG")

    console.stop_status(f"Exported {page_count} pages.")
    doc.close()

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

from __future__ import annotations

import pprint
from io import BytesIO
from pathlib import Path

import pymupdf
import rich_click as click
from PIL import Image

from .. import term
from ..autolevel import apply_levels, find_local_peak, gamma_correction
from ..pdfs import PDFColorspaceGenerate, determine_dpi_from_width, extract_images_from_pdf, generate_image_from_page
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
    console.info(f"Number of pages: {doc.page_count}")
    console.info(f"Metadata: {pprint.pformat(doc.metadata)}")
    page_count = doc.page_count
    console.info(f"Calculating DPI for {page_count} pages...")
    for page_num in range(page_count):
        console.status(f"Calculating DPI... (Page {page_num + 1}/{page_count})")
        page = doc.load_page(page_num)
        images = page.get_image_info(hashes=False, xrefs=True)
        if not images:
            console.info(f"- Page {(page_num + 1):03d}: No images found")
            continue

        maximum_width: int = max(img.get("width", 0) for img in images)
        maximum_height: int = max(img.get("height", 0) for img in images)

        dpi_calc = determine_dpi_from_width(page, maximum_width)
        extra_info = ""
        if len(images) == 1:
            extra_info = f", {images[0].get('cs-name', 'unknown')}"
        console.info(f"- Page {(page_num + 1):03d} ({maximum_width}x{maximum_height}{extra_info}): {dpi_calc} DPI")
        if len(images) > 1:
            console.info(f"   Composite of {len(images)} images:")
            for img_idx, img in enumerate(images):
                img_width = img.get("width", 0)
                img_height = img.get("height", 0)
                colorspace = img.get("cs-name", "unknown")
                bounding_box = img.get("bbox", None)
                transform_mtrx = img.get("transform", None)
                img_dpi = determine_dpi_from_width(page, img_width)
                console.info(f"   - Image {img_idx + 1:02d}: {img_width}x{img_height}, {colorspace}, {img_dpi} DPI")
                if bounding_box:
                    console.info(f"     Bounding box (x0, y0, w, h): {bounding_box}")
                if transform_mtrx:
                    console.info(f"     Transform matrix: {transform_mtrx}")

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
@click.option(
    "--levels",
    "levels",
    is_flag=True,
    default=False,
    help="Automatically adjust levels of the exported images, simulate xpdf --gray method (only for gray PNG)",
)
@time_program
def export_pdf(
    pdf_file: Path,
    dest_output: Path,
    dpi: int,
    colorspace: PDFColorspaceGenerate | None,
    with_alpha: bool,
    force_cmyk: bool,
    levels: bool,
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
            page=page, dpi=dpi, force_colorspace=colorspace, with_alpha=with_alpha, force_cmyk=force_cmyk
        )
        output_name = f"page{(page_num + 1):06d}.{fmt_ext}"
        dest_file = dest_output / output_name

        if levels and fmt_ext == "png":
            # Load with PIL
            img = Image.open(BytesIO(image))
            if img.mode not in ("L", "LA"):
                # not gray, save as is
                dest_file.write_bytes(image)
                img.close()
                continue

            # Remove alpha layer
            if img.mode == "LA":
                img = img.convert("L")
            black_level, _, _ = find_local_peak(img, upper_limit=60, skip_white_check=True)
            if black_level > 60 or black_level <= 0:
                # save
                dest_file.write_bytes(image)
                img.close()
                continue

            # Apply the black level with Pillow
            gamma_correct = gamma_correction(black_level)

            img = img.convert("L")  # if not yet
            adjusted_image = apply_levels(img, black_point=black_level, white_point=255, gamma=gamma_correct)

            adjusted_image.save(dest_file, format="PNG")
        else:
            dest_file.write_bytes(image)

    console.stop_status(f"Exported {page_count} pages.")
    doc.close()


@pdf_manager.command(
    name="extract",
    help="Extract images from PDF",
    cls=NMangaCommandHandler,
)
@pdf_file
@options.dest_output()
@click.option(
    "--no-composite",
    "no_composite",
    is_flag=True,
    default=False,
    help="Do not attempt to composite multiple image parts into one image",
)
@time_program
def extract_pdf_images(
    pdf_file: Path,
    dest_output: Path,
    no_composite: bool,
):
    """
    Extract images from PDF.

    This will extract all images from the PDF and save them in the specified output directory.
    """

    console.info(f"Extracting images from {pdf_file} to {dest_output}")
    doc = pymupdf.Document(str(pdf_file))
    page_count = doc.page_count
    dest_output.mkdir(parents=True, exist_ok=True)

    console.status("Extracting images...")

    for extracted_img in extract_images_from_pdf(doc, no_composite=no_composite):
        page_num = extracted_img.page
        console.status(f"Extracting images... (Page {page_num + 1}/{page_count})")
        img = extracted_img.image
        img_ext = extracted_img.extension
        img_index = extracted_img.index

        output_name = f"page{(page_num + 1):06d}"
        if img_index is not None:
            output_name += f"-{img_index:02d}"
        output_name += f".{img_ext}"
        dest_file = dest_output / output_name
        dest_file.write_bytes(img)
    console.stop_status(f"Extracted images from {page_count} pages.")

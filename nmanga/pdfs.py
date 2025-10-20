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
from enum import Enum
from io import BytesIO
from typing import Generator

import pymupdf
from PIL import Image

__all__ = (
    "ExtractedImage",
    "PDFColorspaceGenerate",
    "determine_dpi_from_width",
    "extract_images_from_pdf",
    "generate_image_from_page",
    "generate_images_from_pdf",
    "load_xref_image",
    "load_xref_with_smask",
)

devicen_allowed_matches = ["DeviceCMYK", "Black"]
devicen_disallowed_matches = [
    "Blue",
    "Red",
    "Green",
    "Yellow",
    "Magenta",
    "Cyan",
]


@dataclass
class ExtractedImage:
    image: bytes
    """:class:`bytes`: The extracted image, or composited image"""
    extension: str
    """:str: The image extension (e.g. 'png', 'jpg', 'tiff', etc.)"""
    page: int
    """:int: The page number (1-based) where the image was extracted from"""
    index: int | None = None
    """:Optional[int]: The index of the image on the page (0-based), or None if the image is a full page image"""


class PDFColorspaceGenerate(str, Enum):
    RGB = "RGB"
    CMYK = "CMYK"
    GRAY = "Gray"

    def as_pymupdf(self, force_cmyk: bool = False) -> pymupdf.Colorspace:
        """Convert to pymupdf.Colorspace

        If coerce_cmyk is True, CMYK will be converted to RGB
        """
        if self == PDFColorspaceGenerate.RGB:
            return pymupdf.csRGB
        if self == PDFColorspaceGenerate.CMYK:
            return pymupdf.csCMYK if force_cmyk else pymupdf.csRGB
        if self == PDFColorspaceGenerate.GRAY:
            return pymupdf.csGRAY
        raise ValueError(f"Unknown colorspace: {self}")


def load_xref_image(doc: pymupdf.Document, xref: int) -> tuple[bytes, str]:
    """
    Load image from xref
    """

    base_image = doc.extract_image(xref)
    image_bytes = base_image["image"]
    image_ext = base_image["ext"]
    return image_bytes, image_ext


def load_xref_with_smask(doc: pymupdf.Document, xref: int, smask: int, skip_mask: bool = False) -> tuple[bytes, str]:
    """
    Handle image with smask (soft mask for transparency)

    Based on: https://github.com/pymupdf/PyMuPDF-Utilities/blob/master/examples/extract-images/extract-from-pages.py
    """

    if smask == 0 or skip_mask:
        return load_xref_image(doc, xref)

    base_image = pymupdf.Pixmap(doc.extract_image(xref)["image"])
    pix0 = base_image
    if pix0.alpha:
        pix0 = pymupdf.Pixmap(pix0, 0)  # remove alpha channel
    mask = pymupdf.Pixmap(doc.extract_image(smask)["image"])

    try:
        pix = pymupdf.Pixmap(pix0, mask)  # apply soft mask
    except:  # noqa: E722
        # Fallback: just use the base image
        pix = base_image

    if pix0.n > 3:
        ext = "pam"
    else:
        ext = "png"
    image_bytes = pix.tobytes(output=ext)

    # Clean up
    del base_image
    del pix0
    del mask
    del pix

    return image_bytes, ext


def prefer_colorspace(colorspaces: list[str]) -> str:
    """
    Given a list of colorspaces, return the preferred one.

    Preference order:
    1. RGB
    3. CMYK
    2. Gray

    Returns the corresponding PIL mode string.
    """

    if not colorspaces:
        return "RGB"  # Default to RGB

    def _determine(cs: str) -> int | None:
        if "DeviceRGB" in cs:
            return 1
        if "DeviceCMYK" in cs:
            return 2
        if "DeviceGray" in cs:
            return 3
        return None

    colorspaces_coerce = sorted((cs for cs in (_determine(c) for c in colorspaces) if cs is not None))
    if colorspaces_coerce:
        if colorspaces_coerce[0] == 1:
            return "RGB"
        if colorspaces_coerce[0] == 2:
            return "CMYK"
        if colorspaces_coerce[0] == 3:
            return "L"
    if "None" in colorspaces:
        return "RGB"
    raise ValueError(f"Unknown colorspaces: {colorspaces}")


def has_alpha(images: list[bytes]) -> bool:
    """
    Check if any image has alpha channel
    """

    for img in images:
        temp_read = Image.open(BytesIO(img))
        if temp_read.mode in ("RGBA", "LA") or (temp_read.mode == "P" and "transparency" in temp_read.info):
            temp_read.close()
            return True
        temp_read.close()
    return False


def extract_images_from_pdf(doc: pymupdf.Document, no_composite: bool = False) -> Generator[ExtractedImage, None, None]:
    """
    This function create a single image from each page of the PDF document.

    The name "extract" is correct since we are actually extracting images from the PDF
    but we did some special handling to make sure we get a full-page image in case
    there is multiple images overlapping on the page.

    If you want to make an image from each page, use `generate_images_from_pdf` instead.

    This is mostly used for PDF from DENPA and other which use PDF as a container for images.

    This return a generator that yields tuples of (PIL Image, image extension).
    """

    total_pages = len(doc)
    for page_num in range(total_pages):
        page = doc.load_page(page_num)

        # Image list since this contains the smask info
        image_lists = page.get_images(full=False)
        # The detailed image info
        image_infos = page.get_image_info(hashes=False, xrefs=True)

        # No images? Create padding
        if len(image_infos) == 0:
            # No images on this page, create a blank white image with the same size as the page
            mat = pymupdf.Matrix(1, 1)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            image = Image.new("RGB", (pix.width, pix.height), (255, 255, 255))
            with BytesIO() as output:
                image.save(output, format="PNG")
                image_bytes = output.getvalue()
            yield ExtractedImage(image=image_bytes, extension="png", page=page_num + 1)
            continue

        # Single image? Return it directly
        if len(image_infos) == 1:
            xref = image_infos[0]["xref"]
            smask = image_lists[0][1]
            image, ext = load_xref_with_smask(doc, xref, smask)
            yield ExtractedImage(image=image, extension=ext, page=page_num + 1)
            continue

        # Preload the images
        images = [load_xref_with_smask(doc, info["xref"], image_lists[idx][1]) for idx, info in enumerate(image_infos)]
        if no_composite:
            for idx, (img, ext) in enumerate(images):
                yield ExtractedImage(image=img, extension=ext, page=page_num + 1, index=idx)
            continue

        # Multiple images? We need to composite them together
        # Create a blank white image with the same size as the page
        mat = pymupdf.Matrix(1, 1)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # load all images first
        prefer_spaces = prefer_colorspace([info["cs-name"] for info in image_infos if "cs-name" in info])

        # Check if any image has alpha channel
        if has_alpha([img for img, _ in images]):
            if prefer_spaces == "CMYK":
                prefer_spaces = "RGB"
            prefer_spaces += "A"  # Add alpha channel

        composite = Image.new(
            prefer_spaces, (pix.width, pix.height), (255, 255, 255, 0) if "A" in prefer_spaces else (255, 255, 255)
        )

        for idx, (img, _) in enumerate(images):
            # We paste the image at the proper position from the
            # Convert to the preferred colorspace
            read_img = Image.open(BytesIO(img))
            if read_img.mode != prefer_spaces:
                read_img = read_img.convert(prefer_spaces)

            infos = image_infos[idx]
            bounding = pymupdf.Rect(*infos["bbox"])

            # TODO: Apply the matrix to the image first

            # Paste the image onto the composite
            composite.paste(read_img, (int(bounding.x0), int(bounding.y0)), read_img if "A" in prefer_spaces else None)

        # Save composite to bytes
        with BytesIO() as output:
            # If cmyk, save as tiff
            if composite.mode == "CMYK":
                composite.save(output, format="TIFF")
                composite_bytes = output.getvalue()
            else:
                composite.save(output, format="PNG")
                composite_bytes = output.getvalue()
        yield ExtractedImage(image=composite_bytes, extension="png", page=page_num + 1)


def generate_image_from_page(
    page: pymupdf.Page,
    dpi: int,
    force_colorspace: PDFColorspaceGenerate | None = None,
    with_alpha: bool = False,
    force_cmyk: bool = False,
) -> tuple[bytes, str]:
    """
    This function create a single image from a single page of the PDF document.

    Return a PIL Image or bytes (for PAM).
    """
    images = page.get_image_info(hashes=False, xrefs=True)

    if force_colorspace is not None:
        real_colorspace: pymupdf.Colorspace = force_colorspace.as_pymupdf(force_cmyk=force_cmyk)
    else:
        maybe_gray = False
        for img in images:
            if (
                img["cs-name"] == "DeviceGray"
                or (
                    all(x in img["cs-name"] for x in devicen_allowed_matches)
                    and not any(x in img["cs-name"] for x in devicen_disallowed_matches)
                )
                or img["bpc"] == 1
            ):
                maybe_gray = True
        if not images:
            # Force grayscale if no images found
            maybe_gray = True
        if maybe_gray:
            real_colorspace = pymupdf.csGRAY
        else:
            real_colorspace = pymupdf.csCMYK if force_cmyk else pymupdf.csRGB

    pix = page.get_pixmap(dpi=dpi, colorspace=real_colorspace, alpha=with_alpha, annots=False)

    format_determine = "jpg" if pix.n > 3 else "png"
    as_bytes_data: bytes = pix.tobytes(output=format_determine, jpg_quality=100)
    return as_bytes_data, format_determine


def determine_dpi_from_width(page: pymupdf.Page, target_width: int) -> int:
    """
    Determine the DPI to use to render the page to match the target width.

    This is a simple calculation based on the page width in points (1/72 inch).
    """

    page_width_points = page.rect.width  # Width in points
    dpi = int((target_width / page_width_points) * 72)  # Convert points to inches and scale to target width
    return max(dpi, 72)  # Ensure a minimum of 72 DPI


def generate_images_from_pdf(
    doc: pymupdf.Document,
    dpi: int,
    force_colorspace: PDFColorspaceGenerate | None = None,
    with_alpha: bool = False,
    coerce_cmyk: bool = True,
) -> Generator[tuple[bytes, str], None, None]:
    """
    This function create a single image from each page of the PDF document.

    This is mostly used for PDF that are not image-based, such as scanned documents or
    PDF generated from text.

    If you want to extract images from the PDF, use `extract_images_from_pdf` instead.

    This return a generator that yields PIL Images or bytes (for PAM).
    """

    total_pages = len(doc)
    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        yield generate_image_from_page(page, dpi, force_colorspace, with_alpha, coerce_cmyk)

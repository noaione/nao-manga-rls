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

# Bulk denoise images in a directory using Waifu2x-TensorRT or ONNX Runtime
# https://github.com/z3lx/waifu2x-tensorrt
# Part of the code is adapted from anon

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import rich_click as click
from defusedxml import ElementTree as ET  # noqa: N817
from PIL import Image

from .. import term
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler

__all__ = ("epub_extractor",)

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)
console = term.get_console()


@dataclass
class ExtractedImage:
    data: bytes
    ext: str
    blank: bool = False

    @classmethod
    def make_blank(cls, ext: str) -> ExtractedImage:
        return cls(data=b"", ext=ext, blank=True)


def find_root_file_path(epub_zip: ZipFile) -> str:
    """Find the rootfile path from the EPUB's container.xml"""

    target_file = "META-INF/container.xml"
    with epub_zip.open(target_file) as container_file:
        container_xml = container_file.read().decode("utf-8")

    root = ET.fromstring(container_xml)
    namespace = {"ns": "urn:oasis:names:tc:opendocument:xmlns:container"}
    rootfile_element = root.find("ns:rootfiles/ns:rootfile", namespace)
    if rootfile_element is None:
        raise ValueError("Rootfile element not found in container.xml")

    full_path_attr = rootfile_element.get("full-path")
    if full_path_attr is None:
        raise ValueError("full-path attribute not found in rootfile element")
    return full_path_attr


def map_spine_to_xhtml_path(epub_zip: ZipFile, root_file_path: str) -> list[str]:
    """Extract the spine mapping to XHTML file paths from the EPUB's package document"""

    opf_path = Path(root_file_path)

    with epub_zip.open(root_file_path) as package_file:
        package_xml = package_file.read().decode("utf-8")

    root = ET.fromstring(package_xml)
    namespace = {"ns": "http://www.idpf.org/2007/opf"}

    # Map item IDs to their hrefs
    id_to_href = {}
    for item in root.findall("ns:manifest/ns:item", namespace):
        item_id = item.get("id")
        href = item.get("href")
        if item_id and href:
            id_to_href[item_id] = href

    # Extract spine order of XHTML files
    spine_xhtml_paths = []
    for itemref in root.findall("ns:spine/ns:itemref", namespace):
        idref = itemref.get("idref")
        if idref and idref in id_to_href:
            href_data = Path(id_to_href[idref])
            resolved_path = (opf_path.parent / href_data).as_posix()
            spine_xhtml_paths.append(resolved_path)
        else:
            # Raise to ensure integrity
            raise ValueError(f"Item with idref '{idref}' not found in manifest")

    return spine_xhtml_paths


def is_xhtml_blank(xhtml_content: str) -> bool:
    """Check if the XHTML content is blank (i.e., contains no significant elements)"""

    root = ET.fromstring(xhtml_content)
    namespace = {"xhtml": "http://www.w3.org/1999/xhtml"}

    # Check for significant elements like <img>, <svg>, etc.
    significant_elements = root.findall(".//xhtml:img", namespace) + root.findall(".//xhtml:svg", namespace)
    # Has text?
    has_text_data = (root.text and root.text.strip()) or any((elem.tail and elem.tail.strip()) for elem in root.iter())
    return len(significant_elements) == 0 and not bool(has_text_data)  # no images and no text


def try_extract_images_from_xhtml(
    epub_zip: ZipFile,
    xhtml_path: str,
) -> list[ExtractedImage]:
    """Try to extract image data from an XHTML file within the EPUB"""

    extracted_images: list[ExtractedImage] = []

    with epub_zip.open(xhtml_path) as xhtml_file:
        xhtml_content = xhtml_file.read().decode("utf-8")

    if is_xhtml_blank(xhtml_content):
        console.log(f"XHTML file {xhtml_path} is blank, skipping image extraction.")
        return [ExtractedImage.make_blank(ext="png")]

    root = ET.fromstring(xhtml_content)
    namespace = {"xhtml": "http://www.w3.org/1999/xhtml"}

    xhtml_path_real = Path(xhtml_path)

    for img_element in root.findall(".//xhtml:img", namespace):
        img_src = img_element.get("src")
        if not img_src:
            continue

        try:
            img_path = Path(img_src)
            img_resolved = (xhtml_path_real.parent / img_path).as_posix()
            with epub_zip.open(img_resolved) as img_file:
                img_data = img_file.read()
                img_ext = Path(img_resolved).suffix.lstrip(".").lower()
                extracted_images.append(ExtractedImage(data=img_data, ext=img_ext))
        except KeyError:
            console.log(f"Image {img_src} not found in EPUB archive, skipping.")

    # Handle blank images (e.g., SVG placeholders)
    for svg_element in root.findall(".//xhtml:svg", namespace):
        img_src = svg_element.get("data-placeholder-src")
        if not img_src:
            continue

        console.log(f"Found SVG placeholder for image {img_src}, marking as blank.")
        extracted_images.append(ExtractedImage(data=b"", ext="png", blank=True))

    return extracted_images


def make_blank_image(original_img: Path, target_path: Path) -> None:
    """Create a blank (1x1 transparent) image with the same format as the original image"""

    with Image.open(original_img) as img:
        blank_img = Image.new("L", (img.width, img.height), color=255)  # white for L mode
        blank_img.save(target_path, format="PNG")
        blank_img.close()


@click.command(
    name="epub-extract",
    help="Extract all referenced images from an EPUB file",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=False, disable_folder=True)
@options.dest_output(optional=False)
@time_program
def epub_extractor(
    path_or_archive: Path,
    dest_output: Path,
) -> None:
    """
    Extract all referenced images from an EPUB file.
    """

    if not path_or_archive.is_file():
        raise click.BadParameter(
            f"{path_or_archive} is not a valid EPUB file.",
            param_hint="path_or_archive",
        )
    if path_or_archive.suffix.lower() != ".epub":
        raise click.BadParameter(
            f"{path_or_archive} is not an EPUB file (expected .epub extension).",
            param_hint="path_or_archive",
        )

    console.info(f"Opening EPUB file: {path_or_archive}")
    dest_output.mkdir(parents=True, exist_ok=True)

    with ZipFile(path_or_archive, "r") as epub_zip:
        # Get META-INF/container.xml to find the rootfile
        root_file_path = find_root_file_path(epub_zip)
        console.info(f"Rootfile located at: {root_file_path}")

        spine_mappings = map_spine_to_xhtml_path(epub_zip, root_file_path)
        console.info(f"Spine contains {len(spine_mappings)} XHTML files.")

        progress = console.make_progress()
        task = progress.add_task("Extracting images...", total=len(spine_mappings))
        img_counters = 0
        last_known_image_path = None
        for xhtml_path in spine_mappings:
            extracted_images = try_extract_images_from_xhtml(epub_zip, xhtml_path)
            for img_data in extracted_images:
                # use i_XXXX.ext to avoid name conflicts
                img_filename = f"i_{img_counters:04d}.{img_data.ext}"

                dest_image_path = dest_output / img_filename
                if img_data.blank and last_known_image_path is not None:
                    console.log(f"Using last known image for blank image reference at {img_filename}")
                    make_blank_image(last_known_image_path, dest_image_path)
                    img_counters += 1
                    continue
                elif img_data.blank:
                    console.log(f"Skipping blank image reference at {img_filename} (no last known image)")
                    continue

                dest_image_path.write_bytes(img_data.data)
                last_known_image_path = dest_image_path
                img_counters += 1

            progress.update(task, advance=1)
        console.stop_progress(progress)
        console.info(f"Extracted {img_counters} images to {dest_output}")

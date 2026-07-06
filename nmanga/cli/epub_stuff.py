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
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import rich_click as click
from defusedxml import ElementTree as ET  # noqa: N817
from PIL import Image

from .. import term
from ..autolevel import apply_levels, find_local_peak, find_local_peak_legacy, gamma_correction
from ..epub_render import (
    attach_chromium_console,
    copy_image_only_page,
    find_root_file_path,
    get_base_image_scale,
    get_image_bounding_box_and_hide,
    get_image_only_page_path,
    get_src_image_path,
    launch_chromium,
    load_xhtml,
    map_spine_to_xhtml_path,
    read_view_port_from_xhtml_bs4,
    resolve_epub_href,
    screenshot_overlay_page,
)
from ..lazy import get_numpy
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler

__all__ = ("epub_group",)

# Setting image max pixel count to ~4/3 GPx for 3bpp (24-bit) to get ~4GB of memory usage tops
Image.MAX_IMAGE_PIXELS = 4 * ((1024**3) // 3)
console = term.get_console()


@click.group(
    name="epub",
    help="EPUB operation toolsets",
)
def epub_group():
    """EPUB operation toolsets"""
    pass


@dataclass
class ExtractedImage:
    data: bytes
    ext: str
    blank: bool = False

    @classmethod
    def make_blank(cls, ext: str) -> ExtractedImage:
        return cls(data=b"", ext=ext, blank=True)


def is_xhtml_blank(xhtml_content: str) -> bool:
    """Check if the XHTML content is blank (i.e., contains no significant elements)"""

    root = ET.fromstring(xhtml_content)
    namespace = {"xhtml": "http://www.w3.org/1999/xhtml", "xhtmlsvg": "http://www.w3.org/2000/svg"}
    # Check for significant elements like <img>, <svg>, etc.
    significant_elements = root.findall(".//xhtml:img", namespace) + root.findall(".//xhtmlsvg:svg", namespace)
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
    namespace = {
        "xhtml": "http://www.w3.org/1999/xhtml",
        "xhtmlsvg": "http://www.w3.org/2000/svg",
    }

    xhtml_path_real = Path(xhtml_path)

    for img_element in root.findall(".//xhtml:img", namespace):
        img_src = img_element.get("src")
        if not img_src:
            continue

        try:
            img_resolved = resolve_epub_href(xhtml_path_real, img_src).as_posix()
            with epub_zip.open(img_resolved) as img_file:
                img_data = img_file.read()
                img_ext = Path(img_resolved).suffix.lstrip(".").lower()
                extracted_images.append(ExtractedImage(data=img_data, ext=img_ext))
        except KeyError:
            console.log(f"Image {img_src} not found in EPUB archive, skipping.")

    # Handle blank images (e.g., SVG placeholders)
    for svg_element in root.findall(".//xhtmlsvg:svg", namespace):
        img_placeholder_src = svg_element.get("data-placeholder-src")
        if img_placeholder_src:
            console.log(f"Found SVG placeholder for image {img_placeholder_src}, marking as blank.")
            extracted_images.append(ExtractedImage(data=b"", ext="png", blank=True))
            continue

        image_child = svg_element.find(".//xhtmlsvg:image", namespace)
        if image_child is not None:
            # Find the referenced image
            img_href = image_child.get("{http://www.w3.org/1999/xlink}href")
            if img_href:
                try:
                    img_resolved = resolve_epub_href(xhtml_path_real, img_href).as_posix()
                    with epub_zip.open(img_resolved) as img_file:
                        img_data = img_file.read()
                        img_ext = Path(img_resolved).suffix.lstrip(".").lower()
                        extracted_images.append(ExtractedImage(data=img_data, ext=img_ext))
                except KeyError:
                    console.log(f"Image {img_href} not found in EPUB archive, skipping.")

    return extracted_images


def make_blank_image(original_img: Path, target_path: Path) -> None:
    """Create a blank (1x1 transparent) image with the same format as the original image"""

    with Image.open(original_img) as img:
        blank_img = Image.new("L", (img.width, img.height), color=255)  # white for L mode
        blank_img.save(target_path, format="PNG")
        blank_img.close()


@epub_group.command(
    name="extract",
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
        container_file = "META-INF/container.xml"
        with epub_zip.open(container_file) as container_fp:
            container_xml = container_fp.read().decode("utf-8")
        root_file_path = find_root_file_path(container_xml)
        console.info(f"Rootfile located at: {root_file_path}")

        with epub_zip.open(root_file_path) as root_fp:
            package_xml = root_fp.read().decode("utf-8")
        spine_mappings = map_spine_to_xhtml_path(package_xml, root_file_path)
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


def overlay_level_image(
    overlay_img: Image.Image,
    *,
    upper_limit: int,
    peak_min_pct: float | None,
    peak_prom_pct: float | None,
    peak_offset: int,
    is_legacy: bool,
) -> Image.Image:
    comp_image = Image.new(overlay_img.mode, (overlay_img.width, overlay_img.height), color=255)
    comp_image.paste(overlay_img, (0, 0), overlay_img)

    if not is_legacy:
        black_level, _, _ = find_local_peak(
            comp_image,
            upper_limit=upper_limit,
            peak_percentage=peak_min_pct,
            peak_prominence=peak_prom_pct,
            skip_white_check=True,
        )
    else:
        black_level, _, _ = find_local_peak_legacy(comp_image, upper_limit=upper_limit, skip_white_peaks=True)

    is_black_bad = black_level <= 0

    if is_black_bad or black_level > upper_limit:
        return overlay_img.copy()

    gamma_correct = gamma_correction(black_level)
    adjusted_img = apply_levels(
        overlay_img,
        black_point=black_level + peak_offset,
        white_point=255,
        gamma=gamma_correct,
    )

    return adjusted_img


@epub_group.command(
    name="render",
    help="Render all pages from an EPUB file to a folder of images",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True, disable_folder=False)
@options.dest_output(optional=False)
@click.option(
    "--default-w",
    "default_w",
    type=options.POSITIVE_INT,
    required=True,
    help="The default width of the image",
)
@click.option(
    "--default-h",
    "default_h",
    type=options.POSITIVE_INT,
    required=True,
    help="The default height of the image",
)
@click.option(
    "-ul",
    "--upper-limit",
    "upper_limit",
    type=click.IntRange(1, 255),
    default=60,
    show_default=True,
    help="The upper limit for finding local peaks in the histogram",
)
@click.option(
    "-pmp",
    "--peak-min-pct",
    "peak_min_pct",
    type=click.FloatRange(0.0, 100.0),
    default=0.25,
    show_default=True,
    help="The minimum percentage of pixels for a peak to be considered valid",
)
@click.option(
    "-ppm",
    "--peak-prominence-pct",
    "peak_prom_pct",
    type=click.FloatRange(0.0, 100.0),
    default=None,
    show_default=True,
    help="Minimum prominence relative to nearby shades to be considered a peak.",
)
@click.option(
    "-po",
    "--peak-offset",
    "peak_offset",
    type=click.IntRange(-100, 100),
    default=0,
    show_default=True,
    help="The offset to add to the detected black level percentage",
)
@click.option(
    "--legacy",
    is_flag=True,
    default=False,
    help="Use legacy autolevel analysis",
)
@click.option(
    "--no-autolevel",
    is_flag=True,
    default=False,
    help="Skip autolevel analysis",
)
@time_program
def epub_render(
    path_or_archive: Path,
    dest_output: Path,
    default_w: int,
    default_h: int,
    upper_limit: int,
    peak_min_pct: float | None,
    peak_prom_pct: float | None,
    peak_offset: int,
    legacy: bool,
    no_autolevel: bool,
) -> None:
    """
    Render all pages from an EPUB file to a folder of images.
    """

    if path_or_archive.is_file():
        raise click.BadParameter(
            f"{path_or_archive} should be a folder from an extracted EPUB file",
            param_hint="path_or_archive",
        )

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise click.ClickException(
            "playwright is not installed. Please install nmanga with `epub-render` extras"
        ) from None

    console.info(f"Opening EPUB file: {path_or_archive}")
    dest_output.mkdir(parents=True, exist_ok=True)

    # Get META-INF/container.xml to find the rootfile
    container_file = path_or_archive / "META-INF" / "container.xml"
    container_xml = container_file.read_text("utf-8")
    root_file_path = find_root_file_path(container_xml)
    console.info(f"Rootfile located at: {root_file_path}")

    package_xml_path = path_or_archive / root_file_path
    package_xml = package_xml_path.read_text("utf-8")
    spine_mappings = map_spine_to_xhtml_path(package_xml, root_file_path)
    console.info(f"Spine contains {len(spine_mappings)} XHTML files.")

    has_numpy = False
    try:
        get_numpy()
        has_numpy = True
    except ImportError:
        console.warning("numpy is not installed. The text will not be leveled.")
        has_numpy = False

    copied_page = 0
    blank_page = 0
    rendered_page = 0
    with sync_playwright() as playwright:
        console.info("Launching Chromium...")
        browser = launch_chromium(playwright, args=["--allow-file-access-from-files"])

        progress = console.make_progress()
        task = progress.add_task("Rendering pages...", total=len(spine_mappings))
        img_counters = 0
        for xhtml_path in spine_mappings:
            # solve with solve_epub_path
            dest_image_path = dest_output / f"i_{img_counters:04d}.png"
            xhtml_full_path = resolve_epub_href(package_xml_path, xhtml_path, epub_root=path_or_archive)
            console.log(f"Opening: {xhtml_full_path} - {xhtml_path} - {package_xml_path} - {path_or_archive}")

            image_only_path = get_image_only_page_path(xhtml_full_path, path_or_archive)
            if image_only_path is not None:
                dest_image_path = copy_image_only_page(image_only_path, dest_output, img_counters)
                console.log(f"Copied image-only page to {dest_image_path}")
                img_counters += 1
                copied_page += 1
                progress.update(task, advance=1)
                continue

            vw_w, vw_h = read_view_port_from_xhtml_bs4(xhtml_full_path, (default_w, default_h))

            img_scale = get_base_image_scale(browser, xhtml_full_path, vw_w, vw_h, cnsl=console)
            console.log(f"Page scale is: {img_scale}, with viewport {vw_w}x{vw_h}")
            if img_scale is None:
                # Make blank white image?
                console.warning(f"No image found in {xhtml_path}, making blank page")
                blank_img = Image.new("L", (default_w, default_h), color=255)  # white for L mode
                blank_img.save(dest_image_path, format="PNG")
                blank_img.close()
                img_counters += 1
                blank_page += 1
                progress.update(task, advance=1)
                continue

            page = browser.new_page(
                viewport={
                    "width": vw_w,
                    "height": vw_h,
                },
                device_scale_factor=img_scale,
            )
            attach_chromium_console(page, console)

            load_xhtml(page, xhtml_full_path)

            box = get_image_bounding_box_and_hide(page)
            src_image_path = get_src_image_path(box)

            src_img = Image.open(src_image_path)
            overlay_pg = screenshot_overlay_page(page, box)
            overlay_img = Image.open(BytesIO(overlay_pg))

            # level the image
            if has_numpy and not no_autolevel:
                overlay_img = overlay_level_image(
                    overlay_img,
                    upper_limit=upper_limit,
                    peak_min_pct=peak_min_pct,
                    peak_prom_pct=peak_prom_pct,
                    peak_offset=peak_offset,
                    is_legacy=legacy,
                )

            src_img.paste(overlay_img, (box["x"], box["y"]), overlay_img)
            src_img.save(dest_image_path, format="PNG")

            img_counters += 1
            rendered_page += 1
            progress.update(task, advance=1)
        console.stop_progress(progress)

    if rendered_page > 0:
        console.info(f"Rendered {rendered_page} pages")
    if copied_page > 0:
        console.info(f"Copied {copied_page} pages")
    if blank_page > 0:
        console.info(f"Generate {blank_page} blank pages")

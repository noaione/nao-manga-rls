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
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse
from urllib.request import url2pathname

from bs4 import BeautifulSoup
from defusedxml import ElementTree as ET  # noqa: N817
from playwright.sync_api import Browser, Error, Page, Playwright

from . import term

__all__ = (
    "BoundingBoxImage",
    "find_root_file_path",
    "get_base_image_scale",
    "get_image_bounding_box_and_hide",
    "get_src_image_path",
    "html_with_local_base",
    "launch_chromium",
    "load_xhtml",
    "map_spine_to_xhtml_path",
    "read_view_port_from_xhtml_bs4",
    "screenshot_overlay_page",
    "solve_epub_path",
)

BASE_HTML_RE = re.compile(r"<base\b[^>]*>", re.IGNORECASE)


class BoundingBoxImage(TypedDict):
    x: int
    y: int
    width: int
    height: int
    src: str


def install_chromium() -> None:
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        check=True,
    )


def launch_chromium(playwright: Playwright, *, args: list[str] | None = None) -> Browser:
    try:
        return playwright.chromium.launch(args=args)
    except Error as e:
        if "Executable doesn't exist" not in str(e):
            raise

        install_chromium()
        return playwright.chromium.launch(args=args)


def solve_epub_path(base_path: Path, relative_path: str) -> Path:
    """Resolve a relative path against a base path within an EPUB archive"""

    base_path_parts = base_path.parts[:-1]  # exclude the file itself
    relative_parts = Path(relative_path).parts

    combined_parts = list(base_path_parts)

    for part in relative_parts:
        if part == ".":
            continue
        elif part == "..":
            if len(combined_parts) > 0:
                combined_parts.pop()
        else:
            combined_parts.append(part)

    return Path(*combined_parts)


def find_root_file_path(container_xml: str) -> str:
    """Find the rootfile path from the EPUB's container.xml"""

    root = ET.fromstring(container_xml)
    namespace = {"ns": "urn:oasis:names:tc:opendocument:xmlns:container"}
    rootfile_element = root.find("ns:rootfiles/ns:rootfile", namespace)
    if rootfile_element is None:
        raise ValueError("Rootfile element not found in container.xml")

    full_path_attr = rootfile_element.get("full-path")
    if full_path_attr is None:
        raise ValueError("full-path attribute not found in rootfile element")
    return full_path_attr


def map_spine_to_xhtml_path(package_xml: str, root_file_path: str) -> list[str]:
    """Extract the spine mapping to XHTML file paths from the EPUB's package document"""

    opf_path = Path(root_file_path)

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


def html_with_local_base(xhtml_path: Path) -> str:
    text = xhtml_path.read_text(encoding="utf-8", errors="ignore")
    base = f'<base href="{xhtml_path.as_uri()}" target="_blank">'

    if BASE_HTML_RE.search(text):
        return BASE_HTML_RE.sub(base, text, count=1)

    return re.sub(r"(<head\b[^>]*>)", r"\1" + base, text, count=1, flags=re.IGNORECASE)


def load_xhtml(page: Page, xhtml_path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".html", delete=False) as fp:
        fp.write(html_with_local_base(xhtml_path))
        temp_path = Path(fp.name)
    page.goto(temp_path.as_uri(), wait_until="load")
    try:
        temp_path.unlink()
    except OSError:
        pass
    page.evaluate("""
        async () => {
            const timeout = ms => new Promise(resolve => setTimeout(resolve, ms));
            const images = Array.from(document.images);
            const imagesReady = Promise.all(images.map(img => {
                if (img.complete) {
                    return Promise.resolve();
                }
                return new Promise(resolve => {
                    img.addEventListener("load", resolve, { once: true });
                    img.addEventListener("error", resolve, { once: true });
                });
            }));
            await Promise.race([imagesReady, timeout(5000)]);

            if (document.fonts && document.fonts.ready) {
                await Promise.race([document.fonts.ready, timeout(5000)]);
            }
        }
    """)


def read_view_port_from_xhtml_bs4(xhtml_path: Path, fallback: tuple[int, int]) -> tuple[int, int]:
    text = xhtml_path.read_text(encoding="utf-8", errors="ignore")

    soup = BeautifulSoup(text, "html.parser")
    meta = soup.find("meta", attrs={"name": "viewport"})
    if meta is None:
        return fallback

    # get content
    meta_content = meta.get("content", "")
    if not meta_content:
        return fallback

    if isinstance(meta_content, list):
        meta_content = meta_content[0]

    width_match = re.search(r"width\s*=\s*(\d+)", meta_content)
    height_match = re.search(r"height\s*=\s*(\d+)", meta_content)

    width = int(width_match.group(1)) if width_match else fallback[0]
    height = int(height_match.group(1)) if height_match else fallback[1]

    return width, height


def get_base_image_scale(browser: Browser, input_xhtml: Path, viewport_width: int, viewport_height: int) -> float:
    page = browser.new_page(
        viewport={
            "width": viewport_width,
            "height": viewport_height,
        },
        device_scale_factor=1,
    )

    load_xhtml(page, input_xhtml)

    box = page.evaluate("""
        () => {
            const images = Array.from(document.images)
                .map(img => {
                    const rect = img.getBoundingClientRect();
                    return {
                        src: img.currentSrc || img.src,
                        width: rect.width,
                        height: rect.height,
                        naturalWidth: img.naturalWidth,
                        naturalHeight: img.naturalHeight,
                        area: rect.width * rect.height,
                    };
                })
                .filter(item => item.width > 0 && item.height > 0 && item.naturalWidth > 0 && item.naturalHeight > 0)
                .sort((a, b) => b.area - a.area);

            if (images.length === 0) {
                throw new Error("No image found on page");
            }

            return images[0];
        }
    """)

    page.close()

    scale_x = box["naturalWidth"] / box["width"]
    scale_y = box["naturalHeight"] / box["height"]

    cnsl = term.get_console()
    if abs(scale_x - scale_y) > 0.01:
        cnsl.warning(f"Non-uniform scale: x={scale_x}, y={scale_y}")

    return scale_x


def get_image_bounding_box_and_hide(page: Page) -> BoundingBoxImage:
    return page.evaluate("""
        () => {
            document.documentElement.style.background = "transparent";
            document.body.style.background = "transparent";
            document.body.style.margin = "0";
            document.body.style.padding = "0";

            const images = Array.from(document.images)
                .map(img => {
                    const rect = img.getBoundingClientRect();
                    return {
                        img,
                        src: img.currentSrc || img.src,
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                        naturalWidth: img.naturalWidth,
                        naturalHeight: img.naturalHeight,
                        area: rect.width * rect.height,
                    };
                })
                .filter(item => {
                    return item.width > 0 && item.height > 0 && item.naturalWidth > 0 && item.naturalHeight > 0;
                })
                .sort((a, b) => b.area - a.area);

            if (images.length === 0) {
                throw new Error("No image found on page");
            }

            const base = images[0];

            // Hide the base page image but keep its layout.
            base.img.style.opacity = "0";

            return {
                src: base.src,
                x: base.x,
                y: base.y,
                width: base.width,
                height: base.height,
            };
        }
    """)


def get_src_image_path(box: BoundingBoxImage) -> Path:
    parsed_src = urlparse(box["src"])
    parsed_path = url2pathname(parsed_src.path)

    return Path(parsed_path).resolve()


def screenshot_overlay_page(page: Page, box: BoundingBoxImage) -> bytes:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    overlay_path = Path(tmp.name)
    tmp.close()

    try:
        page.screenshot(
            path=str(overlay_path),
            omit_background=True,
            type="png",
            quality=100,
            scale="device",
            clip={
                "x": box["x"],
                "y": box["y"],
                "width": box["width"],
                "height": box["height"],
            },
        )

        return overlay_path.read_bytes()
    finally:
        overlay_path.unlink(missing_ok=True)

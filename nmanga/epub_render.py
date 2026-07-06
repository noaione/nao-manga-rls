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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, TypedDict
from urllib.parse import unquote, urldefrag, urlparse, urlsplit
from urllib.request import url2pathname

from bs4 import BeautifulSoup
from defusedxml import ElementTree as ET  # noqa: N817
from PIL import Image

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page, Playwright

from . import term

__all__ = (
    "BoundingBoxImage",
    "copy_image_only_page",
    "find_root_file_path",
    "get_base_image_scale",
    "get_image_bounding_box_and_hide",
    "get_image_only_page_path",
    "get_src_image_path",
    "launch_chromium",
    "load_xhtml",
    "map_spine_to_xhtml_path",
    "read_view_port_from_xhtml_bs4",
    "resolve_epub_href",
    "screenshot_overlay_page",
)

BASE_HTML_RE = re.compile(r"<base\b[^>]*>", re.IGNORECASE)
IMAGE_ONLY_WRAPPER_TAGS = {
    "a",
    "body",
    "br",
    "div",
    "figure",
    "html",
    "main",
    "p",
    "picture",
    "section",
    "span",
}
IGNORED_CONTENT_TAGS = {
    "base",
    "head",
    "link",
    "meta",
    "script",
    "source",
    "style",
    "title",
}


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


def launch_chromium(playwright: "Playwright", *, args: list[str] | None = None) -> "Browser":
    try:
        return playwright.chromium.launch(args=args)
    except Exception as e:
        if "Executable doesn't exist" not in str(e):
            raise

        install_chromium()
        return playwright.chromium.launch(args=args)


def normalize_epub_path(path: PurePosixPath) -> PurePosixPath:
    parts: list[str] = []

    for part in path.parts:
        if part in ("", ".", "/"):
            continue

        if part == "..":
            if not parts:
                raise ValueError(f"Path escapes EPUB root: {path}")
            parts.pop()
        else:
            parts.append(part)

    return PurePosixPath(*parts)


def resolve_epub_href(
    base_file: str | Path | PurePosixPath,
    href: str,
    *,
    epub_root: Path | None = None,
) -> Path:
    """
    Resolve an EPUB href.

    If epub_root is None:
        Returns a PurePosixPath usable with ZipFile.read(...).

    If epub_root is provided:
        Returns a filesystem Path under epub_root.
    """

    if epub_root is None:
        # ZipFile mode.
        base_epub_path = PurePosixPath(str(base_file).replace("\\", "/"))
    else:
        # Extracted-folder mode.
        epub_root = epub_root.resolve()
        base_epub_path = PurePosixPath(Path(base_file).resolve().relative_to(epub_root).as_posix())

    base_epub_path = normalize_epub_path(base_epub_path)

    href = href.strip()
    href, _fragment = urldefrag(href)

    parsed = urlsplit(href)

    # External refs: https://, data:, mailto:, etc.
    if parsed.scheme and parsed.scheme.lower() != "file":
        raise ValueError(f"External hrefs are not supported: {href}")

    if parsed.netloc:
        raise ValueError(f"Network locations are not supported: {href}")

    local_href = unquote(parsed.path).replace("\\", "/")

    if not local_href:
        resolved_epub_path = base_epub_path
    else:
        href_path = PurePosixPath(local_href)

        if local_href.startswith("/"):
            resolved_epub_path = normalize_epub_path(PurePosixPath(local_href.lstrip("/")))
        else:
            base_dir = base_epub_path.parent

            relative_candidate = normalize_epub_path(base_dir / href_path)

            base_top = base_epub_path.parts[0] if base_epub_path.parts else None
            href_top = href_path.parts[0] if href_path.parts else None

            # Handles:
            # base_file = OEBPS/content.opf
            # href      = OEBPS/html/cover.xhtml
            #
            # Instead of producing:
            # OEBPS/OEBPS/html/cover.xhtml
            if base_top and href_top == base_top:
                root_candidate = normalize_epub_path(href_path)
                resolved_epub_path = root_candidate
            else:
                resolved_epub_path = relative_candidate

    if epub_root is None:
        return Path(resolved_epub_path)

    return epub_root.joinpath(*resolved_epub_path.parts)


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


def attach_chromium_console(page: "Page", cnsl: "term.ConsoleInterface"):
    page.on("console", lambda pp: cnsl.log(f"[CHROME-{pp.type.upper()}]", pp.text))


def load_xhtml(page: "Page", xhtml_path: Path) -> None:
    page.goto(xhtml_path.as_uri(), wait_until="load")
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


def _tag_name(tag) -> str:
    name = tag.name or ""
    return name.rsplit(":", 1)[-1].lower()


def get_image_only_page_path(xhtml_path: Path, epub_root: Path) -> Path | None:
    text = xhtml_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(text, "html.parser")
    body = soup.find(_tag_name_filter("body")) or soup

    image_srcs: list[str] = []
    for tag in body.find_all(True):
        name = _tag_name(tag)
        if name in IGNORED_CONTENT_TAGS:
            continue

        if name == "img":
            src = tag.get("src")
            if not src or isinstance(src, list):
                return None
            image_srcs.append(src)
            continue

        if name == "image":
            href = tag.get("href") or tag.get("xlink:href")
            if not href or isinstance(href, list):
                return None
            image_srcs.append(href)
            continue

        if name == "svg" and tag.find(_tag_name_filter("image")) is not None:
            continue

        if name not in IMAGE_ONLY_WRAPPER_TAGS:
            return None

    for tag in body.find_all(lambda tag: _tag_name(tag) in IGNORED_CONTENT_TAGS):
        tag.decompose()

    text_content = body.get_text("", strip=True)
    if text_content:
        return None

    if len(image_srcs) != 1:
        return None

    try:
        return resolve_epub_href(xhtml_path, image_srcs[0], epub_root=epub_root)
    except ValueError:
        return None


def _tag_name_filter(expected_name: str):
    return lambda tag: getattr(tag, "name", None) is not None and _tag_name(tag) == expected_name


def copy_image_only_page(src_image_path: Path, dest_output: Path, img_counter: int) -> Path:
    ext = src_image_path.suffix.lower() or ".png"
    dest_image_path = dest_output / f"i_{img_counter:04d}{ext}"
    shutil.copyfile(src_image_path, dest_image_path)
    return dest_image_path


def get_src_image_path(box: BoundingBoxImage) -> Path:
    parsed_src = urlparse(box["src"])
    parsed_path = url2pathname(parsed_src.path)

    return Path(parsed_path).resolve()


def get_base_image_scale(
    browser: "Browser",
    input_xhtml: Path,
    viewport_width: int,
    viewport_height: int,
    *,
    cnsl: "term.ConsoleInterface | None" = None,
) -> float | None:
    page = browser.new_page(
        viewport={
            "width": viewport_width,
            "height": viewport_height,
        },
        device_scale_factor=1,
    )

    if cnsl is not None:
        attach_chromium_console(page, cnsl)

    load_xhtml(page, input_xhtml)

    box = page.evaluate("""
        () => {
            console.log(document.body.outerHTML);
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
                return null;
            }

            return images[0];
        }
    """)

    if box is None:
        return None

    page.close()
    target_img = get_src_image_path(box)
    opened_img = Image.open(target_img)
    img_w, img_h = opened_img.size
    opened_img.close()

    scale_x = img_w / box["width"]
    scale_y = img_h / box["height"]

    cnsl = term.get_console()
    if abs(scale_x - scale_y) > 0.01:
        cnsl.warning(f"Non-uniform scale: x={scale_x}, y={scale_y}")

    return scale_x


def get_image_bounding_box_and_hide(page: "Page") -> BoundingBoxImage:
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
            console.log("Base image:", base.src);

            // Hide the base page image but keep its layout.
            base.img.style.opacity = "0";
            base.img.style.visibility = "hidden";
            base.img.classList.add("playwright-hidden");

            console.log("Base image hidden", document.body.outerHTML);

            return {
                src: base.src,
                x: base.x,
                y: base.y,
                width: base.width,
                height: base.height,
            };
        }
    """)


def screenshot_overlay_page(page: "Page", box: BoundingBoxImage) -> bytes:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    overlay_path = Path(tmp.name)
    tmp.close()

    try:
        page.screenshot(
            path=str(overlay_path),
            omit_background=True,
            type="png",
            # quality=100,
            scale="device",
            clip={
                "x": box["x"],
                "y": box["y"],
                "width": box["width"],
                "height": box["height"],
            },
            style="""
            .playwright-hidden {
                visibility: hidden !important;
                opacity: 0 !important;
            }
            """,
        )

        return overlay_path.read_bytes()
    finally:
        overlay_path.unlink(missing_ok=True)

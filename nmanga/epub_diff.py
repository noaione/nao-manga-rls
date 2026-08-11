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

# Compare two EPUB files and report the differences between them.

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from difflib import unified_diff
from io import BytesIO
from pathlib import Path

from defusedxml import ElementTree as ET  # ruff: ignore[camelcase-imported-as-acronym]
from PIL import Image, ImageChops

from .epub_render import find_root_file_path, map_spine_to_xhtml_path, resolve_epub_href

__all__ = (
    "EpubDiffEntry",
    "EpubDiffResult",
    "ImagePixelComparison",
    "PageDiffEntry",
    "build_snapshot",
    "compare_image_pixels",
    "diff_epub_files",
    "is_image_entry",
    "is_text_entry",
    "normalize_opf_metadata",
    "sha256_hex",
)

# Extensions whose contents should always be diffed as text.
TEXT_EXTENSIONS = frozenset({
    "css",
    "htm",
    "html",
    "js",
    "json",
    "ncx",
    "opf",
    "svg",
    "txt",
    "xhtml",
    "xml",
})

# Extensions that are always binary, so we never try to decode them as text.
BINARY_EXTENSIONS = frozenset({
    "7z",
    "aac",
    "avi",
    "avif",
    "bin",
    "bmp",
    "eot",
    "flac",
    "gif",
    "gz",
    "jpeg",
    "jpg",
    "jxl",
    "m4a",
    "mkv",
    "mp3",
    "mp4",
    "ogg",
    "otf",
    "png",
    "rar",
    "tif",
    "tiff",
    "ttf",
    "wav",
    "webm",
    "webp",
    "woff",
    "woff2",
    "zip",
})

# Extensions we attempt to load with PIL for a perceptual pixel comparison.
IMAGE_EXTENSIONS = frozenset({
    "avif",
    "bmp",
    "gif",
    "jpeg",
    "jpg",
    "jxl",
    "png",
    "tif",
    "tiff",
    "webp",
})

OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"


def sha256_hex(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def is_text_entry(name: str, data: bytes) -> bool:
    """Return ``True`` when the zip entry should be diffed as text."""
    suffix = Path(name).suffix.lower().lstrip(".")
    if suffix in TEXT_EXTENSIONS:
        return True
    if suffix in BINARY_EXTENSIONS:
        return False
    # fall back to a UTF-8 decode attempt.
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def is_image_entry(name: str) -> bool:
    """Return ``True`` when the entry looks like a raster image (PIL-decodable)."""
    return Path(name).suffix.lower().lstrip(".") in IMAGE_EXTENSIONS


@dataclass(frozen=True)
class ImagePixelComparison:
    differing_pct: float  # percentage of pixels where any color channel differs
    mean_diff: float  # mean absolute per-channel difference (0-255)
    resized: bool  # True when the two images had different dimensions


def compare_image_pixels(old_data: bytes, new_data: bytes) -> ImagePixelComparison | None:
    """Load two images with PIL and compare their actual color data.

    Returns the percentage of differing pixels and the mean absolute channel
    difference. Images with different dimensions are resized to the smaller
    size (``resized=True``) so they can still be compared. Returns ``None``
    when either payload cannot be decoded as an image.
    """
    try:
        with Image.open(BytesIO(old_data)) as old_img, Image.open(BytesIO(new_data)) as new_img:
            if old_img.width == new_img.width and old_img.height == new_img.height:
                image_a = old_img.convert("RGB")
                image_b = new_img.convert("RGB")
                resized = False
            else:
                width = min(old_img.width, new_img.width)
                height = min(old_img.height, new_img.height)
                image_a = old_img.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                image_b = new_img.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                resized = True
    except Exception:
        return None

    diff = ImageChops.difference(image_a, image_b)
    total = image_a.width * image_a.height

    # Count pixels where any channel differs (luminance > 0 means a channel changed).
    gray_diff = diff.convert("L")
    histogram = gray_diff.histogram()
    differing = sum(histogram[1:])
    differing_pct = (differing / total) * 100.0 if total else 0.0

    # Mean absolute difference across all channels.
    diff_hist = diff.histogram()
    channels = len(diff.getbands())
    per_channel = [diff_hist[band * 256 : (band + 1) * 256] for band in range(channels)]
    mean_diff = sum(sum(level * count for level, count in enumerate(hist)) for hist in per_channel) / (total * channels)

    return ImagePixelComparison(differing_pct=differing_pct, mean_diff=mean_diff, resized=resized)


def _image_change_detail(
    old_hash: str | None,
    new_hash: str | None,
    comparison: ImagePixelComparison | None,
) -> str:
    """Build a human-readable detail line for a changed image."""
    hash_part = f"image hash {old_hash[:8] if old_hash else '-'} -> {new_hash[:8] if new_hash else '-'}"
    if comparison is None:
        return hash_part
    resized_note = " (resized)" if comparison.resized else ""
    return (
        f"{hash_part}; {comparison.differing_pct:.2f}% pixels differ{resized_note}, mean Δ {comparison.mean_diff:.2f}"
    )


def _pixel_compare(old_data: bytes | None, new_data: bytes | None) -> ImagePixelComparison | None:
    """Compare two image payloads, returning ``None`` when either is missing."""
    if old_data is None or new_data is None:
        return None
    return compare_image_pixels(old_data, new_data)


def normalize_opf_metadata(text: str) -> str:
    """Strip volatile per-build metadata fields from an OPF document.

    Removes ``dc:identifier``, ``dc:date`` and ``meta[property="dcterms:modified"]``
    from ``<metadata>`` so two EPUBs built at different times from identical
    content compare equal. These fields are stamped per-run by nmanga's own
    exporter (see :mod:`nmanga.templates.epub`).
    """
    root = ET.fromstring(text)
    metadata = root.find(f"{{{OPF_NS}}}metadata")
    if metadata is not None:
        for elem in list(metadata.findall(f"{{{DC_NS}}}identifier")):
            metadata.remove(elem)
        for elem in list(metadata.findall(f"{{{DC_NS}}}date")):
            metadata.remove(elem)
        for elem in list(metadata.findall(f"{{{OPF_NS}}}meta")):
            if elem.get("property") == "dcterms:modified":
                metadata.remove(elem)
    return ET.tostring(root, encoding="unicode")


def _page_image_ref(page_path: str, page_bytes: bytes) -> str | None:
    """Find the first image referenced by a page XHTML, return its entry name."""
    namespace = {
        "xhtml": "http://www.w3.org/1999/xhtml",
        "xhtmlsvg": "http://www.w3.org/2000/svg",
    }
    try:
        root = ET.fromstring(page_bytes)
    except ET.ParseError:
        return None

    img_element = root.find(".//xhtml:img", namespace)
    if img_element is not None:
        img_src = img_element.get("src")
        if img_src:
            try:
                return resolve_epub_href(Path(page_path), img_src).as_posix()
            except ValueError:
                return None

    svg_element = root.find(".//xhtmlsvg:svg", namespace)
    if svg_element is not None:
        image_child = svg_element.find(".//xhtmlsvg:image", namespace)
        if image_child is not None:
            img_href = image_child.get("{http://www.w3.org/1999/xlink}href")
            if img_href:
                try:
                    return resolve_epub_href(Path(page_path), img_href).as_posix()
                except ValueError:
                    return None
    return None


@dataclass
class EpubSnapshot:
    path: Path
    entries: dict[str, bytes]
    opf_path: str
    spine_pages: list[str]
    page_images: dict[str, str | None]


def build_snapshot(path: Path) -> EpubSnapshot:
    """Open an EPUB file and index its entries, spine and page images."""
    if path.suffix.lower() != ".epub":
        raise ValueError(f"{path} is not an EPUB file (expected .epub extension).")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a valid ZIP archive.")

    try:
        with zipfile.ZipFile(path, "r") as epub_zip:
            entries = {info.filename: epub_zip.read(info) for info in epub_zip.infolist()}
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path} is not a valid EPUB archive: {exc}") from exc

    container = entries.get("META-INF/container.xml")
    if container is None:
        raise ValueError("META-INF/container.xml not found in the EPUB archive.")
    root_file_path = find_root_file_path(container.decode("utf-8"))

    opf_bytes = entries.get(root_file_path)
    if opf_bytes is None:
        raise ValueError(f"Rootfile {root_file_path} not found in the EPUB archive.")
    spine_pages = map_spine_to_xhtml_path(opf_bytes.decode("utf-8"), root_file_path)

    page_images: dict[str, str | None] = {}
    for page in spine_pages:
        page_bytes = entries.get(page)
        page_images[page] = _page_image_ref(page, page_bytes) if page_bytes is not None else None

    return EpubSnapshot(
        path=path,
        entries=entries,
        opf_path=root_file_path,
        spine_pages=spine_pages,
        page_images=page_images,
    )


@dataclass
class PageDiffEntry:
    position: int
    old_name: str | None
    new_name: str | None
    status: str
    old_image: str | None = None
    new_image: str | None = None
    old_size: int | None = None
    new_size: int | None = None
    detail: str | None = None
    diff: list[str] = field(default_factory=list)


@dataclass
class EpubDiffEntry:
    name: str
    status: str
    old_size: int | None = None
    new_size: int | None = None
    old_hash: str | None = None
    new_hash: str | None = None
    detail: str | None = None
    diff: list[str] = field(default_factory=list)


@dataclass
class EpubDiffResult:
    pages: list[PageDiffEntry]
    files: list[EpubDiffEntry]

    @property
    def changed(self) -> int:
        return sum(1 for page in self.pages if page.status != "identical") + sum(
            1 for entry in self.files if entry.status != "identical"
        )

    @property
    def identical(self) -> bool:
        return self.changed == 0


def diff_epub_files(
    old_path: Path,
    new_path: Path,
    *,
    strict_metadata: bool = False,
    context: int = 3,
) -> EpubDiffResult:
    """Compare two EPUB files and report the differences between them.

    Spine pages are matched positionally (so mid-book insertions/removals do
    not cascade into name-shift noise); every other zip entry is matched by
    name. Text entries get a git-style unified diff; binary entries are
    compared by hash. OPF metadata is normalized away unless
    ``strict_metadata`` is set.
    """
    old = build_snapshot(old_path)
    new = build_snapshot(new_path)

    def comparison_bytes(name: str, data: bytes) -> bytes:
        if not strict_metadata and name == old.opf_path:
            return normalize_opf_metadata(data.decode("utf-8")).encode("utf-8")
        return data

    def make_unified_diff(name: str, old_data: bytes, new_data: bytes) -> list[str]:
        if not (is_text_entry(name, old_data) and is_text_entry(name, new_data)):
            return []
        return list(
            unified_diff(
                old_data.decode("utf-8", errors="replace").splitlines(),
                new_data.decode("utf-8", errors="replace").splitlines(),
                fromfile=f"old/{name}",
                tofile=f"new/{name}",
                n=context,
                lineterm="",
            )
        )

    pages: list[PageDiffEntry] = []
    max_pages = max(len(old.spine_pages), len(new.spine_pages))
    for position in range(max_pages):
        old_name = old.spine_pages[position] if position < len(old.spine_pages) else None
        new_name = new.spine_pages[position] if position < len(new.spine_pages) else None

        if old_name is None:
            new_image = new.page_images.get(new_name) if new_name else None
            new_size = len(new.entries[new_name]) if new_name and new_name in new.entries else None
            pages.append(PageDiffEntry(position, None, new_name, "added", new_image=new_image, new_size=new_size))
            continue
        if new_name is None:
            old_image = old.page_images.get(old_name) if old_name else None
            old_size = len(old.entries[old_name]) if old_name in old.entries else None
            pages.append(PageDiffEntry(position, old_name, None, "removed", old_image=old_image, old_size=old_size))
            continue

        old_data = old.entries.get(old_name, b"")
        new_data = new.entries.get(new_name, b"")
        old_image = old.page_images.get(old_name)
        new_image = new.page_images.get(new_name)

        old_image_data = old.entries.get(old_image) if old_image and old_image in old.entries else None
        new_image_data = new.entries.get(new_image) if new_image and new_image in new.entries else None
        old_image_hash = sha256_hex(old_image_data) if old_image_data is not None else None
        new_image_hash = sha256_hex(new_image_data) if new_image_data is not None else None

        xhtml_identical = sha256_hex(old_data) == sha256_hex(new_data)
        image_identical = old_image_hash == new_image_hash

        if xhtml_identical and image_identical:
            pages.append(
                PageDiffEntry(
                    position,
                    old_name,
                    new_name,
                    "identical",
                    old_image=old_image,
                    new_image=new_image,
                    old_size=len(old_data),
                    new_size=len(new_data),
                )
            )
        elif not xhtml_identical:
            detail = None
            if not image_identical:
                detail = _image_change_detail(
                    old_image_hash, new_image_hash, _pixel_compare(old_image_data, new_image_data)
                )
            pages.append(
                PageDiffEntry(
                    position,
                    old_name,
                    new_name,
                    "modified",
                    old_image=old_image,
                    new_image=new_image,
                    old_size=len(old_data),
                    new_size=len(new_data),
                    detail=detail,
                    diff=make_unified_diff(old_name, old_data, new_data),
                )
            )
        else:
            pixel_comp = _pixel_compare(old_image_data, new_image_data)
            if pixel_comp is not None and pixel_comp.differing_pct == 0:
                pages.append(
                    PageDiffEntry(
                        position,
                        old_name,
                        new_name,
                        "re-encoded",
                        old_image=old_image,
                        new_image=new_image,
                        old_size=len(old_data),
                        new_size=len(new_data),
                        detail=(
                            "visually identical (re-encoded); "
                            f"image hash {old_image_hash[:8] if old_image_hash else '-'}"
                            f" -> {new_image_hash[:8] if new_image_hash else '-'}"
                        ),
                    )
                )
            else:
                pages.append(
                    PageDiffEntry(
                        position,
                        old_name,
                        new_name,
                        "image-modified",
                        old_image=old_image,
                        new_image=new_image,
                        old_size=len(old_data),
                        new_size=len(new_data),
                        detail=_image_change_detail(old_image_hash, new_image_hash, pixel_comp),
                    )
                )

    referenced_images = {image for image in (*old.page_images.values(), *new.page_images.values()) if image}
    excluded = {*old.spine_pages, *new.spine_pages, *referenced_images}

    files: list[EpubDiffEntry] = []
    for name in sorted(set(old.entries) | set(new.entries)):
        if name in excluded:
            continue
        old_data = old.entries.get(name)
        new_data = new.entries.get(name)

        if old_data is None:
            # Present only in the new archive.
            if new_data is None:
                continue
            files.append(EpubDiffEntry(name, "added", new_size=len(new_data), new_hash=sha256_hex(new_data)))
            continue
        if new_data is None:
            # Present only in the old archive.
            files.append(EpubDiffEntry(name, "removed", old_size=len(old_data), old_hash=sha256_hex(old_data)))
            continue

        old_hash = sha256_hex(old_data)
        new_hash = sha256_hex(new_data)

        if sha256_hex(comparison_bytes(name, old_data)) == sha256_hex(comparison_bytes(name, new_data)):
            files.append(
                EpubDiffEntry(
                    name,
                    "identical",
                    old_size=len(old_data),
                    new_size=len(new_data),
                    old_hash=old_hash,
                    new_hash=new_hash,
                )
            )
            continue

        old_key = comparison_bytes(name, old_data)
        new_key = comparison_bytes(name, new_data)
        diff_lines = make_unified_diff(name, old_key, new_key)
        status = "modified"
        detail = None
        if not diff_lines and is_image_entry(name):
            pixel_comp = compare_image_pixels(old_data, new_data)
            if pixel_comp is not None and pixel_comp.differing_pct == 0:
                status = "re-encoded"
                detail = "visually identical (re-encoded)"
            elif pixel_comp is not None:
                resized_note = " (resized)" if pixel_comp.resized else ""
                detail = (
                    f"{pixel_comp.differing_pct:.2f}% pixels differ{resized_note}, mean Δ {pixel_comp.mean_diff:.2f}"
                )
        if detail is None and not diff_lines:
            detail = f"binary content differs ({len(old_data)} -> {len(new_data)} bytes)"
        files.append(
            EpubDiffEntry(
                name,
                status,
                old_size=len(old_data),
                new_size=len(new_data),
                old_hash=old_hash,
                new_hash=new_hash,
                detail=detail,
                diff=diff_lines,
            )
        )

    return EpubDiffResult(pages=pages, files=files)

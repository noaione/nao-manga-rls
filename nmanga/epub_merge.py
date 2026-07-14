"""Utilities for replacing images in an extracted EPUB."""

from __future__ import annotations

import mimetypes
import posixpath
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from defusedxml import ElementTree as ET  # noqa: N817

from .epub_render import find_root_file_path, resolve_epub_href

__all__ = ("EpubImageReplacement", "merge_images_into_extracted_epub")

TEXT_DOCUMENT_SUFFIXES = {
    ".css",
    ".htm",
    ".html",
    ".ncx",
    ".opf",
    ".pls",
    ".smil",
    ".svg",
    ".xml",
    ".xhtml",
}
IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jfif",
    ".jpeg",
    ".jpg",
    ".jxl",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

ATTRIBUTE_RE = re.compile(
    r"(?P<prefix>\b(?:content|data|href|poster|src)\s*=\s*)"
    r"(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    re.IGNORECASE | re.DOTALL,
)
SRCSET_RE = re.compile(
    r"(?P<prefix>\bsrcset\s*=\s*)(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    re.IGNORECASE | re.DOTALL,
)
CSS_URL_RE = re.compile(
    r"(?P<prefix>url\(\s*)(?P<quote>['\"]?)(?P<value>.*?)(?P=quote)(?P<suffix>\s*\))",
    re.IGNORECASE | re.DOTALL,
)
MANIFEST_ITEM_RE = re.compile(r"<(?P<prefix>[\w.-]+:)?item\b.*?>", re.IGNORECASE | re.DOTALL)
MEDIA_TYPE_RE = re.compile(
    r"(?P<prefix>\bmedia-type\s*=\s*)(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class EpubImageReplacement:
    """One image copied into an extracted EPUB."""

    source_path: Path
    old_path: Path
    new_path: Path
    media_type: str


def _is_image(path: Path) -> bool:
    media_type, _ = mimetypes.guess_type(path.name)
    return path.suffix.casefold() in IMAGE_SUFFIXES or bool(media_type and media_type.startswith("image/"))


def _image_media_type(path: Path) -> str:
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type and media_type.startswith("image/"):
        return media_type
    return f"image/{path.suffix.lstrip('.').lower()}"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _manifest_image_paths(package_path: Path, epub_root: Path) -> list[Path]:
    root = ET.fromstring(package_path.read_bytes())
    paths: list[Path] = []

    for element in root.iter():
        if _local_name(element.tag) != "item":
            continue

        href = element.get("href")
        media_type = element.get("media-type", "")
        if not href:
            continue

        try:
            image_path = resolve_epub_href(package_path, href, epub_root=epub_root)
        except ValueError:
            continue

        if media_type.startswith("image/") or _is_image(image_path):
            paths.append(image_path)

    return paths


def _replacement_plan(epub_root: Path, image_folder: Path, package_path: Path) -> list[EpubImageReplacement]:
    source_by_stem: dict[str, list[Path]] = {}
    for source_path in image_folder.rglob("*"):
        if source_path.is_file() and _is_image(source_path):
            source_by_stem.setdefault(source_path.stem.casefold(), []).append(source_path)

    replacements: list[EpubImageReplacement] = []
    targets: dict[Path, Path] = {}
    seen_old_paths: set[Path] = set()
    for old_path in _manifest_image_paths(package_path, epub_root):
        if old_path in seen_old_paths:
            continue
        seen_old_paths.add(old_path)

        candidates = source_by_stem.get(old_path.stem.casefold(), [])
        if not candidates:
            continue
        if len(candidates) > 1:
            choices = ", ".join(str(path) for path in sorted(candidates))
            raise ValueError(f"Multiple replacement images match {old_path.name}: {choices}")

        source_path = candidates[0]
        new_path = old_path.with_suffix(source_path.suffix.lower())
        if new_path in targets and targets[new_path] != old_path:
            raise ValueError(f"Multiple EPUB images would be replaced by the same file: {new_path}")
        if new_path.exists() and new_path != old_path:
            raise FileExistsError(f"Replacement destination already exists: {new_path}")

        targets[new_path] = old_path
        replacements.append(
            EpubImageReplacement(
                source_path=source_path,
                old_path=old_path,
                new_path=new_path,
                media_type=_image_media_type(new_path),
            )
        )

    return replacements


def _replace_reference(value: str, document_path: Path, epub_root: Path, mapping: dict[Path, Path]) -> str:
    leading = value[: len(value) - len(value.lstrip())]
    trailing = value[len(value.rstrip()) :]
    reference = value.strip()
    if not reference:
        return value

    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return value

    try:
        old_path = resolve_epub_href(document_path, parsed.path, epub_root=epub_root)
    except (ValueError, OSError):
        return value

    new_path = mapping.get(old_path)
    if new_path is None:
        return value

    document_dir = document_path.relative_to(epub_root).parent.as_posix() or "."
    new_epub_path = new_path.relative_to(epub_root).as_posix()
    new_reference = quote(posixpath.relpath(new_epub_path, document_dir), safe="/")
    rewritten = urlunsplit(("", "", new_reference, parsed.query, parsed.fragment))
    return f"{leading}{rewritten}{trailing}"


def _replace_srcset(value: str, document_path: Path, epub_root: Path, mapping: dict[Path, Path]) -> str:
    rewritten_parts: list[str] = []
    changed = False
    for part in value.split(","):
        leading = part[: len(part) - len(part.lstrip())]
        trailing = part[len(part.rstrip()) :]
        tokens = part.strip().split(maxsplit=1)
        if not tokens:
            rewritten_parts.append(part)
            continue

        rewritten_url = _replace_reference(tokens[0], document_path, epub_root, mapping)
        changed = changed or rewritten_url != tokens[0]
        descriptor = f" {tokens[1]}" if len(tokens) == 2 else ""
        rewritten_parts.append(f"{leading}{rewritten_url}{descriptor}{trailing}")

    return ",".join(rewritten_parts) if changed else value


def _replace_manifest_media_types(
    text: str,
    document_path: Path,
    epub_root: Path,
    replacements: dict[Path, EpubImageReplacement],
) -> str:
    def replace_item(match: re.Match[str]) -> str:
        item = match.group(0)
        href_match = next(
            (match for match in ATTRIBUTE_RE.finditer(item) if match.group("prefix").lower().startswith("href")),
            None,
        )
        if href_match is None:
            return item

        try:
            old_path = resolve_epub_href(document_path, href_match.group("value"), epub_root=epub_root)
        except (ValueError, OSError):
            return item

        replacement = replacements.get(old_path)
        if replacement is None:
            return item

        if MEDIA_TYPE_RE.search(item):
            return MEDIA_TYPE_RE.sub(
                lambda media_match: (
                    f"{media_match.group('prefix')}{media_match.group('quote')}"
                    f"{replacement.media_type}{media_match.group('quote')}"
                ),
                item,
                count=1,
            )

        insert_at = item.rfind("/>")
        if insert_at < 0:
            insert_at = item.rfind(">")
        return f'{item[:insert_at]} media-type="{replacement.media_type}"{item[insert_at:]}'

    return MANIFEST_ITEM_RE.sub(replace_item, text)


def _rewrite_document(
    text: str,
    document_path: Path,
    epub_root: Path,
    replacements: dict[Path, EpubImageReplacement],
) -> str:
    mapping = {old_path: replacement.new_path for old_path, replacement in replacements.items()}
    if document_path.suffix.casefold() == ".opf":
        text = _replace_manifest_media_types(text, document_path, epub_root, replacements)

    def replace_srcset(match: re.Match[str]) -> str:
        value = _replace_srcset(match.group("value"), document_path, epub_root, mapping)
        return f"{match.group('prefix')}{match.group('quote')}{value}{match.group('quote')}"

    def replace_attribute(match: re.Match[str]) -> str:
        value = _replace_reference(match.group("value"), document_path, epub_root, mapping)
        return f"{match.group('prefix')}{match.group('quote')}{value}{match.group('quote')}"

    def replace_css_url(match: re.Match[str]) -> str:
        value = _replace_reference(match.group("value"), document_path, epub_root, mapping)
        return f"{match.group('prefix')}{match.group('quote')}{value}{match.group('quote')}{match.group('suffix')}"

    text = SRCSET_RE.sub(replace_srcset, text)
    text = ATTRIBUTE_RE.sub(replace_attribute, text)
    return CSS_URL_RE.sub(replace_css_url, text)


def merge_images_into_extracted_epub(
    epub_root: Path,
    image_folder: Path,
    *,
    remove_originals: bool = False,
) -> list[EpubImageReplacement]:
    """Copy matching images into an extracted EPUB and update its references."""

    epub_root = epub_root.resolve()
    image_folder = image_folder.resolve()
    if epub_root == image_folder:
        raise ValueError("The extracted EPUB and replacement image folders must be different")

    container_path = epub_root / "META-INF" / "container.xml"
    if not container_path.is_file():
        raise FileNotFoundError(f"EPUB container file not found: {container_path}")

    root_file_path = find_root_file_path(container_path.read_text(encoding="utf-8"))
    package_path = resolve_epub_href(container_path, f"/{root_file_path}", epub_root=epub_root)
    if not package_path.is_file():
        raise FileNotFoundError(f"EPUB package document not found: {package_path}")

    plan = _replacement_plan(epub_root, image_folder, package_path)
    if not plan:
        raise ValueError("No replacement images matched images in the EPUB manifest")

    replacement_by_old = {replacement.old_path: replacement for replacement in plan}
    rewritten_documents: dict[Path, str] = {}
    for document_path in epub_root.rglob("*"):
        if not document_path.is_file() or document_path.suffix.casefold() not in TEXT_DOCUMENT_SUFFIXES:
            continue
        text = document_path.read_text(encoding="utf-8")
        rewritten = _rewrite_document(text, document_path, epub_root, replacement_by_old)
        if rewritten != text:
            rewritten_documents[document_path] = rewritten

    for replacement in plan:
        replacement.new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(replacement.source_path, replacement.new_path)
    for document_path, text in rewritten_documents.items():
        document_path.write_text(text, encoding="utf-8")
    if remove_originals:
        for replacement in plan:
            if replacement.old_path != replacement.new_path:
                replacement.old_path.unlink()

    return plan

from __future__ import annotations

from pathlib import Path

from nmanga.epub_render import copy_image_only_page, get_image_only_page_path


def write_xhtml(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
        <html xmlns="http://www.w3.org/1999/xhtml">
          <head><title>test</title></head>
          <body>{body}</body>
        </html>
        """,
        encoding="utf-8",
    )


def test_get_image_only_page_path_accepts_single_wrapped_image(tmp_path: Path) -> None:
    epub_root = tmp_path / "epub"
    xhtml_path = epub_root / "OEBPS" / "html" / "page.xhtml"
    image_path = epub_root / "OEBPS" / "images" / "page.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"jpgdata")
    write_xhtml(xhtml_path, '<div><a href="#"><img src="../images/page.jpg" /></a></div>')

    assert get_image_only_page_path(xhtml_path, epub_root) == image_path


def test_get_image_only_page_path_rejects_text_overlay(tmp_path: Path) -> None:
    epub_root = tmp_path / "epub"
    xhtml_path = epub_root / "OEBPS" / "html" / "page.xhtml"
    write_xhtml(xhtml_path, '<img src="../images/page.jpg" /><p>overlay text</p>')

    assert get_image_only_page_path(xhtml_path, epub_root) is None


def test_get_image_only_page_path_rejects_multiple_images(tmp_path: Path) -> None:
    epub_root = tmp_path / "epub"
    xhtml_path = epub_root / "OEBPS" / "html" / "page.xhtml"
    write_xhtml(xhtml_path, '<img src="../images/a.jpg" /><img src="../images/b.jpg" />')

    assert get_image_only_page_path(xhtml_path, epub_root) is None


def test_copy_image_only_page_preserves_extension_and_bytes(tmp_path: Path) -> None:
    src_image_path = tmp_path / "page.webp"
    dest_output = tmp_path / "out"
    src_image_path.write_bytes(b"webpdata")
    dest_output.mkdir()

    dest_image_path = copy_image_only_page(src_image_path, dest_output, 12)

    assert dest_image_path == dest_output / "i_0012.webp"
    assert dest_image_path.read_bytes() == b"webpdata"

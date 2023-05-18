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

from datetime import datetime
from enum import Enum
from mimetypes import guess_type
from os.path import basename
from pathlib import Path
from typing import Optional, Tuple, Type, Union
from xml.dom.minidom import parseString as xml_dom_parse
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import lxml.etree as ET
import py7zr
from PIL import Image

from .templates.epub import EPUB_CONTAINER, EPUB_CONTENT, EPUB_PAGE, EPUB_STYLES
from .utils import encode_or

__all__ = (
    "MangaExporter",
    "CBZMangaExporter",
    "CB7MangaExporter",
    "EPUBMangaExporter",
    "ExporterType",
    "exporter_factory",
)


class ExporterType(str, Enum):
    raw = "folder"
    cbz = "cbz"
    # cbr = 2
    cb7 = "cb7"
    epub = "epub"

    @classmethod
    def from_choice(cls: Type["ExporterType"], ext: str):
        ext = ext.lower()
        if ext == "cbz":
            return cls.cbz
        elif ext == "cb7":
            return cls.cb7
        elif ext == "epub":
            return cls.epub
        else:
            return cls.raw


class MangaExporter:
    TYPE = ExporterType.raw

    def __init__(self, output_directory: Path):
        self._out_dir = output_directory
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def is_existing(self):
        return self._out_dir.exists()

    def add_image(self, image_name: str, image_data: Union[bytes, Path]):
        target_path = self._out_dir / image_name
        if isinstance(image_data, bytes):
            target_path.write_bytes(image_data)
        else:
            # Copy image
            target_path.write_bytes(image_data.read_bytes())

    def set_comment(self, comment: Union[str, bytes]):
        pass

    def close(self):
        pass


class ArchiveMangaExporter(MangaExporter):
    _file_name: str

    def __init__(self, file_name: str, output_directory: Path):
        # Check if class used is the child class and not this class
        if self.__class__ == ArchiveMangaExporter:
            raise TypeError("Cannot instantiate ArchiveMangaExporter directly.")

        super().__init__(output_directory)

        self._file_name = file_name


class CBZMangaExporter(ArchiveMangaExporter):
    TYPE = ExporterType.cbz

    def __init__(self, file_name: str, output_directory: Path):
        super().__init__(file_name, output_directory)

        self._target_cbz: ZipFile = ZipFile(self._out_dir / f"{file_name}.cbz", "w", compression=ZIP_DEFLATED)

    def is_existing(self):
        parent_dir = self._out_dir.parent
        target_cbz = parent_dir / f"{self._file_name}.cbz"
        if target_cbz.exists():
            return True
        return False

    def add_image(self, image_name: str, image_data: Union[bytes, Path]):
        if isinstance(image_data, bytes):
            self._target_cbz.writestr(basename(image_name), image_data)
        else:
            self._target_cbz.write(str(image_data), basename(image_name))

    def set_comment(self, comment: Union[str, bytes]):
        self._target_cbz.comment = encode_or(comment) or b""

    def close(self):
        self._target_cbz.close()


class CB7MangaExporter(ArchiveMangaExporter):
    TYPE = ExporterType.cb7

    def __init__(self, file_name: str, output_directory: Path):
        super().__init__(file_name, output_directory)

        self._target_cb7: py7zr.SevenZipFile = py7zr.SevenZipFile(self._out_dir / f"{file_name}.cb7", "w")

    def is_existing(self):
        parent_dir = self._out_dir.parent
        target_cb7 = parent_dir / f"{self._file_name}.cb7"
        if target_cb7.exists():
            return True
        return False

    def add_image(self, image_name: str, image_data: Union[bytes, Path]):
        if isinstance(image_data, bytes):
            self._target_cb7.writestr(image_data, basename(image_name))
        else:
            self._target_cb7.write(str(image_data), basename(image_name))

    def close(self):
        self._target_cb7.close()


class EPUBMangaExporter(ArchiveMangaExporter):
    TYPE = ExporterType.epub

    def __init__(self, file_name: str, output_directory: Path, *, manga_title: str):
        super().__init__(file_name, output_directory)

        self._target_epub = ZipFile(self._out_dir / f"{file_name}.epub", "w", compression=ZIP_DEFLATED)
        self._meta_injected: bool = False
        self._page_counter = 1
        self._manga_title = manga_title

        self._base_img_size: Optional[Tuple[int, int]] = None
        self._last_direction = "center"

    def is_existing(self):
        parent_dir = self._out_dir.parent
        target_epub = parent_dir / f"{self._file_name}.epub"
        if target_epub.exists():
            return True
        return False

    def _format_xml(self):
        as_string = ET.tostring(
            self._xml_content_opf, xml_declaration=True, encoding="utf-8", method="xml", pretty_print=True
        )
        as_dom = xml_dom_parse(as_string)
        ugly_xml: str = as_dom.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
        fixed_xml = []
        for ugly in ugly_xml.split("\n"):
            if ugly.strip("\t").strip():
                fixed_xml.append(ugly)
        return "\n".join(fixed_xml)

    def _initialize_meta(self):
        if self._meta_injected:
            return
        self._target_epub.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
        self._target_epub.writestr("OEBPS/Styles/styles.css", EPUB_STYLES)
        self._target_epub.writestr("META-INF/container.xml", EPUB_CONTAINER)
        self._initialize_opf()
        self._meta_injected = True

    def _initialize_opf(self):
        current_date = datetime.utcnow().timestamp()
        identifier = self._manga_title.lower() + f"-{int(current_date)}"

        fixed_content_opf = EPUB_CONTENT.format(
            title=self._manga_title,
            identifier=identifier,
            time=int(current_date),
        )

        content_opf_xml = ET.fromstring(fixed_content_opf.encode("utf-8"))
        self._xml_content_opf = content_opf_xml

        self.__add_item_to_manifest("styles.css", "Styles/styles.css", "text/css")

    def _inject_meta(self, number: int, filename: str, width: int, height: int):
        page_title = f"{self._manga_title} - Page #{number}"
        page_id = f"page-{number:03d}"
        if number == 1:
            page_title = f"{self._manga_title} - Cover Page"
            page_id = "page-cover"

        page_xhtml_text = EPUB_PAGE.format(
            title=page_title,
            filename=filename,
            width=width,
            height=height,
        )
        self._target_epub.writestr(f"OEBPS/Text/page_{number:03d}.xhtml", page_xhtml_text.encode("utf-8"))
        self.__add_item_to_manifest(
            page_id,
            f"Text/page_{number:03d}.xhtml",
            "application/xhtml+xml",
            page_id.replace("page-", "image-"),
        )

    def __add_item_to_manifest(self, idname: str, filename: str, mimetype: str, fallback: Optional[str] = None):
        # ref: <item id="cover" href="Text/cover.xhtml" media-type="application/xhtml+xml"/>
        # ref: <item id="Color1.jpg" href="Images/Color1.jpg" media-type="image/jpeg"/>
        manifest_root = self._xml_content_opf.find(".//{http://www.idpf.org/2007/opf}manifest")
        item = ET.SubElement(self._xml_content_opf, "item")
        item.set("id", idname)
        item.set("href", filename)
        item.set("media-type", mimetype)
        if fallback:
            item.set("fallback", fallback)
            item.set("properties", "svg")
        manifest_root.append(item)
        if mimetype == "application/xhtml+xml":
            self.__add_item_to_spine(idname)

    def __add_item_to_spine(self, idref: str):
        # ref: <itemref idref="cover"/>
        spine_root = self._xml_content_opf.find(".//{http://www.idpf.org/2007/opf}spine")
        item = ET.SubElement(self._xml_content_opf, "itemref")
        item.set("linear", "yes")
        item.set("idref", idref)
        if "cover" in idref:
            item.set("properties", "rendition:page-spread-center")
        else:
            if self._last_direction == "center":
                self._last_direction = "right"
                direction = "right"
            elif self._last_direction == "right":
                self._last_direction = "left"
                direction = "left"
            else:
                self._last_direction = "right"
                direction = "right"
            if self._mark_center_spread:
                item.set("properties", "rendition:page-spread-center")
                self._last_direction = "center"
            else:
                item.set("properties", f"page-spread-{direction}")
        spine_root.append(item)

    def __inject_size_metadata(self, width: int, height: int):
        metadata_root = self._xml_content_opf.find(".//{http://www.idpf.org/2007/opf}metadata")
        item_res = ET.SubElement(metadata_root, "meta")
        item_res.set("name", "original-resolution")
        item_res.set("content", f"{width}x{height}")
        item_viewport = ET.SubElement(metadata_root, "meta")
        item_viewport.set("property", "fixed-layout-jp:viewport")
        # Put inside tag
        item_viewport.text = f"width={width}, height={height}"
        metadata_root.append(item_res)
        metadata_root.append(item_viewport)

    def add_image(self, image_name: str, image_data: Union[bytes, Path]):
        self._initialize_meta()
        image = f"OEBPS/Images/{basename(image_name)}"

        if isinstance(image_data, bytes):
            self._target_epub.writestr(image, image_data)
        else:
            self._target_epub.write(str(image_data), image)
        im = Image.open(image_data)
        width, height = im.size
        if self._base_img_size is None:
            self._base_img_size = (width, height)
            # Inject the size properties
            self.__inject_size_metadata(width, height)
        base_target = (self._base_img_size[0] * 2) - 80
        im.close()
        # If base target are smaller than base, just set it to + 150
        if base_target < self._base_img_size[0]:
            base_target = self._base_img_size + 150
        if width > base_target:
            # Double spread mode
            self._mark_center_spread = True
        self._inject_meta(self._page_counter, basename(image_name), width, height)
        mimetype, _ = guess_type(image)
        mimetype = mimetype or "application/octet-stream"
        if self._page_counter == 1:
            idref_image = "image-cover"
        else:
            idref_image = f"image-{self._page_counter:03d}"
        self.__add_item_to_manifest(idref_image, f"Images/{image}", mimetype)
        self._page_counter += 1
        self._mark_center_spread = False

    def close(self):
        self._target_epub.writestr("OEBPS/content.opf", self._format_xml())
        self._target_epub.close()

    def set_comment(self, comment: Union[str, bytes]):
        self._target_epub.comment = encode_or(comment) or b""


def exporter_factory(
    file_name: str,
    output_directory: Path,
    mode: Union[str, ExporterType] = ExporterType.cbz,
    **kwargs,
):
    if isinstance(mode, str):
        mode = ExporterType.from_choice(mode)
    if mode == ExporterType.cbz:
        return CBZMangaExporter(file_name, output_directory)
    elif mode == ExporterType.cb7:
        return CB7MangaExporter(file_name, output_directory)
    elif mode == ExporterType.epub:
        return EPUBMangaExporter(file_name, output_directory, **kwargs)
    else:
        return MangaExporter(output_directory)

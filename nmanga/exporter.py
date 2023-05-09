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

from enum import Enum
from os.path import basename
from pathlib import Path
from typing import Union
from zipfile import ZIP_DEFLATED, ZipFile

from .utils import encode_or

__all__ = (
    "MangaExporter",
    "CBZMangaExporter",
    "ExporterType",
)


class ExporterType(Enum):
    raw = 0
    cbz = 1


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


class CBZMangaExporter(MangaExporter):
    TYPE = ExporterType.cbz

    def __init__(self, file_name: str, output_directory: Path):
        super().__init__(output_directory)

        self._file_name = file_name
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
        self._target_cbz.comment = encode_or(comment)

    def close(self):
        self._target_cbz.close()

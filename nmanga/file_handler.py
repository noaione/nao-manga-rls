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

import random
import tempfile
import zipfile
from enum import Enum
from mimetypes import types_map
from os import path
from pathlib import Path
from string import ascii_letters, digits
from typing import List

import py7zr
from unrar.cffi import rarfile

__all__ = (
    "YieldType",
    "collect_image_from_cbz",
    "collect_image_from_rar",
    "collect_image_from_7z",
    "collect_image_from_folder",
    "is_cbz",
    "is_rar",
    "is_7zarchive",
    "is_archive",
    "collect_image",
    "collect_image_archive",
    "collect_all_comics",
    "create_temp_dir",
    "remove_folder_and_contents",
    "random_name",
)


class YieldType(Enum):
    CBZ = 1
    FOLDER = 2
    RAR = 3
    SEVENZIP = 4


class UnknownArchiveType(Exception):
    def __init__(self, file: Path) -> None:
        self.file = file
        super().__init__(f"An unknown archive format found: {file}")


def is_image(file_name: str) -> bool:
    return types_map.get(path.splitext(file_name)[-1], "").startswith("image/")


def collect_image_from_cbz(cbz_file: zipfile.ZipFile):
    all_contents = cbz_file.filelist.copy()
    all_contents.sort(key=lambda x: path.basename(x.filename))
    valid_images = [x for x in all_contents if not x.is_dir() and is_image(x.filename)]
    valid_images.sort(key=lambda x: path.basename(x.filename))
    total_count = len(valid_images)
    for content in valid_images:
        yield content, cbz_file, total_count, YieldType.CBZ


def collect_image_from_rar(rar_file: rarfile.RarFile):
    all_contents: List[rarfile.RarInfo] = rar_file.infolist()
    all_contents.sort(key=lambda x: path.basename(x.filename))
    valid_images = [x for x in all_contents if not x.is_dir() and is_image(x.filename)]
    valid_images.sort(key=lambda x: path.basename(x.filename))
    total_count = len(valid_images)
    for content in valid_images:
        yield content, rar_file, total_count, YieldType.RAR


def collect_image_from_7z(sevenzip_file: py7zr.SevenZipFile):
    all_contents = sevenzip_file.list()
    all_contents.sort(key=lambda x: path.basename(x.filename))
    valid_images = [x for x in all_contents if not x.is_directory and is_image(x.filename)]
    valid_images.sort(key=lambda x: path.basename(x.filename))
    total_count = len(valid_images)
    for content in valid_images:
        yield content, sevenzip_file, total_count, YieldType.SEVENZIP


def collect_image_from_folder(folder_path: Path):
    all_contents = list(folder_path.glob("*"))
    all_contents.sort(key=lambda x: path.basename(x))
    valid_images = [x for x in all_contents if is_image(x.name)]
    valid_images.sort(key=lambda x: path.basename(x.name))
    total_count = len(valid_images)
    for file in valid_images:
        yield file, folder_path, total_count, YieldType.FOLDER


def is_cbz(file: Path):
    if not file.is_file():
        return False
    return zipfile.is_zipfile(file)


def is_rar(file: Path):
    if not file.is_file():
        return False
    return rarfile.is_rarfile(str(file))


def is_7zarchive(file: Path):
    if not file.is_file():
        return False
    return py7zr.is_7zfile(file)


def is_archive(file: Path):
    return is_cbz(file) or is_rar(file) or is_7zarchive(file)


def collect_image_archive(file: Path):
    if not file.is_file():
        return

    if is_cbz(file):
        with zipfile.ZipFile(str(file)) as cbz_file:
            yield from collect_image_from_cbz(cbz_file)
    elif is_rar(file):
        with rarfile.RarFile(str(file)) as rar_file:
            yield from collect_image_from_rar(rar_file)
    elif is_7zarchive(file):
        with py7zr.SevenZipFile(str(file)) as archive:
            yield from collect_image_from_7z(archive)
    else:
        raise UnknownArchiveType(file)


def collect_image(path_or_archive: Path):
    if path_or_archive.is_file():
        yield from collect_image_archive(path_or_archive)
    else:
        yield from collect_image_from_folder(path_or_archive)


def collect_all_comics(folder: Path):
    for file in folder.glob("*.cb[z|r|7]"):
        if is_archive(file):
            yield file


def create_temp_dir() -> Path:
    return Path(tempfile.mkdtemp())


def remove_folder_and_contents(folder: Path):
    if not folder.exists() or not folder.is_dir():
        return
    for folder in folder.iterdir():
        if folder.is_dir():
            remove_folder_and_contents(folder)
        else:
            folder.unlink(missing_ok=True)
    folder.rmdir()


def random_name(length: int = 8):
    return "".join(random.choices(ascii_letters + digits, k=length))

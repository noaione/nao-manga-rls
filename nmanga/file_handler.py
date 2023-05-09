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
import tarfile
import tempfile
import zipfile
from copy import deepcopy
from enum import Enum
from mimetypes import types_map
from os import path
from pathlib import Path
from string import ascii_letters, digits
from typing import Generator, List, Optional, Tuple, Union

import ftfy
import py7zr
from unrar.cffi import rarfile

from .utils import decode_or, encode_or

__all__ = (
    "YieldType",
    "MangaArchive",
    "collect_image_from_cbz",
    "collect_image_from_rar",
    "collect_image_from_7z",
    "collect_image_from_tar",
    "collect_image_from_folder",
    "is_cbz",
    "is_rar",
    "is_7zarchive",
    "is_tararchive",
    "is_archive",
    "collect_image",
    "collect_image_archive",
    "collect_all_comics",
    "create_temp_dir",
    "remove_folder_and_contents",
    "random_name",
)
extended_types_map = deepcopy(types_map)
# modern image formats
extended_types_map[".avif"] = "image/avif"
extended_types_map[".webp"] = "image/webp"
extended_types_map[".heic"] = "image/heic"
extended_types_map[".heif"] = "image/heif"
extended_types_map[".jxl"] = "image/jxl"
# jpeg2000
extended_types_map[".jp2"] = "image/jp2"


class YieldType(Enum):
    CBZ = 1
    FOLDER = 2
    RAR = 3
    SEVENZIP = 4
    TAR = 5  # Gzip, LZMA, XZ, BZ2


class UnknownArchiveType(Exception):
    def __init__(self, file: Path) -> None:
        self.file = file
        super().__init__(f"An unknown archive format found: {file}")


class WrappedRarFile(rarfile.RarFile):
    def __init__(self, filename):
        super().__init__(filename)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def is_image(file_name: str) -> bool:
    return extended_types_map.get(path.splitext(file_name)[-1], "").startswith("image/")


def collect_image_from_cbz(cbz_file: zipfile.ZipFile):
    all_contents = cbz_file.filelist.copy()
    valid_images = [x for x in all_contents if not x.is_dir() and is_image(x.filename)]
    valid_images.sort(key=lambda x: path.basename(x.filename))
    total_count = len(valid_images)
    for content in valid_images:
        yield content, cbz_file, total_count, YieldType.CBZ


def collect_image_from_rar(rar_file: rarfile.RarFile):
    all_contents: List[rarfile.RarInfo] = rar_file.infolist()
    valid_images = [x for x in all_contents if not x.is_dir() and is_image(x.filename)]
    valid_images.sort(key=lambda x: path.basename(x.filename))
    total_count = len(valid_images)
    for content in valid_images:
        yield content, rar_file, total_count, YieldType.RAR


def collect_image_from_7z(sevenzip_file: py7zr.SevenZipFile):
    all_contents = sevenzip_file.list()
    valid_images = [x for x in all_contents if not x.is_directory and is_image(x.filename)]
    valid_images.sort(key=lambda x: path.basename(x.filename))
    total_count = len(valid_images)
    for content in valid_images:
        yield content, sevenzip_file, total_count, YieldType.SEVENZIP


def collect_image_from_tar(tararchive_file: tarfile.TarFile):
    all_contents = tararchive_file.getmembers()
    valid_images = [x for x in all_contents if not x.isdir() and is_image(x.name)]
    valid_images.sort(key=lambda x: path.basename(x.name))
    total_count = len(valid_images)
    for content in valid_images:
        yield content, tararchive_file, total_count, YieldType.CBZ


def collect_image_from_folder(folder_path: Path):
    all_contents = list(folder_path.glob("*"))
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


def is_tararchive(file: Path):
    if not file.is_file():
        return False
    return tarfile.is_tarfile(str(file))


def is_archive(file: Path):
    return is_cbz(file) or is_rar(file) or is_7zarchive(file) or is_tararchive(file)


def collect_image_archive(file: Path):
    if not file.is_file():
        return

    if is_cbz(file):
        with zipfile.ZipFile(str(file)) as cbz_file:
            yield from collect_image_from_cbz(cbz_file)
    elif is_rar(file):
        with WrappedRarFile(str(file)) as rar_file:
            yield from collect_image_from_rar(rar_file)
    elif is_7zarchive(file):
        with py7zr.SevenZipFile(str(file)) as archive:
            yield from collect_image_from_7z(archive)
    elif is_tararchive(file):
        with tarfile.open(str(file)) as tararchive_file:
            yield from collect_image_from_tar(tararchive_file)
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


AccessorType = Union[zipfile.ZipFile, rarfile.RarFile, py7zr.SevenZipFile, tarfile.TarFile, Path]
AccessorFile = Union[zipfile.ZipInfo, py7zr.FileInfo, rarfile.RarInfo, tarfile.TarInfo, str, bytes]
AccessorImage = Union[zipfile.ZipInfo, py7zr.FileInfo, rarfile.RarInfo, tarfile.TarInfo, Path]


class MangaImage:
    """
    Wrapper for image path, archive, or something like that.
    """

    def __init__(self, file: AccessorImage):
        self.__accessor = file

    @property
    def name(self) -> str:
        """Return the final component of the image filename.

        Will include the extension.
        """
        if isinstance(self.__accessor, zipfile.ZipInfo):
            return path.basename(self.__accessor.filename)
        elif isinstance(self.__accessor, py7zr.FileInfo):
            return path.basename(self.__accessor.filename)
        elif isinstance(self.__accessor, rarfile.RarInfo):
            return path.basename(self.__accessor.filename)
        elif isinstance(self.__accessor, tarfile.TarInfo):
            return path.basename(self.__accessor.name)
        elif isinstance(self.__accessor, Path):
            return self.__accessor.name
        else:
            raise TypeError(f"Unknown type: {type(self.__accessor)}")

    def __str__(self):
        return self.filename

    @property
    def filename(self):
        """Shortcut for the image filename."""
        try:
            return ftfy.fix_encoding(self.name)
        except Exception:
            # In a absurd case that it fail miserably, just return the original name.
            return self.name

    @property
    def stem(self):
        """Return the final component of the image filename without the extension."""
        if isinstance(self.__accessor, Path):
            return self.__accessor.stem
        return path.splitext(self.filename)[0]

    @property
    def suffix(self):
        """Return the extension of the image filename.

        Include the leading dot.
        """
        if isinstance(self.__accessor, Path):
            return self.__accessor.suffix
        extension = path.splitext(self.filename)[-1]
        return extension if extension.startswith(".") else f".{extension}"

    def access(self):
        """Return the accessor or internal file object."""
        return self.__accessor

    def is_archive(self):
        """Return True if the image is an archive."""
        return not isinstance(self.__accessor, Path)


class MangaArchive:
    """
    Wrapper for multiple archive format and folder.
    Allow simple access to some property, and access to read
    one of the file in the archive or folder.
    """

    def __init__(self, file_or_folder: Path):
        self.__accessor: AccessorType = None
        self.__path = file_or_folder

    def __check_open(self):
        if self.__accessor is None:
            self.open()

    def open(self):
        if self.__accessor is not None:
            return self.__accessor
        if self.__path.is_file():
            if is_cbz(self.__path):
                self.__accessor = zipfile.ZipFile(str(self.__path))
            elif is_rar(self.__path):
                self.__accessor = rarfile.RarFile(str(self.__path))
            elif is_7zarchive(self.__path):
                self.__accessor = py7zr.SevenZipFile(str(self.__path))
            elif is_tararchive(self.__path):
                self.__accessor = tarfile.open(str(self.__path))
            else:
                raise UnknownArchiveType(self.__path)
        else:
            self.__accessor = self.__path
        return self.__accessor

    def close(self):
        if self.__accessor is not None:
            if isinstance(self.__accessor, (zipfile.ZipFile, py7zr.SevenZipFile, tarfile.TarFile)):
                self.__accessor.close()
            self.__accessor = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def read(self, file: Union[AccessorFile, MangaImage]) -> bytes:
        """Read a specific file from the archive or folder.

        Return the bytes data.
        """
        if isinstance(file, MangaImage):
            return self.__actual_read(file.access())
        return self.__actual_read(file)

    def __actual_read(self, file: AccessorFile) -> bytes:
        self.__check_open()
        if isinstance(file, py7zr.FileInfo) and isinstance(self.__accessor, py7zr.SevenZipFile):
            file_data = self.__accessor.read([file.filename])
            self.__accessor.reset()
            return list(file_data.values())[0].read()
        elif isinstance(file, tarfile.TarInfo) and isinstance(self.__accessor, tarfile.TarFile):
            file_data = self.__accessor.extractfile(file)
            file_data.seek(0)
            return file_data.read()
        elif isinstance(file, Path):
            return file.read_bytes()
        elif isinstance(self.__accessor, Path) and isinstance(file, (str, bytes)):
            if isinstance(file, bytes):
                file = file.decode()
            actual_path = self.__accessor / file
            return actual_path.read_bytes()
        return self.__accessor.read(file)

    def contents(self):
        """
        Get all contents in the files in the archive or folder.
        Might also includes other folders and files.
        """
        self.__check_open()
        if isinstance(self.__accessor, Path):
            return list(self.__accessor.glob("*"))
        elif isinstance(self.__accessor, zipfile.ZipFile):
            return self.__accessor.filelist
        elif isinstance(self.__accessor, rarfile.RarFile):
            return self.__accessor.infolist()
        elif isinstance(self.__accessor, py7zr.SevenZipFile):
            return self.__accessor.list()
        elif isinstance(self.__accessor, tarfile.TarFile):
            return self.__accessor.getmembers()

    def __iter__(self) -> Generator[Tuple[MangaImage, int], None, None]:
        self.__check_open()
        if isinstance(self.__accessor, Path):
            for file, _, count, _ in collect_image_from_folder(self.__path):
                yield MangaImage(file), count
        elif isinstance(self.__accessor, zipfile.ZipFile):
            for file, _, count, _ in collect_image_from_cbz(self.__accessor):
                yield MangaImage(file), count
        elif isinstance(self.__accessor, rarfile.RarFile):
            for file, _, count, _ in collect_image_from_rar(self.__accessor):
                yield MangaImage(file), count
        elif isinstance(self.__accessor, py7zr.SevenZipFile):
            for file, _, count, _ in collect_image_from_7z(self.__accessor):
                yield MangaImage(file), count
        elif isinstance(self.__accessor, tarfile.TarFile):
            for file, _, count, _ in collect_image_from_tar(self.__accessor):
                yield MangaImage(file), count
        else:
            raise NotImplementedError("Not implemented for this archive type")

    @property
    def comment(self) -> Optional[str]:
        self.__check_open()
        if isinstance(self.__accessor, (zipfile.ZipFile, rarfile.RarFile)):
            return decode_or(self.__accessor.comment)
        return None

    @comment.setter
    def comment(self, new_comment: Optional[Union[str, bytes]]):
        self.__check_open()
        if isinstance(self.__accessor, (zipfile.ZipFile, rarfile.RarFile)):
            print("yo")
            self.__accessor.comment = encode_or(new_comment) or b""


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

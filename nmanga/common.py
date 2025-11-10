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

import logging
import logging.handlers
import multiprocessing as mp
import re
import signal
import subprocess as sp
import threading
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Match, Pattern, cast, overload

from . import config, term, utils
from .constants import TARGET_FORMAT, TARGET_FORMAT_ALT, TARGET_TITLE, MangaPublication

__all__ = (
    "ALLOWED_TAG_EXTENSIONS",
    "BRACKET_MAPPINGS",
    "ChapterRange",
    "PseudoChapterMatch",
    "RegexCollection",
    "actual_or_fallback",
    "check_cbz_exist",
    "create_chapter",
    "format_archive_filename",
    "format_daiz_like_filename",
    "format_volume_text",
    "inject_metadata",
    "inquire_chapter_ranges",
    "is_pingo_alpha",
    "make_metadata_command",
    "optimize_images",
    "run_pingo_and_verify",
    "safe_int",
    "threaded_worker",
)


console = term.get_console()
conf = config.get_config()


BRACKET_MAPPINGS = {
    "square": ["[", "]"],
    "round": ["(", ")"],
    "curly": ["{", "}"],
}
# List of image extension that supports exif tagging
ALLOWED_TAG_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "tiff", "avif", "jxl"]


def _worker_initializer(log_queue: term.MessageQueue, log_level: int):
    """Initializer for worker processes to handle keyboard interrupts properly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handler = logging.handlers.QueueHandler(log_queue)
    root_logger.addHandler(handler)

    root_logger.setLevel(log_level)


@contextmanager
def threaded_worker(console: term.Console, threads: int):
    """Initialize worker processes to handle keyboard interrupts properly."""
    with mp.Manager() as manager:
        log_queue = manager.Queue()
        root_logger = logging.getLogger()

        listener = threading.Thread(
            target=term.thread_queue_callback,
            args=(log_queue, console),
            daemon=True,
        )
        listener.start()

        with mp.Pool(
            processes=threads, initializer=_worker_initializer, initargs=(log_queue, root_logger.level)
        ) as pool:
            try:
                yield pool, log_queue
            except KeyboardInterrupt as ke:
                console.warning("Process interrupted by user, terminating workers...")
                pool.terminate()
                pool.join()
                raise RuntimeError("Process interrupted by user.") from ke
            except Exception as e:
                console.error(f"An error occurred: {e}, terminating workers...")
                traceback.print_exc()
                pool.terminate()
                pool.join()
                raise e
            finally:
                log_queue.put(("__CLOSE__", ""))
                listener.join()
                pool.close()


def assert_proc(stuff: IO[bytes] | None) -> None:
    if stuff is None:
        raise ValueError("Subprocess stream is None, cannot proceed.")


class PseudoChapterMatch:
    """
    Simulate a Regex match object for the chapter regex.

    Used mostly in `manualsplit` command.
    """

    def __init__(self):
        self._contents: dict[str, str] = {}

    def set(self, key: str, value: str):
        self._contents[key] = value

    def get(self, key: str):
        return self._contents.get(key)

    def group(self, key: str | int):
        if isinstance(key, int):
            try:
                actual = list(self._contents.keys())[key]
            except IndexError:
                return None
            return self.get(actual)
        return self.get(key)


def format_daiz_like_numbering(
    number: int | float, digit: int = 3, use_minus: bool = True, separator: str = conf.defaults.ch_special_tag
):
    if isinstance(number, int):
        return f"{int(number):0{digit}d}"
    base, floating = str(number).split(".")
    floating = int(floating)
    if floating - 4 >= 1 and use_minus:
        # Handle split chapter (.1, .2, etc)
        floating -= 4
    return f"{int(base):0{digit}d}{separator}{floating}"


class ChapterRange:
    def __init__(
        self, number: int | float, name: str | None = None, range: list[int] | None = None, is_single: bool = False
    ):
        self.number = number
        self.name = name
        self.range: list[int] = range or []
        self.is_single = is_single

    def __repr__(self):
        if isinstance(self.number, float):
            return f"<ChapterRange c{self.number} - {self.name} [p{self.page_num_str}]>"
        return f"<ChapterRange c{self.number:03d} - {self.name} [p{self.page_num_str}]>"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ChapterRange):
            return self.number == other.number
        if isinstance(other, (int, float)):
            return self.number == other
        return False

    @property
    def page_num_str(self) -> str:
        if self.is_single:
            return f"{self.range[0]:03d}-end"
        return f"{self.range[0]:03d}-{self.range[-1]:03d}"

    @property
    def bnum(self):
        return format_daiz_like_numbering(self.number)

    @property
    def base(self):
        if isinstance(self.number, int):
            return self.number
        b, _ = str(self.number).split(".")
        return int(b)

    @property
    def floating(self) -> int | None:
        if not isinstance(self.number, float):
            return None
        _, f = str(self.number).split(".")
        return int(f)


def check_cbz_exist(base_path: Path, filename: str):
    full_path = base_path / f"{filename}.cbz"
    return full_path.exists() and full_path.is_file()


def actual_or_fallback(actual_ch: str | None, chapter_num: int) -> str:
    proper_ch = f"{chapter_num:03d}"
    if not actual_ch:
        return proper_ch
    if "." in actual_ch:
        try:
            base, floating = str(float(actual_ch)).split(".", 1)
        except ValueError:
            return proper_ch
        try:
            return f"{int(base):03d}.{floating}"
        except ValueError:  # pragma: no cover (unreachable)
            return proper_ch
    try:
        return f"{int(actual_ch):03d}"
    except ValueError:
        return proper_ch


def create_chapter(match: Match[str] | PseudoChapterMatch, has_publisher: bool = False):
    chapter_num = int(cast(str, match.group("ch")))
    chapter_extra = match.group("ex")
    chapter_vol = match.group("vol")
    chapter_actual = match.group("actual")
    chapter_vol_ex = match.group("volex")
    if chapter_vol is not None:
        if utils.is_not_volume_number(chapter_vol):
            chapter_vol = 0
        else:
            if chapter_vol_ex is not None:
                chapter_vol = chapter_vol.replace(chapter_vol_ex, "")
                chapter_vol_ex = int(chapter_vol_ex[1:]) + 4
                chapter_vol = float(f"{chapter_vol[1:]}.{chapter_vol_ex}")
            else:
                chapter_vol = int(chapter_vol[1:])

    chapter_title: str | None = None
    try:
        chapter_title = match.group("title")
        if chapter_title is not None:
            chapter_title = utils.clean_title(chapter_title)
    except IndexError:  # pragma: no cover (unreachable)
        pass

    act_ch_num = actual_or_fallback(chapter_actual, chapter_num)

    if chapter_vol is not None:
        chapter_data = f"{format_daiz_like_numbering(chapter_vol, 2, False, '.')}.{act_ch_num}"
    else:
        chapter_data = act_ch_num
    if chapter_extra is not None and chapter_actual is None:
        add_num = int(chapter_extra[1:])
        if "." not in chapter_extra:
            add_num += 4
        chapter_data += f".{add_num}"
    if chapter_title is not None:
        chapter_data += f" - {chapter_title}"
    if chapter_title is None and has_publisher and chapter_extra is not None:
        ch_ex = int(chapter_extra[1:])
        chapter_data += f" - Extra {ch_ex}"

    return chapter_data


def safe_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def int_or_float(value: str) -> int | float | None:
    if "." in value:
        try:
            return float(value)
        except ValueError:
            pass
    return safe_int(value)


def parse_ch_ranges(data: str) -> tuple[list[int], bool]:
    split_range = data.split("-")
    if len(split_range) < 2:
        return [int(data)], True

    first, second = split_range
    return list(range(int(first), int(second) + 1)), False


def validate_ch_ranges(current: str):
    split_range = current.strip().split("-")
    if len(split_range) < 2:
        return safe_int(current) is not None

    first, second = split_range
    if not isinstance(second, str):  # pragma: no cover (this can't be entered anyway)
        return False

    first = safe_int(first)
    second = safe_int(second)
    if first is None or second is None:
        return False
    return True


def inquire_chapter_ranges(
    initial_prompt: str, continue_prompt: str, ask_title: bool = False
) -> list[ChapterRange]:  # pragma: no cover
    chapter_ranges: list[ChapterRange] = []
    while True:
        console.info(initial_prompt)

        ch_number = console.inquire("Chapter number", lambda y: int_or_float(y) is not None)
        ch_number = cast(int | float, int_or_float(ch_number))

        ch_ranges = console.inquire("Chapter ranges (x-y or x)", validate_ch_ranges)
        actual_ranges, is_single = parse_ch_ranges(ch_ranges)

        ch_title: str | None = None
        if ask_title:
            ch_title = console.inquire("Chapter title", lambda y: len(y.strip()) > 0)
        simple_range = ChapterRange(ch_number, ch_title, actual_ranges, is_single)
        chapter_ranges.append(simple_range)

        do_more = console.confirm(continue_prompt)
        if not do_more:
            break

    return chapter_ranges


def make_metadata_command(exiftool_path: str, image_title: str, image_email: str) -> list[str]:  # pragma: no cover
    base_cmd = [exiftool_path]
    update_tags = {
        "XPComment": image_email,
        "Artist": image_email,
        "XPAuthor": image_email,
        "XPTitle": image_title,
        "ImageDescription": image_title,
        "Title": image_title,
        "Description": image_title,
    }
    for tag, value in update_tags.items():
        base_cmd.append(f"-{tag}={value}")
    base_cmd.append("-overwrite_original_in_place")
    return base_cmd


def inject_metadata(exiftool_dir: str, current_directory: Path, image_title: str, image_email: str):  # pragma: no cover
    resolve_dir = current_directory.resolve()
    any_jpg = len(list(resolve_dir.glob("*.jpg"))) > 0
    any_tiff = len(list(resolve_dir.glob("*.tiff"))) > 0
    png_files = list(resolve_dir.glob("*.png"))
    any_png = len(png_files) > 0
    if not any_jpg and not any_tiff and not any_png:
        console.warning("No valid images found in directory, skipping metadata injection")
        return
    base_cmd = [exiftool_dir]
    update_tags = {
        "XPComment": image_email,
        "Artist": image_email,
        "XPAuthor": image_email,
        "XPTitle": image_title,
        "ImageDescription": image_title,
        "Title": image_title,
        "Description": image_title,
    }
    for tag, value in update_tags.items():
        base_cmd.append(f"-{tag}={value}")
    base_cmd.append("-overwrite_original_in_place")
    if any_jpg:
        full_dir = current_directory.resolve() / "*.jpg"
        base_cmd.append(str(full_dir))
        console.info("Injecting metadata into JP(e)G files...")
        proc = sp.Popen(base_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        proc.wait()
        base_cmd.pop()
    if any_tiff:
        full_dir = current_directory.resolve() / "*.tiff"
        base_cmd.append(str(full_dir))
        console.info("Injecting metadata into TIFF files...")
        proc = sp.Popen(base_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        proc.wait()
    if any_png:
        full_dir = current_directory.resolve() / "*.png"
        base_cmd.append(str(full_dir))
        console.info("Injecting metadata into PNG files...")
        proc = sp.Popen(base_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        proc.wait()


def run_pingo_and_verify(pingo_cmd: list[str]):  # pragma: no cover
    proc = sp.Popen(pingo_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    proc.wait()

    if proc.stdout is None or proc.stderr is None:
        raise ValueError("Subprocess stream is None, cannot proceed.")

    stdout = proc.stdout.read().decode("utf-8")
    stderr = proc.stderr.read().decode("utf-8")
    # Merge both stdout and stderr
    output = stdout + stderr

    # The pingo output is formatted like this:
    # pingo - (87.76s):
    # -----------------------------------------------------------------
    # 188 files => 12.32 MB - (5.39%) saved
    # -----------------------------------------------------------------
    # We want to extract the last line, which contains the percentage
    for line in output.splitlines():
        line = line.strip()
        if line.endswith("\n"):
            line = line[:-1]
        if line.casefold().endswith("saved"):
            return line
    return None  # unknown result


def is_pingo_alpha(pingo_path: str) -> bool:  # pragma: no cover
    console.info("Checking if pingo is alpha version...")
    proc = sp.Popen([pingo_path, "-help"], stdout=sp.PIPE, stderr=sp.PIPE)
    proc.wait()

    if proc.stdout is None or proc.stderr is None:
        raise ValueError("Subprocess stream is None, cannot proceed.")

    stdout = proc.stdout.read().decode("utf-8")
    stderr = proc.stderr.read().decode("utf-8")
    # Merge both stdout and stderr
    output = stdout + stderr

    # The output for help format is like this:
    # -----------------------------------------------------------------
    # pingo aXX (v1) - experimental web image optimizer (64-bit)
    # -----------------------------------------------------------------

    is_alpha_ver = "bad command. type 'pingo' for help" in output.casefold()
    console.info(f"Using {'alpha' if is_alpha_ver else 'stable'} version of pingo")
    return is_alpha_ver


def optimize_images(pingo_path: str, target_directory: Path, aggresive: bool = False):  # pragma: no cover
    alpha_ver = is_pingo_alpha(pingo_path)
    resolve_dir = target_directory.resolve()
    any_jpg = len(list(resolve_dir.glob("*.jpg"))) > 0
    any_png = len(list(resolve_dir.glob("*.png"))) > 0
    any_webp = len(list(resolve_dir.glob("*.webp"))) > 0

    base_cmd = [pingo_path, "-strip"] if alpha_ver else [pingo_path, "-notime", "-lossless", "-s4"]
    if any_jpg:
        pingo_cmd = base_cmd[:]
        if alpha_ver:
            pingo_cmd.append("-s0")
        if aggresive:
            if alpha_ver:
                pingo_cmd.append("-jpgtype=1")
            else:
                pingo_cmd.remove("-lossless")
                pingo_cmd.append("-q=97")
        pingo_cmd.append(str(resolve_dir / "*.jpg"))
        console.status("Optimizing JP(e)G files...")
        proc = run_pingo_and_verify(pingo_cmd)
        end_msg = "Optimized JPG files!"
        if proc is not None:
            end_msg += f" [{proc}]"
        console.stop_status(end_msg)
        console.enter()

    if any_png:
        pingo_cmd = base_cmd[:]
        if alpha_ver:
            pingo_cmd.append("-sb")
        pingo_cmd.append(str(resolve_dir / "*.png"))
        console.status("Optimizing PNG files...")
        proc = run_pingo_and_verify(pingo_cmd)
        end_msg = "Optimized PNG files!"
        if proc is not None:
            end_msg += f" [{proc}]"
        console.stop_status(end_msg)
        console.enter()

    if any_webp:
        pingo_cmd = base_cmd[:]
        if alpha_ver:
            pingo_cmd.append("-s9")
        if aggresive and not alpha_ver:
            pingo_cmd.remove("-lossless")
            pingo_cmd.append("-webp")
        pingo_cmd.append(str(resolve_dir / "*.webp"))
        console.status("Optimizing WEBP files...")
        proc = run_pingo_and_verify(pingo_cmd)
        end_msg = "Optimized WEBP files!"
        if proc is not None:
            end_msg += f" [{proc}]"
        console.stop_status(end_msg)
        console.enter()


def format_archive_filename(
    *,
    manga_title: str,
    manga_year: int,
    publication_type: MangaPublication,
    ripper_credit: str,
    bracket_type: str,
    manga_volume_text: str | None = None,
    extra_metadata: str | None = None,
    rls_revision: int | None = None,
):
    pair_left, pair_right = BRACKET_MAPPINGS.get(bracket_type.lower(), BRACKET_MAPPINGS["square"])

    if extra_metadata is not None:
        extra_metadata = "(" + extra_metadata.strip() + ") "

    act_vol = ""
    if manga_volume_text is not None:
        act_vol = f" {manga_volume_text}"
    archive_filename = TARGET_TITLE.format(
        mt=manga_title,
        vol=act_vol,
        year=manga_year,
        pt=publication_type.archive,
        c=ripper_credit,
        cpa=pair_left,
        cpb=pair_right,
        ex=extra_metadata or "",
    )

    if rls_revision is not None and rls_revision > 1:
        archive_filename += " (v%d)" % rls_revision

    return archive_filename


@overload
def format_volume_text(
    *,
    manga_volume: int,
) -> str: ...


@overload
def format_volume_text(
    *,
    manga_volume: float,
) -> str: ...


@overload
def format_volume_text(
    *,
    manga_chapter: int,
) -> str: ...


@overload
def format_volume_text(
    *,
    manga_chapter: float,
) -> str: ...


@overload
def format_volume_text(
    *,
    manga_volume: None,
    manga_chapter: None,
) -> None: ...


@overload
def format_volume_text(
    *,
    manga_volume: int | float | None,
    manga_chapter: int | float | None,
) -> str | None: ...


def format_volume_text(
    *,
    manga_volume: int | float | None = None,
    manga_chapter: int | float | None = None,
) -> str | None:
    tag_sep = conf.defaults.ch_special_tag

    volume_text: str | None = None
    if manga_chapter is not None:
        if isinstance(manga_chapter, float):
            float_string = str(manga_chapter)
            base_float, decimal_float = float_string.split(".")
            dec_float = int(decimal_float)
            if dec_float - 4 > 0:
                dec_float -= 4
            volume_text = f"{int(base_float):03d}{tag_sep}{dec_float}"
        else:
            volume_text = f"{manga_chapter:03d}"
        if conf.defaults.ch_add_c_prefix:  # pragma: no cover
            volume_text = f"c{volume_text}"  # pragma: no cover
    if manga_volume is not None:
        volume_text = f"v{format_daiz_like_numbering(manga_volume, 2, False, '.')}"

    return volume_text


def format_daiz_like_filename(
    *,
    manga_title: str,
    manga_publisher: str,
    manga_year: int,
    chapter_info: ChapterRange,
    page_number: str,
    publication_type: MangaPublication,
    ripper_credit: str,
    bracket_type: str,
    manga_volume: int | float | None = None,
    extra_metadata: str | None = None,
    image_quality: str | None = None,  # {HQ}/{LQ} thing
    rls_revision: int | None = None,
    chapter_extra_maps: dict[int, list[ChapterRange]] | None = None,
    extra_archive_metadata: str | None = None,
    fallback_volume_name: str = "OShot",
):
    pub_type = ""
    chapter_extra_maps = chapter_extra_maps or {}
    if publication_type.image:
        pub_type = f"[{publication_type.image}]"

    chapter_num = f"{chapter_info.base:03d}"

    chapter_ex_data = ""
    if chapter_extra_maps:
        pack_data = chapter_extra_maps[chapter_info.base]
        pack_data.sort(key=lambda x: x.number)
        if len(pack_data) > 1:
            smallest = pack_data[1].floating
            for pack in pack_data:
                if pack.floating is not None and smallest is not None and pack.floating < smallest:
                    smallest = pack.floating
            if smallest is not None and chapter_info.floating is not None:
                # Check if we should append the custom float data
                if smallest >= 5:
                    # We don't need to append the float data
                    float_act = chapter_info.floating - 4
                    chapter_num += f"x{float_act}"
                else:
                    idx = pack_data.index(chapter_info)
                    chapter_ex_data = f" (c{chapter_num}.{chapter_info.floating})"
                    chapter_num += f"x{idx + 1}"
        else:
            floaty = chapter_info.floating
            if floaty is not None:
                if floaty >= 5:
                    chapter_num += f"x{floaty - 4}"
                else:
                    chapter_ex_data = f" (c{chapter_num}.{floaty})"
                    chapter_num += "x1"
    else:
        if chapter_info.floating is not None:
            if chapter_info.floating >= 5:
                chapter_num += f"x{chapter_info.floating - 4}"
            else:
                chapter_ex_data = f" (c{chapter_num}.{chapter_info.floating})"
                chapter_num += "x1"

    act_vol = fallback_volume_name
    if manga_volume is not None:
        act_vol = f"v{format_daiz_like_numbering(manga_volume, 2, separator='x')}"

    extra_name = " "
    if extra_metadata is not None:
        extra_name = f" [{extra_metadata}]"

    if not pub_type and not extra_name.strip():
        extra_name = ""
    elif pub_type and extra_name.strip():
        pub_type = f" {pub_type}"

    # if extra_archive_metadata is not None:
    #     manga_title = f"{manga_title} [{extra_archive_metadata.lower()}]"

    image_filename = TARGET_FORMAT_ALT.format(
        mt=manga_title,
        ch=chapter_num,
        chex=chapter_ex_data,
        vol=act_vol,
        pg=page_number,
        ex=extra_name,
        pt=pub_type,
        pb=manga_publisher or "Unknown Publisher",
        c=ripper_credit,
    )
    if chapter_info.name is not None:
        image_filename = TARGET_FORMAT.format(
            mt=manga_title,
            ch=chapter_num,
            chex=chapter_ex_data,
            vol=act_vol,
            pg=page_number,
            ex=extra_name,
            t=chapter_info.name,
            pt=pub_type,
            pb=manga_publisher or "Unknown Publisher",
            c=ripper_credit,
        )

    if image_quality is not None:
        image_filename += " {" + image_quality + "}"

    if rls_revision is not None and rls_revision > 1:
        image_filename += " {r%d}" % rls_revision

    return image_filename, format_archive_filename(
        manga_title=manga_title,
        manga_year=manga_year,
        publication_type=publication_type,
        ripper_credit=ripper_credit,
        bracket_type=bracket_type,
        manga_volume_text=format_volume_text(manga_volume=manga_volume, manga_chapter=chapter_info.number),
        rls_revision=rls_revision,
        extra_metadata=extra_archive_metadata,
    )


_PublicationRegexMatch = [
    "dig",
    "web",
    "c2c",
    "mag",
    "scan",
    "paper",
    "raw",
    # Exist on danke-repack
    re.escape("raw-d"),
    re.escape("raw-dig"),
    re.escape("raw-digital"),
    re.escape("raw-m"),
    re.escape("raw-mag"),
    re.escape("raw-magazine"),
]


class RegexCollection:
    _VolumeRegex = r"CHANGETHIS v(\d+\.?[\d]{1,2}?).*"  # pragma: no cover
    _OneShotRegex = r"CHANGETHIS .*"  # pragma: no cover
    # fmt: off
    _ChapterTitleRe = r"CHANGETHIS - c(?P<ch>\d+)(?P<ex>[\#x.][\d]{1,2})? \(?c?(?P<actual>[\d]{1,4}[\.][\d]{1,4})?\)?" \
                      r" ?\(?(?P<vol>v[\d]+(?P<volex>[\#x][\d]{1,3})?|[A-Za-z\-]+)?\)?" \
                      r" ?- p[\d]+x?[\d]?\-?[\d]+x?[\d]?.* \[(?:PUBREPLACE)] (?:\[(?P<title>.*)\] )" \
                      r"?\[CHANGEPUBLISHER.*"  # pragma: no cover
    _ChapterBasicRe = r"CHANGETHIS - c(?P<ch>\d+)(?P<ex>[\#x.][\d]{1,2})? \(?c?(?P<actual>[\d]{1,4}[\.][\d]{1,4})?\)?" \
                      r" ?\(?(?P<vol>v[\d]+(?P<volex>[\#x][\d]{1,3})?|[A-Za-z\-]+)?\)?" \
                      r" ?- p[\d]+x?[\d]?\-?[\d]+x?[\d]?.*"  # pragma: no cover
    # fmt: on

    @classmethod
    def volume_re(cls, title: str, limit_credit: str | None = None) -> Pattern[str]:
        re_fmt = cls._VolumeRegex.replace("CHANGETHIS", re.escape(title))
        if limit_credit is not None:
            re_fmt += r"[\[\(]" + limit_credit + r".*"
        return re.compile(re_fmt)

    @overload
    @classmethod
    def chapter_re(cls, title: str) -> Pattern[str]:  # pragma: no cover
        ...

    @overload
    @classmethod
    def chapter_re(cls, title: str, publisher: str) -> Pattern[str]:  # pragma: no cover
        ...

    @overload
    @classmethod
    def chapter_re(cls, title: str, publisher: None) -> Pattern[str]:  # pragma: no cover
        ...

    @classmethod
    def chapter_re(cls, title: str, publisher: str | None = None) -> Pattern[str]:
        if publisher is None:
            return re.compile(cls._ChapterBasicRe.replace("CHANGETHIS", re.escape(title)))
        return re.compile(
            cls._ChapterTitleRe.replace("CHANGETHIS", re.escape(title))
            .replace("CHANGEPUBLISHER", re.escape(publisher))
            .replace("PUBREPLACE", "|".join(_PublicationRegexMatch))
        )

    @classmethod
    def cmx_re(cls) -> Pattern[str]:
        return re.compile(
            r"(?P<t>[\w\W\D\d\S\s]+?)(?:\- (?P<vol>v[\d]{1,3}))?(?P<volex>\.[\d]{1,2})? \- "
            r"p(?P<a>[\d]{1,3})\-?(?P<b>[\d]{1,3})?"
        )

    @classmethod
    def page_re(cls) -> Pattern[str]:
        return re.compile(r"(?P<any>.*)p(?P<a>[\d]{1,3})\-?(?P<b>[\d]{1,3})?(?P<anyback>.*)")

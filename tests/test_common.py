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

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from nmanga.common import (
    ChapterRange,
    PseudoChapterMatch,
    actual_or_fallback,
    check_cbz_exist,
    create_chapter,
    format_archive_filename,
    format_daiz_like_filename,
    format_daiz_like_numbering,
    format_volume_text,
    int_or_float,
    parse_ch_ranges,
    safe_int,
    validate_ch_ranges,
)
from nmanga.config import get_config
from nmanga.constants import MANGA_PUBLICATION_TYPES


class TestFormatDaizLikeFilename:
    _TITLE = "Test Title"
    _PUBLISHER = "Real Publisher"
    _YEAR = 2023
    _CURRENT = ChapterRange(1, "Introduction", [0, 41], True)
    _CURRENT_NO_TITLE = ChapterRange(1, None, [0, 41], True)
    _CURRENT_EXTRA = ChapterRange(1.5, "Extra Story", [0, 51], False)
    _CURRENT_EXTRA_NO_TITLE = ChapterRange(1.5, None, [0, 51], False)
    _CURRENT_SPLIT_A = ChapterRange(2.1, "Split A", [52, 69], False)
    _CURRENT_SPLIT_B = ChapterRange(2.2, "Split B", [70, 110], False)
    _CURRENT_SPLIT_EXTRA = ChapterRange(2.5, "Split C", [111, 114], False)
    _CURRENT_SPLIT_EXTRA_TWO = ChapterRange(2.6, "Split D", [115], True)

    _BRACKET = "square"
    _RIPPER = "nao"
    _PUB_TYPE = MANGA_PUBLICATION_TYPES["digital"]
    _PUB_TYPE_NONE = MANGA_PUBLICATION_TYPES["none"]

    _REAL_WORLD_MAPPING_CASE_A: ClassVar[Dict[str, Any]] = {
        "title": "Captain Corinth",
        "publisher": "One Peace Books",
        "ripper": "nao",
        "volume": 2,
        "chapters": [
            ChapterRange(7, "The Village of Talas", [0, 26]),
            ChapterRange(8, "Escort, Part 1", [27, 50]),
            ChapterRange(9, "Escort, Part 2", [51, 76]),
            ChapterRange(10, "Gotania", [77, 100]),
            ChapterRange(11, "The Adventurers Guild", [101, 132]),
            ChapterRange(12, "The Fall of Starvake", [133, 156]),
            ChapterRange(13, "Reasons", [157, 182]),
            ChapterRange(13.5, "Cleria's Conversation With the Spirits", [183], True),
        ],
        "expects": [
            "Captain Corinth - c007 (v02) - p000 [Cover] [dig] [The Village of Talas] [One Peace Books] [nao]",
            "Captain Corinth - c008 (v02) - p027 [dig] [Escort, Part 1] [One Peace Books] [nao]",
            "Captain Corinth - c009 (v02) - p051 [dig] [Escort, Part 2] [One Peace Books] [nao]",
            "Captain Corinth - c010 (v02) - p077 [dig] [Gotania] [One Peace Books] [nao]",
            "Captain Corinth - c011 (v02) - p101 [dig] [The Adventurers Guild] [One Peace Books] [nao]",
            "Captain Corinth - c012 (v02) - p133 [dig] [The Fall of Starvake] [One Peace Books] [nao]",
            "Captain Corinth - c013 (v02) - p157 [dig] [Reasons] [One Peace Books] [nao]",
            "Captain Corinth - c013x1 (v02) - p183 [dig] [Cleria's Conversation With the Spirits] [One Peace Books] [nao]",  # noqa
        ],
        "extra": {0: "Cover"},
    }

    def _make_packing_extra(self):
        packing_extra: Dict[int, List[ChapterRange]] = {}
        for key in self.__dir__():
            if not key.startswith("_CURRENT_SPLIT"):
                continue
            value = getattr(self, key)
            if isinstance(value, ChapterRange):
                if value.base not in packing_extra:
                    packing_extra[value.base] = []
                packing_extra[value.base].append(value)

        return packing_extra

    def test_normal_chapter(self):
        sel = self._CURRENT
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
        )

        assert filename == f"Test Title - c001 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_normal_chapter_with_volume(self):
        sel = self._CURRENT
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            manga_volume=1,
        )

        assert filename == f"Test Title - c001 (v01) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_normal_chapter_with_quality_mark(self):
        sel = self._CURRENT
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            image_quality="HQ",
        )

        expect = f"Test Title - c001 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        expect += " {HQ}"
        assert filename == expect

    def test_normal_chapter_with_revision(self):
        sel = self._CURRENT
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            rls_revision=2,
        )

        expect = f"Test Title - c001 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        expect += " {r2}"
        assert filename == expect

    def test_normal_chapter_with_quality_mark_and_revision(self):
        sel = self._CURRENT
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            image_quality="HQ",
            rls_revision=2,
        )

        expect = f"Test Title - c001 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        expect += " {HQ} {r2}"
        assert filename == expect

    def test_normal_chapter_no_title(self):
        sel = self._CURRENT_NO_TITLE
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
        )

        assert filename == f"Test Title - c001 (OShot) - p000 [dig] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_normal_chapter_pub_type_none(self):
        sel = self._CURRENT_NO_TITLE
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE_NONE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
        )

        assert filename == f"Test Title - c001 (OShot) - p000 [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_normal_chapter_with_extra(self):
        sel = self._CURRENT
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            extra_metadata="Cover",
        )

        assert (
            filename
            == f"Test Title - c001 (OShot) - p000 [Cover] [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        )

    def test_normal_chapter_no_title_with_extra(self):
        sel = self._CURRENT_NO_TITLE
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            extra_metadata="Cover",
        )

        assert filename == f"Test Title - c001 (OShot) - p000 [Cover] [dig] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_normal_chapter_pub_type_none_with_extra(self):
        sel = self._CURRENT_NO_TITLE
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE_NONE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            extra_metadata="Cover",
        )

        assert filename == f"Test Title - c001 (OShot) - p000 [Cover] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_extra_chapter(self):
        sel = self._CURRENT_EXTRA
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
        )

        assert filename == f"Test Title - c001x1 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_extra_chapter_less_than_5(self):
        sel = self._CURRENT_SPLIT_A
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
        )

        assert (
            filename
            == f"Test Title - c002x1 (c002.1) (OShot) - p052 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        )

    def test_extra_chapter_with_volume(self):
        sel = self._CURRENT_EXTRA
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            manga_volume=1,
        )

        assert filename == f"Test Title - c001x1 (v01) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_extra_chapter_with_quality_mark(self):
        sel = self._CURRENT_EXTRA
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            image_quality="HQ",
        )

        expect = f"Test Title - c001x1 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        expect += " {HQ}"
        assert filename == expect

    def test_extra_chapter_with_revision(self):
        sel = self._CURRENT_EXTRA
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            rls_revision=2,
        )

        expect = f"Test Title - c001x1 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        expect += " {r2}"
        assert filename == expect

    def test_extra_chapter_with_quality_mark_and_revision(self):
        sel = self._CURRENT_EXTRA
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            image_quality="HQ",
            rls_revision=2,
        )

        expect = f"Test Title - c001x1 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        expect += " {HQ} {r2}"
        assert filename == expect

    def test_extra_chapter_no_title(self):
        sel = self._CURRENT_EXTRA_NO_TITLE
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
        )

        assert filename == f"Test Title - c001x1 (OShot) - p000 [dig] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_extra_chapter_pub_type_none(self):
        sel = self._CURRENT_EXTRA_NO_TITLE
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE_NONE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
        )

        assert filename == f"Test Title - c001x1 (OShot) - p000 [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_extra_chapter_with_extra(self):
        sel = self._CURRENT_EXTRA
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            extra_metadata="Cover",
        )

        assert (
            filename
            == f"Test Title - c001x1 (OShot) - p000 [Cover] [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        )

    def test_extra_chapter_no_title_with_extra(self):
        sel = self._CURRENT_EXTRA_NO_TITLE
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            extra_metadata="Cover",
        )

        assert filename == f"Test Title - c001x1 (OShot) - p000 [Cover] [dig] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_extra_chapter_pub_type_none_with_extra(self):
        sel = self._CURRENT_EXTRA_NO_TITLE
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE_NONE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            extra_metadata="Cover",
        )

        assert filename == f"Test Title - c001x1 (OShot) - p000 [Cover] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_split_chapter_part_a(self):
        sel = self._CURRENT_SPLIT_A
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            chapter_extra_maps=self._make_packing_extra(),
        )

        assert filename == "Test Title - c002x1 (c002.1) (OShot) - p052 [dig] [Split A] [Real Publisher] [nao]"

    def test_split_chapter_part_b(self):
        sel = self._CURRENT_SPLIT_B
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            chapter_extra_maps=self._make_packing_extra(),
        )

        assert filename == "Test Title - c002x2 (c002.2) (OShot) - p070 [dig] [Split B] [Real Publisher] [nao]"

    def test_split_chapter_no_extra_with_maps(self):
        packing_extra = {
            2: [self._CURRENT_SPLIT_A],
        }
        sel = self._CURRENT_SPLIT_A
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            chapter_extra_maps=packing_extra,
        )

        assert filename == "Test Title - c002x1 (c002.1) (OShot) - p052 [dig] [Split A] [Real Publisher] [nao]"

    def test_split_chapter_extra_with_maps(self):
        packing_extra = {
            2: [self._CURRENT_SPLIT_EXTRA],
        }
        sel = self._CURRENT_SPLIT_EXTRA
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            chapter_extra_maps=packing_extra,
        )

        assert filename == "Test Title - c002x1 (OShot) - p111 [dig] [Split C] [Real Publisher] [nao]"

    def test_split_chapter_extra_with_maps_multi(self):
        packing_extra = {
            2: [self._CURRENT_SPLIT_EXTRA, self._CURRENT_SPLIT_EXTRA_TWO],
        }
        sel = self._CURRENT_SPLIT_EXTRA_TWO
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            chapter_extra_maps=packing_extra,
        )

        assert filename == "Test Title - c002x2 (OShot) - p115 [dig] [Split D] [Real Publisher] [nao]"

    def test_chapter_with_extra_archive_metadata(self):
        sel = self._CURRENT
        filename, _ = format_daiz_like_filename(
            manga_title=self._TITLE,
            manga_publisher=self._PUBLISHER,
            manga_year=self._YEAR,
            chapter_info=sel,
            page_number=f"{sel.range[0]:03d}",
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            extra_metadata="Cover",
            extra_archive_metadata="JPN",
        )

        assert (
            filename
            == f"Test Title - c001 (OShot) - p000 [Cover] [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        )

    def test_real_world_use_case_a(self):
        title: str = self._REAL_WORLD_MAPPING_CASE_A["title"]
        publisher: str = self._REAL_WORLD_MAPPING_CASE_A["publisher"]
        ripper: str = self._REAL_WORLD_MAPPING_CASE_A["ripper"]
        volume: int = self._REAL_WORLD_MAPPING_CASE_A["volume"]
        chapters: List[ChapterRange] = self._REAL_WORLD_MAPPING_CASE_A["chapters"]
        expects: List[str] = self._REAL_WORLD_MAPPING_CASE_A["expects"]
        extra: Dict[int, str] = self._REAL_WORLD_MAPPING_CASE_A["extra"]

        packing_extra = {}
        for chapter in chapters:
            if chapter.base not in packing_extra:
                packing_extra[chapter.base] = []
            packing_extra[chapter.base].append(chapter)

        for chapter, expect in zip(chapters, expects):
            pg_num = chapter.range[0]
            pg_extra = extra.get(pg_num, None)
            filename, _ = format_daiz_like_filename(
                manga_title=title,
                manga_publisher=publisher,
                manga_year=self._YEAR,
                chapter_info=chapter,
                page_number=f"{pg_num:03d}",
                bracket_type=self._BRACKET,
                manga_volume=volume,
                extra_metadata=pg_extra,
                publication_type=self._PUB_TYPE,
                ripper_credit=ripper,
                image_quality=None,
                rls_revision=1,
                chapter_extra_maps=packing_extra,
            )

            assert filename == expect


class TestFormatArchiveFilename:
    _TITLE = "Test Title"
    _YEAR = 2023
    _PUB_TYPE = MANGA_PUBLICATION_TYPES["digital"]
    _PUB_TYPE_SCAN = MANGA_PUBLICATION_TYPES["scan"]
    _RIPPER = "nao"
    _BRACKET = "square"

    def test_no_volume_text(self):
        filename = format_archive_filename(
            manga_title=self._TITLE,
            manga_year=self._YEAR,
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
        )

        assert filename == f"Test Title ({self._YEAR}) (Digital) [{self._RIPPER}]"

    def test_no_volume_text_with_scan(self):
        filename = format_archive_filename(
            manga_title=self._TITLE,
            manga_year=self._YEAR,
            publication_type=self._PUB_TYPE_SCAN,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
        )

        assert filename == f"Test Title ({self._YEAR}) (c2c) [{self._RIPPER}]"

    def test_with_volume_text(self):
        filename = format_archive_filename(
            manga_title=self._TITLE,
            manga_year=self._YEAR,
            publication_type=self._PUB_TYPE,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            manga_volume_text="v01",
        )

        assert filename == f"Test Title v01 ({self._YEAR}) (Digital) [{self._RIPPER}]"

    def test_with_volume_text_with_scan(self):
        filename = format_archive_filename(
            manga_title=self._TITLE,
            manga_year=self._YEAR,
            publication_type=self._PUB_TYPE_SCAN,
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            manga_volume_text="v01",
        )

        assert filename == f"Test Title v01 ({self._YEAR}) (c2c) [{self._RIPPER}]"

    def test_with_volume_text_and_extra(self) -> None:
        filename = format_archive_filename(
            manga_title=self._TITLE,
            manga_year=self._YEAR,
            publication_type=MANGA_PUBLICATION_TYPES["digital-raw"],
            ripper_credit=self._RIPPER,
            bracket_type=self._BRACKET,
            manga_volume_text="v01",
            extra_metadata="JPN",
        )

        assert filename == f"Test Title v01 ({self._YEAR}) (JPN) (Digital) [{self._RIPPER}]"


class TestFormatDaizLikeNumbering:
    def test_int(self):
        assert format_daiz_like_numbering(1, separator="x") == "001"

    def test_int_volume(self):
        assert format_daiz_like_numbering(1, digit=2, separator="x") == "01"

    def test_float(self):
        assert format_daiz_like_numbering(1.5, separator="x") == "001x1"

    def test_float_no_minus(self):
        assert format_daiz_like_numbering(1.5, digit=2, use_minus=False, separator=".") == "01.5"

    def test_float_split(self):
        assert format_daiz_like_numbering(1.3, digit=2, separator="x") == "01x3"


class TestFormatVolumeText:
    # Make a with statement to mock the config
    conf = get_config()

    def test_manga_volume_int(self):
        assert format_volume_text(manga_volume=1) == "v01"

    def test_manga_volume_float(self):
        assert format_volume_text(manga_volume=1.5) == "v01.5"

    def test_manga_chapter_int(self):
        pre = "c" if self.conf.defaults.ch_add_c_prefix else ""
        assert format_volume_text(manga_chapter=1) == f"{pre}001"

    def test_manga_chapter_float(self):
        pre = "c" if self.conf.defaults.ch_add_c_prefix else ""
        sep = self.conf.defaults.ch_special_tag
        assert format_volume_text(manga_chapter=1.5) == f"{pre}001{sep}1"


class _MissingT:
    __slots__ = ()

    def __eq__(self, other) -> bool:
        return False

    def __bool__(self) -> bool:
        return False

    def __hash__(self) -> int:
        return 0

    def __repr__(self):
        return "..."


MISSING = _MissingT()


class TestCreateChapter:
    def _mock_match(
        self,
        ch_num: str,
        ch_title: Optional[str] = MISSING,
        ch_actual: Optional[str] = MISSING,
        ch_extra: Optional[str] = MISSING,
        vol_num: Optional[str] = MISSING,
        vol_extra: Optional[str] = MISSING,
    ):
        match = PseudoChapterMatch()
        match.set("ch", str(ch_num))
        if ch_title is not MISSING:
            match.set("title", ch_title)
        if ch_actual is not MISSING:
            match.set("actual", ch_actual)
        if ch_extra is not MISSING:
            match.set("ex", ch_extra)
        if vol_num is not MISSING:
            match.set("vol", vol_num)
        if vol_extra is not MISSING:
            match.set("volex", vol_extra)
        return match

    def test_basic_match(self):
        assert create_chapter(self._mock_match("1")) == "001"

    def test_with_volume(self):
        assert create_chapter(self._mock_match("1", vol_num="v01")) == "01.001"

    def test_with_volume_oneshot(self):
        assert create_chapter(self._mock_match("1", vol_num="OShot")) == "00.001"

    def test_with_volume_extra(self):
        assert create_chapter(self._mock_match("1", vol_num="v01", vol_extra="x1")) == "01.5.001"

    def test_with_chapter_extra(self):
        assert create_chapter(self._mock_match("1", ch_extra="x1")) == "001.5"
        assert create_chapter(self._mock_match("1", ch_extra=".1")) == "001.1"

    def test_with_chapter_extra_and_volume(self):
        assert create_chapter(self._mock_match("1", ch_extra="x1", vol_num="v01")) == "01.001.5"

    def test_with_chapter_extra_and_volume_extra(self):
        assert create_chapter(self._mock_match("1", ch_extra="x1", vol_num="v01", vol_extra="x1")) == "01.5.001.5"

    def test_with_chapter_actual(self):
        assert create_chapter(self._mock_match("1", ch_actual="1.1")) == "001.1"

    def test_with_chapter_actual_and_volume(self):
        assert create_chapter(self._mock_match("1", ch_actual="1.1", vol_num="v01")) == "01.001.1"

    def test_with_chapter_actual_and_volume_extra(self):
        assert create_chapter(self._mock_match("1", ch_actual="1.1", vol_num="v01", vol_extra="x1")) == "01.5.001.1"

    def test_with_chapter_title(self):
        assert create_chapter(self._mock_match("1", ch_title="Test")) == "001 - Test"

    def test_with_chapter_pub_no_title(self):
        assert create_chapter(self._mock_match("1", ch_extra="x1"), True) == "001.5 - Extra 1"
        assert create_chapter(self._mock_match("1", ch_extra=".1"), True) == "001.1 - Extra 1"


class TestValidateChapterRange:
    def test_multi_length(self):
        assert validate_ch_ranges("1-3")

    def test_single_length(self):
        assert validate_ch_ranges("1")

    def test_single_length_float(self):
        assert not validate_ch_ranges("1.5")

    def test_multi_length_float(self):
        assert not validate_ch_ranges("1.5-3.5")


class TestRangeParsing:
    def test_single_length(self):
        r_num, is_single = parse_ch_ranges("1")
        assert r_num == [1] and is_single

    def test_multi_length(self):
        r_num, is_single = parse_ch_ranges("1-3")
        assert r_num == [1, 2, 3] and not is_single


class TestNumberConverter:
    def test_safe_int(self):
        assert safe_int("1") == 1

    def test_safe_int_null(self):
        assert safe_int("1.2") is None

    def test_int_or_float(self):
        assert int_or_float("1.2") == 1.2

    def test_int_or_float_int(self):
        assert int_or_float("1") == 1

    def test_int_or_float_null(self):
        assert int_or_float("a") is None

    def test_int_or_float_fail(self):
        assert int_or_float("1.2.3") is None


class TestPseudoChapterMatch:
    def test_set_and_get(self):
        match = PseudoChapterMatch()
        match.set("title", "test")
        assert match.get("title") == "test"

    def test_get_none(self):
        match = PseudoChapterMatch()
        assert match.get("title") is None

    def test_get_group(self):
        match = PseudoChapterMatch()
        match.set("title", "test")
        match.set("group", "test group")

        assert match.group(0) == "test"
        assert match.group(1) == "test group"
        assert match.group(2) is None
        assert match.group("title") == "test"
        assert match.group("group") == "test group"
        assert match.group("notexist") is None


class TestChapterRange:
    _CHAPTER = TestFormatDaizLikeFilename._CURRENT
    _CHAPTER_EXTRA = TestFormatDaizLikeFilename._CURRENT_EXTRA
    _CHAPTER_SPLIT = TestFormatDaizLikeFilename._CURRENT_SPLIT_A

    def test_repr(self):
        assert repr(self._CHAPTER) == "<ChapterRange c001 - Introduction>"

    def test_repr_extra(self):
        assert repr(self._CHAPTER_EXTRA) == "<ChapterRange c1.5 - Extra Story>"

    def test_eq(self):
        _CHAPTER_ONE = ChapterRange(1, "Unknown", [0, 41], True)
        assert self._CHAPTER != self._CHAPTER_EXTRA
        assert self._CHAPTER == _CHAPTER_ONE

        assert self._CHAPTER == 1

    def test_bnum_attr(self):
        assert self._CHAPTER.bnum == "001"

    def test_bnum_extra_attr(self):
        assert self._CHAPTER_EXTRA.bnum == "001x1"
        assert self._CHAPTER_SPLIT.bnum == "002x1"


def test_cbz_check():
    assert not check_cbz_exist(Path.cwd(), "test")


class TestActualOrFallback:
    def test_valid_chapter(self):
        assert actual_or_fallback("2", 1) == "002"

    def test_invalid_chapter(self):
        assert actual_or_fallback(None, 1) == "001"
        assert actual_or_fallback("xxxx", 2) == "002"

    def test_valid_chapter_float(self):
        assert actual_or_fallback("2.5", 1) == "002.5"

    def test_invalid_chapter_float(self):
        assert actual_or_fallback("2.x", 1) == "001"

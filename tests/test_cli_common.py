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

from typing import Dict, List
from nmanga.cli.common import ChapterRange, format_daiz_like_filename
from nmanga.cli.constants import MANGA_PUBLICATION_TYPES


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

    _REAL_WORLD_MAPPING_CASE_A = {
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
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
        )

        assert filename == f"Test Title - c001 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_normal_chapter_with_volume(self):
        sel = self._CURRENT
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            1,
        )

        assert filename == f"Test Title - c001 (v01) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_normal_chapter_with_quality_mark(self):
        sel = self._CURRENT
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            None,
            None,
            "HQ",
        )

        expect = f"Test Title - c001 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        expect += " {HQ}"
        assert filename == expect

    def test_normal_chapter_with_revision(self):
        sel = self._CURRENT
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            None,
            None,
            None,
            2,
        )

        expect = f"Test Title - c001 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        expect += " {r2}"
        assert filename == expect

    def test_normal_chapter_with_quality_mark_and_revision(self):
        sel = self._CURRENT
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            None,
            None,
            "HQ",
            2,
        )

        expect = f"Test Title - c001 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        expect += " {HQ} {r2}"
        assert filename == expect

    def test_normal_chapter_no_title(self):
        sel = self._CURRENT_NO_TITLE
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
        )

        assert filename == f"Test Title - c001 (OShot) - p000 [dig] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_normal_chapter_pub_type_none(self):
        sel = self._CURRENT_NO_TITLE
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE_NONE,
            self._RIPPER,
            self._BRACKET,
        )

        assert filename == f"Test Title - c001 (OShot) - p000 [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_normal_chapter_with_extra(self):
        sel = self._CURRENT
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            None,
            "Cover",
        )

        assert (
            filename
            == f"Test Title - c001 (OShot) - p000 [Cover] [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        )

    def test_normal_chapter_no_title_with_extra(self):
        sel = self._CURRENT_NO_TITLE
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            None,
            "Cover",
        )

        assert filename == f"Test Title - c001 (OShot) - p000 [Cover] [dig] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_normal_chapter_pub_type_none_with_extra(self):
        sel = self._CURRENT_NO_TITLE
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE_NONE,
            self._RIPPER,
            self._BRACKET,
            None,
            "Cover",
        )

        assert filename == f"Test Title - c001 (OShot) - p000 [Cover] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_extra_chapter(self):
        sel = self._CURRENT_EXTRA
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
        )

        assert filename == f"Test Title - c001x1 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_extra_chapter_less_than_5(self):
        sel = self._CURRENT_SPLIT_A
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
        )

        assert (
            filename
            == f"Test Title - c002x1 (c002.1) (OShot) - p052 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        )

    def test_extra_chapter_with_volume(self):
        sel = self._CURRENT_EXTRA
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            1,
        )

        assert filename == f"Test Title - c001x1 (v01) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_extra_chapter_with_quality_mark(self):
        sel = self._CURRENT_EXTRA
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            None,
            None,
            "HQ",
        )

        expect = f"Test Title - c001x1 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        expect += " {HQ}"
        assert filename == expect

    def test_extra_chapter_with_revision(self):
        sel = self._CURRENT_EXTRA
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            None,
            None,
            None,
            2,
        )

        expect = f"Test Title - c001x1 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        expect += " {r2}"
        assert filename == expect

    def test_extra_chapter_with_quality_mark_and_revision(self):
        sel = self._CURRENT_EXTRA
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            None,
            None,
            "HQ",
            2,
        )

        expect = f"Test Title - c001x1 (OShot) - p000 [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        expect += " {HQ} {r2}"
        assert filename == expect

    def test_extra_chapter_no_title(self):
        sel = self._CURRENT_EXTRA_NO_TITLE
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
        )

        assert filename == f"Test Title - c001x1 (OShot) - p000 [dig] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_extra_chapter_pub_type_none(self):
        sel = self._CURRENT_EXTRA_NO_TITLE
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE_NONE,
            self._RIPPER,
            self._BRACKET,
        )

        assert filename == f"Test Title - c001x1 (OShot) - p000 [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_extra_chapter_with_extra(self):
        sel = self._CURRENT_EXTRA
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            None,
            "Cover",
        )

        assert (
            filename
            == f"Test Title - c001x1 (OShot) - p000 [Cover] [dig] [{sel.name}] [{self._PUBLISHER}] [{self._RIPPER}]"
        )

    def test_extra_chapter_no_title_with_extra(self):
        sel = self._CURRENT_EXTRA_NO_TITLE
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            None,
            "Cover",
        )

        assert filename == f"Test Title - c001x1 (OShot) - p000 [Cover] [dig] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_extra_chapter_pub_type_none_with_extra(self):
        sel = self._CURRENT_EXTRA_NO_TITLE
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE_NONE,
            self._RIPPER,
            self._BRACKET,
            None,
            "Cover",
        )

        assert filename == f"Test Title - c001x1 (OShot) - p000 [Cover] [{self._PUBLISHER}] [{self._RIPPER}]"

    def test_split_chapter_part_a(self):
        sel = self._CURRENT_SPLIT_A
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            chapter_extra_maps=self._make_packing_extra(),
        )

        assert filename == "Test Title - c002x1 (c002.1) (OShot) - p052 [dig] [Split A] [Real Publisher] [nao]"

    def test_split_chapter_part_b(self):
        sel = self._CURRENT_SPLIT_B
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            chapter_extra_maps=self._make_packing_extra(),
        )

        assert filename == "Test Title - c002x2 (c002.2) (OShot) - p070 [dig] [Split B] [Real Publisher] [nao]"

    def test_split_chapter_no_extra_with_maps(self):
        packing_extra = {
            2: [self._CURRENT_SPLIT_A],
        }
        sel = self._CURRENT_SPLIT_A
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            chapter_extra_maps=packing_extra,
        )

        assert filename == "Test Title - c002x1 (c002.1) (OShot) - p052 [dig] [Split A] [Real Publisher] [nao]"

    def test_split_chapter_extra_with_maps(self):
        packing_extra = {
            2: [self._CURRENT_SPLIT_EXTRA],
        }
        sel = self._CURRENT_SPLIT_EXTRA
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            chapter_extra_maps=packing_extra,
        )

        assert filename == "Test Title - c002x1 (OShot) - p111 [dig] [Split C] [Real Publisher] [nao]"

    def test_split_chapter_extra_with_maps_multi(self):
        packing_extra = {
            2: [self._CURRENT_SPLIT_EXTRA, self._CURRENT_SPLIT_EXTRA_TWO],
        }
        sel = self._CURRENT_SPLIT_EXTRA_TWO
        filename, _ = format_daiz_like_filename(
            self._TITLE,
            self._PUBLISHER,
            self._YEAR,
            sel,
            f"{sel.range[0]:03d}",
            self._PUB_TYPE,
            self._RIPPER,
            self._BRACKET,
            chapter_extra_maps=packing_extra,
        )

        assert filename == "Test Title - c002x2 (OShot) - p115 [dig] [Split D] [Real Publisher] [nao]"

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

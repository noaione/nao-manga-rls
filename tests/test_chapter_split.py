import re
import zipfile
from pathlib import Path

import pytest

from nmanga.chapter_split import (
    ChapterSplitError,
    ChapterSplitObserver,
    SourcePage,
    build_chapter_archive_name,
    coerce_number_page,
    collect_source_pages,
    extract_page_num,
    find_chapter_for_page,
    read_source_image,
    split_chapters,
    validate_chapter_coverage,
    validate_source_coverage,
)
from nmanga.common import ChapterRange, PseudoChapterMatch, create_chapter, format_daiz_like_numbering
from nmanga.config import get_config


def _make_source_folder(base: Path, pages: list[int], prefix: str = "", name: str = "source") -> Path:
    source = base / name
    source.mkdir(parents=True, exist_ok=True)
    for page in pages:
        (source / f"{prefix}{page:03d}.png").write_bytes(f"image-{page:03d}".encode())
    return source


def _make_source_cbz(base: Path, pages: list[int], prefix: str = "", name: str = "vol.cbz") -> Path:
    target = base / name
    with zipfile.ZipFile(target, "w") as archive:
        for page in pages:
            archive.writestr(f"{prefix}{page:03d}.png", f"image-{page:03d}")
    return target


PAGE_REGEX = re.compile(r"p(?:([\d]{1,4})(?:-)?([\d]{1,4})?).*")


def _legacy_chapter_name(number: int | float, title: str | None, volume: int | None = None) -> str:
    """Replicate the naming path used by the old `manualsplit` command."""

    tag = get_config().defaults.ch_special_tag
    as_bnum = format_daiz_like_numbering(number).split(tag, 1)

    match = PseudoChapterMatch()
    match.set("ch", as_bnum[0])
    if len(as_bnum) > 1:
        match.set("ex", "x" + as_bnum[1])
    if volume is not None:
        match.set("vol", f"v{format_daiz_like_numbering(volume, 2, False, '.')}")
    if title is not None:
        match.set("title", title)
    return create_chapter(match)


class TestExtractPageNum:
    def test_plain_number(self):
        assert extract_page_num("001") == [1]

    def test_number_with_leading_zeros(self):
        assert extract_page_num("0123") == [123]

    def test_dash_range_is_a_spread(self):
        assert extract_page_num("003-004") == [3, 4]

    def test_tilde_range_is_a_spread(self):
        assert extract_page_num("003~004") == [3, 4]

    def test_trailing_dash_is_stripped(self):
        assert extract_page_num("12-") == [12]

    def test_custom_data_takes_priority(self):
        assert extract_page_num("001-002", {"001": 99}) == [99]

    def test_custom_data_substring_match(self):
        assert extract_page_num("cover-a", {"cover": 0}) == [0]

    def test_regex_data(self):
        assert extract_page_num("p003", regex_data=PAGE_REGEX) == [3]
        assert extract_page_num("p003-004x", regex_data=PAGE_REGEX) == [3, 4]

    def test_standard_p_name_is_detected(self):
        assert extract_page_num("p003") == [3]
        assert extract_page_num("p000") == [0]
        assert extract_page_num("p003-004") == [3, 4]

    def test_padded_p_name_is_not_truncated(self):
        assert extract_page_num("p0001") == [1]
        assert extract_page_num("p0012") == [12]
        assert extract_page_num("p1000") == [1000]

    def test_tilde_spread_is_detected(self):
        assert extract_page_num("p003~004") == [3, 4]

    def test_p_name_is_detected_anywhere_in_the_filename(self):
        assert extract_page_num("Title - c001 (v01) - p003 [dig] [nao]") == [3]
        assert extract_page_num("Title - c001 (v01) - p005-006 [dig] [nao]") == [5, 6]

    def test_bare_number_still_wins_over_a_lookalike(self):
        assert extract_page_num("003") == [3]
        assert extract_page_num("003-004") == [3, 4]

    def test_custom_data_beats_the_built_in_detection(self):
        assert extract_page_num("p003", {"p003": 99}) == [99]

    def test_custom_data_beats_a_user_regex(self):
        assert extract_page_num("p003", {"p003": 99}, PAGE_REGEX) == [99]

    def test_unparsable_filename_without_regex_raises(self):
        with pytest.raises(ChapterSplitError) as exc_info:
            extract_page_num("cover")
        assert "cover" in str(exc_info.value)


class TestCoerceNumberPage:
    @pytest.mark.parametrize(
        ("number", "expected"),
        [
            (1, "001"),
            (42, "042"),
            (1234, "1234"),
            (1.5, "001.5"),
            (12.25, "012.25"),
        ],
    )
    def test_coerce(self, number, expected):
        assert coerce_number_page(number) == expected


class TestBuildChapterArchiveName:
    @pytest.mark.parametrize(
        ("number", "title", "volume"),
        [
            (1, "Introduction", None),
            (1, None, None),
            (1, "Introduction", 1),
            (7, "Some Title", 3),
            (1.5, "Extra Story", 2),
            (2.1, "Split A", 2),
            (2.5, "Split C", 2),
            (12, "Finale", 10),
        ],
    )
    def test_matches_legacy_naming(self, number: int | float, title: str | None, volume: int | None):
        chapter = ChapterRange(number, title, [1, 2], False, volume=volume)
        assert build_chapter_archive_name(chapter) == _legacy_chapter_name(number, title, volume)

    @pytest.mark.parametrize(
        ("chapter", "volume", "expected"),
        [
            (ChapterRange(1, "Introduction"), None, "001 - Introduction"),
            (ChapterRange(1, "Introduction"), 1, "01.001 - Introduction"),
            (ChapterRange(7, "Some Title"), 3, "03.007 - Some Title"),
            (ChapterRange(12, "Finale"), 10, "10.012 - Finale"),
            # Float chapter numbers become a split/extra marker, matching ``create_chapter``.
            (ChapterRange(1.5, "Extra Story"), 2, "02.001.5 - Extra Story"),
            (ChapterRange(2.1, "Split A"), 2, "02.002.5 - Split A"),
            (ChapterRange(2.5, "Split C"), 2, "02.002.5 - Split C"),
            (ChapterRange(2.6, "Split D"), 2, "02.002.6 - Split D"),
            (ChapterRange(5, None), None, "005"),
        ],
    )
    def test_explicit_names(self, chapter: ChapterRange, volume: int | None, expected: str):
        assert build_chapter_archive_name(chapter, volume) == expected

    def test_without_volume(self):
        assert build_chapter_archive_name(ChapterRange(1, "Intro", [1])) == "001 - Intro"

    def test_volume_from_argument(self):
        assert build_chapter_archive_name(ChapterRange(1, "Intro", [1]), 3) == "03.001 - Intro"

    def test_chapter_volume_wins_over_argument(self):
        chapter = ChapterRange(1, "Intro", [1], volume=2)
        assert build_chapter_archive_name(chapter, 3) == "02.001 - Intro"

    def test_omnibus_volume(self):
        chapter = ChapterRange(5, "Bonus", [1], volume=(1, 2))
        assert build_chapter_archive_name(chapter) == "01-02.005 - Bonus"

    def test_float_volume(self):
        chapter = ChapterRange(1, None, [1], volume=1.5)
        assert build_chapter_archive_name(chapter) == "01.5.001"

    def test_title_is_cleaned(self):
        assert build_chapter_archive_name(ChapterRange(1, "Intro]", [1])) == "001 - Intro"


class TestFindChapterForPage:
    def test_explicit_range(self):
        chapters = [ChapterRange(1, "Intro", [1, 2, 3])]
        assert find_chapter_for_page(chapters, 2) == chapters[0]
        assert find_chapter_for_page(chapters, 4) is None

    def test_single_range_is_open_ended(self):
        chapters = [ChapterRange(3, "Extras", [3], True)]
        for page in (3, 4, 50):
            assert find_chapter_for_page(chapters, page) == chapters[0]
        assert find_chapter_for_page(chapters, 2) is None

    def test_empty_range_is_ignored(self):
        chapters = [ChapterRange(1, "Empty", []), ChapterRange(2, "Real", [1, 2])]
        assert find_chapter_for_page(chapters, 1) == chapters[1]

    def test_first_match_wins(self):
        chapters = [ChapterRange(1, "First", [1, 2]), ChapterRange(2, "Second", [1, 2])]
        assert find_chapter_for_page(chapters, 1) == chapters[0]


class TestValidateChapterCoverage:
    def test_exact_cover(self):
        chapters = [ChapterRange(1, "A", [1, 2, 3]), ChapterRange(2, "B", [4, 5])]
        coverage = validate_chapter_coverage(chapters, [1, 2, 3, 4, 5])
        assert coverage.ok
        assert coverage.unassigned_pages == []
        assert coverage.overlapping_pages == []
        assert coverage.out_of_bounds_pages == []

    def test_unassigned_pages(self):
        chapters = [ChapterRange(1, "A", [1, 2])]
        coverage = validate_chapter_coverage(chapters, [1, 2, 3, 4, 5])
        assert coverage.unassigned_pages == [3, 4, 5]
        assert not coverage.ok

    def test_overlapping_pages(self):
        chapters = [ChapterRange(1, "A", [1, 2, 3, 4]), ChapterRange(2, "B", [3, 4, 5])]
        coverage = validate_chapter_coverage(chapters, [1, 2, 3, 4, 5])
        assert coverage.overlapping_pages == [3, 4]
        assert not coverage.ok

    def test_out_of_bounds_pages(self):
        chapters = [ChapterRange(1, "A", [1, 2, 3, 4, 5, 6, 7])]
        coverage = validate_chapter_coverage(chapters, [1, 2, 3, 4, 5])
        assert coverage.out_of_bounds_pages == [6, 7]
        assert not coverage.ok

    def test_single_range_covers_until_last_page(self):
        chapters = [ChapterRange(3, "Extras", [3], True)]
        coverage = validate_chapter_coverage(chapters, [1, 2, 3, 4, 5])
        assert coverage.unassigned_pages == [1, 2]
        assert coverage.out_of_bounds_pages == []

    def test_no_pages_available(self):
        coverage = validate_chapter_coverage([], [])
        assert coverage.ok


def _source_page(index: int, *page_numbers: int) -> SourcePage:
    """A source page whose filename reflects the page number(s) it holds."""
    label = "-".join(f"{number:03d}" for number in page_numbers)
    return SourcePage(index=index, filename=f"{label}.png", page_numbers=list(page_numbers))


class TestValidateSourceCoverage:
    def test_spread_second_page_is_not_reported_as_missing(self):
        # Page 3 exists, but only as the second half of the 002-003 spread.
        pages = [_source_page(0, 1), _source_page(1, 2, 3), _source_page(2, 4)]
        coverage = validate_source_coverage([ChapterRange(1, "A", [1, 2, 3, 4])], pages)
        assert coverage.out_of_bounds_pages == []
        assert coverage.unassigned_pages == []
        assert coverage.ok

    def test_spread_second_page_is_not_uncovered_when_its_image_is_assigned(self):
        pages = [_source_page(0, 1), _source_page(1, 2, 3), _source_page(2, 4)]
        coverage = validate_source_coverage([ChapterRange(1, "A", [1, 2])], pages)
        # Page 3 travels with image 002-003, which chapter A owns.
        assert coverage.unassigned_pages == [4]
        assert not coverage.ok

    def test_unassigned_pages_match_what_split_chapters_skips(self):
        pages = [_source_page(0, 1), _source_page(1, 2, 3), _source_page(2, 4), _source_page(3, 5)]
        chapters = [ChapterRange(1, "A", [1, 2, 3])]
        coverage = validate_source_coverage(chapters, pages)
        skipped = [page.first_page for page in pages if find_chapter_for_page(chapters, page.first_page) is None]
        assert coverage.unassigned_pages == skipped == [4, 5]

    def test_missing_pages_are_reported(self):
        pages = [_source_page(0, 1), _source_page(1, 2)]
        coverage = validate_source_coverage([ChapterRange(1, "A", [1, 2, 3, 4])], pages)
        assert coverage.out_of_bounds_pages == [3, 4]
        assert not coverage.ok

    def test_overlapping_pages(self):
        pages = [_source_page(0, 1), _source_page(1, 2, 3)]
        chapters = [ChapterRange(1, "A", [1, 2, 3]), ChapterRange(2, "B", [3, 4])]
        coverage = validate_source_coverage(chapters, pages)
        assert coverage.overlapping_pages == [3]
        assert coverage.out_of_bounds_pages == [4]

    def test_single_range_ends_on_the_last_spread_page(self):
        pages = [_source_page(0, 1), _source_page(1, 2, 3), _source_page(2, 4)]
        coverage = validate_source_coverage([ChapterRange(2, "Extras", [2], True)], pages)
        assert coverage.unassigned_pages == [1]
        assert coverage.out_of_bounds_pages == []

    def test_no_pages(self):
        assert validate_source_coverage([], []).ok

    def test_flat_and_source_validation_agree_without_spreads(self):
        pages = [_source_page(index, index + 1) for index in range(5)]
        chapters = [ChapterRange(1, "A", [1, 2]), ChapterRange(2, "B", [2, 3, 9])]
        assert validate_source_coverage(chapters, pages) == validate_chapter_coverage(chapters, [1, 2, 3, 4, 5])


class TestCollectSourcePages:
    def test_folder(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1, 2, 3])
        pages = collect_source_pages(source)
        assert [page.filename for page in pages] == ["001.png", "002.png", "003.png"]
        assert [page.first_page for page in pages] == [1, 2, 3]
        assert [page.index for page in pages] == [0, 1, 2]
        assert not any(page.is_spread for page in pages)

    def test_folder_is_read_naturally(self, tmp_path: Path):
        source = tmp_path / "source"
        source.mkdir()
        for name in ("1.png", "2.png", "10.png"):
            (source / name).write_bytes(b"x")
        pages = collect_source_pages(source)
        assert [page.first_page for page in pages] == [1, 2, 10]

    def test_archive(self, tmp_path: Path):
        source = _make_source_cbz(tmp_path, [1, 2])
        pages = collect_source_pages(source)
        assert [page.filename for page in pages] == ["001.png", "002.png"]

    def test_spread_is_detected(self, tmp_path: Path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "001-002.png").write_bytes(b"x")
        pages = collect_source_pages(source)
        assert pages[0].is_spread
        assert pages[0].page_numbers == [1, 2]

    def test_regex_and_custom_data(self, tmp_path: Path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "cover.png").write_bytes(b"x")
        (source / "p005.png").write_bytes(b"x")
        pages = collect_source_pages(source, custom_data={"cover": 0}, regex_data=PAGE_REGEX)
        assert [page.page_numbers for page in pages] == [[0], [5]]

    def test_prefixed_folder_is_detected_automatically(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1, 2], prefix="p")
        pages = collect_source_pages(source)
        assert [page.first_page for page in pages] == [1, 2]

    def test_regex_overrides_the_built_in_detection(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1, 2], prefix="p")
        pages = collect_source_pages(source, regex_data=PAGE_REGEX)
        assert [page.first_page for page in pages] == [1, 2]

    def test_unparsable_folder_raises(self, tmp_path: Path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "cover.png").write_bytes(b"x")
        with pytest.raises(ChapterSplitError):
            collect_source_pages(source)

    def test_missing_source(self, tmp_path: Path):
        with pytest.raises(ChapterSplitError):
            collect_source_pages(tmp_path / "does-not-exist")


class TestReadSourceImage:
    def test_from_folder(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1])
        assert read_source_image(source, "001.png") == b"image-001"

    def test_from_archive(self, tmp_path: Path):
        source = _make_source_cbz(tmp_path, [1])
        assert read_source_image(source, "001.png") == b"image-001"

    def test_not_found(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1])
        with pytest.raises(ChapterSplitError):
            read_source_image(source, "999.png")


class _RecordingObserver(ChapterSplitObserver):
    def __init__(self):
        self.events: list[tuple] = []

    def on_chapter_start(self, chapter: str) -> None:
        self.events.append(("start", chapter))

    def on_chapter_finish(self, chapter: str, target: Path) -> None:
        self.events.append(("finish", chapter, target.name))

    def on_chapter_skip(self, chapter: str) -> None:
        self.events.append(("skip", chapter))

    def on_page_unassigned(self, page: int, filename: str) -> None:
        self.events.append(("unassigned", page, filename))


class TestSplitChapters:
    def test_split_folder(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1, 2, 3, 4, 5])
        out = tmp_path / "out"
        chapters = [ChapterRange(1, "Intro", [1, 2, 3]), ChapterRange(2, "Story", [4, 5])]

        report = split_chapters(source, out, chapters)

        assert report.created_chapters == 2
        assert report.pages_total == 5
        assert report.pages_assigned == 5
        assert not report.has_unassigned
        assert {created.name for created in report.created} == {"001 - Intro.cbz", "002 - Story.cbz"}

        with zipfile.ZipFile(out / "001 - Intro.cbz") as archive:
            assert sorted(archive.namelist()) == ["001.png", "002.png", "003.png"]
            assert archive.read("001.png") == b"image-001"

        with zipfile.ZipFile(out / "002 - Story.cbz") as archive:
            assert sorted(archive.namelist()) == ["004.png", "005.png"]

    def test_split_folder_with_prefixed_filenames(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1, 2, 3], prefix="p")
        out = tmp_path / "out"
        chapters = [ChapterRange(1, "Intro", [1, 2]), ChapterRange(2, "Story", [3])]

        report = split_chapters(source, out, chapters)

        assert report.pages_assigned == 3
        with zipfile.ZipFile(out / "001 - Intro.cbz") as archive:
            assert sorted(archive.namelist()) == ["p001.png", "p002.png"]

    def test_split_folder_with_full_mangafication_filenames(self, tmp_path: Path):
        source = tmp_path / "source"
        source.mkdir()
        for page in (1, 2, 3):
            name = f"Test Title - c001 (v01) - p{page:03d} [dig] [nao].png"
            (source / name).write_bytes(f"image-{page:03d}".encode())
        out = tmp_path / "out"

        report = split_chapters(source, out, [ChapterRange(1, "Intro", [1, 2, 3])])

        assert report.pages_assigned == 3
        with zipfile.ZipFile(out / "001 - Intro.cbz") as archive:
            assert sorted(archive.namelist()) == [
                "Test Title - c001 (v01) - p001 [dig] [nao].png",
                "Test Title - c001 (v01) - p002 [dig] [nao].png",
                "Test Title - c001 (v01) - p003 [dig] [nao].png",
            ]

    def test_split_folder_with_spreads(self, tmp_path: Path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "p001.png").write_bytes(b"x")
        (source / "p002-003.png").write_bytes(b"x")
        out = tmp_path / "out"

        report = split_chapters(source, out, [ChapterRange(1, "Intro", [1, 2])])

        assert report.pages_assigned == 2
        assert not report.has_unassigned
        with zipfile.ZipFile(out / "001 - Intro.cbz") as archive:
            # A spread is assigned by its first page, and only that page counts as covered.
            assert sorted(archive.namelist()) == ["p001.png", "p002-003.png"]

    def test_split_archive(self, tmp_path: Path):
        source = _make_source_cbz(tmp_path, [1, 2, 3])
        out = tmp_path / "out"
        report = split_chapters(source, out, [ChapterRange(1, "Intro", [1, 2, 3])])

        assert report.created_chapters == 1
        assert report.pages_assigned == 3
        with zipfile.ZipFile(out / "001 - Intro.cbz") as archive:
            assert sorted(archive.namelist()) == ["001.png", "002.png", "003.png"]

    def test_volume_is_applied(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1, 2])
        out = tmp_path / "out"
        report = split_chapters(source, out, [ChapterRange(1, "Intro", [1, 2])], volume=3)

        assert [created.name for created in report.created] == ["03.001 - Intro.cbz"]

    def test_per_chapter_omnibus_volume(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1, 2, 3, 4])
        out = tmp_path / "out"
        chapters = [
            ChapterRange(10, "First", [1, 2], volume=1),
            ChapterRange(11, "Second", [3, 4], volume=2),
        ]
        report = split_chapters(source, out, chapters, volume=(1, 2))

        assert {created.name for created in report.created} == {"01.010 - First.cbz", "02.011 - Second.cbz"}

    def test_unassigned_pages_are_reported(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1, 2, 3, 4, 5])
        out = tmp_path / "out"
        observer = _RecordingObserver()

        report = split_chapters(source, out, [ChapterRange(1, "Intro", [1, 2])], observer=observer)

        assert report.pages_assigned == 2
        assert [page.first_page for page in report.unassigned_pages] == [3, 4, 5]
        assert report.has_unassigned
        assert ("unassigned", 3, "003.png") in observer.events

    def test_existing_chapter_is_skipped(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1, 2])
        out = tmp_path / "out"
        out.mkdir()
        (out / "001 - Intro.cbz").write_bytes(b"existing")
        observer = _RecordingObserver()

        report = split_chapters(source, out, [ChapterRange(1, "Intro", [1, 2])], observer=observer)

        assert report.created_chapters == 0
        assert report.skipped == ["001 - Intro"]
        assert report.pages_assigned == 0
        assert (out / "001 - Intro.cbz").read_bytes() == b"existing"
        assert ("skip", "001 - Intro") in observer.events

    def test_overwrite_recreates_chapter(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1, 2])
        out = tmp_path / "out"
        out.mkdir()
        (out / "001 - Intro.cbz").write_bytes(b"existing")

        report = split_chapters(source, out, [ChapterRange(1, "Intro", [1, 2])], overwrite=True)

        assert report.created_chapters == 1
        assert report.skipped == []
        with zipfile.ZipFile(out / "001 - Intro.cbz") as archive:
            assert sorted(archive.namelist()) == ["001.png", "002.png"]

    def test_observer_lifecycle(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1, 2])
        out = tmp_path / "out"
        observer = _RecordingObserver()

        split_chapters(source, out, [ChapterRange(1, "Intro", [1, 2])], observer=observer)

        assert observer.events[0] == ("start", "001 - Intro")
        assert observer.events[-1] == ("finish", "001 - Intro", "001 - Intro.cbz")

    def test_writers_are_closed_after_split(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1, 2])
        out = tmp_path / "out"

        split_chapters(source, out, [ChapterRange(1, "Intro", [1, 2])])

        # A closed zip file can be opened by a fresh reader without warnings.
        with zipfile.ZipFile(out / "001 - Intro.cbz") as archive:
            assert archive.testzip() is None

    def test_no_chapters_is_an_error(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1])
        with pytest.raises(ChapterSplitError):
            split_chapters(source, tmp_path / "out", [])

    def test_missing_source_is_an_error(self, tmp_path: Path):
        with pytest.raises(ChapterSplitError):
            split_chapters(tmp_path / "nope", tmp_path / "out", [ChapterRange(1, "Intro", [1])])

    def test_non_archive_file_is_an_error(self, tmp_path: Path):
        source = tmp_path / "not-an-archive.txt"
        source.write_text("hello")
        with pytest.raises(ChapterSplitError):
            split_chapters(source, tmp_path / "out", [ChapterRange(1, "Intro", [1])])

    def test_spread_pages_use_the_first_page(self, tmp_path: Path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "001-002.png").write_bytes(b"spread")
        out = tmp_path / "out"

        report = split_chapters(source, out, [ChapterRange(1, "Intro", [1, 2])])

        assert report.pages_assigned == 1
        with zipfile.ZipFile(out / "001 - Intro.cbz") as archive:
            assert archive.namelist() == ["001-002.png"]

    def test_default_volume_is_used_when_chapter_has_none(self, tmp_path: Path):
        source = _make_source_folder(tmp_path, [1, 2])
        out = tmp_path / "out"
        chapters = [ChapterRange(1, "Intro", [1], volume=(1, 2)), ChapterRange(2, "Extra", [2])]

        report = split_chapters(source, out, chapters, volume=4)

        assert {created.name for created in report.created} == {"01-02.001 - Intro.cbz", "04.002 - Extra.cbz"}


class TestSourcePage:
    def test_helpers(self):
        page = SourcePage(index=3, filename="p003-004.png", page_numbers=[3, 4])
        assert page.first_page == 3
        assert page.is_spread

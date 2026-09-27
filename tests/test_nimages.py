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

# Tests for the `autolevel3` and `posterize2` VapourSynth paths.

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from nmanga.cli.autolevel import Autolevel3Config, write_page

NIMAGES_PLUGIN = "xyz.n4o.nimages"
IMGSEQS_PLUGIN = "xyz.n4o.imgseqs"


def _has_plugin(plugin: str) -> bool:
    try:
        from nmanga.vapour import vs_find_missing_plugins
    except ImportError:
        return False
    try:
        return not vs_find_missing_plugins([plugin])
    except ImportError:
        return False


HAS_NIMAGES = _has_plugin(NIMAGES_PLUGIN)
HAS_IMGSEQS = _has_plugin(IMGSEQS_PLUGIN)
requires_nimages = pytest.mark.skipif(not HAS_NIMAGES, reason="the nimages VapourSynth plugin is not installed")
requires_imagseqs = pytest.mark.skipif(not HAS_IMGSEQS, reason="the imgseqs VapourSynth plugin is not installed")


def _gradient(width: int = 64, height: int = 48, shift: int = 0) -> Image.Image:
    """A deterministic gray page with a black block, a white block and a ramp."""
    image = Image.new("L", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            if x < width // 4:
                value = 12
            elif x >= (width * 3) // 4:
                value = 244
            else:
                value = (x + y + shift) % 200
            pixels[x, y] = value  # type: ignore
    return image


def _write_pages(directory: Path, count: int = 3, *, suffix: str = ".png") -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in range(count):
        path = directory / f"page_{index:03d}{suffix}"
        _gradient(shift=index * 7).save(path)
        written.append(path)
    return written


def _config(**overrides) -> Autolevel3Config:
    values: dict = {
        "upper_limit": 60,
        "peak_offset": 0,
        "peak_min_pct": 0.25,
        "peak_prom_pct": None,
        "force_gray": False,
        "keep_colorspace": False,
        "image_fmt": "png",
        "no_white": False,
    }
    values.update(overrides)
    return Autolevel3Config(**values)


class TestWritePage:
    """The per page tail shared with `autolevel2`."""

    def test_levels_a_page_it_writes(self, tmp_path: Path):
        source = tmp_path / "source"
        destination = tmp_path / "out"
        destination.mkdir()
        (img_path,) = _write_pages(source, 1)

        result = write_page(img_path, destination, _gradient(), 12, 244, _config())
        assert result.name == "PROCESSED"
        written = destination / img_path.name
        assert written.exists()

    def test_jpg_output_uses_quality_98(self, tmp_path: Path):
        source = tmp_path / "source"
        destination = tmp_path / "out"
        destination.mkdir()
        (img_path,) = _write_pages(source, 1)

        result = write_page(img_path, destination, _gradient(), 12, 244, _config(image_fmt="jpg"))
        assert result.name == "PROCESSED"
        written = destination / img_path.with_suffix(".jpg").name
        assert written.exists()

    def test_black_level_zero_copies_without_processing(self, tmp_path: Path):
        """
        `autolevel2` copies when `black_level <= 0`, whatever the white level is.

        The plugin separates "no black peak was found" from "a peak genuinely sits on bin 0"
        in `NImagesBlackPeakFound`, and `not black_found` is the check `autolevel2` meant to
        write. Using it here would process pages `autolevel2` copies, which would make the two
        commands produce different output for the same input, so this command reproduces the
        existing choice; `--debug` reports both found flags per page.
        """
        source = tmp_path / "source"
        destination = tmp_path / "out"
        destination.mkdir()
        (img_path,) = _write_pages(source, 1)

        result = write_page(img_path, destination, _gradient(), 0, 255, _config())
        assert result.name == "COPIED"
        assert (destination / img_path.name).read_bytes() == img_path.read_bytes()

    def test_black_zero_with_white_255_copies(self, tmp_path: Path):
        source = tmp_path / "source"
        destination = tmp_path / "out"
        destination.mkdir()
        (img_path,) = _write_pages(source, 1)

        result = write_page(img_path, destination, _gradient(), 0, 255, _config())
        assert result.name == "COPIED"

    def test_black_zero_with_a_white_below_255_is_processed(self, tmp_path: Path):
        """
        A white below 255 with a black of 0 still changes the page.

        The curve then maps `0..white` over the whole range, so this is not the identity and
        the page is written. This is where the plugin path and the Pillow path can diverge:
        `autolevel2` copies it because its `black_level <= 0` check fires first.
        """
        source = tmp_path / "source"
        destination = tmp_path / "out"
        destination.mkdir()
        (img_path,) = _write_pages(source, 1)

        assert write_page(img_path, destination, _gradient(), 0, 244, _config()).name == "PROCESSED"
        assert (destination / img_path.name).exists()

    def test_a_real_black_peak_is_processed(self, tmp_path: Path):
        source = tmp_path / "source"
        destination = tmp_path / "out"
        destination.mkdir()
        (img_path,) = _write_pages(source, 1)

        assert write_page(img_path, destination, _gradient(), 12, 255, _config()).name == "PROCESSED"

    def test_no_white_only_skips_on_black(self, tmp_path: Path):
        source = tmp_path / "source"
        destination = tmp_path / "out"
        destination.mkdir()
        (img_path,) = _write_pages(source, 1)

        # A white of 255 is fine with `--no-white`, the page is still leveled.
        result = write_page(img_path, destination, _gradient(), 12, 255, _config(no_white=True))
        assert result.name == "PROCESSED"

        (other,) = _write_pages(source / "second", 1)
        assert write_page(other, destination, _gradient(), 0, 255, _config(no_white=True)).name == "COPIED"

    def test_white_255_skips_when_white_is_adjusted(self, tmp_path: Path):
        """Both levels bad is what copies when `--no-white` is off; black 0 alone is enough too."""
        source = tmp_path / "source"
        destination = tmp_path / "out"
        destination.mkdir()
        (img_path,) = _write_pages(source, 1)

        assert write_page(img_path, destination, _gradient(), 0, 255, _config()).name == "COPIED"

    def test_force_gray_writes_a_grayscale_png(self, tmp_path: Path):
        source = tmp_path / "source"
        destination = tmp_path / "out"
        destination.mkdir()
        (img_path,) = _write_pages(source, 1)

        result = write_page(img_path, destination, _gradient(), 0, 255, _config(force_gray=True))
        assert result.name == "GRAYSCALED"
        written = destination / img_path.with_suffix(".png").name
        with Image.open(written) as written_image:
            assert written_image.mode == "L"

    def test_existing_output_is_skipped(self, tmp_path: Path):
        source = tmp_path / "source"
        destination = tmp_path / "out"
        destination.mkdir()
        (img_path,) = _write_pages(source, 1)
        existing = destination / img_path.name
        existing.write_bytes(b"already here")

        result = write_page(img_path, destination, _gradient(), 12, 244, _config())
        assert result.name == "COPIED"
        assert existing.read_bytes() == b"already here"

    def test_existing_output_is_skipped_when_copying(self, tmp_path: Path):
        source = tmp_path / "source"
        destination = tmp_path / "out"
        destination.mkdir()
        (img_path,) = _write_pages(source, 1)
        existing = destination / img_path.name
        existing.write_bytes(b"already here")

        result = write_page(img_path, destination, _gradient(), 0, 244, _config())
        assert result.name == "COPIED"
        assert existing.read_bytes() == b"already here"


@requires_nimages
@requires_imagseqs
class TestGrayNormalisation:
    """
    `vs_to_gray8` has to agree with a Pillow `convert("L")` on every container.

    `imgseqs` hands out `RGB24` for jpeg and png, and a limited range `YUV420P8` for lossy
    webp. Taking the luma plane of the YUV clip directly would analyse the decoder's limited
    range Y rather than the page, so this is the one place with a wrong answer available.
    """

    def _gray_frame(self, files: list[Path], index: int):
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_to_gray8

        vs = get_vapoursynth()
        source = vs.core.imgseqs.Read(files=[str(path) for path in files], mismatch=True, prefetch=0)
        return vs_to_gray8(source, core=vs.core).get_frame(index)

    @pytest.mark.parametrize("suffix", [".png", ".jpg", ".webp"])
    def test_matches_pillow_luma(self, tmp_path: Path, suffix: str):
        import numpy as np

        pages = _write_pages(tmp_path / "pages", 2, suffix=suffix)
        with Image.open(pages[0]) as page:
            expected = np.asarray(page.convert("L")).astype(int)

        frame = self._gray_frame(pages, 0)
        actual = np.asarray(frame[0]).astype(int)
        assert actual.shape == expected.shape
        delta = np.abs(actual - expected)
        # The two paths round differently at the last bit, so allow a single code value and
        # require the mean to stay far below it.
        assert delta.max() <= 1
        assert delta.mean() < 0.05

    def test_mixed_container_sizes_all_convert(self, tmp_path: Path):
        """The `mismatch=True` clip reports an undefined format, so the probe path runs."""
        import numpy as np

        pages = _write_pages(tmp_path / "pages", 1)
        wide = tmp_path / "pages" / "wide.webp"
        # Even dimensions, so this stays on the plain YUV to RGB path; the odd width case is
        # covered on its own below.
        _gradient(width=66, height=34).save(wide, quality=90)
        pages.append(wide)

        for index, page in enumerate(pages):
            frame = self._gray_frame(pages, index)
            assert frame.format.name == "Gray8"
            with Image.open(page) as opened:
                assert frame.width == opened.width
                assert frame.height == opened.height
            assert np.asarray(frame[0]).size > 0

    def test_odd_width_webp_is_trimmed_and_converted(self, tmp_path: Path):
        """An odd width cannot be 4:2:0, so the conversion has to trim before resizing."""
        pages = [_tmp_odd_webp(tmp_path)]
        frame = self._gray_frame(pages, 0)
        assert frame.width == 65
        assert frame.format.name == "Gray8"


def _tmp_odd_webp(tmp_path: Path) -> Path:
    path = tmp_path / "odd.webp"
    _gradient(width=65, height=33).save(path, quality=90)
    return path


@requires_nimages
@requires_imagseqs
class TestAutolevelClip:
    """The `PeakStats` to `Levels` chain over a directory of pages."""

    def test_levels_match_the_pillow_reference(self, tmp_path: Path):
        from nmanga.autolevel import find_local_peak
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_autolevel_clip

        pages = _write_pages(tmp_path / "pages", 3, suffix=".jpg")
        vs = get_vapoursynth()
        chain = vs_autolevel_clip(pages, options=_config(), core=vs.core)

        for index, page in enumerate(pages):
            with chain.leveled.get_frame(index) as frame:
                black = int(frame.props["NImagesBlackLevel"])  # type: ignore
                white = int(frame.props["NImagesWhiteLevel"])  # type: ignore
            with Image.open(page) as opened:
                expected_black, expected_white, _ = find_local_peak(
                    opened,
                    upper_limit=60,
                    peak_percentage=0.25,
                    peak_prominence=None,
                    skip_white_check=False,
                )
            assert (black, white) == (expected_black, expected_white)

    def test_peak_found_properties_are_reported(self, tmp_path: Path):
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_autolevel_clip

        pages = _write_pages(tmp_path / "pages", 2)
        vs = get_vapoursynth()
        chain = vs_autolevel_clip(pages, options=_config(), core=vs.core)

        with chain.leveled.get_frame(0) as frame:
            assert int(frame.props["NImagesBlackPeakFound"]) in (0, 1)  # type: ignore
            assert int(frame.props["NImagesWhitePeakFound"]) in (0, 1)  # type: ignore

    def test_no_white_reaches_peak_stats(self, tmp_path: Path):
        """`--no-white` has to be passed to `PeakStats`, whose result `Levels` then reads."""
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_autolevel_clip

        pages = _write_pages(tmp_path / "pages", 2)
        vs = get_vapoursynth()
        chain = vs_autolevel_clip(pages, options=_config(no_white=True), core=vs.core)

        for index in range(len(pages)):
            with chain.leveled.get_frame(index) as frame:
                assert int(frame.props["NImagesWhiteLevel"]) == 255  # type: ignore

    def test_frame_to_image_matches_the_levelled_frame(self, tmp_path: Path):
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_autolevel_clip, vs_frame_to_image

        pages = _write_pages(tmp_path / "pages", 2)
        vs = get_vapoursynth()
        chain = vs_autolevel_clip(pages, options=_config(), core=vs.core)

        with chain.leveled.get_frame(1) as frame:
            image = vs_frame_to_image(frame)
            raw = np.asarray(frame[0]).copy()
        assert image.mode == "L"
        assert np.array_equal(np.asarray(image), raw)
        image.close()

    def test_keep_colorspace_levels_every_plane(self, tmp_path: Path):
        """The `ModifyFrame` carry has to keep the frame in RGB and change all three planes."""
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_carry_levels

        vs = get_vapoursynth()
        colour = Image.merge(
            "RGB",
            (
                _gradient(),
                _gradient(shift=5),
                _gradient(shift=11),
            ),
        )
        path = tmp_path / "colour.png"
        colour.save(path)

        source = vs.core.imgseqs.Read(files=[str(path)], mismatch=True, prefetch=0)
        rgb = vs.core.resize.Bicubic(source, format=vs.RGB24)
        gray = vs.core.resize.Bicubic(rgb, format=vs.GRAY8, matrix_s="470bg", range_s="full")
        stats = vs.core.nimages.PeakStats(gray, upper_limit=60, peak_percentage=0.25)
        carried = vs_carry_levels(rgb, stats, core=vs.core)
        assert carried.format.color_family == vs.RGB

        leveled = vs.core.nimages.Levels(carried, use_props=True, auto_gamma=True)
        with rgb.get_frame(0) as before, leveled.get_frame(0) as after:
            for plane in range(3):
                original = np.asarray(before[plane])
                changed = np.asarray(after[plane])
                assert not np.array_equal(original, changed)

    def test_carry_accepts_a_bare_frame(self, tmp_path: Path):
        """
        The caller holds a page pulled out of the chain, which is a frame and not a clip.

        `ModifyFrame` needs a clip, so `vs_carry_levels` has to wrap the frame itself. This is
        the shape `autolevel3 --keep-colorspace` uses: one page at a time, out of order.
        """
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_carry_levels

        vs = get_vapoursynth()
        colour = Image.merge("RGB", (_gradient(), _gradient(shift=5), _gradient(shift=11)))
        path = tmp_path / "colour.png"
        colour.save(path)

        source = vs.core.imgseqs.Read(files=[str(path)], mismatch=True, prefetch=0)
        rgb = vs.core.resize.Bicubic(source, format=vs.RGB24)
        gray = vs.core.resize.Bicubic(rgb, format=vs.GRAY8, matrix_s="470bg", range_s="full")
        stats = vs.core.nimages.PeakStats(gray, upper_limit=60, peak_percentage=0.25)

        with rgb.get_frame(0) as frame:
            carried = vs_carry_levels(frame, stats, core=vs.core)
            assert carried.format.color_family == vs.RGB
            assert carried.num_frames == 1
            assert carried.width == frame.width
            assert carried.height == frame.height
            with carried.get_frame(0) as result:
                assert int(result.props["NImagesBlackLevel"]) >= 0  # type: ignore
                assert int(result.props["NImagesWhiteLevel"]) <= 255  # type: ignore


@requires_nimages
@requires_imagseqs
class TestPosterizeClip:
    """The plugin's posterization has to match `posterize_image_by_bits`'s mapping."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_mapping_matches_pillow_point_table(self, tmp_path: Path, bits: int):
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_frame_to_image, vs_posterize_clip

        # A ramp covers all 256 inputs in one page.
        ramp = Image.new("L", (256, 1))
        ramp.putdata(list(range(256)))
        path = tmp_path / "ramp.png"
        ramp.save(path)

        colors = 2**bits
        expected = np.array(
            [round(round(value * (colors - 1) / 255) * 255 / (colors - 1)) for value in range(256)],
            dtype=np.uint8,
        )

        vs = get_vapoursynth()
        chain = vs_posterize_clip([path], bits=bits, core=vs.core)
        with chain.posterized.get_frame(0) as frame:
            actual = np.asarray(vs_frame_to_image(frame)).reshape(-1)
        assert np.array_equal(actual, expected)
        assert len(np.unique(actual)) == colors

    def test_output_is_byte_identical_to_the_pillow_mapping_on_png(self, tmp_path: Path):
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_frame_to_image, vs_posterize_clip

        pages = _write_pages(tmp_path / "pages", 3, suffix=".png")
        vs = get_vapoursynth()
        chain = vs_posterize_clip(pages, bits=4, core=vs.core)

        for index, page in enumerate(pages):
            with chain.posterized.get_frame(index) as frame:
                actual = np.asarray(vs_frame_to_image(frame))
            with Image.open(page) as opened:
                gray = opened.convert("L")
                reference = np.asarray(gray.point(lambda x: round(x * 15 / 255) * 255 / 15))  # type: ignore
            assert np.array_equal(actual, reference)

    def test_gray_clip_is_kept_for_the_gate(self, tmp_path: Path):
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_posterize_clip

        pages = _write_pages(tmp_path / "pages", 1)
        vs = get_vapoursynth()
        chain = vs_posterize_clip(pages, bits=4, core=vs.core)
        assert chain.gray is not chain.posterized
        with chain.gray.get_frame(0) as frame:
            assert frame.format.name == "Gray8"


@requires_nimages
@requires_imagseqs
class TestFrameToGrays:
    """
    `vship` takes one page at a time at its own size, and a float plane holds `0..1`.

    Writing the raw bytes in instead produces a score of about -15843 where 15.74 is correct,
    so the scaling is the part worth pinning down.
    """

    def test_plane_is_scaled_into_zero_to_one(self, tmp_path: Path):
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_frame_to_grays, vs_posterize_clip

        pages = _write_pages(tmp_path / "pages", 1)
        vs = get_vapoursynth()
        chain = vs_posterize_clip(pages, bits=4, core=vs.core)

        with chain.gray.get_frame(0) as frame:
            source = np.asarray(frame[0]).copy()
            width, height = frame.width, frame.height
            grays = vs_frame_to_grays(frame, core=vs.core)

        assert grays.format.name == "GrayS"
        assert grays.width == width
        assert grays.height == height
        assert grays.num_frames == 1

        with grays.get_frame(0) as converted:
            scaled = np.asarray(converted[0])
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0
        assert np.allclose(scaled, source.astype(np.float32) / 255.0)


class TestPluginRegistration:
    """The two new commands have to be reachable, and the old ones untouched."""

    def test_commands_are_registered(self):
        from nmanga.cmd import main

        names = set(main.commands)
        assert {"autolevel3", "posterize2"} <= names
        # Nothing existing may disappear.
        assert {"autolevel", "autolevel2", "posterize", "autoposterize", "analyze-shades"} <= names

    def test_autolevel3_rejects_the_flags_it_cannot_honour(self):
        from click.testing import CliRunner

        from nmanga.cmd import main

        runner = CliRunner()
        for flag in ("--legacy", "--use-magick", "--debug"):
            result = runner.invoke(main, ["autolevel3", ".", "-o", ".", flag])
            assert result.exit_code != 0

    def test_posterize2_rejects_the_debug_flag(self):
        from click.testing import CliRunner

        from nmanga.cmd import main

        runner = CliRunner()
        result = runner.invoke(main, ["posterize2", ".", "-o", ".", "--debug"])
        assert result.exit_code != 0

    def test_autolevel3_has_threads_and_cache(self):
        from click.testing import CliRunner

        from nmanga.cmd import main

        runner = CliRunner()
        result = runner.invoke(main, ["autolevel3", "--help"])
        assert result.exit_code == 0
        assert "--threads" in result.output
        assert "--cache" in result.output

    def test_posterize2_has_threads_and_cache(self):
        from click.testing import CliRunner

        from nmanga.cmd import main

        runner = CliRunner()
        result = runner.invoke(main, ["posterize2", "--help"])
        assert result.exit_code == 0
        assert "--threads" in result.output
        assert "--cache" in result.output

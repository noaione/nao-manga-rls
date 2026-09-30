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


@requires_nimages
@requires_imagseqs
class TestClipToGrays:
    """
    The clip form of `vs_frame_to_grays`, which is what makes a clip score comparable to the
    per page score the gate already computes.
    """

    def test_clip_form_matches_the_per_page_form(self, tmp_path: Path):
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_clip_to_grays, vs_frame_to_grays, vs_to_gray8

        pages = _write_pages(tmp_path / "pages", 3)
        vs = get_vapoursynth()
        source = vs.core.imgseqs.Read(files=[str(path) for path in pages], mismatch=True, prefetch=0)
        gray = vs_to_gray8(source, core=vs.core)
        reference = vs_clip_to_grays(gray, core=vs.core)

        assert reference.format.name == "GrayS"
        assert reference.num_frames == len(pages)
        for index in range(len(pages)):
            with reference.get_frame(index) as clip_frame:
                clip_plane = np.asarray(clip_frame[0]).copy()
            with gray.get_frame(index) as gray_frame:
                per_page = vs_frame_to_grays(gray_frame, core=vs.core)
            with per_page.get_frame(0) as per_page_frame:
                per_page_plane = np.asarray(per_page_frame[0]).copy()
            assert clip_plane.shape == per_page_plane.shape
            # The clip form is one conversion node over every page, the per page form is a
            # `ModifyFrame` copy, and the two round the last float bit differently.
            assert np.allclose(clip_plane, per_page_plane, atol=1e-6)

    def test_full_range_scaling_is_not_rewritten(self, tmp_path: Path):
        """A limited range conversion turns 255 into 235, and the score with it."""
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_clip_to_grays, vs_to_gray8

        page = tmp_path / "white.png"
        Image.new("L", (16, 16), 255).save(page)
        vs = get_vapoursynth()
        source = vs.core.imgseqs.Read(files=[str(page)], mismatch=True, prefetch=0)
        reference = vs_clip_to_grays(vs_to_gray8(source, core=vs.core), core=vs.core)

        with reference.get_frame(0) as frame:
            plane = np.asarray(frame[0])
        assert np.allclose(plane, 1.0)

    def test_a_scored_pair_stays_in_the_usable_range(self, tmp_path: Path):
        """
        A score in the thousands means the scaling is wrong.

        The recorded failure mode is `-15843` where `15.74` was correct, from handing `vship` a
        plane that was not scaled into `0..1`. This can only catch a gross regression, but it is
        the cheap guard on the property the per page path already lost once.
        """
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_clip_to_grays, vs_metric_clip, vs_to_gray8

        pages = _write_pages(tmp_path / "pages", 2)
        vs = get_vapoursynth()
        source = vs.core.imgseqs.Read(files=[str(path) for path in pages], mismatch=True, prefetch=0)
        gray = vs_to_gray8(source, core=vs.core)
        reference = vs_clip_to_grays(gray, core=vs.core)
        node = vs_metric_clip(reference, reference, "ssimulacra2")
        with node.get_frame(0) as frame:
            score = float(frame.props["_SSIMULACRA2"])  # type: ignore
        assert 80.0 < score <= 100.0

    def test_frame_to_image_round_trips_the_bytes(self, tmp_path: Path):
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_clip_to_grays, vs_grays_frame_to_image, vs_to_gray8

        pages = _write_pages(tmp_path / "pages", 3)
        vs = get_vapoursynth()
        source = vs.core.imgseqs.Read(files=[str(path) for path in pages], mismatch=True, prefetch=0)
        gray = vs_to_gray8(source, core=vs.core)
        reference = vs_clip_to_grays(gray, core=vs.core)

        for index in range(len(pages)):
            with reference.get_frame(index) as frame:
                image = vs_grays_frame_to_image(frame)
            with gray.get_frame(index) as gray_frame:
                expected = np.asarray(gray_frame[0]).copy()
            assert np.array_equal(np.asarray(image), expected)


@requires_nimages
@requires_imagseqs
class TestClipMetrics:
    """
    `vship` scores the whole clip in one call, so the fan-out costs one dispatch per candidate
    rather than one per page.
    """

    def _reference(self, pages: list[Path]):
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_clip_to_grays, vs_to_gray8

        vs = get_vapoursynth()
        source = vs.core.imgseqs.Read(files=[str(path) for path in pages], mismatch=True, prefetch=0)
        gray = vs_to_gray8(source, core=vs.core)
        return gray, vs_clip_to_grays(gray, core=vs.core)

    def test_one_call_scores_every_page(self, tmp_path: Path):
        from nmanga.vapour import vs_metric_clip

        pages = _write_pages(tmp_path / "pages", 4)
        _gray, reference = self._reference(pages)
        branch = vs_metric_clip(reference, reference, "ssimulacra2")
        assert branch.num_frames == len(pages)
        with branch.get_frame(len(pages) - 1) as frame:
            assert "_SSIMULACRA2" in frame.props

    def test_clip_score_matches_the_per_page_score(self, tmp_path: Path):
        """The two paths have to agree, or a candidate is scored against a different reference."""
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_clip_to_grays, vs_frame_to_grays, vs_metric_clip, vs_ssimulacra2

        pages = _write_pages(tmp_path / "pages", 3)
        vs = get_vapoursynth()
        gray, reference = self._reference(pages)
        posterized = vs.core.nimages.Posterize(gray, bits=2, method=0)
        branch = vs_clip_to_grays(posterized, core=vs.core)
        clip_node = vs_metric_clip(reference, branch, "ssimulacra2")

        for index in range(len(pages)):
            with clip_node.get_frame(index) as frame:
                clip_score = float(frame.props["_SSIMULACRA2"])  # type: ignore
            with gray.get_frame(index) as gray_frame, posterized.get_frame(index) as branch_frame:
                per_page = vs_ssimulacra2(
                    vs_frame_to_grays(gray_frame, core=vs.core),
                    vs_frame_to_grays(branch_frame, core=vs.core),
                )
            # The GPU reduction order differs slightly between the two, so this is a tight
            # tolerance rather than equality.
            assert abs(clip_score - per_page) < 0.01

    def test_butteraugli_is_a_distance(self, tmp_path: Path):
        """Its direction is confirmed before it is allowed to decide a depth."""
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import METRIC_HIGHER_IS_BETTER, vs_clip_to_grays, vs_metric_clip

        pages = _write_pages(tmp_path / "pages", 2)
        vs = get_vapoursynth()
        gray, reference = self._reference(pages)
        node = vs_metric_clip(reference, reference, "butteraugli")
        with node.get_frame(0) as frame:
            identical = float(frame.props["_BUTTERAUGLI_QNorm"])  # type: ignore

        posterized = vs.core.nimages.Posterize(gray, bits=1, method=0)
        degraded = vs_metric_clip(reference, vs_clip_to_grays(posterized, core=vs.core), "butteraugli")
        with degraded.get_frame(0) as frame:
            lossy = float(frame.props["_BUTTERAUGLI_QNorm"])  # type: ignore

        assert identical == pytest.approx(0.0)
        assert lossy > identical
        assert METRIC_HIGHER_IS_BETTER["butteraugli"] is False

    def test_cvvdp_is_higher_is_better(self, tmp_path: Path):
        """
        CVVDP is a 0 to 10 quality score, and getting this backwards picks the worst depth.

        The recorded failure: with the direction wired the wrong way the floor became `<= 9.9`,
        a page scoring `4.251` at one bit passed it, and "fewest bits that passes" then chose one
        bit over the three bits that scored `9.873`. The anchor is that a page scored against
        itself is exactly 10 and degrading it lowers the score.
        """
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import METRIC_DEFAULT_MINIMUM, METRIC_HIGHER_IS_BETTER, vs_clip_to_grays, vs_metric_clip

        assert METRIC_HIGHER_IS_BETTER["cvvdp"] is True
        assert METRIC_DEFAULT_MINIMUM["cvvdp"] == pytest.approx(9.9)

        pages = _write_pages(tmp_path / "pages", 2)
        vs = get_vapoursynth()
        gray, reference = self._reference(pages)

        with vs_metric_clip(reference, reference, "cvvdp").get_frame(0) as frame:
            identical = float(frame.props["_CVVDP"])  # type: ignore
        assert identical == pytest.approx(10.0, abs=0.01)

        posterized = vs.core.nimages.Posterize(gray, bits=1, method=0)
        one_bit = vs_metric_clip(reference, vs_clip_to_grays(posterized, core=vs.core), "cvvdp")
        with one_bit.get_frame(0) as frame:
            degraded = float(frame.props["_CVVDP"])  # type: ignore
        # Better means a larger number, so a degraded page is below the floor and is rejected.
        assert degraded < identical
        assert degraded < METRIC_DEFAULT_MINIMUM["cvvdp"]

    def test_the_floor_rejects_a_degraded_candidate_and_finds_the_best_one(self):
        """The reduction has to pick the fewest bits that clears a higher-is-better floor."""
        import numpy as np

        from nmanga.vapour import vs_select_posterize_bits

        # The KamiKatsu v03 p001 scores that exposed the inverted direction.
        scores = {
            "cvvdp": {
                1: np.array([4.251]),
                2: np.array([9.230]),
                3: np.array([9.873]),
                4: np.array([9.979]),
            }
        }
        chosen = vs_select_posterize_bits(scores, candidates=[1, 2, 3, 4], minimums={"cvvdp": 9.9})
        # One bit scores 4.251 and must not pass; four bits is the fewest that clears the floor.
        assert np.array_equal(chosen, [4])

    def test_cvvdp_models_a_four_k_display(self):
        """The model is stated here rather than inherited from the plugin's own default."""
        from nmanga.vapour import CVVDP_MODEL

        assert CVVDP_MODEL == "standard_4k"


@requires_nimages
@requires_imagseqs
class TestPosterizeGroup:
    """One decode shared by the reference and every candidate branch."""

    def test_every_candidate_is_built_off_one_chain(self, tmp_path: Path):
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_posterize_group

        pages = _write_pages(tmp_path / "pages", 3)
        vs = get_vapoursynth()
        group = vs_posterize_group(pages, bits_candidates=[2, 4, 6], method=0, core=vs.core)

        assert group.gray is not group.reference
        assert set(group.branches) == {2, 4, 6}
        assert group.size == (64, 48)
        # The reference and the branches all report the same page size and length.
        for branch in group.branches.values():
            assert branch.num_frames == len(pages)
            assert (branch.width, branch.height) == group.size
            with branch.get_frame(0) as frame:
                assert frame.format.name == "GrayS"

    def test_scores_are_per_candidate_per_page(self, tmp_path: Path):
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_posterize_group, vs_score_candidates

        pages = _write_pages(tmp_path / "pages", 3)
        vs = get_vapoursynth()
        group = vs_posterize_group(pages, bits_candidates=[1, 4], method=0, core=vs.core)
        scores = vs_score_candidates(group, metrics=["ssimulacra2"], core=vs.core)

        assert set(scores) == {"ssimulacra2"}
        assert set(scores["ssimulacra2"]) == {1, 4}
        for bits in (1, 4):
            assert len(scores["ssimulacra2"][bits]) == len(pages)
        # A coarser posterization cannot score better than a finer one on the same pages.
        import numpy as np

        assert np.all(scores["ssimulacra2"][4] > scores["ssimulacra2"][1])


class TestSelectPosterizeBits:
    """The reduction over the candidate axis, without a GPU or a plugin in the way."""

    def _scores(self, **per_metric):
        import numpy as np

        return {
            metric: {bits: np.array([value], dtype=float) for bits, value in values.items()}
            for metric, values in per_metric.items()
        }

    def test_fewest_bits_that_passes_wins(self):
        import numpy as np

        from nmanga.vapour import vs_select_posterize_bits

        scores = self._scores(ssimulacra2={2: 85.0, 3: 90.0, 4: 99.0})
        chosen = vs_select_posterize_bits(scores, candidates=[2, 3, 4], minimums={"ssimulacra2": 80.0})
        assert np.array_equal(chosen, [2])

    def test_no_candidate_passing_is_the_fallback(self):
        import numpy as np

        from nmanga.vapour import vs_select_posterize_bits

        scores = self._scores(ssimulacra2={2: 10.0, 4: 20.0})
        chosen = vs_select_posterize_bits(scores, candidates=[2, 4], minimums={"ssimulacra2": 80.0})
        assert np.array_equal(chosen, [-1])

    def test_a_lower_is_better_metric_keeps_its_direction(self):
        import numpy as np

        from nmanga.vapour import vs_select_posterize_bits

        scores = self._scores(butteraugli={2: 3.0, 3: 0.4, 4: 0.1})
        chosen = vs_select_posterize_bits(scores, candidates=[2, 3, 4], minimums={"butteraugli": 1.0})
        # 2 fails, 3 and 4 both pass, so the fewest bits that passes is 3.
        assert np.array_equal(chosen, [3])

    def test_every_thresholded_metric_has_to_pass(self):
        import numpy as np

        from nmanga.vapour import vs_select_posterize_bits

        scores = self._scores(ssimulacra2={2: 85.0, 3: 88.0}, butteraugli={2: 2.0, 3: 0.4})
        chosen = vs_select_posterize_bits(
            scores,
            candidates=[2, 3],
            minimums={"ssimulacra2": 80.0, "butteraugli": 1.0},
        )
        assert np.array_equal(chosen, [3])

    def test_a_metric_without_a_threshold_only_reports(self):
        import numpy as np

        from nmanga.vapour import vs_select_posterize_bits

        scores = self._scores(ssimulacra2={2: 85.0, 3: 90.0}, cvvdp={2: 10.0, 3: 10.0})
        chosen = vs_select_posterize_bits(scores, candidates=[2, 3], minimums={"ssimulacra2": 80.0})
        assert np.array_equal(chosen, [2])

    def test_the_preference_order_only_breaks_ties(self):
        import numpy as np

        from nmanga.vapour import vs_select_posterize_bits

        scores = self._scores(ssimulacra2={2: 85.0, 3: 90.0, 4: 70.0})
        chosen = vs_select_posterize_bits(
            scores,
            candidates=[4, 3, 2],
            minimums={"ssimulacra2": 80.0},
            order=[4, 3, 2],
        )
        # 4 fails and both 2 and 3 pass, so the caller's preference picks 3.
        assert np.array_equal(chosen, [3])

    def test_the_decision_is_per_page(self):
        import numpy as np

        from nmanga.vapour import vs_select_posterize_bits

        scores = {"ssimulacra2": {2: np.array([85.0, 20.0]), 3: np.array([90.0, 30.0])}}
        chosen = vs_select_posterize_bits(scores, candidates=[2, 3], minimums={"ssimulacra2": 80.0})
        assert np.array_equal(chosen, [2, -1])

    def test_an_unknown_metric_raises(self):
        import numpy as np
        import pytest

        from nmanga.vapour import vs_select_posterize_bits

        scores = {"ssimulacra2": {2: np.array([85.0])}}
        with pytest.raises(ValueError, match="butteraugli"):
            vs_select_posterize_bits(scores, candidates=[2], minimums={"butteraugli": 1.0})

    def test_no_candidates_raises(self):
        import numpy as np
        import pytest

        from nmanga.vapour import vs_select_posterize_bits

        scores = {"ssimulacra2": {2: np.array([85.0])}}
        with pytest.raises(ValueError, match="candidate"):
            vs_select_posterize_bits(scores, candidates=[], minimums={"ssimulacra2": 80.0})


class TestBitsOption:
    """`--bits` is the whole candidate set: one depth, or several to choose between."""

    def test_a_single_depth_is_one_candidate(self):
        from nmanga.cli.posterize import parse_bits

        assert parse_bits("4") == [4]

    def test_a_range_expands_inclusively(self):
        from nmanga.cli.posterize import parse_bits

        assert parse_bits("2-5") == [2, 3, 4, 5]

    def test_a_plain_list_keeps_the_order_it_was_written_in(self):
        from nmanga.cli.posterize import parse_bits

        assert parse_bits("5,4,3,2") == [5, 4, 3, 2]
        assert parse_bits("2,3,4") == [2, 3, 4]

    def test_ranges_and_depths_mix(self):
        from nmanga.cli.posterize import parse_bits

        assert parse_bits("1-2,4") == [1, 2, 4]

    def test_duplicates_collapse(self):
        from nmanga.cli.posterize import parse_bits

        assert parse_bits("3,3,3") == [3]

    @pytest.mark.parametrize("raw", ["9", "0", "3-1", "abc", "", "2-", "-3"])
    def test_a_bad_selection_is_rejected(self, raw: str):
        import click as click_module

        from nmanga.cli.posterize import parse_bits

        with pytest.raises(click_module.BadParameter):
            parse_bits(raw)

    def test_a_trailing_comma_is_ignored(self):
        from nmanga.cli.posterize import parse_bits

        assert parse_bits("4,") == [4]
        assert parse_bits("2,3,") == [2, 3]

    def test_a_range_is_fewest_bits_and_a_list_is_a_preference(self):
        from nmanga.cli.posterize import resolve_candidates

        assert resolve_candidates("2-5") == ([2, 3, 4, 5], True)
        assert resolve_candidates("5,4,3,2") == ([5, 4, 3, 2], False)
        assert resolve_candidates("2,3,4") == ([2, 3, 4], False)

    def test_one_depth_has_nothing_to_choose_between(self):
        from nmanga.cli.posterize import resolve_candidates

        assert resolve_candidates("3") == ([3], True)
        # The default is the fixed depth posterize this command always was.
        assert resolve_candidates("4") == ([4], True)

    def test_automatic_degradation_is_a_range(self):
        from nmanga.cli.posterize import AUTO_BITS_RANGE, resolve_candidates

        assert AUTO_BITS_RANGE == "1-6"
        candidates, fewest = resolve_candidates(AUTO_BITS_RANGE)
        assert (candidates, fewest) == ([1, 2, 3, 4, 5, 6], True)

    def test_metric_thresholds_parse(self):
        from nmanga.cli.posterize import parse_metric_minimums

        assert parse_metric_minimums("ssimulacra2=85,butteraugli=1.5") == {
            "ssimulacra2": 85.0,
            "butteraugli": 1.5,
        }

    def test_metrics_default_to_ssimulacra2(self):
        """Asking for candidates without naming a metric is a request to score them."""
        from nmanga.cli.posterize import parse_metrics

        assert parse_metrics(None) == ["ssimulacra2"]
        assert parse_metrics("ssimulacra2,butteraugli") == ["ssimulacra2", "butteraugli"]
        assert parse_metrics("CVVDP") == ["cvvdp"]

    @pytest.mark.parametrize("raw", ["", ",", "nope", "ssimulacra2,nope"])
    def test_a_bad_metric_name_is_rejected(self, raw: str):
        import click as click_module

        from nmanga.cli.posterize import parse_metrics

        with pytest.raises(click_module.BadParameter):
            parse_metrics(raw)

    @pytest.mark.parametrize("raw", ["ssimulacra2", "nope=1", "ssimulacra2=x"])
    def test_a_bad_metric_threshold_is_rejected(self, raw: str):
        import click as click_module

        from nmanga.cli.posterize import parse_metric_minimums

        with pytest.raises(click_module.BadParameter):
            parse_metric_minimums(raw)

    def test_ssimulacra2_has_a_default_threshold(self):
        from nmanga.cli.posterize import resolve_metric_minimums

        minimums = resolve_metric_minimums(["ssimulacra2", "cvvdp"], minimums={})
        assert minimums["ssimulacra2"] == pytest.approx(80.0)
        assert minimums["cvvdp"] == pytest.approx(9.9)

    def test_an_explicit_threshold_wins(self):
        from nmanga.cli.posterize import resolve_metric_minimums

        minimums = resolve_metric_minimums(["ssimulacra2", "cvvdp"], minimums={"cvvdp": 9.5, "ssimulacra2": 72.0})
        assert minimums == {"ssimulacra2": 72.0, "cvvdp": 9.5}

    def test_butteraugli_has_a_default_threshold(self):
        from nmanga.cli.posterize import resolve_metric_minimums

        assert resolve_metric_minimums(["butteraugli"], minimums={}) == {"butteraugli": 1.0}


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

    @pytest.mark.parametrize("flag", ["--use-ssimulacra2", "--ssim-min", "--bits-candidates", "--auto-bits"])
    def test_posterize2_no_longer_takes_the_old_options(self, flag: str):
        """`--bits` carries the candidates now, and the old gate's options are gone."""
        from click.testing import CliRunner

        from nmanga.cmd import main

        runner = CliRunner()
        result = runner.invoke(main, ["posterize2", ".", "-o", ".", flag])
        assert result.exit_code != 0
        # The Pillow command keeps its own copy of the gate options.
        if flag in ("--use-ssimulacra2", "--ssim-min"):
            pillow = runner.invoke(main, ["posterize", "--help"])
            assert flag in pillow.output

    def test_posterize2_takes_one_or_several_depths(self):
        """`--bits` is a single depth or a candidate set, and it defaults to one depth."""
        from click.testing import CliRunner

        from nmanga.cmd import main

        runner = CliRunner()
        result = runner.invoke(main, ["posterize2", "--help"])
        assert result.exit_code == 0
        assert "--metrics" in result.output
        assert "--metric-min" in result.output
        assert "2-5" in result.output
        assert "Default: 4" in result.output

    def test_one_depth_is_scored_when_a_metric_is_asked_for(self, tmp_path: Path):
        """
        A single depth is a yes or no gate, not an instruction to skip the metric.

        The page is scored against that one candidate and copied when it fails, which is the
        same fallback every other candidate set takes.
        """
        from click.testing import CliRunner

        from nmanga.cmd import main

        pages = _write_pages(tmp_path / "pages", 2)
        runner = CliRunner()

        # An impossible floor has to send every page to the source bytes.
        strict = tmp_path / "strict"
        result = runner.invoke(
            main,
            [
                "posterize2",
                str(pages),
                "-o",
                str(strict),
                "-t",
                "1",
                "--bits",
                "4",
                "--metrics",
                "ssimulacra2",
                "--metric-min",
                "ssimulacra2=100",
            ],
        )
        assert result.exit_code == 0
        assert "Copied 2 images without posterization." in result.output
        for page in pages:
            assert (strict / page.name).read_bytes() == page.read_bytes()

        # A floor nothing can miss writes every page at that depth, like the fixed path.
        loose = tmp_path / "loose"
        result = runner.invoke(
            main,
            [
                "posterize2",
                str(pages),
                "-o",
                str(loose),
                "-t",
                "1",
                "--bits",
                "4",
                "--metrics",
                "ssimulacra2",
                "--metric-min",
                "ssimulacra2=0",
            ],
        )
        assert result.exit_code == 0
        assert "Posterized 2 images." in result.output
        for page in pages:
            assert (loose / page.name).read_bytes() != page.read_bytes()

    def test_one_depth_without_a_metric_scores_nothing(self, tmp_path: Path):
        """`--bits 4` on its own stays the fixed depth posterize it always was."""
        from click.testing import CliRunner

        from nmanga.cmd import main

        pages = _write_pages(tmp_path / "pages", 2)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["posterize2", str(pages), "-o", str(tmp_path / "out"), "-t", "1", "--bits", "4"],
        )
        assert result.exit_code == 0
        assert "Scoring candidate" not in result.output
        assert "Copied" not in result.output
        assert "Posterized 2 images to 4 bits." in result.output

    def test_autolevel3_has_prefetch_threads_and_cache(self):
        from click.testing import CliRunner

        from nmanga.cmd import main

        runner = CliRunner()
        result = runner.invoke(main, ["autolevel3", "--help"])
        assert result.exit_code == 0
        assert "--prefetch" in result.output
        assert "--threads" in result.output
        assert "--cache" in result.output

    def test_posterize2_has_prefetch_threads_and_cache(self):
        from click.testing import CliRunner

        from nmanga.cmd import main

        runner = CliRunner()
        result = runner.invoke(main, ["posterize2", "--help"])
        assert result.exit_code == 0
        assert "--prefetch" in result.output
        assert "--threads" in result.output
        assert "--cache" in result.output

    def test_prefetch_defaults_to_16(self):
        from click.testing import CliRunner

        from nmanga.cmd import main

        runner = CliRunner()
        result = runner.invoke(main, ["autolevel3", "--help"])
        assert "Default: 16" in result.output

    def test_prefetch_rejects_a_negative_value(self):
        from click.testing import CliRunner

        from nmanga.cmd import main

        runner = CliRunner()
        result = runner.invoke(main, ["autolevel3", ".", "-o", ".", "--prefetch", "-1"])
        assert result.exit_code != 0


class TestBoundedWritePool:
    """The write pool that keeps the encode off the frame loop."""

    def test_runs_every_submitted_call(self):
        from nmanga.common import BoundedWritePool

        seen: list[int] = []
        with BoundedWritePool(4) as pool:
            for index in range(20):
                oldest = pool.submit(seen.append, index)
                if oldest is not None:
                    oldest.result()
            for future in pool.pending():
                future.result()
        assert sorted(seen) == list(range(20))

    def test_hands_back_the_oldest_page_once_full(self):
        from nmanga.common import BoundedWritePool

        with BoundedWritePool(3) as pool:
            assert pool.submit(lambda: None) is None
            assert pool.submit(lambda: None) is None
            # The third page fills the pool, so the first one comes back to be waited on, and
            # the pool stays at its bound from there on.
            assert pool.submit(lambda: None) is not None
            assert pool.submit(lambda: None) is not None
            assert len(pool.pending()) == 2

    def test_hands_pages_back_in_order(self):
        """The bound is only a memory bound if the oldest page is the one returned."""
        from nmanga.common import BoundedWritePool

        with BoundedWritePool(2) as pool:
            assert pool.submit(int, 1) is None
            second = pool.submit(int, 2)
            assert second is not None
            assert second.result() == 1

    def test_a_failing_write_is_raised_on_result(self):
        from nmanga.common import BoundedWritePool

        def boom():
            raise ValueError("write failed")

        with pytest.raises(ValueError, match="write failed"):
            # One worker, so the first page is already handed back to be waited on.
            with BoundedWritePool(1) as pool:
                oldest = pool.submit(boom)
                assert oldest is not None
                oldest.result()

    def test_workers_is_at_least_one(self):
        from nmanga.common import BoundedWritePool

        # Zero workers would mean no write ever completes, so it is clamped to one.
        with BoundedWritePool(0) as pool:
            assert pool.submit(int, 7) is not None

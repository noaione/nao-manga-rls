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
from typing import TYPE_CHECKING

import pytest
from PIL import Image

if TYPE_CHECKING:
    from nmanga.cli.autolevel import Autolevel3Config

NIMAGES_PLUGIN = "xyz.n4o.nimages"
IMGSEQS_PLUGIN = "xyz.n4o.imgseqs"
VSHIP_PLUGIN = "com.lumen.vship"


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
HAS_VSHIP = _has_plugin(VSHIP_PLUGIN)
requires_nimages = pytest.mark.skipif(not HAS_NIMAGES, reason="the nimages VapourSynth plugin is not installed")
requires_imagseqs = pytest.mark.skipif(not HAS_IMGSEQS, reason="the imgseqs VapourSynth plugin is not installed")
requires_vship = pytest.mark.skipif(not HAS_VSHIP, reason="the vship VapourSynth plugin is not installed")


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


def _colour_gradient(width: int = 64, height: int = 48) -> Image.Image:
    """An RGB page with hard edges whose channel offsets stay clear of both ends."""
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            if x < width // 4:
                base = 70
            elif x >= (width * 3) // 4:
                base = 170
            else:
                base = 70 + ((x * 3 + y * 2) % 100)
            pixels[x, y] = (base + 10, base, base + 20)  # type: ignore
    return image


def _write_pages(directory: Path, count: int = 3, *, suffix: str = ".png") -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in range(count):
        path = directory / f"page_{index:03d}{suffix}"
        _gradient(shift=index * 7).save(path)
        written.append(path)
    return written


def _config(**overrides) -> "Autolevel3Config":
    """The `autolevel3` config a chain test builds.

    The fields are the rewritten config's: the output mode replaced the old
    `force_gray` / `keep_colorspace` pair, and `gpu` reaches both plugin nodes.
    """
    from nmanga.cli.autolevel import Autolevel3Config, OutputMode

    values: dict = {
        "upper_limit": 60,
        "peak_offset": 0,
        "peak_min_pct": 0.25,
        "peak_prom_pct": None,
        "no_white": False,
        "cache_mb": 512,
        "gpu": False,
        "output_mode": OutputMode.COLOR,
    }
    values.update(overrides)
    return Autolevel3Config(**values)


class TestAutolevelDecisions:
    """The two pure helpers the per page decision and the level read are made of."""

    def test_unusable_levels_reproduce_the_autolevel2_rule(self):
        from nmanga.cli.autolevel import _levels_are_unusable

        config = _config()
        # Both levels bad, or a black on zero: copies.
        assert _levels_are_unusable(0, 255, config) is True
        # A black on zero with a white that still changes the page: processed.
        assert _levels_are_unusable(0, 244, config) is False
        # A real black peak at the top of the white range: processed.
        assert _levels_are_unusable(12, 255, config) is False
        # A black above the upper limit: copies.
        assert _levels_are_unusable(61, 255, config) is True
        assert _levels_are_unusable(60, 255, config) is False
        # `--no-white` makes a black on zero decisive and leaves white alone.
        assert _levels_are_unusable(0, 255, _config(no_white=True)) is True
        assert _levels_are_unusable(12, 255, _config(no_white=True)) is False

    def test_page_levels_reads_both_levels_and_both_flags(self):
        from nmanga.cli.autolevel import _page_levels

        frame = _FakeFrame({
            "NImagesBlackLevel": 12,
            "NImagesWhiteLevel": 244,
            "NImagesBlackPeakFound": 1,
            "NImagesWhitePeakFound": 0,
        })
        assert _page_levels(frame) == (12, 244, True, False)


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

    def test_odd_width_lossless_webp_converts_through_the_rgb_path(self, tmp_path: Path):
        """
        A lossy webp is 4:2:0 and the plugin cannot declare an odd one, so the odd edge
        this project can actually be handed is a lossless (RGB) webp.

        That page needs no subsampling trim: it converts through the plain RGB path and
        the width survives.
        """
        import numpy as np

        pages = [_tmp_odd_webp(tmp_path)]
        frame = self._gray_frame(pages, 0)
        assert frame.width == 65
        assert frame.format.name == "Gray8"

        with Image.open(pages[0]) as opened:
            expected = np.asarray(opened.convert("L")).astype(int)
        actual = np.asarray(frame[0]).astype(int)
        assert actual.shape == expected.shape
        assert np.abs(actual - expected).max() <= 1

    def test_a_lone_odd_lossy_webp_is_refused_at_read(self, tmp_path: Path):
        """
        The plugin documents this boundary: an odd 4:2:0 frame cannot exist in VapourSynth.

        `Read` refuses the clip before any frame is requested, which is the non-crashing
        form of the failure a mixed clip used to abort on. `_trim_and_convert`'s odd-YUV
        branch is therefore unreachable by construction.
        """
        import vapoursynth

        from nmanga.lazy import get_vapoursynth

        odd = tmp_path / "odd-lossy.webp"
        _gradient(width=65, height=33).save(odd, quality=90)
        vs = get_vapoursynth()
        with pytest.raises(vapoursynth.Error, match="VSVideoInfo"):
            vs.core.imgseqs.Read(files=[str(odd)], mismatch=True, prefetch=0)

    def test_odd_size_pages_in_a_mixed_clip_probe_each_frame(self, tmp_path: Path):
        """
        A clip mixing containers reports an undefined format, so the choice is per frame.

        The odd lossless (RGB) page and the even lossy (YUV) page each convert at their
        own size, which is what the per-frame probe in `vs_to_gray8` exists for.
        """
        import numpy as np

        odd_rgb = _tmp_odd_webp(tmp_path)
        yuv = tmp_path / "even.webp"
        _gradient(width=66, height=34).save(yuv, quality=90)
        pages = [odd_rgb, yuv]

        first = self._gray_frame(pages, 0)
        assert (first.width, first.height, first.format.name) == (65, 33, "Gray8")
        with Image.open(odd_rgb) as opened:
            expected = np.asarray(opened.convert("L")).astype(int)
        assert np.abs(np.asarray(first[0]).astype(int) - expected).max() <= 1

        second = self._gray_frame(pages, 1)
        assert (second.width, second.height, second.format.name) == (66, 34, "Gray8")

    def test_colour_page_first_still_converts_a_later_webp(self, tmp_path: Path):
        """
        A clip's frames are not all one format: a colour page first must not send the webp
        behind it down the plain resize, which refuses its 4:2:0 size.
        """
        colour = tmp_path / "colour.png"
        _gradient(width=64, height=48).convert("RGB").save(colour)

        later = tmp_path / "later.webp"
        _gradient(width=66, height=34).save(later, quality=90)
        frame = self._gray_frame([colour, later], 1)
        assert (frame.width, frame.height, frame.format.name) == (66, 34, "Gray8")


def _tmp_odd_webp(tmp_path: Path) -> Path:
    """An odd-width lossless webp.

    A lossy webp is 4:2:0 and `imgseqs` cannot declare an odd frame in that format, so the
    odd page a graph can be handed is the RGB one a lossless webp decodes to.
    """
    path = tmp_path / "odd.webp"
    _gradient(width=65, height=33).save(path, lossless=True)
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

    def test_gray_mode_uses_one_plane_and_keeps_the_levels(self, tmp_path: Path):
        """`--gray` picks the analysis clip; the levels it finds do not change with the mode."""
        from nmanga.cli.autolevel import OutputMode
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_autolevel_clip

        pages = _write_pages(tmp_path / "pages", 3)
        colour_path = tmp_path / "pages" / "colour.png"
        Image.merge("RGB", (_gradient(), _gradient(shift=5), _gradient(shift=11))).save(colour_path)
        pages.append(colour_path)

        vs = get_vapoursynth()
        colour_chain = vs_autolevel_clip(pages, options=_config(), core=vs.core)
        gray_chain = vs_autolevel_clip(pages, options=_config(output_mode=OutputMode.GRAY), gray=True, core=vs.core)

        for index, page in enumerate(pages):
            with colour_chain.leveled.get_frame(index) as frame:
                colour_format = frame.format.name
                colour_levels = (
                    int(frame.props["NImagesBlackLevel"]),
                    int(frame.props["NImagesWhiteLevel"]),
                )
            with gray_chain.leveled.get_frame(index) as frame:
                gray_format = frame.format.name
                gray_levels = (
                    int(frame.props["NImagesBlackLevel"]),
                    int(frame.props["NImagesWhiteLevel"]),
                )

            assert gray_format == "Gray8"
            assert gray_levels == colour_levels
            # A gray page is already one plane, so the colour mode keeps it; only the RGB
            # page changes family when `--gray` is asked for.
            assert colour_format == ("RGB24" if page == colour_path else "Gray8")

    def test_a_colour_page_keeps_its_container_family_and_round_trips(self, tmp_path: Path):
        """
        The write stage stores the frame the chain selected, in its own container family.

        A gray page stays one plane and an `RGB24` page stays RGB, so the PNG the writer
        publishes decodes back to the same samples. A YUV page stays YUV through the chain --
        the analysis reads plane 0 and `Levels` maps native code values -- and `PNGWrite`
        converts it to RGB with the frame's own matrix and range when it publishes.
        """
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_autolevel_clip, vs_frame_to_image, vs_png_writer

        pages = tmp_path / "pages"
        (gray_path,) = _write_pages(pages, 1)
        rgb_path = pages / "rgb.png"
        _colour_gradient().save(rgb_path)
        webp_path = pages / "even.webp"
        # An even size, so this webp decodes as a plain YUV page; the odd width case
        # belongs to the command level mixed directory test.
        _gradient(width=66, height=34).save(webp_path, quality=90)

        # (page, chain family, Pillow mode of the chain frame, mode of the published PNG)
        expected = (
            (gray_path, "Gray8", "L", "L"),
            (rgb_path, "RGB24", "RGB", "RGB"),
            (webp_path, "YUV420P8", None, "RGB"),
        )
        vs = get_vapoursynth()
        for page, family, image_mode, written_mode in expected:
            chain = vs_autolevel_clip([page], options=_config(), core=vs.core)
            output = tmp_path / f"out-{page.stem}"
            writer = vs_png_writer(chain.leveled, output, core=vs.core)

            expected_pixels = None
            with chain.leveled.get_frame(0) as frame:
                assert frame.format.name == family
                size = (frame.width, frame.height)
                if image_mode is not None:
                    # A gray or RGB frame is stored verbatim, so its samples are the reference.
                    image = vs_frame_to_image(frame)
                    expected_pixels = np.asarray(image).copy()
                    assert image.mode == image_mode

            with writer.get_frame(0) as frame:
                assert frame.props["ImgSeqPNGWritePath"] == str(output / f"{page.stem}.png")
            (written,) = list(output.iterdir())
            assert written.name == f"{page.stem}.png"
            with Image.open(written) as opened:
                assert opened.mode == written_mode
                assert opened.size == size
                if expected_pixels is not None:
                    assert np.array_equal(np.asarray(opened), expected_pixels)

    def test_imgseqpath_survives_every_migrated_branch(self, tmp_path: Path):
        """`output_dir` names a still from this property, so each branch has to keep it."""
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import (
            vs_deblur_clip,
            vs_posterize_clip,
        )

        pages = tmp_path / "pages"
        (gray_path,) = _write_pages(pages, 1)
        colour_path = pages / "colour.png"
        _colour_gradient().save(colour_path)
        webp_path = pages / "even.webp"
        _gradient(width=66, height=34).save(webp_path, quality=90)
        files = [gray_path, colour_path, webp_path]

        vs = get_vapoursynth()
        source = vs.core.imgseqs.Read(files=[str(page) for page in files], mismatch=True, prefetch=0)
        posterize = vs_posterize_clip(files, bits=4, method=0, core=vs.core)
        deblur = vs_deblur_clip(
            files,
            method=1,
            radius=0.8,
            strength=None,
            iterations=6,
            threshold=2.0,
            overshoot=0.0,
            core=vs.core,
        )
        analysis = source

        branches = (
            ("analysis", analysis),
            ("Posterize", posterize.posterized),
            ("Deblur", deblur.deblurred),
        )
        for label, clip in branches:
            for index, page in enumerate(files):
                with clip.get_frame(index) as frame:
                    assert frame.props["ImgSeqPath"] == str(page), label


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
        chain = vs_posterize_clip([path], bits=bits, method=0, core=vs.core)
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
        chain = vs_posterize_clip(pages, bits=4, method=0, core=vs.core)

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
        chain = vs_posterize_clip(pages, bits=4, method=0, core=vs.core)
        assert chain.gray is not chain.posterized
        with chain.gray.get_frame(0) as frame:
            assert frame.format.name == "Gray8"


@requires_nimages
@requires_imagseqs
class TestDeblurClip:
    """
    `Deblur` is the one filter that must not be normalised to gray.

    It reads RGB24, moves the luma and carries the change onto the three planes as an equal
    offset, so a chain that converts first silently discards the colour the filter preserves.
    """

    def test_colour_page_keeps_its_planes(self, tmp_path: Path):
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_deblur_clip, vs_frame_to_image

        page = tmp_path / "colour.png"
        source = _colour_gradient()
        source.save(page)

        vs = get_vapoursynth()
        chain = vs_deblur_clip(
            [page],
            method=1,
            radius=0.8,
            # A gentle blend, so the equal offset the plugin adds stays inside every channel's
            # gamut and the comparison is about the offset rather than about clipping.
            strength=0.3,
            iterations=6,
            threshold=2.0,
            overshoot=0.0,
            core=vs.core,
        )
        with chain.deblurred.get_frame(0) as frame:
            assert frame.format.name == "RGB24"
            image = vs_frame_to_image(frame)

        assert image.mode == "RGB"
        actual = np.asarray(image).astype(np.int16)
        expected = np.asarray(source).astype(np.int16)
        assert actual.shape == expected.shape

        # An equal offset on R, G and B leaves every channel difference untouched. The page
        # is chosen away from both ends so the gamut clamp never engages.
        for first, second in ((0, 1), (1, 2), (0, 2)):
            assert np.array_equal(
                actual[..., first] - actual[..., second], expected[..., first] - expected[..., second]
            )

        # And the sharpening actually happened, rather than the frame being carried through.
        assert not np.array_equal(actual, expected)

    def test_gray_page_stays_a_single_plane(self, tmp_path: Path):
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_deblur_clip, vs_frame_to_image

        (page,) = _write_pages(tmp_path / "pages", 1)
        vs = get_vapoursynth()
        chain = vs_deblur_clip(
            [page],
            method=1,
            radius=0.8,
            strength=None,
            iterations=6,
            threshold=2.0,
            overshoot=0.0,
            core=vs.core,
        )
        with chain.deblurred.get_frame(0) as frame:
            assert frame.format.num_planes == 1
            image = vs_frame_to_image(frame)

        assert image.mode == "L"

    def test_unset_strength_uses_the_per_method_default(self, tmp_path: Path):
        """`strength=None` has to reach the plugin as an absent argument, not as a number."""
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_deblur_clip, vs_frame_to_image

        (page,) = _write_pages(tmp_path / "pages", 1)
        vs = get_vapoursynth()

        def pull(strength: float | None) -> "np.ndarray":
            chain = vs_deblur_clip(
                [page],
                method=1,
                radius=0.8,
                strength=strength,
                iterations=6,
                threshold=2.0,
                overshoot=0.0,
                core=vs.core,
            )
            with chain.deblurred.get_frame(0) as frame:
                return np.asarray(vs_frame_to_image(frame)).copy()

        # 0.85 is the unsharp mask's documented default, so leaving it off has to match it.
        assert np.array_equal(pull(None), pull(0.85))
        assert not np.array_equal(pull(None), pull(0.2))

    def test_frame_to_image_stacks_all_three_planes(self, tmp_path: Path):
        """
        The deleted `vs_frame_to_rgb_image` was the only reader that stacked the planes.

        `vs_frame_to_image` does it now, so the colour write path cannot silently write
        plane 0 as a gray page: the three channels are three different planes.
        """
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_deblur_clip, vs_frame_to_image

        page = tmp_path / "colour.png"
        _colour_gradient().save(page)
        vs = get_vapoursynth()
        chain = vs_deblur_clip(
            [page],
            method=1,
            radius=0.8,
            strength=None,
            iterations=6,
            threshold=2.0,
            overshoot=0.0,
            core=vs.core,
        )
        with chain.deblurred.get_frame(0) as frame:
            image = vs_frame_to_image(frame)
            planes = [np.asarray(frame[plane]).copy() for plane in range(3)]

        assert image.mode == "RGB"
        actual = np.asarray(image)
        for plane, expected in enumerate(planes):
            assert np.array_equal(actual[..., plane], expected)
        assert not np.array_equal(actual[..., 0], actual[..., 1])


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
        chain = vs_posterize_clip(pages, bits=4, method=0, core=vs.core)

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

    def test_the_raw_branches_are_the_eight_bit_posterize_outputs(self, tmp_path: Path):
        """The writer stores `group.raw`, so the bytes it publishes are the plugin's own."""
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_posterize_group

        pages = _write_pages(tmp_path / "pages", 2)
        vs = get_vapoursynth()
        group = vs_posterize_group(pages, bits_candidates=[2, 4], method=0, core=vs.core)

        assert set(group.raw) == {2, 4}
        for node in group.raw.values():
            assert node.num_frames == len(pages)
            for index, page in enumerate(pages):
                with node.get_frame(index) as frame:
                    assert frame.format.name == "Gray8"
                    assert frame.props["ImgSeqPath"] == str(page)

    def test_mixed_containers_of_one_size_convert(self, tmp_path: Path):
        """
        `vs_posterize_group` sits on `vs_to_gray8`, and its clips carry the size the
        undefined-size decode cannot report to `vship`.

        Both pages are 65x33 so `vship` gets one size: the png and the lossless webp are
        both `RGB24`, which is the odd-width page this plugin can actually hand out."""
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_posterize_group, vs_score_candidates

        odd_webp = _tmp_odd_webp(tmp_path)
        odd_png = tmp_path / "odd.png"
        _gradient(width=65, height=33).save(odd_png)

        vs = get_vapoursynth()
        group = vs_posterize_group([odd_webp, odd_png], bits_candidates=[2, 4], method=0, core=vs.core)

        assert group.size == (65, 33)
        with group.gray.get_frame(0) as frame:
            assert (frame.width, frame.height, frame.format.name) == (65, 33, "Gray8")
        # Every branch is built on the same clip, and the raw `GRAY8` frame is what is written.
        with group.branches[4].get_frame(0) as frame:
            assert frame.format.name == "GrayS"

        scores = vs_score_candidates(group, metrics=["ssimulacra2"], core=vs.core)
        assert np.all(scores["ssimulacra2"][4] > scores["ssimulacra2"][2])

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

    # The command aborts without these, and the abort is what the plugin-less CI saw as an
    # exit code of 1, so the execution is gated exactly like the other plugin-backed tests.
    @requires_nimages
    @requires_imagseqs
    @requires_vship
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
                str(tmp_path / "pages"),
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
                str(tmp_path / "pages"),
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

    # Scoring is off here, so the fixed depth path only needs the two decode plugins.
    @requires_nimages
    @requires_imagseqs
    def test_one_depth_without_a_metric_scores_nothing(self, tmp_path: Path):
        """`--bits 4` on its own stays the fixed depth posterize it always was."""
        from click.testing import CliRunner

        from nmanga.cmd import main

        _write_pages(tmp_path / "pages", 2)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["posterize2", str(tmp_path / "pages"), "-o", str(tmp_path / "out"), "-t", "1", "--bits", "4"],
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

    def test_deblur2_is_on_the_imops_group(self):
        from nmanga.cmd import main

        commands = set(main.commands["imops"].commands)  # pyright: ignore[reportAttributeAccessIssue]
        assert "deblur2" in commands
        # The Pillow command it sits beside has to stay reachable.
        assert "deblur" in commands

    def test_deblur2_has_prefetch_threads_cache_and_force(self):
        from click.testing import CliRunner

        from nmanga.cmd import main

        runner = CliRunner()
        result = runner.invoke(main, ["imops", "deblur2", "--help"])
        assert result.exit_code == 0
        for flag in ("--method", "--radius", "--strength", "--iterations", "--threshold", "--overshoot"):
            assert flag in result.output
        assert "--prefetch" in result.output
        assert "--threads" in result.output
        assert "--cache" in result.output
        assert "--force" in result.output
        # The plugin resolves the per method default, so the option is not defaulted here.
        assert "Default: 0.65" not in result.output

    @requires_nimages
    @requires_imagseqs
    def test_deblur2_writes_a_png_per_page(self, tmp_path: Path):
        from click.testing import CliRunner

        from nmanga.cmd import main

        pages = _write_pages(tmp_path / "pages", 2)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["imops", "deblur2", str(tmp_path / "pages"), "-o", str(tmp_path / "out"), "-t", "1", "--method", "edge"],
        )
        assert result.exit_code == 0, result.output
        assert "Deblurred 2 images." in result.output
        for page in pages:
            written = tmp_path / "out" / f"{page.stem}.png"
            assert written.exists()
            with Image.open(written) as opened:
                assert opened.format == "PNG"


class _FakeFrame:
    """The context manager `PageWrite.result()` receives from one request."""

    def __init__(self, props: dict | None = None):
        self.props = props or {}

    def __enter__(self) -> "_FakeFrame":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeRequest:
    """One `get_frame_async` result: optionally fails, optionally hands back props."""

    def __init__(self, error: Exception | None = None, props: dict | None = None):
        self._error = error
        self._props = props

    def result(self) -> _FakeFrame:
        if self._error is not None:
            raise self._error
        return _FakeFrame(self._props)


class _FakeWriter:
    """Records every requested index and hands back the prepared requests in order."""

    def __init__(self, *requests: _FakeRequest):
        self._requests = list(requests)
        self.requested: list[int] = []

    def get_frame_async(self, index: int) -> _FakeRequest:
        self.requested.append(index)
        return self._requests.pop(0)


class TestPngWriteWindow:
    """The bounded replacement for `BoundedWritePool`, without a plugin or an encode."""

    def test_hands_back_the_oldest_page_once_full(self):
        from nmanga.vapour import PngWriteWindow

        writer = _FakeWriter(_FakeRequest(), _FakeRequest(), _FakeRequest(), _FakeRequest())
        with PngWriteWindow(3) as window:
            assert window.submit(writer, 0, "first") is None
            assert window.submit(writer, 1, "second") is None
            # The third request fills the window, so the first page comes back to be waited on.
            oldest = window.submit(writer, 2, "third")
            assert oldest is not None
            assert oldest.index == 0
            assert oldest.result() == "first"
            assert writer.requested == [0, 1, 2]
            assert [page.result() for page in window.pending()] == ["second", "third"]

    def test_hands_pages_back_in_order(self):
        """The bound is only a memory bound if the oldest page is the one returned."""
        from nmanga.vapour import PngWriteWindow

        writer = _FakeWriter(_FakeRequest(), _FakeRequest(), _FakeRequest())
        with PngWriteWindow(2) as window:
            assert window.submit(writer, 10, 10) is None
            second = window.submit(writer, 11, 11)
            assert second is not None
            assert second.index == 10
            assert second.result() == 10
            third = window.submit(writer, 12, 12)
            assert third is not None and third.index == 11
            assert third.result() == 11
            assert [page.result() for page in window.pending()] == [12]

    def test_result_returns_the_outcome_and_leaves_the_destination(self, tmp_path):
        """`output_dir` already published the file; `result()` must not touch it again."""
        from nmanga.vapour import PngWriteWindow

        destination = tmp_path / "page.png"
        destination.write_bytes(b"written by the writer")
        writer = _FakeWriter(_FakeRequest())

        with PngWriteWindow(1) as window:
            page = window.submit(writer, 0, "PROCESSED")
            assert page is not None
            assert page.result() == "PROCESSED"

        assert destination.read_bytes() == b"written by the writer"

    def test_result_raises_the_writers_error(self):
        from nmanga.vapour import PngWriteWindow

        writer = _FakeWriter(_FakeRequest(error=ValueError("write failed")))
        with pytest.raises(ValueError, match="write failed"):
            with PngWriteWindow(1) as window:
                page = window.submit(writer, 0, "nope")
                assert page is not None
                page.result()

    def test_a_clean_exit_with_a_request_still_open_raises(self):
        from nmanga.vapour import PngWriteWindow

        writer = _FakeWriter(_FakeRequest())
        window = PngWriteWindow(2)
        with pytest.raises(RuntimeError, match="still open"):
            with window:
                window.submit(writer, 0, "open")
        assert [page.index for page in window.pending()] == [0]

    def test_an_error_exit_discards_the_open_requests(self):
        """The request is waited on without being reported, and the original error wins."""
        from nmanga.vapour import PngWriteWindow

        writer = _FakeWriter(_FakeRequest(error=ValueError("encode failed")))
        with pytest.raises(ValueError, match="boom"):
            with PngWriteWindow(2) as window:
                window.submit(writer, 0, "never")
                raise ValueError("boom")

    def test_depth_is_at_least_one(self):
        from nmanga.vapour import PngWriteWindow

        writer = _FakeWriter(_FakeRequest())
        window = PngWriteWindow(0)
        assert window.depth == 1
        assert window.submit(writer, 0, "first") is not None


@requires_imagseqs
class TestPngWriter:
    """`output_dir` names each still from its `ImgSeqPath`; the plugin owns the publish."""

    def _read(self, pages: list[Path]):
        from nmanga.lazy import get_vapoursynth

        vs = get_vapoursynth()
        return vs, vs.core.imgseqs.Read(files=[str(page) for page in pages], mismatch=True, prefetch=0)

    def test_a_still_is_named_from_its_source_stem(self, tmp_path: Path):
        import numpy as np

        from nmanga.vapour import vs_png_writer

        pages = _write_pages(tmp_path / "pages", 2)
        _vs, source = self._read(pages)
        output = tmp_path / "out"
        writer = vs_png_writer(source, output)
        assert writer.num_frames == len(pages)

        for index, page in enumerate(pages):
            with source.get_frame(index) as source_frame:
                width, height = source_frame.width, source_frame.height
                planes = [np.asarray(source_frame[plane]).copy() for plane in range(source_frame.format.num_planes)]
            with writer.get_frame(index) as written:
                # The output frame is the input frame plus the write; it keeps its planes.
                assert written.width == width
                assert written.height == height
                assert written.props["ImgSeqPath"] == str(page)
                assert int(written.props["ImgSeqPNGWritePerformed"]) == 1
                assert written.props["ImgSeqPNGWriteSaved"]

            destination = output / f"{page.stem}.png"
            assert destination.exists()
            with Image.open(destination) as opened:
                if len(planes) == 1:
                    assert opened.mode == "L"
                    actual = np.asarray(opened)
                else:
                    assert opened.mode == "RGB"
                    actual = np.asarray(opened)
            assert np.array_equal(actual, planes[0] if len(planes) == 1 else np.stack(planes, axis=-1))

    def test_the_output_directory_is_created_when_the_node_is_built(self, tmp_path: Path):
        from nmanga.vapour import vs_png_writer

        pages = _write_pages(tmp_path / "pages", 1)
        _vs, source = self._read(pages)
        output = tmp_path / "does-not-exist-yet"
        assert not output.exists()

        vs_png_writer(source, output)
        # The directory exists, but no PNG is written until a frame is requested.
        assert output.is_dir()
        assert list(output.iterdir()) == []

    def test_overwrite_false_refuses_an_existing_destination(self, tmp_path: Path):
        from nmanga.vapour import vs_png_writer

        pages = _write_pages(tmp_path / "pages", 1)
        _vs, source = self._read(pages)
        output = tmp_path / "out"
        output.mkdir()
        destination = output / f"{pages[0].stem}.png"
        destination.write_bytes(b"keep me")

        refusing = vs_png_writer(source, output, overwrite=False)
        with pytest.raises(Exception, match="already exists"):
            with refusing.get_frame(0):
                pass
        assert destination.read_bytes() == b"keep me"

        # `overwrite` is the permission, so the same node replaces it when given.
        replacing = vs_png_writer(source, output, overwrite=True)
        with replacing.get_frame(0):
            pass
        assert destination.read_bytes() != b"keep me"
        with Image.open(destination) as opened:
            assert opened.format == "PNG"

    def test_a_frame_without_imgseqpath_is_refused(self, tmp_path: Path):
        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_png_writer

        vs = get_vapoursynth()
        blank = vs.core.std.BlankClip(width=8, height=8, format=vs.GRAY8, length=1)
        writer = vs_png_writer(blank, tmp_path / "out")
        with pytest.raises(Exception, match="ImgSeqPath"):
            with writer.get_frame(0):
                pass

    def test_two_frames_with_one_destination_are_refused(self, tmp_path: Path):
        """One writer never lets a second page silently replace the first."""
        from nmanga.vapour import vs_png_writer

        first = _write_pages(tmp_path / "one", 1)[0]
        second = _write_pages(tmp_path / "two", 1)[0]
        assert first.name == second.name
        _vs, source = self._read([first, second])
        writer = vs_png_writer(source, tmp_path / "out")

        with writer.get_frame(0):
            pass
        with pytest.raises(Exception, match="same destination"):
            with writer.get_frame(1):
                pass

    def test_the_window_settles_a_real_writer_request(self, tmp_path: Path):
        from nmanga.vapour import PngWriteWindow, vs_png_writer

        pages = _write_pages(tmp_path / "pages", 2)
        _vs, source = self._read(pages)
        output = tmp_path / "out"
        writer = vs_png_writer(source, output)

        with PngWriteWindow(1) as window:
            page = window.submit(writer, 0, "PROCESSED")
            assert page is not None
            assert page.result() == "PROCESSED"

        assert (output / f"{pages[0].stem}.png").exists()


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


def _staging_leftovers(root: Path) -> list[Path]:
    """A `PngWriteStaging` directory was named like this; no route may leave one now."""
    return sorted(path for path in root.rglob(".nmanga-pngwrite-*") if path.is_dir())


def _flat_pages(directory: Path, count: int = 1, *, value: int = 128, mode: str = "L") -> list[Path]:
    """Flat pages: no black peak and a white of 255, so `autolevel3` skips them."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in range(count):
        path = directory / f"flat_{index:03d}.png"
        Image.new(mode, (64, 48), value if mode == "L" else (value, value, value)).save(path)
        written.append(path)
    return written


def _colour_level_page(directory: Path) -> Path:
    """A colour page whose luma is `_gradient`'s, so it has usable levels and stays RGB."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "colour.png"
    Image.merge("RGB", (_gradient(), _gradient(shift=5), _gradient(shift=11))).save(path)
    return path


def _vulkan_device_available() -> bool:
    """Whether `nimages` can build and run a `gpu=1` node, which needs Vulkan 1.4."""
    try:
        from nmanga.lazy import get_vapoursynth

        vs = get_vapoursynth()
        clip = vs.core.std.BlankClip(width=16, height=16, format=vs.GRAY8, length=1)
        node = vs.core.nimages.PeakStats(clip, upper_limit=60, peak_percentage=0.25, gpu=1)
        with node.get_frame(0):
            pass
    except Exception:
        return False
    return True


HAS_VULKAN = _vulkan_device_available()
requires_vulkan = pytest.mark.skipif(not HAS_VULKAN, reason="no usable Vulkan 1.4 device")


class TestDuplicateStemReporting:
    """`{stem}.png` is the destination for several routes, so the volume is checked once first."""

    def test_unique_stems_are_not_reported(self, tmp_path: Path):
        from nmanga.vapour import vs_duplicate_png_stems

        pages = _write_pages(tmp_path / "pages", 3)
        assert vs_duplicate_png_stems(pages) == {}

    def test_a_jpg_beside_a_png_is_one_destination(self, tmp_path: Path):
        from nmanga.vapour import vs_duplicate_png_stems

        directory = tmp_path / "pages"
        directory.mkdir()
        jpg = directory / "page.jpg"
        png = directory / "page.png"
        other = directory / "other.png"
        _gradient().save(jpg)
        _gradient().save(png)
        _gradient().save(other)

        duplicates = vs_duplicate_png_stems([jpg, png, other])
        assert set(duplicates) == {"page"}
        assert set(duplicates["page"]) == {jpg, png}

    def test_posterize2_uses_the_same_decision(self, tmp_path: Path):
        from nmanga.cli.posterize import _duplicate_destination_stems

        directory = tmp_path / "pages"
        directory.mkdir()
        jpg = directory / "page.jpg"
        png = directory / "page.png"
        _gradient().save(jpg)
        _gradient().save(png)
        assert _duplicate_destination_stems([jpg, png]) == {"page": [jpg, png]}

    @requires_nimages
    @requires_imagseqs
    def test_posterize2_refuses_before_the_first_write(self, tmp_path: Path):
        from click.testing import CliRunner

        from nmanga.cmd import main

        directory = tmp_path / "pages"
        directory.mkdir()
        _gradient().save(directory / "page.jpg")
        _gradient().save(directory / "page.png")
        output = tmp_path / "out"

        result = CliRunner().invoke(
            main,
            ["posterize2", str(directory), "-o", str(output), "-t", "1", "--bits", "4"],
        )
        assert result.exit_code != 0
        assert "share one output name" in result.output
        assert "would all write page.png" in result.output
        # Refused before anything was built or written.
        assert not output.exists() or list(output.iterdir()) == []


@requires_nimages
@requires_imagseqs
class TestDeblur2Acceptance:
    """`deblur2` writes straight to `output_dir`; there is no staging and no Pillow encode."""

    def _run(self, source: Path, output: Path, *extra: str):
        from click.testing import CliRunner

        from nmanga.cmd import main

        return CliRunner().invoke(
            main,
            ["imops", "deblur2", str(source), "-o", str(output), "-t", "1", "--method", "edge", *extra],
        )

    def _expected(self, page: Path):
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_deblur_clip, vs_frame_to_image

        vs = get_vapoursynth()
        chain = vs_deblur_clip(
            [page],
            method=1,
            radius=0.8,
            strength=0.85,
            iterations=6,
            threshold=2.0,
            overshoot=0.0,
            core=vs.core,
        )
        with chain.deblurred.get_frame(0) as frame:
            return np.asarray(vs_frame_to_image(frame)).copy()

    def test_writes_pixel_for_pixel_without_staging(self, tmp_path: Path):
        import numpy as np

        pages = _write_pages(tmp_path / "pages", 2)
        output = tmp_path / "out"
        result = self._run(tmp_path / "pages", output)
        assert result.exit_code == 0, result.output
        assert "Deblurred 2 images." in result.output

        assert sorted(path.name for path in output.iterdir()) == sorted(f"{page.stem}.png" for page in pages)
        assert _staging_leftovers(tmp_path) == []

        for page in pages:
            expected = self._expected(page)
            with Image.open(output / f"{page.stem}.png") as opened:
                assert opened.format == "PNG"
                assert np.array_equal(np.asarray(opened), expected)

    def test_skip_without_force_and_replace_with_force(self, tmp_path: Path):
        pages = _write_pages(tmp_path / "pages", 2)
        output = tmp_path / "out"
        assert self._run(tmp_path / "pages", output).exit_code == 0

        destination = output / f"{pages[0].stem}.png"
        destination.write_bytes(b"keep me")
        skipped = self._run(tmp_path / "pages", output)
        assert skipped.exit_code == 0, skipped.output
        assert "Skipping existing file" in skipped.output
        assert destination.read_bytes() == b"keep me"

        forced = self._run(tmp_path / "pages", output, "--force")
        assert forced.exit_code == 0, forced.output
        assert destination.read_bytes() != b"keep me"
        with Image.open(destination) as opened:
            assert opened.format == "PNG"

    def test_a_16_bit_source_writes_a_16_bit_png(self, tmp_path: Path):
        import numpy as np

        pages = tmp_path / "pages"
        pages.mkdir()
        source = pages / "page_000.png"
        array = np.arange(64 * 48, dtype=np.uint16).reshape(48, 64) * 7
        Image.fromarray(array).save(source)

        output = tmp_path / "out"
        result = self._run(pages, output)
        assert result.exit_code == 0, result.output

        # The old `vs_frame_to_rgb_image` path raised on a 16-bit frame; the writer stores it.
        with Image.open(output / "page_000.png") as opened:
            assert opened.mode == "I;16"
            decoded = np.asarray(opened)
        assert decoded.dtype == np.uint16
        assert decoded.shape == array.shape
        assert decoded.max() > 255


@requires_nimages
@requires_imagseqs
class TestPosterize2Acceptance:
    """Processed pages land directly in `output_dir`; copied pages keep their source bytes."""

    def _run(self, source: Path, output: Path, *extra: str):
        from click.testing import CliRunner

        from nmanga.cmd import main

        return CliRunner().invoke(
            main,
            ["posterize2", str(source), "-o", str(output), "-t", "1", "--bits", "4", "-m", "even", *extra],
        )

    def test_a_processed_page_lands_directly_and_matches_the_plugin(self, tmp_path: Path):
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_frame_to_image, vs_posterize_clip

        pages = _write_pages(tmp_path / "pages", 2)
        output = tmp_path / "out"
        result = self._run(tmp_path / "pages", output)
        assert result.exit_code == 0, result.output
        assert "Posterized 2 images to 4 bits." in result.output

        assert sorted(path.name for path in output.iterdir()) == sorted(f"{page.stem}.png" for page in pages)
        assert _staging_leftovers(tmp_path) == []

        vs = get_vapoursynth()
        for page in pages:
            chain = vs_posterize_clip([page], bits=4, method=0, core=vs.core)
            with chain.posterized.get_frame(0) as frame:
                expected = np.asarray(vs_frame_to_image(frame)).copy()
            with Image.open(output / f"{page.stem}.png") as opened:
                assert np.array_equal(np.asarray(opened), expected)

    def test_a_copied_page_keeps_its_suffix_and_bytes(self, tmp_path: Path):
        directory = tmp_path / "pages"
        directory.mkdir()
        jpg = directory / "page_000.jpg"
        png = directory / "page_001.png"
        _gradient().save(jpg)
        _gradient(shift=9).save(png)
        output = tmp_path / "out"

        result = self._run(
            directory,
            output,
            "--metrics",
            "ssimulacra2",
            "--metric-min",
            "ssimulacra2=100",
        )
        assert result.exit_code == 0, result.output
        assert "Copied 2 images without posterization." in result.output
        assert (output / "page_000.jpg").read_bytes() == jpg.read_bytes()
        assert (output / "page_001.png").read_bytes() == png.read_bytes()

    def test_a_selected_page_overwrites_an_existing_png(self, tmp_path: Path):
        import numpy as np

        pages = _write_pages(tmp_path / "pages", 1)
        output = tmp_path / "out"
        output.mkdir()
        destination = output / f"{pages[0].stem}.png"
        destination.write_bytes(b"the old Pillow save would have replaced this")

        result = self._run(tmp_path / "pages", output)
        assert result.exit_code == 0, result.output
        assert destination.read_bytes() != b"the old Pillow save would have replaced this"
        with Image.open(destination) as opened:
            assert opened.format == "PNG"
            assert np.asarray(opened).ndim == 2

    @requires_vship
    def test_the_fan_out_writes_processed_pages_directly(self, tmp_path: Path):
        import numpy as np
        from click.testing import CliRunner

        from nmanga.cmd import main

        pages = _write_pages(tmp_path / "pages", 2)
        output = tmp_path / "out"
        result = CliRunner().invoke(
            main,
            [
                "posterize2",
                str(tmp_path / "pages"),
                "-o",
                str(output),
                "-t",
                "1",
                "--bits",
                "2-4",
                "--metrics",
                "ssimulacra2",
                "--metric-min",
                "ssimulacra2=0",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Posterized 2 images" in result.output
        assert sorted(path.name for path in output.iterdir()) == sorted(f"{page.stem}.png" for page in pages)
        assert _staging_leftovers(tmp_path) == []
        for page in pages:
            with Image.open(output / f"{page.stem}.png") as opened:
                decoded = np.asarray(opened)
            with Image.open(page) as source:
                assert not np.array_equal(decoded, np.asarray(source.convert("L")))
                assert len(np.unique(decoded)) <= 16


@requires_nimages
@requires_imagseqs
class TestAutolevel3Acceptance:
    """The PNG route decides first and writes straight to the destination; no staging."""

    def _run(self, source: Path, output: Path, *extra: str):
        from click.testing import CliRunner

        from nmanga.cmd import main

        return CliRunner().invoke(
            main,
            ["autolevel3", str(source), "-o", str(output), "-t", "1", *extra],
        )

    def _expected(self, page: Path):
        import numpy as np

        from nmanga.lazy import get_vapoursynth
        from nmanga.vapour import vs_autolevel_clip, vs_frame_to_image

        vs = get_vapoursynth()
        chain = vs_autolevel_clip([page], options=_config(), core=vs.core)
        with chain.leveled.get_frame(0) as frame:
            return np.asarray(vs_frame_to_image(frame)).copy()

    def test_a_levelled_page_is_written_directly_without_staging(self, tmp_path: Path):
        import numpy as np

        pages = _write_pages(tmp_path / "pages", 2)
        output = tmp_path / "out"
        result = self._run(tmp_path / "pages", output)
        assert result.exit_code == 0, result.output
        assert "Autoleveled 2 images." in result.output
        assert sorted(path.name for path in output.iterdir()) == sorted(f"{page.stem}.png" for page in pages)
        assert _staging_leftovers(tmp_path) == []

        for page in pages:
            with Image.open(output / f"{page.stem}.png") as opened:
                assert np.array_equal(np.asarray(opened), self._expected(page))

    def test_a_skipped_page_is_copied_byte_for_byte_in_colour_mode(self, tmp_path: Path):
        (page,) = _flat_pages(tmp_path / "pages", 1)
        output = tmp_path / "out"
        result = self._run(tmp_path / "pages", output)
        assert result.exit_code == 0, result.output
        assert "Copied 1 images without autolevel." in result.output
        assert (output / page.name).read_bytes() == page.read_bytes()

    def test_gray_mode_re_encodes_a_skipped_colour_page(self, tmp_path: Path):
        (page,) = _flat_pages(tmp_path / "pages", 1, mode="RGB")
        output = tmp_path / "out"
        result = self._run(tmp_path / "pages", output, "--gray")
        assert result.exit_code == 0, result.output
        assert "Grayscaled 1 images." in result.output
        written = output / page.with_suffix(".png").name
        with Image.open(written) as opened:
            assert opened.mode == "L"
        assert written.read_bytes() != page.read_bytes()

    def test_gray_mode_copies_an_already_gray_skipped_page(self, tmp_path: Path):
        (page,) = _flat_pages(tmp_path / "pages", 1)
        output = tmp_path / "out"
        result = self._run(tmp_path / "pages", output, "--gray")
        assert result.exit_code == 0, result.output
        assert "Copied 1 images without autolevel." in result.output
        assert (output / page.name).read_bytes() == page.read_bytes()

    def test_colour_mode_keeps_a_levelled_colour_page_in_rgb(self, tmp_path: Path):
        page = _colour_level_page(tmp_path / "pages")
        output = tmp_path / "out"
        result = self._run(tmp_path / "pages", output)
        assert result.exit_code == 0, result.output
        assert "Autoleveled 1 images." in result.output
        with Image.open(output / page.name) as opened:
            assert opened.mode == "RGB"

    def test_an_existing_destination_is_not_replaced(self, tmp_path: Path):
        pages = _write_pages(tmp_path / "pages", 1)
        output = tmp_path / "out"
        assert self._run(tmp_path / "pages", output).exit_code == 0

        existing = output / f"{pages[0].stem}.png"
        existing.write_bytes(b"keep me")
        rerun = self._run(tmp_path / "pages", output)
        assert rerun.exit_code == 0, rerun.output
        assert "Skipping existing file" in rerun.output
        assert existing.read_bytes() == b"keep me"

    def test_a_mixed_png_and_webp_directory_at_an_odd_width(self, tmp_path: Path):
        directory = tmp_path / "pages"
        directory.mkdir()
        _gradient(width=65, height=33).save(directory / "odd.png")
        _gradient(width=66, height=34).save(directory / "even.webp", quality=90)
        output = tmp_path / "out"

        # `--gray` sends the YUV webp through the analysis clip and writes both as one
        # plane, so the odd png and the even webp exercise the per-frame probe.
        result = self._run(directory, output, "--gray")
        assert result.exit_code == 0, result.output
        assert sorted(path.name for path in output.iterdir()) == ["even.png", "odd.png"]
        with Image.open(output / "odd.png") as odd:
            assert (odd.mode, odd.size) == ("L", (65, 33))
        with Image.open(output / "even.png") as even:
            assert (even.mode, even.size) == ("L", (66, 34))


class TestPngSaveGuard:
    """A PNG page that still goes through Pillow fails loudly here."""

    def _forbid_pillow_save(self, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("a PNG page went through PIL.Image.Image.save")

        monkeypatch.setattr(Image.Image, "save", fail)

    @requires_nimages
    @requires_imagseqs
    def test_deblur2_never_calls_pillow_save(self, tmp_path: Path, monkeypatch):
        from click.testing import CliRunner

        from nmanga.cmd import main

        pages = _write_pages(tmp_path / "pages", 2)
        self._forbid_pillow_save(monkeypatch)
        result = CliRunner().invoke(
            main,
            ["imops", "deblur2", str(tmp_path / "pages"), "-o", str(tmp_path / "out"), "-t", "1"],
        )
        assert result.exit_code == 0, result.output + repr(result.exception)
        assert len(list((tmp_path / "out").iterdir())) == len(pages)

    @requires_nimages
    @requires_imagseqs
    def test_posterize2_fixed_depth_never_calls_pillow_save(self, tmp_path: Path, monkeypatch):
        from click.testing import CliRunner

        from nmanga.cmd import main

        _write_pages(tmp_path / "pages", 2)
        self._forbid_pillow_save(monkeypatch)
        result = CliRunner().invoke(
            main,
            ["posterize2", str(tmp_path / "pages"), "-o", str(tmp_path / "out"), "-t", "1", "--bits", "4"],
        )
        assert result.exit_code == 0, result.output + repr(result.exception)

    @requires_nimages
    @requires_imagseqs
    @requires_vship
    def test_posterize2_fan_out_never_calls_pillow_save(self, tmp_path: Path, monkeypatch):
        from click.testing import CliRunner

        from nmanga.cmd import main

        _write_pages(tmp_path / "pages", 2)
        self._forbid_pillow_save(monkeypatch)
        result = CliRunner().invoke(
            main,
            [
                "posterize2",
                str(tmp_path / "pages"),
                "-o",
                str(tmp_path / "out"),
                "-t",
                "1",
                "--bits",
                "2-4",
                "--metrics",
                "ssimulacra2",
                "--metric-min",
                "ssimulacra2=0",
            ],
        )
        assert result.exit_code == 0, result.output + repr(result.exception)

    @requires_nimages
    @requires_imagseqs
    def test_autolevel3_png_never_calls_pillow_save(self, tmp_path: Path, monkeypatch):
        from click.testing import CliRunner

        from nmanga.cmd import main

        _write_pages(tmp_path / "pages", 2)
        self._forbid_pillow_save(monkeypatch)
        for index, mode in enumerate(((), ("--gray",))):
            output = tmp_path / f"out-{index}"
            result = CliRunner().invoke(
                main,
                ["autolevel3", str(tmp_path / "pages"), "-o", str(output), "-t", "1", *mode],
            )
            assert result.exit_code == 0, result.output + repr(result.exception)
            assert len(list(output.iterdir())) == 2


class TestCommandSurface:
    """The flags and deleted helpers the migration is supposed to leave behind."""

    def test_autolevel3_replaced_force_gray_and_keep_colorspace(self):
        from click.testing import CliRunner

        from nmanga.cmd import main

        runner = CliRunner()
        help_text = runner.invoke(main, ["autolevel3", "--help"]).output
        assert "--color" in help_text
        assert "--gray" in help_text
        assert "--gpu" in help_text
        assert "Vulkan" in help_text
        assert "--force-gray" not in help_text
        assert "--keep-colorspace" not in help_text
        for flag in ("--force-gray", "--keep-colorspace"):
            result = runner.invoke(main, ["autolevel3", ".", "-o", ".", flag])
            assert result.exit_code != 0

    def test_posterize2_exposes_gpu(self):
        from click.testing import CliRunner

        from nmanga.cmd import main

        help_text = CliRunner().invoke(main, ["posterize2", "--help"]).output
        assert "--gpu" in help_text
        assert "Vulkan" in help_text

    def test_the_pillow_png_write_helpers_are_gone(self):
        import nmanga.cli.image_operations as image_operations
        import nmanga.cli.posterize as posterize

        assert not hasattr(image_operations, "_save_deblur_png")
        for name in ("_save_png", "_submit_fallback", "_submit_selection"):
            assert not hasattr(posterize, name)

    def test_the_old_colour_helpers_are_gone_from_vapour(self):
        import nmanga.vapour as vapour

        assert not hasattr(vapour, "level_color_page")
        assert not hasattr(vapour, "vs_frame_to_rgb_image")
        assert hasattr(vapour, "vs_grays_frame_to_image")


class TestGpuFlag:
    """`--gpu` must fail with the plugin's own message, never fall back to the host."""

    @requires_nimages
    @requires_imagseqs
    @pytest.mark.skipif(HAS_VULKAN, reason="this machine has a usable Vulkan device")
    def test_autolevel3_gpu_without_a_device_reports_the_plugin_message(self, tmp_path: Path):
        from click.testing import CliRunner

        from nmanga.cmd import main

        _write_pages(tmp_path / "pages", 1)
        result = CliRunner().invoke(
            main,
            ["autolevel3", str(tmp_path / "pages"), "-o", str(tmp_path / "out"), "-t", "1", "--gpu"],
        )
        assert result.exit_code != 0
        assert "no usable Vulkan device" in result.output + repr(result.exception)

    @requires_nimages
    @requires_imagseqs
    @pytest.mark.skipif(HAS_VULKAN, reason="this machine has a usable Vulkan device")
    def test_posterize2_gpu_without_a_device_reports_the_plugin_message(self, tmp_path: Path):
        from click.testing import CliRunner

        from nmanga.cmd import main

        _write_pages(tmp_path / "pages", 1)
        result = CliRunner().invoke(
            main,
            ["posterize2", str(tmp_path / "pages"), "-o", str(tmp_path / "out"), "-t", "1", "--gpu"],
        )
        assert result.exit_code != 0
        assert "no usable Vulkan device" in result.output + repr(result.exception)

    @requires_nimages
    @requires_imagseqs
    @requires_vulkan
    def test_autolevel3_gpu_matches_the_host_route(self, tmp_path: Path):
        import numpy as np

        pages = _write_pages(tmp_path / "pages", 2)
        host = tmp_path / "host"
        device = tmp_path / "device"
        assert self._run_autolevel3(tmp_path / "pages", host).exit_code == 0
        assert self._run_autolevel3(tmp_path / "pages", device, "--gpu").exit_code == 0
        for page in pages:
            name = f"{page.stem}.png"
            with Image.open(host / name) as host_image:
                expected = np.asarray(host_image)
            with Image.open(device / name) as device_image:
                assert np.array_equal(np.asarray(device_image), expected)

    @staticmethod
    def _run_autolevel3(source: Path, output: Path, *extra: str):
        from click.testing import CliRunner

        from nmanga.cmd import main

        return CliRunner().invoke(
            main,
            ["autolevel3", str(source), "-o", str(output), "-t", "1", *extra],
        )

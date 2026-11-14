"""
MIT License

Copyright (c) 2022-present noaione, anon

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

# Regression tests for histogram peak selection and image ownership.

from __future__ import annotations

from io import BytesIO
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nmanga.autolevel import _find_histogram_peak, find_local_peak


def _image_from_counts(counts: dict[int, int]) -> Image.Image:
    pixels = bytearray()
    for shade, count in counts.items():
        pixels.extend(bytes([shade]) * count)
    return Image.frombytes("L", (len(pixels), 1), pixels)


def _tallest_plateau(hist: list[int]) -> int | None:
    tallest = max(hist, default=0)
    if tallest == 0:
        return None
    start = hist.index(tallest)
    end = start
    while end + 1 < len(hist) and hist[end + 1] == tallest:
        end += 1
    return (start + end) // 2


def test_histogram_peak_matches_exhaustive_plateau_oracle() -> None:
    # With zero-padded nonnegative counts, a tallest peak either passes or wins the fallback.
    for length in range(9):
        for counts in product(range(3), repeat=length):
            hist = list(counts)
            expected = _tallest_plateau(hist)
            for min_height, min_prominence in ((0, 0), (1, 1), (2, 0), (0, 2), (100, 100)):
                assert _find_histogram_peak(hist, min_height, min_prominence) == expected, (
                    hist,
                    min_height,
                    min_prominence,
                )


@pytest.mark.parametrize("skip_white_check", [False, True])
def test_find_local_peak_matches_generated_histograms(skip_white_check: bool) -> None:
    random = np.random.default_rng(20261114)
    for _ in range(256):
        hist = random.integers(0, 6, size=256).tolist()
        upper_limit = int(random.choice([-1, 0, 1, 30, 60, 127, 254, 255, 256, 300]))
        peak_percentage = random.choice([None, -1, 0, 0.25, 50, 100, 200])
        peak_prominence = random.choice([None, -1, 0, 0.1, 50, 100, 200])
        image = _image_from_counts(dict(enumerate(hist)))

        black_level, white_level = 0, 255
        if upper_limit >= 0:
            black_peak = _tallest_plateau(hist[: min(256, upper_limit + 1)])
            if black_peak is not None:
                black_level = black_peak
            if not skip_white_check:
                white_start = max(0, 255 - upper_limit)
                white_peak = _tallest_plateau(hist[white_start:])
                if white_peak is not None:
                    white_level = white_start + white_peak

        assert find_local_peak(
            image,
            upper_limit=upper_limit,
            peak_percentage=peak_percentage,
            peak_prominence=peak_prominence,
            skip_white_check=skip_white_check,
        ) == (black_level, white_level, False)


@pytest.mark.parametrize(
    ("counts", "upper_limit", "expected"),
    [
        ({5: 5, 20: 10, 220: 15, 240: 30}, 60, (20, 240)),
        ({0: 12, 60: 20, 195: 25, 255: 30}, 60, (60, 255)),
        ({0: 20, 60: 12, 195: 30, 255: 25}, 60, (0, 195)),
        ({60: 5, 61: 500, 194: 600, 195: 5}, 60, (60, 195)),
        ({10: 10, 11: 10, 12: 10, 220: 10, 221: 10}, 60, (11, 220)),
        ({0: 10, 1: 10, 59: 20, 60: 20, 195: 20, 196: 20, 254: 10, 255: 10}, 60, (59, 195)),
        ({8: 10, 12: 10, 220: 15, 230: 15}, 60, (8, 220)),
        ({128: 10}, 60, (0, 255)),
        ({0: 3, 1: 50, 254: 50, 255: 3}, 0, (0, 255)),
        ({15: 10, 240: 10}, -1, (0, 255)),
        ({100: 20}, 255, (100, 100)),
        ({100: 20}, 300, (100, 100)),
    ],
    ids=[
        "tallest",
        "upper-edges",
        "lower-edges",
        "ignore-outside",
        "plateau-midpoint",
        "edge-plateaus",
        "ties-use-lower-shade",
        "empty-regions",
        "zero-limit",
        "negative-limit",
        "full-range",
        "oversized-limit",
    ],
)
@pytest.mark.parametrize(
    ("peak_percentage", "peak_prominence"),
    [(0.25, None), (None, None), (0, 0), (100, 100), (200, 200), (-1, -1)],
    ids=["default", "disabled", "zero", "fallback", "above-100", "negative"],
)
def test_find_local_peak_selection(
    counts: dict[int, int],
    upper_limit: int,
    expected: tuple[int, int],
    peak_percentage: float | None,
    peak_prominence: float | None,
) -> None:
    image = _image_from_counts(counts)

    result = find_local_peak(
        image, upper_limit=upper_limit, peak_percentage=peak_percentage, peak_prominence=peak_prominence
    )

    assert result == (*expected, False)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "LA", "P", "I", "I;16", "F"])
def test_find_local_peak_preserves_caller_image(mode: str) -> None:
    image = _image_from_counts({15: 10, 240: 20}).convert(mode)
    pixels = image.tobytes()

    assert find_local_peak(image) == (15, 240, mode != "L")
    assert image.mode == mode
    assert image.tobytes() == pixels


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P"])
def test_find_local_peak_uses_luminance(mode: str) -> None:
    image = Image.new("RGB", (2, 1))
    image.putdata([(0, 0, 255), (255, 255, 0)])
    image = image.convert(mode)

    assert find_local_peak(image) == (29, 226, True)


def test_find_local_peak_bilevel_image() -> None:
    image = _image_from_counts({0: 10, 255: 20}).convert("1")

    assert find_local_peak(image) == (0, 255, True)


def test_find_local_peak_skips_white() -> None:
    image = _image_from_counts({15: 10, 240: 20})

    assert find_local_peak(image, skip_white_check=True) == (15, 255, False)


def test_find_local_peak_empty_image() -> None:
    assert find_local_peak(Image.new("L", (0, 0))) == (0, 255, False)


@pytest.mark.parametrize("image_format", ["PNG", "GIF", "TIFF"])
@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_find_local_peak_file_and_stream(tmp_path: Path, image_format: str, mode: str) -> None:
    image = _image_from_counts({15: 10, 240: 20}).convert(mode)
    stream = BytesIO()
    image.save(stream, format=image_format)
    path = tmp_path / "page"
    path.write_bytes(stream.getvalue())
    with Image.open(path) as opened:
        expected = (15, 240, opened.mode != "L")
        assert find_local_peak(opened) == expected
        assert opened.convert("L").tobytes() == image.convert("L").tobytes()

    assert find_local_peak(path) == expected
    assert find_local_peak(stream) == expected
    assert not stream.closed
    stream.seek(0)
    with Image.open(stream) as reopened:
        assert reopened.convert("L").tobytes() == image.convert("L").tobytes()


@pytest.mark.parametrize("image_format", ["GIF", "TIFF"])
def test_find_local_peak_preserves_current_frame(image_format: str) -> None:
    stream = BytesIO()
    Image.new("L", (10, 10), 15).save(
        stream, format=image_format, save_all=True, append_images=[Image.new("L", (10, 10), 240)]
    )
    stream.seek(0)
    with Image.open(stream) as opened:
        opened.seek(1)
        assert find_local_peak(opened) == (0, 240, opened.mode != "L")
        assert opened.tell() == 1
        opened.seek(0)
        assert find_local_peak(opened) == (15, 255, opened.mode != "L")
    assert not stream.closed

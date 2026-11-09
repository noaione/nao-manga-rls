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

# Tests for the bulk ML color detection helper.

from __future__ import annotations

from pathlib import Path

from PIL import Image

from nmanga.ogsov import detect_image_color_ml_bulk


class _StubKernel:
    def __init__(self):
        self.calls = 0

    def predict(self, bgr_image, *, fast_method: bool = False) -> tuple[bool, int]:
        self.calls += 1
        return True, 90


def _write_image(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (8, 8), color).save(path)
    return path


def test_bulk_runs_every_page_on_a_single_thread(tmp_path: Path):
    images = [_write_image(tmp_path / f"p0{index}.png", (255, 0, 0)) for index in range(1, 4)]
    kernel = _StubKernel()

    results = list(detect_image_color_ml_bulk(images, kernel=kernel, threads=1))

    assert sorted(path.name for path, _ in results) == ["p01.png", "p02.png", "p03.png"]
    assert all(detected.is_color for _, detected in results)
    assert kernel.calls == 3


def test_bulk_runs_every_page_on_a_thread_pool(tmp_path: Path):
    images = [_write_image(tmp_path / f"p0{index}.png", (255, 0, 0)) for index in range(1, 6)]
    kernel = _StubKernel()

    results = list(detect_image_color_ml_bulk(images, kernel=kernel, threads=3))

    assert sorted(path.name for path, _ in results) == [f"p0{index}.png" for index in range(1, 6)]
    assert all(detected.is_color for _, detected in results)
    assert kernel.calls == 5


def test_bulk_skips_the_kernel_for_grayscale_pages(tmp_path: Path):
    gray = _write_image(tmp_path / "p01.png", (128, 128, 128))
    kernel = _StubKernel()

    results = list(detect_image_color_ml_bulk([gray], kernel=kernel, threads=2))

    assert len(results) == 1
    assert results[0][1].is_color is False
    assert kernel.calls == 0

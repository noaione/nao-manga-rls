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

# Tests for the orchestrator color actions and their VapourSynth detection mode.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nmanga.common import RegexCollection
from nmanga.orchestrator.actions._base import ToolsKind
from nmanga.orchestrator.actions.colors import (
    ActionColorDetect,
    ActionColorJpegify,
    ActionMoveColor,
    _iter_ml_color_pages,
    _known_color_pages,
    _page_images,
    _vapoursynth_method,
)
from nmanga.orchestrator.common import ColorDetectionKind

VAPOURSYNTH_TOOLS = {
    "vapoursynth": ToolsKind.PACKAGE,
    "xyz.n4o.mangacolor": ToolsKind.VAPOURSYNTH,
    "xyz.n4o.imgseqs": ToolsKind.VAPOURSYNTH,
}


NO_COLOR_MODEL = SimpleNamespace(metafields=SimpleNamespace(color_model=None))
OGSOV_MODEL = SimpleNamespace(metafields=SimpleNamespace(color_model=SimpleNamespace(kind=ColorDetectionKind.OGSOV)))
OGSOX_MODEL = SimpleNamespace(metafields=SimpleNamespace(color_model=SimpleNamespace(kind=ColorDetectionKind.OGSOX)))


def test_vapoursynth_mode_requests_the_mangacolor_plugins():
    assert ActionColorDetect(mode="vapoursynth").get_tools(OGSOV_MODEL) == VAPOURSYNTH_TOOLS
    assert ActionMoveColor(mode="vapoursynth").get_tools(OGSOV_MODEL) == VAPOURSYNTH_TOOLS


def test_auto_mode_without_a_color_model_needs_nothing():
    assert ActionColorDetect().get_tools(NO_COLOR_MODEL) == {}
    assert ActionMoveColor().get_tools(NO_COLOR_MODEL) == {}
    assert ActionColorJpegify().get_tools(NO_COLOR_MODEL) == {"cjpegli": ToolsKind.BINARY}


def test_ogsov_kernel_requires_torch_and_scipy():
    kernel_tools = {
        "numpy": ToolsKind.PACKAGE,
        "scipy": ToolsKind.PACKAGE,
        "torch": ToolsKind.PACKAGE,
    }

    assert ActionColorDetect().get_tools(OGSOV_MODEL) == kernel_tools
    assert ActionMoveColor().get_tools(OGSOV_MODEL) == kernel_tools
    assert ActionColorJpegify().get_tools(OGSOV_MODEL) == {"cjpegli": ToolsKind.BINARY, **kernel_tools}


def test_ogsox_kernel_requires_onnxruntime():
    kernel_tools = {
        "onnxruntime": ToolsKind.PACKAGE,
        "numpy": ToolsKind.PACKAGE,
    }

    assert ActionColorDetect().get_tools(OGSOX_MODEL) == kernel_tools
    assert ActionMoveColor().get_tools(OGSOX_MODEL) == kernel_tools
    assert ActionColorJpegify().get_tools(OGSOX_MODEL) == {"cjpegli": ToolsKind.BINARY, **kernel_tools}


def test_threads_default_follows_the_machine_size(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("nmanga.common.mp.cpu_count", lambda: 16)

    assert ActionColorDetect().threads == 12
    assert ActionMoveColor().threads == 12


def test_known_color_pages_prefers_the_volume_config():
    volume = SimpleNamespace(colors=[2, 4])
    context = SimpleNamespace(detected_colors=[1])

    assert _known_color_pages(volume, context) == {2, 4}


def test_known_color_pages_reuses_an_earlier_detection():
    volume = SimpleNamespace(colors="auto")
    context = SimpleNamespace(detected_colors=[1, 3])

    assert _known_color_pages(volume, context) == {1, 3}


def test_known_color_pages_detects_when_nothing_is_known():
    volume = SimpleNamespace(colors="auto")
    context = SimpleNamespace(detected_colors=None)

    assert _known_color_pages(volume, context) is None


def test_vapoursynth_method_follows_the_configured_kernel():
    assert _vapoursynth_method(OGSOX_MODEL) == "ogsox"
    assert _vapoursynth_method(OGSOV_MODEL) == "ogsov"
    assert _vapoursynth_method(NO_COLOR_MODEL) == "ogsov"


def test_page_images_keeps_only_page_numbered_files(tmp_path: Path):
    for name in ("p01.png", "p02.jpg", "cover.png"):
        (tmp_path / name).write_bytes(b"")

    pages = _page_images(tmp_path, RegexCollection.page_re())

    assert pages == [(tmp_path / "p01.png", 1), (tmp_path / "p02.jpg", 2)]


class _StubKernel:
    def predict(self, bgr_image, *, fast_method: bool = False) -> tuple[bool, int]:
        return True, 90


def test_iter_ml_color_pages_maps_page_numbers(tmp_path: Path):
    for name in ("p01.png", "p02.png"):
        Image.new("RGB", (8, 8), (255, 0, 0)).save(tmp_path / name)

    pages = [(tmp_path / "p01.png", 1), (tmp_path / "p02.png", 2)]
    results = dict(_iter_ml_color_pages(pages, kernel=_StubKernel(), fast_method=False, threads=2))

    assert set(results) == {1, 2}
    assert all(detected.is_color for detected in results.values())

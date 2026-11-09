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

# Tests for the asynchronous frame iterator.

from __future__ import annotations

import vapoursynth as vs

from nmanga.lazy import get_vapoursynth
from nmanga.vapour import vs_iter_frames_async


def _blank(length: int, width: int = 32, height: int = 32) -> vs.VideoNode:
    core = get_vapoursynth().core
    return core.std.BlankClip(width=width, height=height, format=vs.RGB24, length=length)


def test_async_iteration_covers_every_frame_once():
    clip = _blank(10)
    frames = list(vs_iter_frames_async(clip, window=4))

    indices = [n for n, _ in frames]
    assert sorted(indices) == list(range(10))
    assert len(indices) == len(set(indices))
    assert all(frame.width == 32 for _, frame in frames)


def test_async_iteration_handles_window_wider_than_the_clip():
    clip = _blank(5)
    frames = list(vs_iter_frames_async(clip, window=64))
    assert sorted(n for n, _ in frames) == list(range(5))


def test_async_iteration_clamps_a_zero_window():
    clip = _blank(3)
    frames = list(vs_iter_frames_async(clip, window=0))
    assert sorted(n for n, _ in frames) == list(range(3))


class _FrameNodeProxy:
    """A `VideoNode` stand-in that records every async request."""

    def __init__(self, node: vs.VideoNode, async_requests: list[int]):
        self.num_frames = node.num_frames
        self._node = node
        self._async_requests = async_requests

    def get_frame(self, n: int) -> vs.VideoFrame:
        return self._node.get_frame(n)

    def get_frame_async(self, n: int):
        self._async_requests.append(n)
        return self._node.get_frame_async(n)


def test_window_of_one_uses_the_blocking_request():
    clip = _blank(4)
    async_requests: list[int] = []

    frames = list(vs_iter_frames_async(_FrameNodeProxy(clip, async_requests), window=1))

    assert [n for n, _ in frames] == list(range(4))
    assert async_requests == []

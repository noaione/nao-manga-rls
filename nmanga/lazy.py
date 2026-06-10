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

# Lazy loader

from __future__ import annotations

import functools

__all__ = (
    "get_numpy",
    "get_scipy_signal",
)


@functools.lru_cache(maxsize=1)
def get_numpy():
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ImportError("numpy is required to do the following action. Please install numpy.") from exc


@functools.lru_cache(maxsize=1)
def get_scipy_signal():
    try:
        from scipy import signal  # type: ignore

        return signal
    except ImportError as exc:
        raise ImportError("scipy is required to do the following action. Please install scipy.") from exc


@functools.lru_cache(maxsize=1)
def get_vapoursynth():
    try:
        import vapoursynth as vs  # type: ignore

        return vs
    except ImportError as exc:
        raise ImportError("vapoursynth is required to do the following action. Please install vapoursynth.") from exc

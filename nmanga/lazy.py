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

# Lazy loader
#
# The non-ML helpers live here. The ONNX Runtime initialization moved to `nmanga.ml.ort`; it is
# re-exported as a thin forward (and the initialization state through `__getattr__`) so the old
# entry point keeps working without duplicating the state, the registration or the cache.

from __future__ import annotations

import functools
from typing import Any, Protocol, TypeVar, overload

from .ml import ort as _ort
from .ml.ort import get_onnxruntime

__all__ = (
    "get_einops_rearrange",
    "get_numpy",
    "get_onnxruntime",
    "get_scipy_ndimage",
    "get_scipy_signal",
    "get_torch",
    "get_vapoursynth",
)

Tensor = TypeVar("Tensor")

# Attributes that used to be defined here but are now owned by `nmanga.ml.ort`. Reading them
# through the module `__getattr__` keeps one initialization state per process instead of a
# second copy that silently diverges from the loader that actually builds the sessions.
_FORWARDED_ORT_NAMES = frozenset({"__preloaded_ort__", "__telemetry_disabled__", "_disable_ort_telemetry"})


def __getattr__(name: str) -> object:
    """Forward the moved ONNX Runtime initialization entries to :mod:`nmanga.ml.ort`."""
    if name in _FORWARDED_ORT_NAMES:
        return getattr(_ort, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class RearrangeCallable(Protocol):
    @overload
    def __call__(self, tensor: list[Tensor], pattern: str, **axes_lengths: Any) -> Tensor: ...
    @overload
    def __call__(self, tensor: Tensor, pattern: str, **axes_lengths: Any) -> Tensor: ...
    def __call__(self, tensor: Tensor | list[Tensor], pattern: str, **axes_lengths: Any) -> Tensor: ...


@functools.lru_cache(maxsize=1)
def get_numpy():
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ImportError("numpy is required to do the following action. Please install numpy.") from exc


@functools.lru_cache(maxsize=1)
def get_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("torch is required to do the following action. Please install pytorch.") from exc


@functools.lru_cache(maxsize=1)
def get_scipy_signal():
    try:
        from scipy import signal  # type: ignore

        return signal
    except ImportError as exc:
        raise ImportError("scipy is required to do the following action. Please install scipy.") from exc


@functools.lru_cache(maxsize=1)
def get_scipy_ndimage():
    try:
        from scipy import ndimage  # type: ignore

        return ndimage
    except ImportError as exc:
        raise ImportError("scipy is required to do the following action. Please install scipy.") from exc


@functools.lru_cache(maxsize=1)
def get_einops_rearrange() -> RearrangeCallable:
    try:
        from einops import rearrange

        return rearrange
    except ImportError as exc:
        raise ImportError("einops is required to do the following action. Please install einops.") from exc


@functools.lru_cache(maxsize=1)
def get_vapoursynth():
    try:
        import vapoursynth as vs  # type: ignore

        return vs
    except ImportError as exc:
        raise ImportError("vapoursynth is required to do the following action. Please install vapoursynth.") from exc

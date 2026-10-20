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

import ctypes
import functools
import site
import sys
from importlib import util as importutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, overload

if TYPE_CHECKING:
    import onnxruntime as ort

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
__preloaded_ort__: "ort" | None = None  # pyright: ignore[reportInvalidTypeForm]


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


def get_onnxruntime() -> "type[ort]":  # type: ignore
    global __preloaded_ort__

    if __preloaded_ort__ is not None:
        return __preloaded_ort__

    for sp in site.getsitepackages():
        ort_dll = Path(sp) / "onnxruntime" / "capi" / "onnxruntime.dll"
        if ort_dll.exists() and sys.platform == "win32":
            ctypes.WinDLL(str(ort_dll))
            break

    # Preload TensorRT if available
    is_tensorrt_available = importutil.find_spec("tensorrt")
    if is_tensorrt_available is not None:
        import tensorrt  # type: ignore # ruff: ignore[unused-import]

    import onnxruntime as ort  # type: ignore

    is_trt_ep_available = importutil.find_spec("onnxruntime_ep_nv_tensorrt_rtx")

    if sys.platform != "darwin":
        ort.preload_dlls()

    if is_trt_ep_available is not None:
        import onnxruntime_ep_nv_tensorrt_rtx as trt_ep  # type: ignore

        ort.register_execution_provider_library(trt_ep.get_ep_name(), trt_ep.get_library_path())  # type: ignore
    __preloaded_ort__ = ort

    return ort  # type: ignore

"""
MIT License

Copyright (c) 2022-present noaione
Copyright (c) 2025-present anon

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

# Compatibility facade for the ONNX Runtime implementation that now lives in `nmanga.ml`.
#
# Nothing is implemented here: every name is a thin forward to the canonical module, so there is
# exactly one definition, one cache and one initialization state per process. New code should
# import from `nmanga.ml.contracts`, `nmanga.ml.onnx_session`, `nmanga.ml.onnx_cache` or
# `nmanga.ml.onnx_backend` instead.

from __future__ import annotations

from typing import TYPE_CHECKING

from .ml.contracts import MLDataType
from .ml.onnx_backend import denoise_single_image, denoise_single_image_with_overlap
from .ml.onnx_cache import (
    build_compilation_cache_key,
    get_data_dir,
    get_model_fingerprint,
    get_tf32_tag,
)
from .ml.onnx_session import (
    get_model_fixed_batch_size,
    get_model_information,
    get_model_scale_factor,
    get_model_static_input,
    get_torch_memory_limit_and_rtx,
    prepare_model_runtime,
    prepare_model_runtime_builders,
    resolve_model_input_sizes,
    resolve_model_static_shapes,
)

if TYPE_CHECKING:
    from .ml.onnx_session import InferenceSessionWithScale as InferenceSessionWithScale

__all__ = (
    "MLDataType",
    "build_compilation_cache_key",
    "denoise_single_image",
    "denoise_single_image_with_overlap",
    "get_data_dir",
    "get_model_fingerprint",
    "get_model_fixed_batch_size",
    "get_model_information",
    "get_model_scale_factor",
    "get_model_static_input",
    "get_tf32_tag",
    "get_torch_memory_limit_and_rtx",
    "prepare_model_runtime",
    "prepare_model_runtime_builders",
    "resolve_model_input_sizes",
    "resolve_model_static_shapes",
)

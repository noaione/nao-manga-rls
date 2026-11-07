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

# ONNX Runtime initialization: DLL preload, telemetry and provider registration.

from __future__ import annotations

import ctypes
import site
import sys
from importlib import util as importutil
from pathlib import Path
from typing import Any

__all__ = ("get_onnxruntime",)

__preloaded_ort__: Any = None
__telemetry_disabled__ = False


def _disable_ort_telemetry(ort_module: Any) -> None:
    """
    Turn off the ONNX Runtime telemetry events exactly once per process.

    Serializing telemetry costs time inside every session and, more importantly for a
    reproducible benchmark, it is global state that is easier to rule out than to measure.
    Older ONNX Runtime builds do not expose the switch, so it is optional.

    :param ort_module: The imported :mod:`onnxruntime` module
    """

    global __telemetry_disabled__

    if __telemetry_disabled__:
        return

    disable_telemetry = getattr(ort_module, "disable_telemetry_events", None)
    if callable(disable_telemetry):
        disable_telemetry()
    __telemetry_disabled__ = True


def get_onnxruntime() -> Any:
    global __preloaded_ort__

    if __preloaded_ort__ is not None:
        _disable_ort_telemetry(__preloaded_ort__)
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

    # Keep this before any session/compiler exists, so it cannot be missed by a caller
    _disable_ort_telemetry(ort)

    is_trt_ep_available = importutil.find_spec("onnxruntime_ep_nv_tensorrt_rtx")
    is_mlx_ep_available = importutil.find_spec("onnxruntime_ep_mlx")
    is_openvino_ep_available = importutil.find_spec("onnxruntime_ep_openvino")
    is_webgpu_ep_available = importutil.find_spec("onnxruntime_ep_webgpu")

    if sys.platform != "darwin":
        ort.preload_dlls()

    excluded = []
    if is_trt_ep_available is not None:
        import onnxruntime_ep_nv_tensorrt_rtx as trt_rtx_ep  # type: ignore

        rtx_name = trt_rtx_ep.get_ep_name()
        ort.register_execution_provider_library(rtx_name, trt_rtx_ep.get_library_path())  # type: ignore
        excluded.append(rtx_name)
    if is_mlx_ep_available is not None:
        # MLX is an Apple Silicon plugin EP, it has to be registered before any session is built
        import onnxruntime_ep_mlx  # type: ignore

        ort.register_execution_provider_library(onnxruntime_ep_mlx.EP_NAME, onnxruntime_ep_mlx.library_path())
    if is_openvino_ep_available is not None:
        import onnxruntime_ep_openvino as openvino_ep  # type: ignore

        ov_name = openvino_ep.get_ep_name()
        ort.register_execution_provider_library(ov_name, openvino_ep.get_library_path())  # type: ignore
        excluded.append(ov_name)
    if is_webgpu_ep_available is not None:
        import onnxruntime_ep_webgpu as webgpu_ep  # type: ignore

        wgpu_name = webgpu_ep.get_ep_name()
        ort.register_execution_provider_library(wgpu_name, webgpu_ep.get_library_path())  # type: ignore
        excluded.append(wgpu_name)

    # WinML discovery is imported here, inside the initialization path, so that importing this
    # module never pulls in the Windows ML package.
    from .winml import get_winml_ep_libraries

    winml_libs = get_winml_ep_libraries(excluded)
    for name, path in winml_libs:
        ort.register_execution_provider_library(name, str(path))
    __preloaded_ort__ = ort

    return ort  # type: ignore

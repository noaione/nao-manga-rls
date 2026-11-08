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

# Friendly backend names, so callers never have to spell out the raw ONNX Runtime
# `<Vendor>ExecutionProvider` identifiers in a config file or on the command line.

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

__all__ = (
    "AUTO_BACKEND_PRIORITY",
    "BackendSpec",
    "BackendStatus",
    "InferenceBackend",
    "describe_backends",
    "detect_backend_status",
    "get_backend_spec",
)

_NVIDIA_PLATFORMS = ("linux", "win32")
_INTEL_PLATFORMS = ("linux", "win32", "darwin")
_APPLE_PLATFORMS = ("darwin",)


class InferenceBackend(str, Enum):
    """
    The inference backends that can be requested by name.

    Every value is a nmanga name, never an ONNX Runtime provider name: the mapping to the
    provider (or plugin device) each one needs lives in :data:`_BACKEND_SPECS`.
    """

    AUTO = "auto"
    """Pick the best backend the machine can actually run, in :data:`AUTO_BACKEND_PRIORITY` order."""

    TRT_RTX = "trt-rtx"
    """NVIDIA TensorRT RTX, which keeps the precision the graph declares."""

    TRT = "trt"
    """NVIDIA TensorRT, which builds a cached engine per optimization profile."""

    CUDA = "cuda"
    """NVIDIA CUDA."""

    AMDGPU = "amdgpu"
    """Reserved for the native AMD provider, which is not wired up yet."""

    MIGRAPHX = "migraphx"
    """AMD MIGraphX, the ROCm backend."""

    OPENVINO = "openvino"
    """Intel OpenVINO, on a CPU, a GPU or an NPU."""

    COREML = "coreml"
    """Apple CoreML."""

    MLX = "mlx"
    """Apple MLX."""

    WEBGPU = "webgpu"
    """WebGPU, the last resort on a machine without a vendor backend."""

    @classmethod
    def from_str(cls, value: str) -> InferenceBackend:
        """
        Parse a backend name as written by a user.

        :param value: The backend name, case insensitive and `_`/`-` tolerant
        :return: The matching backend
        :raises ValueError: If the name does not belong to any backend
        """

        normalized = value.strip().lower().replace("_", "-")
        for member in cls:
            if member.value == normalized:
                return member

        accepted = ", ".join(member.value for member in cls)
        raise ValueError(f"Unknown inference backend {value!r}, expected one of: {accepted}")

    @property
    def is_automatic(self) -> bool:
        """Whether this value asks for automatic selection instead of one backend."""

        return self is InferenceBackend.AUTO

    @property
    def label(self) -> str:
        """The human readable name of this backend, for log lines and errors."""

        return get_backend_spec(self).label


@dataclass(frozen=True)
class BackendSpec:
    """What nmanga knows about one backend before ONNX Runtime is imported."""

    label: str
    """The human readable name shown to the user."""

    provider_names: tuple[str, ...]
    """The raw provider names this backend answers to; any match means it is available."""

    platforms: tuple[str, ...] = ()
    """The platforms the backend runs on, empty when nmanga does not restrict it."""

    note: str = ""
    """A short remark used in help output and in "not available here" errors."""

    def is_supported_here(self, platform: str) -> bool:
        """
        Whether the backend can exist on the given platform at all.

        :param platform: A :data:`sys.platform` value
        :return: `True` when the backend is not restricted, or the platform is in the list
        """

        return not self.platforms or platform in self.platforms


_BACKEND_SPECS: dict[InferenceBackend, BackendSpec] = {
    InferenceBackend.TRT_RTX: BackendSpec(
        "TensorRT RTX",
        ("NvTensorRTRTXExecutionProvider",),
        _NVIDIA_PLATFORMS,
        "keeps the precision the model declares, compiles an engine on first use",
    ),
    InferenceBackend.TRT: BackendSpec(
        "TensorRT",
        ("TensorrtExecutionProvider",),
        _NVIDIA_PLATFORMS,
        "builds and caches one engine per optimization profile",
    ),
    InferenceBackend.CUDA: BackendSpec("CUDA", ("CUDAExecutionProvider",), _NVIDIA_PLATFORMS),
    InferenceBackend.AMDGPU: BackendSpec(
        "AMD GPU",
        ("AMDGPUExecutionProvider",),
        ("linux", "win32"),
        "use migraphx for AMD GPUs today",
    ),
    InferenceBackend.MIGRAPHX: BackendSpec("MIGraphX", ("MIGraphXExecutionProvider",), ("linux",)),
    InferenceBackend.OPENVINO: BackendSpec("OpenVINO", ("OpenVINOExecutionProvider",), _INTEL_PLATFORMS),
    InferenceBackend.COREML: BackendSpec("CoreML", ("CoreMLExecutionProvider",), _APPLE_PLATFORMS),
    InferenceBackend.MLX: BackendSpec("MLX", ("MLXExecutionProvider",), _APPLE_PLATFORMS),
    InferenceBackend.WEBGPU: BackendSpec("WebGPU", ("WebGpuExecutionProvider",)),
}

AUTO_BACKEND_PRIORITY: tuple[InferenceBackend, ...] = (
    InferenceBackend.TRT,
    InferenceBackend.CUDA,
    InferenceBackend.MIGRAPHX,
    InferenceBackend.OPENVINO,
    InferenceBackend.MLX,
    InferenceBackend.COREML,
    InferenceBackend.WEBGPU,
)
"""
The order `auto` tries backends in.

TensorRT RTX is deliberately absent: it needs an engine compiled from the graph on first use,
which is a slow, disk-hungry step that should never happen behind an automatic choice. Asking
for `trt-rtx` (or `auto` on a machine that only offers RTX) is what opts into it.
"""

_BACKEND_DESCRIPTIONS: dict[InferenceBackend, str] = {
    InferenceBackend.AUTO: "pick the best backend this machine offers, in this order: "
    + ", ".join(backend.value for backend in AUTO_BACKEND_PRIORITY),
    InferenceBackend.TRT_RTX: "NVIDIA TensorRT RTX",
    InferenceBackend.TRT: "NVIDIA TensorRT",
    InferenceBackend.CUDA: "NVIDIA CUDA",
    InferenceBackend.AMDGPU: "native AMD provider, not implemented yet",
    InferenceBackend.MIGRAPHX: "AMD MIGraphX (ROCm)",
    InferenceBackend.OPENVINO: "Intel OpenVINO (CPU, GPU or NPU)",
    InferenceBackend.COREML: "Apple CoreML",
    InferenceBackend.MLX: "Apple MLX",
    InferenceBackend.WEBGPU: "WebGPU",
}


def get_backend_spec(backend: InferenceBackend) -> BackendSpec:
    """
    Look up what nmanga knows about a backend.

    :param backend: The backend to describe
    :return: The spec of the backend
    :raises ValueError: If `backend` is `auto`, which is not a provider of its own
    """

    if backend.is_automatic:
        raise ValueError("`auto` is a selection strategy, it has no execution provider of its own")
    return _BACKEND_SPECS[backend]


@dataclass(frozen=True)
class BackendStatus:
    """Whether one backend can actually run on the machine that loaded ONNX Runtime."""

    backend: InferenceBackend
    """The backend this status describes."""

    provider: str | None
    """The raw provider name that makes it available, `None` when nothing matched."""

    @property
    def is_available(self) -> bool:
        """Whether ONNX Runtime exposes what this backend needs."""

        return self.provider is not None

    @property
    def spec(self) -> BackendSpec:
        """What nmanga knows about this backend."""

        return get_backend_spec(self.backend)


def detect_backend_status(
    available_providers: Iterable[str],
    ep_device_names: Iterable[str] = (),
) -> list[BackendStatus]:
    """
    Match the backends nmanga knows against what an ONNX Runtime build actually exposes.

    A backend can be reachable by provider name or only through the plugin device API, so both
    lists are consulted; the provider name is reported either way so the output is copy-pasteable
    into an ONNX Runtime bug report.

    :param available_providers: The names from :func:`onnxruntime.get_available_providers`
    :param ep_device_names: The `ep_name` of every device from :func:`onnxruntime.get_ep_devices`
    :return: One status per backend in :class:`InferenceBackend` order, `auto` excluded
    """

    available = set(available_providers) | set(ep_device_names)
    statuses: list[BackendStatus] = []
    for backend in InferenceBackend:
        if backend.is_automatic:
            continue
        matched = next(
            (name for name in get_backend_spec(backend).provider_names if name in available),
            None,
        )
        statuses.append(BackendStatus(backend=backend, provider=matched))
    return statuses


def describe_backends() -> str:
    """
    Render every backend name with what it maps to, for command line help.

    Entries are separated by a blank line because the help renderer reflows single newlines
    into one paragraph, which would turn the table into an unreadable block of text.

    :return: One bullet per backend
    """

    lines: list[str] = []
    for backend in InferenceBackend:
        description = _BACKEND_DESCRIPTIONS[backend]
        spec = _BACKEND_SPECS.get(backend)
        if spec is not None and spec.note:
            description = f"{description} ({spec.note})"
        lines.append(f"- {backend.value} — {description}")
    return "\n\n".join(lines)

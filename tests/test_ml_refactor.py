"""
Tests for the `nmanga.ml` relocation.

They pin down the properties that make the move safe rather than any image arithmetic: the
compatibility facades forward the canonical objects, the ORT initialization runs once and is
shared by every entry point, the plugin execution providers are registered in the same order with
the same exclusions, and neither the package nor a facade drags the heavy runtime in on import.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

import nmanga.denoiser
import nmanga.lazy
import nmanga.ml.contracts
import nmanga.ml.onnx_backend
import nmanga.ml.onnx_cache
import nmanga.ml.onnx_session
import nmanga.ml.ort
import nmanga.ml.winml
import nmanga.winml

REPO_ROOT = Path(__file__).resolve().parents[1]
HEAVY_MODULES = ("torch", "onnxruntime", "tensorrt", "windowsml")


def _run_python(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


# --------------------------------------------------------------------------------------
# Compatibility import paths
# --------------------------------------------------------------------------------------


def test_denoiser_facade_forwards_the_canonical_objects():
    assert nmanga.denoiser.MLDataType is nmanga.ml.contracts.MLDataType
    assert nmanga.denoiser.prepare_model_runtime is nmanga.ml.onnx_session.prepare_model_runtime
    assert nmanga.denoiser.prepare_model_runtime_builders is nmanga.ml.onnx_session.prepare_model_runtime_builders
    assert nmanga.denoiser.get_model_information is nmanga.ml.onnx_session.get_model_information
    assert nmanga.denoiser.get_model_static_input is nmanga.ml.onnx_session.get_model_static_input
    assert nmanga.denoiser.resolve_model_static_shapes is nmanga.ml.onnx_session.resolve_model_static_shapes
    assert nmanga.denoiser.build_compilation_cache_key is nmanga.ml.onnx_cache.build_compilation_cache_key
    assert nmanga.denoiser.get_model_fingerprint is nmanga.ml.onnx_cache.get_model_fingerprint
    assert nmanga.denoiser.get_tf32_tag is nmanga.ml.onnx_cache.get_tf32_tag
    assert nmanga.denoiser.denoise_single_image is nmanga.ml.onnx_backend.denoise_single_image
    assert nmanga.denoiser.denoise_single_image_with_overlap is nmanga.ml.onnx_backend.denoise_single_image_with_overlap


def test_winml_facade_forwards_the_canonical_object():
    assert nmanga.winml.get_winml_ep_libraries is nmanga.ml.winml.get_winml_ep_libraries


def test_lazy_facade_forwards_the_canonical_loader():
    assert nmanga.lazy.get_onnxruntime is nmanga.ml.ort.get_onnxruntime


def test_facades_do_not_duplicate_the_cache_state():
    assert "_external_data_cache" not in vars(nmanga.denoiser)
    assert "_model_fingerprint_cache" not in vars(nmanga.denoiser)
    assert "__preloaded_ort__" not in vars(nmanga.lazy)
    assert "__telemetry_disabled__" not in vars(nmanga.lazy)


# --------------------------------------------------------------------------------------
# No eager heavy imports
# --------------------------------------------------------------------------------------


def test_importing_the_ml_modules_does_not_load_the_heavy_stack():
    modules = (
        "nmanga.ml",
        "nmanga.ml.contracts",
        "nmanga.ml.onnx_session",
        "nmanga.ml.onnx_cache",
        "nmanga.ml.onnx_backend",
        "nmanga.cli.torch",
        "nmanga.ml.ort",
        "nmanga.ml.winml",
        "nmanga.denoiser",
        "nmanga.winml",
        "nmanga.lazy",
    )
    code = f"import {', '.join(modules)}\nimport sys\nprint(sorted(m for m in {HEAVY_MODULES!r} if m in sys.modules))"

    assert _run_python(code) == "[]"


def test_ort_initialization_does_not_import_the_compatibility_loader():
    code = "import nmanga.ml.ort\nimport sys\nprint('nmanga.lazy' in sys.modules)"

    assert _run_python(code) == "False"


def test_winml_discovery_does_not_import_onnxruntime():
    code = "import nmanga.ml.winml\nimport sys\nprint('onnxruntime' in sys.modules)"

    assert _run_python(code) == "False"


# --------------------------------------------------------------------------------------
# One-time ORT initialization, shared state and plugin registration
# --------------------------------------------------------------------------------------


class _FakeOrtRuntime:
    def __init__(self) -> None:
        self.telemetry_calls = 0
        self.preload_dll_calls = 0
        self.registered: list[tuple[str, str]] = []

    def disable_telemetry_events(self) -> None:
        self.telemetry_calls += 1

    def preload_dlls(self) -> None:
        self.preload_dll_calls += 1

    def register_execution_provider_library(self, name: str, path: str) -> None:
        self.registered.append((name, path))

    def get_available_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]


def _make_ep_module(ep_name: str, library: str) -> types.ModuleType:
    """Build a stand-in for one of the plugin execution provider packages."""
    module = types.ModuleType(f"fake_{ep_name}")
    module.EP_NAME = ep_name  # type: ignore[attr-defined]
    module.get_ep_name = lambda: ep_name  # type: ignore[attr-defined]
    module.get_library_path = lambda: library  # type: ignore[attr-defined]
    module.library_path = lambda: library  # type: ignore[attr-defined]
    return module


def test_ort_initialization_is_shared_and_runs_once(monkeypatch: pytest.MonkeyPatch):
    fake_ort = _FakeOrtRuntime()
    plugin_modules = {
        "onnxruntime_ep_nv_tensorrt_rtx": _make_ep_module(
            "NvTensorRTRTXExecutionProvider", "C:/plugins/tensorrt_rtx.dll"
        ),
        "onnxruntime_ep_mlx": _make_ep_module("MLXExecutionProvider", "C:/plugins/mlx.dylib"),
        "onnxruntime_ep_openvino": _make_ep_module("OpenVINOExecutionProvider", "C:/plugins/openvino.dll"),
        "onnxruntime_ep_webgpu": _make_ep_module("WebGpuExecutionProvider", "C:/plugins/webgpu.dll"),
    }
    winml_exclusions: list[list[str]] = []

    def fake_winml(exclusion_list: list[str] | None = None) -> list[tuple[str, Path]]:
        winml_exclusions.append(list(exclusion_list or []))
        return [("Windows ML Execution Provider", Path("C:/plugins/winml.dll"))]

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)  # type: ignore[arg-type]
    for module_name, plugin_module in plugin_modules.items():
        monkeypatch.setitem(sys.modules, module_name, plugin_module)
    monkeypatch.setattr(
        nmanga.ml.ort.importutil, "find_spec", lambda name: object() if name in plugin_modules else None
    )
    # Keep the real onnxruntime DLL out of this process while the initialization path runs.
    monkeypatch.setattr(nmanga.ml.ort.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(nmanga.ml.ort, "__preloaded_ort__", None)
    monkeypatch.setattr(nmanga.ml.ort, "__telemetry_disabled__", False)
    monkeypatch.setattr(nmanga.ml.winml, "get_winml_ep_libraries", fake_winml)

    first = nmanga.ml.ort.get_onnxruntime()
    second = nmanga.ml.ort.get_onnxruntime()
    from_facade = nmanga.lazy.get_onnxruntime()

    assert first is fake_ort
    assert second is fake_ort
    assert from_facade is fake_ort

    # The DLL preload happens once, and telemetry is disabled exactly once even though the
    # cached path calls the guard again.
    assert fake_ort.preload_dll_calls == 1
    assert fake_ort.telemetry_calls == 1

    # Every plugin is registered once, in order, and WinML discovery receives the exclusion list
    # of the providers that were already registered by name.
    assert fake_ort.registered == [
        ("NvTensorRTRTXExecutionProvider", "C:/plugins/tensorrt_rtx.dll"),
        ("MLXExecutionProvider", "C:/plugins/mlx.dylib"),
        ("OpenVINOExecutionProvider", "C:/plugins/openvino.dll"),
        ("WebGpuExecutionProvider", "C:/plugins/webgpu.dll"),
        ("Windows ML Execution Provider", str(Path("C:/plugins/winml.dll"))),
    ]
    assert winml_exclusions == [
        ["NvTensorRTRTXExecutionProvider", "OpenVINOExecutionProvider", "WebGpuExecutionProvider"]
    ]

    # The facade reads the canonical state instead of keeping a copy of it.
    assert nmanga.lazy.__preloaded_ort__ is fake_ort
    assert nmanga.lazy.__telemetry_disabled__ is True


def test_winml_discovery_skips_non_windows_without_importing_windowsml(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delitem(sys.modules, "windowsml", raising=False)

    assert nmanga.ml.winml.get_winml_ep_libraries(["SomeExecutionProvider"]) == []
    assert "windowsml" not in sys.modules

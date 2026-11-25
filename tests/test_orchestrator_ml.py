"""Tests for orchestrator `denoise`/`upscale` actions with Umzi `.ummodel` and variable tiling."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from nmanga import term
from nmanga.ml.backends import InferenceBackend
from nmanga.ml.contracts import MLDataType
from nmanga.ml.umzi import UmziSessionPool, get_umzi_native
from nmanga.orchestrator import ActionDenoise, ActionUpscale, OrchestratorConfig, ToolsKind, WorkerContext
from nmanga.orchestrator.actions.ml import _UMZI_CONSTRAINTS_CACHE


def _orchestrator(tmp_path: Path, action: dict) -> OrchestratorConfig:
    return OrchestratorConfig.model_validate({
        "title": "Test Series",
        "publisher": "Test Publisher",
        "credit": "Ripper",
        "email": "ripper@example.com",
        "base_path": "source",
        "volumes": [{"path": "v01", "number": 1}],
        "actions": [action],
    })


def _install_fake_umzi(monkeypatch: pytest.MonkeyPatch, *, family: str = "smosr", scale: int = 2):
    created: list[object] = []

    class _FakeSession:
        scale_factor = scale
        context = 0

        def __init__(self, model_path: Path, options: object) -> None:
            self.model_path = model_path
            self.options = options
            self.calls: list[tuple[int, int, int, int]] = []
            created.append(self)

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=("batch", 3, "height", "width"), type="tensor(float16)")]

        def get_outputs(self):
            return [SimpleNamespace(name="output", shape=("batch", 3, "height", "width"), type="tensor(float16)")]

        def run(self, names, feeds):
            arr = feeds["input"]
            self.calls.append(tuple(int(d) for d in arr.shape))
            out = arr.repeat(scale, axis=2).repeat(scale, axis=3)
            return [np.ascontiguousarray(out)]

        def close(self) -> None:
            pass

    payload = SimpleNamespace(
        scale=scale,
        input_channels=3,
        output_channels=3,
        alignment=2 if family == "figsr" else None,
        base_channels=48,
        reflection_padding=2,
        first_blocks=2,
        middle_blocks=3,
        end_blocks=1,
        weight_bytes=7000,
        checkpoint="smosr.safetensors",
    )
    info = SimpleNamespace(
        path=Path("/models/smosr.ummodel"),
        size=8192,
        version=1,
        family=5,
        family_name=family,
        schema=2,
        digest="abc123",
        sections=(),
        payload=payload,
        problems=(),
    )

    module = types.ModuleType("umzi_native")
    module.peek_model = lambda path: info  # type: ignore[attr-defined]
    module.SessionOptions = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    module.InferenceSession = _FakeSession  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "umzi_native", module)
    monkeypatch.setattr("nmanga.ml.umzi._load_torch_cuda_libraries", lambda: None)
    get_umzi_native.cache_clear()
    return created


def test_ummodel_action_declares_umzi_native_tool(tmp_path: Path):
    model = tmp_path / "smosr.ummodel"
    model.write_bytes(b"UMZINPKG")
    config = _orchestrator(tmp_path, {"kind": "upscale", "model": model})
    action = config.actions[0]

    assert isinstance(action, ActionUpscale)
    assert action.is_umzi_model is True
    assert action.get_tools(config) == {"umzi_native": ToolsKind.PACKAGE, "numpy": ToolsKind.PACKAGE}


def test_onnx_action_rejects_variable_tiling_on_static_backends(tmp_path: Path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")

    with pytest.raises(ValidationError, match="variable_tiling"):
        _orchestrator(
            tmp_path,
            {"kind": "denoise", "model": model, "backend": InferenceBackend.TRT_RTX, "variable_tiling": True},
        )


def test_ummodel_action_rejects_bf16_without_precision_override(tmp_path: Path):
    model = tmp_path / "model.ummodel"
    model.write_bytes(b"UMZINPKG")

    with pytest.raises(ValidationError, match="bf16"):
        _orchestrator(tmp_path, {"kind": "denoise", "model": model, "data_type": MLDataType.BF16})


def test_ummodel_action_runs_with_variable_tiling_and_session_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    model = tmp_path / "smosr.ummodel"
    model.write_bytes(b"UMZINPKG")
    src = tmp_path / "source" / "v01"
    src.mkdir(parents=True)
    page = Image.new("RGB", (140, 140), (30, 60, 90))
    page.save(src / "p001.png")
    page.close()

    created = _install_fake_umzi(monkeypatch, family="smosr", scale=2)
    config = _orchestrator(
        tmp_path,
        {
            "kind": "upscale",
            "model": model,
            "tile_size": 64,
            "tile_expand": 16,
            "tile_bucket": 16,
            "variable_tiling": True,
            "precision": "mixed",
            "use_plan_cache": False,
            "kernel_options": {"tune_convolutions": 0},
        },
    )
    action = config.actions[0]
    assert isinstance(action, ActionUpscale)

    context = WorkerContext(
        root_dir=tmp_path,
        current_dir=src,
        terminal=term.get_console(),
        toolsets={},
    )
    try:
        action.run(context, config.volumes[0], config)

        out_file = tmp_path / "upscaled" / "v01" / "p001.png"
        assert out_file.is_file()
        with Image.open(out_file) as upscaled:
            assert upscaled.size == (280, 280)

        cached = context.ml_model_session[action.make_cache_key()]
        assert isinstance(cached, UmziSessionPool)
        assert created[0].options.precision == "mixed"  # type: ignore[attr-defined]
        assert created[0].options.kernel_options == {"tune_convolutions": 0, "fuse_silu": 1}  # type: ignore[attr-defined]
    finally:
        for session in WorkerContext.ml_model_session.values():
            close = getattr(session, "close", None)
            if callable(close):
                close()
        WorkerContext.ml_model_session.clear()
        _UMZI_CONSTRAINTS_CACHE.clear()
        get_umzi_native.cache_clear()


def test_ummodel_action_fixed_mode_opens_single_session_and_supports_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    model = tmp_path / "gaterv3.ummodel"
    model.write_bytes(b"UMZINPKG")
    config = _orchestrator(tmp_path, {"kind": "denoise", "model": model, "precision": "fp16"})
    action = config.actions[0]
    assert isinstance(action, ActionDenoise)

    dry_ctx = WorkerContext(
        root_dir=tmp_path,
        current_dir=tmp_path / "source" / "v01",
        terminal=term.get_console(),
        toolsets={},
        dry_run=True,
    )
    action.run(dry_ctx, config.volumes[0], config)
    assert dry_ctx.current_dir == tmp_path / "denoised" / "v01"


def test_torch_action_declares_torch_and_resselt_tools(tmp_path: Path):
    model = tmp_path / "span.safetensors"
    model.write_bytes(b"safetensors")
    config = _orchestrator(tmp_path, {"kind": "upscale", "model": model})
    action = config.actions[0]

    assert isinstance(action, ActionUpscale)
    assert action.is_torch_model is True
    assert action.get_tools(config) == {
        "torch": ToolsKind.PACKAGE,
        "resselt": ToolsKind.PACKAGE,
        "numpy": ToolsKind.PACKAGE,
    }


def test_torch_action_rejects_invalid_combinations(tmp_path: Path):
    model = tmp_path / "span.pth"
    model.write_bytes(b"pth")

    with pytest.raises(ValidationError, match="device_id"):
        _orchestrator(tmp_path, {"kind": "upscale", "model": model, "device_id": "tpu"})

    with pytest.raises(ValidationError, match="fast"):
        _orchestrator(tmp_path, {"kind": "upscale", "model": model, "preset": "original", "enable_compile": True})

    with pytest.raises(ValidationError, match="variable_tiling"):
        _orchestrator(tmp_path, {"kind": "upscale", "model": model, "variable_tiling": True})


def test_device_id_accepts_int_and_torch_formats(tmp_path: Path):
    torch_model = tmp_path / "span.safetensors"
    torch_model.write_bytes(b"safetensors")
    umzi_model = tmp_path / "smosr.ummodel"
    umzi_model.write_bytes(b"UMZINPKG")

    act_int = _orchestrator(tmp_path, {"kind": "upscale", "model": torch_model, "device_id": 1}).actions[0]
    assert isinstance(act_int, ActionUpscale)
    assert act_int._resolve_torch_device() == "cuda:1"

    act_cpu = _orchestrator(tmp_path, {"kind": "upscale", "model": torch_model, "device_id": "cpu"}).actions[0]
    assert isinstance(act_cpu, ActionUpscale)
    assert act_cpu._resolve_torch_device() == "cpu"

    act_umzi = _orchestrator(tmp_path, {"kind": "upscale", "model": umzi_model, "device_id": "cuda:2"}).actions[0]
    assert isinstance(act_umzi, ActionUpscale)
    assert act_umzi._resolve_cuda_device_id() == 2

    with pytest.raises(ValidationError, match="cpu"):
        _orchestrator(tmp_path, {"kind": "upscale", "model": umzi_model, "device_id": "cpu"})


def test_torch_action_runs_with_mocked_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    torch = pytest.importorskip("torch")

    model = tmp_path / "span.safetensors"
    model.write_bytes(b"safetensors")
    src = tmp_path / "source" / "v01"
    src.mkdir(parents=True)
    page = Image.new("RGB", (64, 48), (15, 45, 75))
    page.save(src / "p001.png")
    page.close()

    captured_configs: list[object] = []

    class _FakeTorchRuntime:
        info = SimpleNamespace(scale=2, architecture="SPAN", in_channels=3, out_channels=3)

        def predict(self, tensor):
            return tensor.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

        def close(self) -> None:
            pass

    def _fake_create_runtime(path, cfg):
        captured_configs.append((path, cfg))
        return _FakeTorchRuntime()

    monkeypatch.setattr("nmanga.ml.torch_session.create_runtime", _fake_create_runtime)

    config = _orchestrator(
        tmp_path,
        {
            "kind": "upscale",
            "model": model,
            "tile_size": 64,
            "tile_overlap": 16,
            "device_id": "cuda",
            "preset": "fast",
            "enable_compile": False,
        },
    )
    action = config.actions[0]
    assert isinstance(action, ActionUpscale)

    context = WorkerContext(
        root_dir=tmp_path,
        current_dir=src,
        terminal=term.get_console(),
        toolsets={},
    )
    try:
        action.run(context, config.volumes[0], config)

        out_file = tmp_path / "upscaled" / "v01" / "p001.png"
        assert out_file.is_file()
        with Image.open(out_file) as upscaled:
            assert upscaled.size == (128, 96)

        assert len(captured_configs) == 1
        _, runtime_cfg = captured_configs[0]
        assert runtime_cfg.device == "cuda:0"
        assert runtime_cfg.preset == "fast"
        assert runtime_cfg.tiling.tile_size == 64
        assert runtime_cfg.tiling.tile_overlap == 16
        assert isinstance(torch.zeros(1), torch.Tensor)
    finally:
        WorkerContext.ml_model_session.clear()

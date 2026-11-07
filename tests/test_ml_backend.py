"""Tests for the PyTorch execution backend: device resolution, layout and the runtime contract."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nmanga.ml.contracts import ModelInfo, TilingConfig, TorchRuntimeConfig  # ruff: ignore[module-import-not-at-top-of-file] - after importorskip
from nmanga.ml.torch_session import (  # ruff: ignore[module-import-not-at-top-of-file]
    TorchRuntime,
    configure_torch_runtime,
    create_runtime,
    resolve_channels_last,
    resolve_device,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("NMANGA_TEST_MODELS", REPO_ROOT.parent / "umzi-bench" / "models"))
FDAT_M = MODELS_DIR / "2x_IllustrationJaNai_V3detail_FDAT_M_unshuffle_40k_fp16.safetensors"
SPAN_S = MODELS_DIR / "2x_IllustrationJaNai_V3detail_SPAN_S_40k_fp16.safetensors"
HAT_L = MODELS_DIR / "4x_IllustrationJaNai_V3detail_HAT_L_28k_bf16.safetensors"

requires_m = pytest.mark.skipif(not FDAT_M.is_file(), reason=f"missing test checkpoint {FDAT_M.name}")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")

TINY_INFO = ModelInfo(architecture="TEST", scale=2, in_channels=3, out_channels=3)


def test_importing_the_module_stays_lightweight():
    script = "import sys, nmanga.ml.torch_session; print('torch' in sys.modules, 'resselt' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    )

    assert result.stdout.strip() == "False False"


class Upscaler(torch.nn.Module):
    """A tiny stand-in with the same CPU-tensor contract as a real checkpoint."""

    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = torch.nn.functional.interpolate(self.conv(x), scale_factor=self.scale, mode="nearest")
        return scaled


def _tiny_runtime(**config: object) -> TorchRuntime:
    settings = TorchRuntimeConfig(**{"device": "cpu", "preset": "original", **config})
    return TorchRuntime(Upscaler(), TINY_INFO, config=settings)


def _psnr(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    delta = prediction.double() - reference.double()
    mse = delta.square().mean().item()
    if mse <= 0.0:
        return math.inf
    return -10.0 * math.log10(mse)


# --------------------------------------------------------------------------------------
# Device resolution
# --------------------------------------------------------------------------------------


def test_cpu_is_always_available():
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_prefers_cuda_only_when_it_is_available():
    device = resolve_device("auto")

    assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        assert device.index == torch.cuda.current_device()


@requires_cuda
def test_an_explicit_cuda_device_is_resolved():
    assert resolve_device("cuda:0") == torch.device("cuda", 0)
    assert resolve_device("cuda") == torch.device("cuda", torch.cuda.current_device())


def test_explicit_cuda_fails_clearly_when_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda:0")


def test_a_missing_cuda_device_fails_clearly(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="does not exist"):
        resolve_device("cuda:3")


def test_an_unknown_device_specification_is_rejected():
    with pytest.raises(ValueError, match="Unsupported device specification"):
        resolve_device("tpu")


def test_a_non_string_device_specification_is_rejected():
    with pytest.raises(TypeError):
        resolve_device(0)  # type: ignore[arg-type]


def test_the_runtime_configures_fp32_without_tf32():
    configure_torch_runtime()

    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


# --------------------------------------------------------------------------------------
# Runtime contract
# --------------------------------------------------------------------------------------


def test_the_runtime_exposes_the_expected_contract():
    runtime = _tiny_runtime()

    assert runtime.info is TINY_INFO
    assert runtime.preset == "original"
    assert runtime.device == torch.device("cpu")
    assert runtime.dtype == torch.float32
    assert runtime.channels_last is False
    assert runtime.compiled is False
    runtime.close()


def test_predict_returns_a_scaled_fp32_cpu_tensor():
    runtime = _tiny_runtime()
    sample = torch.rand(1, 3, 16, 20)
    preserved = sample.clone()

    result = runtime.predict(sample)

    assert result.shape == (1, 3, 32, 40)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert result.is_inference() is False
    result.add_(0)  # in-place use outside inference_mode must be allowed
    assert torch.equal(sample, preserved)
    runtime.close()


def test_the_runtime_is_callable():
    runtime = _tiny_runtime()

    assert runtime(torch.zeros(1, 3, 8, 8)).shape == (1, 3, 16, 16)
    runtime.close()


def test_close_is_idempotent_and_predict_after_close_fails():
    runtime = _tiny_runtime()
    runtime.close()
    runtime.close()

    with pytest.raises(RuntimeError, match="closed"):
        runtime.predict(torch.zeros(1, 3, 8, 8))


def test_the_runtime_is_a_context_manager():
    with _tiny_runtime() as runtime:
        assert runtime.predict(torch.zeros(1, 3, 8, 8)).shape == (1, 3, 16, 16)

    with pytest.raises(RuntimeError, match="closed"):
        runtime.predict(torch.zeros(1, 3, 8, 8))


@pytest.mark.parametrize(
    ("sample", "error"),
    [
        (torch.zeros(1, 3, 8, 8, dtype=torch.float64), ValueError),
        (torch.zeros(3, 8, 8), ValueError),
        (torch.zeros(2, 3, 8, 8), ValueError),
        (torch.zeros(1, 1, 8, 8), ValueError),
        (torch.zeros(1, 3, 0, 8), ValueError),
        ("not a tensor", TypeError),
    ],
)
def test_predict_rejects_inputs_outside_the_contract(sample: object, error: type[Exception]):
    runtime = _tiny_runtime()

    with pytest.raises(error):
        runtime.predict(sample)  # type: ignore[arg-type]
    runtime.close()


def test_an_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="Unknown preset"):
        TorchRuntime(Upscaler(), TINY_INFO, config=TorchRuntimeConfig(device="cpu", preset="turbo"))  # type: ignore[arg-type]


def test_compilation_requires_the_fast_preset():
    settings = TorchRuntimeConfig(device="cpu", preset="original", enable_compile=True)

    with pytest.raises(ValueError, match="only wired up for the `fast` preset"):
        TorchRuntime(Upscaler(), TINY_INFO, config=settings)


def test_compilation_requires_cuda():
    fast_info = ModelInfo(architecture="FDAT", scale=2, in_channels=3, out_channels=3)

    with pytest.raises(ValueError, match="needs a CUDA device"):
        TorchRuntime(Upscaler(), fast_info, config=TorchRuntimeConfig(device="cpu", enable_compile=True))


# --------------------------------------------------------------------------------------
# Real checkpoints
# --------------------------------------------------------------------------------------


@requires_m
def test_create_runtime_runs_the_fast_preset_to_end():
    runtime = create_runtime(
        FDAT_M,
        TorchRuntimeConfig(device="cpu", preset="fast", tiling=TilingConfig(tile_size=32, tile_overlap=0)),
    )
    try:
        assert runtime.info.architecture == "FDAT"
        assert runtime.info.scale == 2
        assert runtime.preset == "fast"
        assert runtime.dtype == torch.float32
        assert runtime.channels_last is False
        assert runtime.compiled is False
        assert runtime.load_seconds is not None and runtime.load_seconds >= 0
        result = runtime.predict(torch.rand(1, 3, 32, 32))
        assert result.shape == (1, 3, 64, 64)
        assert torch.isfinite(result).all()
    finally:
        runtime.close()


@requires_m
def test_create_runtime_runs_the_original_preset_to_end():
    runtime = create_runtime(
        FDAT_M,
        TorchRuntimeConfig(device="cpu", preset="original", tiling=TilingConfig(tile_size=32, tile_overlap=0)),
    )
    try:
        assert runtime.preset == "original"
        assert runtime.channels_last is False
        assert runtime.compiled is False
        assert runtime.predict(torch.zeros(1, 3, 32, 32)).shape == (1, 3, 64, 64)
    finally:
        runtime.close()


@pytest.mark.skipif(not HAT_L.is_file(), reason=f"missing test checkpoint {HAT_L.name}")
def test_create_runtime_rejects_the_fast_preset_for_an_architecture_without_a_recipe():
    with pytest.raises(ValueError, match="--preset original"):
        create_runtime(HAT_L, TorchRuntimeConfig(device="cpu", preset="fast"))


@requires_m
def test_create_runtime_rejects_a_missing_checkpoint(tmp_path: Path):
    with pytest.raises(ValueError, match="Checkpoint not found"):
        create_runtime(tmp_path / "nope.safetensors", TorchRuntimeConfig(device="cpu"))


# --------------------------------------------------------------------------------------
# CUDA recipes
# --------------------------------------------------------------------------------------


@requires_m
@requires_cuda
def test_the_fast_cuda_runtime_uses_channels_last():
    runtime = create_runtime(
        FDAT_M,
        TorchRuntimeConfig(device="cuda", preset="fast", tiling=TilingConfig(tile_size=64)),
    )
    try:
        assert runtime.device.type == "cuda"
        assert runtime.channels_last is True
        assert runtime.compiled is False
        result = runtime.predict(torch.rand(1, 3, 64, 64))
        assert result.device.type == "cpu"
        assert result.shape == (1, 3, 128, 128)
        assert torch.isfinite(result).all()
    finally:
        runtime.close()


@requires_m
@requires_cuda
def test_the_original_cuda_runtime_does_not_use_channels_last():
    runtime = create_runtime(
        FDAT_M,
        TorchRuntimeConfig(device="cuda", preset="original", tiling=TilingConfig(tile_size=64)),
    )
    try:
        assert runtime.channels_last is False
    finally:
        runtime.close()


@requires_cuda
def test_predict_rejects_a_cuda_tensor():
    runtime = _tiny_runtime()

    with pytest.raises(ValueError, match="CPU tensor"):
        runtime.predict(torch.zeros(1, 3, 8, 8, device="cuda"))
    runtime.close()


@requires_m
@requires_cuda
def test_the_compiled_runtime_matches_the_eager_one():
    settings = {"device": "cuda", "preset": "fast", "tiling": TilingConfig(tile_size=64)}
    sample = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(7))
    eager = create_runtime(FDAT_M, TorchRuntimeConfig(**settings))
    try:
        reference = eager.predict(sample)
    finally:
        eager.close()
    try:
        compiled = create_runtime(FDAT_M, TorchRuntimeConfig(enable_compile=True, **settings))
    except RuntimeError as exc:
        pytest.skip(f"torch.compile is unavailable in this environment: {exc}")
    try:
        assert compiled.compiled is True
        assert compiled.compile_seconds is not None
        optimized = compiled.predict(sample)
    finally:
        compiled.close()

    assert optimized.shape == reference.shape
    assert _psnr(reference, optimized) > 90.0


# --------------------------------------------------------------------------------------
# Layout policy
# --------------------------------------------------------------------------------------


def _arch(name: str) -> ModelInfo:
    return ModelInfo(architecture=name, scale=2, in_channels=3, out_channels=3)


def test_only_fdat_selects_channels_last_by_default():
    """Channels-last is a measured win for FDAT and unmeasured for everything else, so the
    other architectures stay contiguous until their own comparison justifies the change."""
    assert resolve_channels_last(_arch("FDAT"), preset="fast", device_type="cuda", override=None) is True
    for architecture in ("SPAN", "GateRV3", "HAT"):
        assert resolve_channels_last(_arch(architecture), preset="fast", device_type="cuda", override=None) is False


def test_an_explicit_override_wins_over_the_adapter_policy():
    assert resolve_channels_last(_arch("SPAN"), preset="fast", device_type="cuda", override=True) is True
    assert resolve_channels_last(_arch("FDAT"), preset="fast", device_type="cuda", override=False) is False


def test_the_untouched_recipe_never_uses_channels_last():
    for architecture in ("FDAT", "SPAN", "GateRV3"):
        info = _arch(architecture)
        assert resolve_channels_last(info, preset="original", device_type="cuda", override=False) is False
        # An explicit override cannot turn on a layout the untouched model was not measured with.
        assert resolve_channels_last(info, preset="original", device_type="cuda", override=True) is False


def test_a_cpu_runtime_never_uses_channels_last():
    assert resolve_channels_last(_arch("FDAT"), preset="fast", device_type="cpu", override=True) is False


def test_the_runtime_reports_where_its_layout_came_from():
    automatic = _tiny_runtime()
    assert automatic.channels_last_explicit is False

    overridden = _tiny_runtime(use_channels_last=False)
    assert overridden.channels_last_explicit is True
    assert overridden.channels_last is False
    overridden.close()
    automatic.close()


@pytest.mark.skipif(not SPAN_S.is_file(), reason=f"missing test checkpoint {SPAN_S.name}")
def test_create_runtime_runs_the_span_fast_preset_to_end():
    runtime = create_runtime(
        SPAN_S,
        TorchRuntimeConfig(device="cpu", preset="fast", tiling=TilingConfig(tile_size=32, tile_overlap=0)),
    )
    try:
        assert runtime.info.architecture == "SPAN"
        assert runtime.preset == "fast"
        # SPAN is not FDAT: the adapter's layout policy is contiguous even on CUDA.
        assert runtime.channels_last is False
        sample = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(11))
        before = {name: tensor.clone() for name, tensor in runtime._model.state_dict().items()}
        result = runtime.predict(sample)
        assert result.shape == (1, 3, 64, 64)
        assert result.device.type == "cpu"
        assert torch.isfinite(result).all()
        # Repeated predictions are deterministic and never move a weight.
        assert torch.equal(result, runtime.predict(sample))
        for name, tensor in runtime._model.state_dict().items():
            assert torch.equal(before[name], tensor), name
    finally:
        runtime.close()

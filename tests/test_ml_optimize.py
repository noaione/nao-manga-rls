"""Tests for the resselt FDAT lean rewrite preset."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nmanga.ml.contracts import ModelInfo  # ruff: ignore[module-import-not-at-top-of-file] - after importorskip
from nmanga.ml.model import load_checkpoint  # ruff: ignore[module-import-not-at-top-of-file]
from nmanga.ml.opt_arch import adapter_for, build_fast_model, supports_fast_preset  # ruff: ignore[module-import-not-at-top-of-file]
from nmanga.ml.opt_arch import span as span_adapter  # ruff: ignore[module-import-not-at-top-of-file]

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("NMANGA_TEST_MODELS", REPO_ROOT.parent / "umzi-bench" / "models"))
FDAT_M = MODELS_DIR / "2x_IllustrationJaNai_V3detail_FDAT_M_unshuffle_40k_fp16.safetensors"
FDAT_XL = MODELS_DIR / "4x_IllustrationJaNai_V3detail_FDAT_XL_27k_bf16.safetensors"
SPAN_S = MODELS_DIR / "2x_IllustrationJaNai_V3detail_SPAN_S_40k_fp16.safetensors"

requires_m = pytest.mark.skipif(not FDAT_M.is_file(), reason=f"missing test checkpoint {FDAT_M.name}")
requires_xl = pytest.mark.skipif(not FDAT_XL.is_file(), reason=f"missing test checkpoint {FDAT_XL.name}")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")

TILE = 256
"""The validated tile size; the parity gate is stated for it."""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_importing_the_module_stays_lightweight():
    script = "import sys, nmanga.ml.opt_arch; print('torch' in sys.modules, 'resselt' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    )

    assert result.stdout.strip() == "False False"


def test_the_registry_knows_which_architectures_have_a_recipe():
    for architecture in ("FDAT", "SPAN", "GateRV3", "GATERV3"):
        assert supports_fast_preset(ModelInfo(architecture=architecture, scale=2, in_channels=3, out_channels=3))
    for architecture in ("HAT", "SPANPlus", "GFISR", "DAT2"):
        assert not supports_fast_preset(ModelInfo(architecture=architecture, scale=4, in_channels=3, out_channels=3))


def test_an_unsupported_architecture_is_rejected_with_a_hint():
    info = ModelInfo(architecture="HAT", scale=4, in_channels=3, out_channels=3)

    with pytest.raises(ValueError, match="--preset original"):
        build_fast_model(torch.nn.Identity(), info)


def _reference_cases(size: int = TILE) -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(20261107)
    cases: list[tuple[str, torch.Tensor]] = [("random", torch.rand((1, 3, size, size), generator=generator))]
    for value in (0.0, 0.5, 1.0):
        cases.append((f"constant:{value}", torch.full((1, 3, size, size), value)))
    grid = ((torch.arange(size)[:, None] + torch.arange(size)[None, :]) % 2).float()
    cases.append(("checkerboard", grid.expand(1, 3, size, size).contiguous()))
    return cases


def _psnr(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    delta = prediction.double() - reference.double()
    mse = delta.square().mean().item()
    if mse <= 0.0:
        return math.inf
    return -10.0 * math.log10(mse)


def _predict_all(model: torch.nn.Module, cases: list[tuple[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    model = model.eval().to(DEVICE)
    outputs: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for name, sample in cases:
            outputs[name] = model(sample.to(DEVICE)).float().cpu()
    return outputs


@requires_m
def test_build_fast_model_requires_an_eval_model():
    model, info = load_checkpoint(FDAT_M)
    model.train()

    with pytest.raises(ValueError, match="evaluation model"):
        build_fast_model(model, info)


@requires_m
def test_build_fast_model_requires_fp32():
    model, info = load_checkpoint(FDAT_M)
    model.half()

    with pytest.raises(ValueError, match="FP32"):
        build_fast_model(model, info)


@requires_m
def test_build_fast_model_does_not_mutate_the_source():
    model, info = load_checkpoint(FDAT_M)
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    gelu_count = sum(1 for module in model.modules() if isinstance(module, torch.nn.GELU))

    fast = build_fast_model(model, info)

    assert fast is not model
    for name, tensor in model.state_dict().items():
        assert torch.equal(state[name], tensor), name
    assert sum(1 for module in model.modules() if isinstance(module, torch.nn.GELU)) == gelu_count
    assert not any(type(module).__name__.startswith("Lean") for module in model.modules())
    source_pointers = {parameter.data_ptr() for parameter in model.parameters()}
    fast_pointers = {parameter.data_ptr() for parameter in fast.parameters()}
    assert not source_pointers & fast_pointers
    assert fast.training is False
    assert all(parameter.requires_grad is False for parameter in fast.parameters())


@requires_m
def test_fast_model_rewrites_the_reselt_modules():
    model, info = load_checkpoint(FDAT_M)
    fast = build_fast_model(model, info)

    def count(module: torch.nn.Module, name: str) -> int:
        return sum(1 for child in module.modules() if type(child).__name__ == name)

    # Every GELU becomes the in-place variant; the DAT blocks, their feed-forward and the spatial
    # attentions are replaced; the channel attention is left untouched.
    assert count(model, "GELU") == count(fast, "InplaceGELU")
    assert count(model, "FastSpatialWindowAttention") == count(fast, "LeanSpatialAttention") > 0
    assert count(fast, "FastSpatialWindowAttention") == 0
    assert count(model, "FastChannelAttention") == count(fast, "FastChannelAttention") > 0
    assert count(model, "SimplifiedDATBlock") == count(fast, "LeanDATBlock") > 0
    assert count(fast, "SimplifiedDATBlock") == 0
    assert count(model, "SimplifiedFFN") == count(fast, "LeanFFN") > 0
    assert count(fast, "SimplifiedFFN") == 0
    assert count(model, "SimplifiedAIM") == count(fast, "SimplifiedAIM") > 0
    # No weight is dropped or added by the rewrites.
    assert {name for name, _ in model.named_parameters()} == {name for name, _ in fast.named_parameters()}
    assert count(fast, "Dropout") == 0


@requires_m
def test_a_broken_block_structure_is_reported():
    model, info = load_checkpoint(FDAT_M)
    model.groups[0].blocks[0] = torch.nn.Identity()

    with pytest.raises(ValueError, match="SimplifiedDATBlock"):
        build_fast_model(model, info)


@requires_m
def test_parity_on_the_small_fdat():
    model, info = load_checkpoint(FDAT_M)
    fast = build_fast_model(model, info)
    cases = _reference_cases()

    reference = _predict_all(model, cases)
    optimized = _predict_all(fast, cases)

    assert optimized.keys() == reference.keys()
    for name in reference:
        psnr = _psnr(reference[name], optimized[name])
        assert psnr > 90.0, f"{name}: {psnr} dB"


@requires_xl
@requires_cuda
def test_parity_on_the_xl_fdat():
    model, info = load_checkpoint(FDAT_XL)
    fast = build_fast_model(model, info)
    cases = _reference_cases()

    reference = _predict_all(model, cases)
    optimized = _predict_all(fast, cases)

    for name in reference:
        psnr = _psnr(reference[name], optimized[name])
        assert psnr > 90.0, f"{name}: {psnr} dB"


@requires_xl
def test_the_xl_checkpoint_is_rewritten_the_same_way():
    model, info = load_checkpoint(FDAT_XL)
    fast = build_fast_model(model, info)

    assert sum(1 for module in fast.modules() if type(module).__name__ == "LeanDATBlock") == 72
    sample = _reference_cases(32)[0][1]
    with torch.inference_mode():
        reference = model(sample)
        optimized = fast(sample)

    assert torch.isfinite(optimized).all()
    assert optimized.shape == reference.shape
    assert _psnr(reference, optimized) > 90.0


@pytest.mark.skipif(not SPAN_S.is_file(), reason=f"missing test checkpoint {SPAN_S.name}")
def test_span_gets_its_own_adapter_and_no_fdat_rewrites():
    model, info = load_checkpoint(SPAN_S)
    assert supports_fast_preset(info)
    assert adapter_for(info.architecture) is span_adapter

    fast = build_fast_model(model, info)

    # SPAN has its own recipe: the FDAT rewrites must not leak into it.
    fdat_rewrites = {"InplaceGELU", "LeanSpatialAttention", "LeanFFN", "LeanDATBlock"}
    assert not any(type(module).__name__ in fdat_rewrites for module in fast.modules())
    assert any(type(module).__name__ == "LeanSPAB" for module in fast.modules())
    # The source model is independent of the candidate.
    assert any(type(module).__name__ == "SPAB" for module in model.modules())
    assert not any(type(module).__name__ == "LeanSPAB" for module in model.modules())

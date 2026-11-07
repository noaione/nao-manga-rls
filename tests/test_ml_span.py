"""Tests for the resselt SPAN lean rewrite preset."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nmanga.ml.contracts import ModelInfo  # ruff: ignore[module-import-not-at-top-of-file] - after importorskip
from nmanga.ml.model import load_checkpoint, read_model_info  # ruff: ignore[module-import-not-at-top-of-file]
from nmanga.ml.opt_arch import (  # ruff: ignore[module-import-not-at-top-of-file]
    build_fast_model,
    layout_for,
    supports_fast_preset,
)
from nmanga.ml.opt_arch import span as span_adapter  # ruff: ignore[module-import-not-at-top-of-file]

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("NMANGA_TEST_MODELS", REPO_ROOT.parent / "umzi-bench" / "models"))
SPAN_S = MODELS_DIR / "2x_IllustrationJaNai_V3detail_SPAN_S_40k_fp16.safetensors"

requires_span = pytest.mark.skipif(not SPAN_S.is_file(), reason=f"missing test checkpoint {SPAN_S.name}")

TILE = 64
"""A small parity tile; the rewrite is exact, so the size only has to exercise the arithmetic."""


def test_importing_the_module_stays_lightweight():
    script = "import sys, nmanga.ml.opt_arch.span; print('torch' in sys.modules, 'resselt' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    )

    assert result.stdout.strip() == "False False"


def test_span_is_registered_and_stays_contiguous():
    assert supports_fast_preset(ModelInfo(architecture="SPAN", scale=2, in_channels=3, out_channels=3))
    # Channels-last is only a measured win for FDAT; SPAN starts at the contiguous layout.
    assert layout_for("SPAN") is False
    assert layout_for("FDAT") is True


def test_the_adapter_is_dispatched_by_architecture():
    from nmanga.ml.opt_arch import adapter_for

    assert adapter_for("SPAN") is span_adapter
    assert adapter_for("span") is span_adapter
    assert adapter_for("SPANPlus") is None


def test_an_unsupported_architecture_is_rejected_with_a_hint():
    info = ModelInfo(architecture="HAT", scale=4, in_channels=3, out_channels=3)

    with pytest.raises(ValueError, match="--preset original"):
        span_adapter.build_fast_model(torch.nn.Identity(), info)


def _span_classes() -> tuple[type, type, type]:
    from resselt.archs.span.arch import SPAB, SPAN, Conv3XC

    return (SPAN, SPAB, Conv3XC)


def _synthetic_span(*, norm: bool, feature_channels: int = 16) -> torch.nn.Module:
    """Build a small SPAN of the installed resselt class, with random weights."""
    span, _spab, _conv3xc = _span_classes()
    torch.manual_seed(20261107)
    return span(num_in_ch=3, num_out_ch=3, feature_channels=feature_channels, upscale=2, norm=norm).eval()


def _synthetic_info(scale: int = 2) -> ModelInfo:
    return ModelInfo(architecture="SPAN", scale=scale, in_channels=3, out_channels=3, loader_version="1.4.1")


def _reference_cases(size: int = TILE) -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(20261107)
    cases: list[tuple[str, torch.Tensor]] = [
        ("random", torch.rand((1, 3, size, size), generator=generator)),
        # Signed input, so a rewrite that silently clamps or folds the residual is caught.
        ("signed", torch.rand((1, 3, size, size), generator=generator) * 2.0 - 1.0),
        # Odd, non-square, and not a multiple of the pixel-shuffle factor.
        ("odd", torch.rand((1, 3, size - 1, size + 3), generator=generator)),
    ]
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
    model = model.eval()
    outputs: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for name, sample in cases:
            outputs[name] = model(sample).float()
    return outputs


@requires_span
def test_build_fast_model_requires_an_eval_model():
    model, info = load_checkpoint(SPAN_S)
    model.train()

    with pytest.raises(ValueError, match="evaluation model"):
        build_fast_model(model, info)


@requires_span
def test_build_fast_model_requires_fp32():
    model, info = load_checkpoint(SPAN_S)
    model.half()

    with pytest.raises(ValueError, match="FP32"):
        build_fast_model(model, info)


@requires_span
def test_build_fast_model_does_not_mutate_the_source():
    model, info = load_checkpoint(SPAN_S)
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    _span, spab, conv3xc = _span_classes()
    conv3xc_count = sum(1 for module in model.modules() if isinstance(module, conv3xc))
    spab_count = sum(1 for module in model.modules() if isinstance(module, spab))

    fast = build_fast_model(model, info)

    assert fast is not model
    for name, tensor in model.state_dict().items():
        assert torch.equal(state[name], tensor), name
    # The source keeps every training branch and its per-prediction reparameterization.
    assert sum(1 for module in model.modules() if isinstance(module, conv3xc)) == conv3xc_count
    assert sum(1 for module in model.modules() if isinstance(module, spab)) == spab_count
    source_pointers = {parameter.data_ptr() for parameter in model.parameters()}
    fast_pointers = {parameter.data_ptr() for parameter in fast.parameters()}
    assert not source_pointers & fast_pointers
    assert fast.training is False
    assert all(parameter.requires_grad is False for parameter in fast.parameters())


@requires_span
def test_the_rewrite_collapses_every_wrapper():
    _span, spab, conv3xc = _span_classes()
    model, info = load_checkpoint(SPAN_S)

    def count(module: torch.nn.Module, kind: type) -> int:
        return sum(1 for child in module.modules() if isinstance(child, kind))

    # The supplied checkpoint: 1 + 6 * 3 + 1 convolutions and one block per stage.
    assert count(model, conv3xc) == 20
    assert count(model, spab) == 6

    fast = build_fast_model(model, info)

    assert count(fast, conv3xc) == 0
    assert count(fast, spab) == 0
    assert count(fast, type(fast.block_1)) == 6
    assert type(fast.block_1).__name__ == "LeanSPAB"
    # Only the six inference kernels and the two convolutions outside the blocks remain.
    assert count(fast, torch.nn.Conv2d) == 20 + 2


@requires_span
def test_the_rewrite_drops_the_training_parameter_storage():
    model, info = load_checkpoint(SPAN_S)

    fast = build_fast_model(model, info)

    # The fused kernels plus the weights the rewrite carries over; the branches are gone.
    assert sum(parameter.numel() for parameter in fast.parameters()) * 4 == 1_642_800
    assert sum(parameter.numel() for parameter in fast.parameters()) < sum(
        parameter.numel() for parameter in model.parameters()
    )


@requires_span
def test_a_stale_inference_kernel_is_recomputed():
    """
    resselt's SPAN refreshes its fused kernel on every prediction, so the tensors a checkpoint
    stores are only a snapshot. Corrupting them must not change either side of the comparison.
    """
    _span, _spab, conv3xc = _span_classes()
    model, info = load_checkpoint(SPAN_S)
    corrupted = 0
    with torch.inference_mode():
        for module in model.modules():
            if isinstance(module, conv3xc):
                module.eval_conv.weight.add_(torch.randn_like(module.eval_conv.weight))
                corrupted += 1

    assert corrupted == 20
    fast = build_fast_model(model, info)
    cases = _reference_cases(32)

    reference = _predict_all(model, cases)
    optimized = _predict_all(fast, cases)

    for name in reference:
        assert _psnr(reference[name], optimized[name]) > 90.0, name


@requires_span
def test_a_broken_block_structure_is_reported():
    model, info = load_checkpoint(SPAN_S)
    model.block_6.end = False

    with pytest.raises(ValueError, match="end=True"):
        build_fast_model(model, info)


@requires_span
def test_an_out_of_place_activation_is_reported():
    model, info = load_checkpoint(SPAN_S)
    model.block_2.act1 = torch.nn.SiLU(inplace=False)

    with pytest.raises(ValueError, match="out-of-place SiLU"):
        build_fast_model(model, info)


def test_a_pixel_unshuffle_is_not_reported_as_one():
    _span, _spab, _conv3xc = _span_classes()
    model = _synthetic_span(norm=False)
    assert span_adapter.unshuffle_of(model, Path("model.safetensors")) == 1

    model.conv_1 = torch.nn.Sequential(torch.nn.PixelUnshuffle(2), model.conv_1)
    with pytest.raises(ValueError, match="PixelUnshuffle"):
        span_adapter.unshuffle_of(model, Path("model.safetensors"))


@requires_span
def test_the_terminal_block_returns_the_activated_feature():
    """
    resselt's `SPAB.forward` activates its first branch in place and returns that buffer, so the
    tuple's second value is the activated feature. The lean block has to return the same thing.
    """
    model, info = load_checkpoint(SPAN_S)
    fast = build_fast_model(model, info)
    sample = torch.rand(1, model.block_6.in_channels, 32, 32)

    with torch.inference_mode():
        reference_out, reference_feature = model.block_6(sample)
        optimized = fast.block_6(sample)

    assert isinstance(optimized, tuple) and len(optimized) == 2
    optimized_out, optimized_feature = optimized
    assert torch.equal(optimized_feature, reference_feature)
    assert torch.equal(optimized_out, reference_out)
    # The terminal block is the only one that returns a tuple.
    with torch.inference_mode():
        assert not isinstance(fast.block_1(sample), tuple)


@requires_span
def test_the_lean_block_does_not_mutate_its_input():

    model, _info = load_checkpoint(SPAN_S)
    (lean_spab,) = span_adapter._lean_classes()
    block = lean_spab(model.block_1)
    sample = torch.rand(1, model.block_1.in_channels, 8, 8)

    with torch.inference_mode():
        reference = block.forward(sample.clone())
        preserved = sample.clone()
        optimized = block.forward(sample)

    # The gate reuses the third convolution's buffer, never the caller's tensor.
    assert torch.equal(sample, preserved)
    assert torch.equal(reference, optimized)
    assert torch.isfinite(optimized).all()


@requires_span
def test_repeated_predictions_leave_the_weights_stable():
    model, info = load_checkpoint(SPAN_S)
    fast = build_fast_model(model, info)
    before = {name: tensor.clone() for name, tensor in fast.state_dict().items()}
    sample = _reference_cases(32)[0][1]

    with torch.inference_mode():
        first = fast(sample)
        for _ in range(3):
            fast(sample)

    for name, tensor in fast.state_dict().items():
        assert torch.equal(before[name], tensor), name
    # Nothing calls update_params any more: the wrapper that used to own it is gone.
    assert not any(hasattr(module, "update_params") for module in fast.modules())
    with torch.inference_mode():
        assert torch.equal(first, fast(sample))


@requires_span
def test_parity_on_the_supplied_checkpoint():
    model, info = load_checkpoint(SPAN_S)
    fast = build_fast_model(model, info)
    cases = _reference_cases()

    reference = _predict_all(model, cases)
    optimized = _predict_all(fast, cases)

    assert optimized.keys() == reference.keys()
    for name in reference:
        assert optimized[name].shape == reference[name].shape, name
        assert torch.isfinite(optimized[name]).all(), name
        assert _psnr(reference[name], optimized[name]) > 90.0, name


def test_parity_on_a_normalized_synthetic_model():
    """
    The supplied JaNai checkpoint has normalization disabled. The rewrite must not depend on
    that, so the normalized path is exercised on a small synthetic model too.
    """
    model = _synthetic_span(norm=True)
    info = _synthetic_info()

    assert model.is_norm is True
    fast = build_fast_model(model, info)
    assert fast.is_norm is True
    assert isinstance(fast.mean, torch.Tensor) and torch.equal(fast.mean, model.mean)
    assert fast.img_range == model.img_range

    cases = _reference_cases(32)
    reference = _predict_all(model, cases)
    optimized = _predict_all(fast, cases)

    for name in reference:
        assert _psnr(reference[name], optimized[name]) > 90.0, name


def test_a_synthetic_model_without_normalization_matches_the_rewrite():
    model = _synthetic_span(norm=False)
    info = _synthetic_info()

    assert model.is_norm is False
    fast = build_fast_model(model, info)

    cases = _reference_cases(32)
    reference = _predict_all(model, cases)
    optimized = _predict_all(fast, cases)

    for name in reference:
        assert _psnr(reference[name], optimized[name]) > 90.0, name


@requires_span
def test_the_header_hook_declines_so_the_loader_owns_the_description():
    info = read_model_info(SPAN_S)

    assert (
        span_adapter.info_from_header(  # every argument is required, but nothing is read
            SPAN_S,
            {},
            architecture="SPAN",
            stored_dtypes=(),
            checkpoint_sha256="",
            loader_version=None,
        )
        is None
    )
    assert info == load_checkpoint(SPAN_S)[1]
    assert info.unshuffle == 1
    assert info.in_channels == 3
    assert info.scale == 2


@requires_span
def test_the_span_contract_is_cross_checked():
    model, info = load_checkpoint(SPAN_S)
    span_adapter.cross_check(model, info, SPAN_S)

    wrong = ModelInfo(architecture="SPAN", scale=4, in_channels=3, out_channels=3, loader_version="1.4.1")
    with pytest.raises(ValueError, match="does not match the constructed model"):
        span_adapter.cross_check(model, wrong, SPAN_S)


@requires_span
def test_the_span_adapter_only_touches_span_modules():
    """The FDAT rewrites must never appear in a SPAN candidate."""
    model, info = load_checkpoint(SPAN_S)
    fast = build_fast_model(model, info)

    fdat_names = {"InplaceGELU", "LeanSpatialAttention", "LeanFFN", "LeanDATBlock"}
    assert not any(type(module).__name__ in fdat_names for module in fast.modules())

"""Tests for the resselt GateRV3 lean rewrite preset."""

from __future__ import annotations

import copy
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nmanga.ml.contracts import ModelInfo  # ruff: ignore[module-import-not-at-top-of-file] - after importorskip
from nmanga.ml.model import load_checkpoint, read_model_info  # ruff: ignore[module-import-not-at-top-of-file]
from nmanga.ml.opt_arch import build_fast_model, supports_fast_preset  # ruff: ignore[module-import-not-at-top-of-file]
from nmanga.ml.opt_arch import gaterv3 as gaterv3_adapter  # ruff: ignore[module-import-not-at-top-of-file]

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("NMANGA_TEST_MODELS", REPO_ROOT.parent / "umzi-bench" / "models"))
GATERV3 = MODELS_DIR / "1x_umzi_digital_decompress_gaterv3_1.pth"

requires_gaterv3 = pytest.mark.skipif(not GATERV3.is_file(), reason=f"missing test checkpoint {GATERV3.name}")

SIZE = 64
"""A small parity tile; the folds only change rounding, so the size only has to exercise them."""


def _adapter_classes() -> tuple[type, ...]:
    """The resselt classes the assertions below name, imported only where torch already exists."""
    from resselt.archs.gaterv3.arch import (
        SPAB,
        Attention,
        Block,
        Conv3XC,
        GatedCNNBlock,
        GateRV3,
        InceptionDWConv2d,
        MetaGated,
        RMSNorm,
    )

    return (GateRV3, Conv3XC, RMSNorm, MetaGated, GatedCNNBlock, Block, SPAB, Attention, InceptionDWConv2d)


def _reference_cases(size: int = SIZE) -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(20261107)
    cases: list[tuple[str, torch.Tensor]] = [
        ("random", torch.rand((1, 3, size, size), generator=generator)),
        # Signed input, so a fold that silently clamps or drops a residual is caught.
        ("signed", torch.rand((1, 3, size, size), generator=generator) * 2.0 - 1.0),
        # Odd, non-square, and not a multiple of the internal downsampling factor.
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


def _count(module: torch.nn.Module, kind: type) -> int:
    return sum(1 for child in module.modules() if isinstance(child, kind))


@pytest.fixture(scope="module")
def checkpoint() -> tuple[torch.nn.Module, ModelInfo]:
    """The untouched resselt model, loaded once for the whole module."""
    return load_checkpoint(GATERV3)


@pytest.fixture(scope="module")
def fast(checkpoint: tuple[torch.nn.Module, ModelInfo]) -> torch.nn.Module:
    """The rewritten candidate, built once: `build_fast_model` never mutates its source."""
    return build_fast_model(*checkpoint)


def test_importing_the_adapter_stays_lightweight():
    script = "import sys, nmanga.ml.opt_arch.gaterv3; print('torch' in sys.modules, 'resselt' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    )

    assert result.stdout.strip() == "False False"


def test_gate_rv3_is_registered_and_stays_contiguous():
    from nmanga.ml.opt_arch import adapter_for, layout_for

    assert supports_fast_preset(ModelInfo(architecture="GateRV3", scale=1, in_channels=3, out_channels=3))
    assert supports_fast_preset(ModelInfo(architecture="GATERV3", scale=1, in_channels=3, out_channels=3))
    assert adapter_for("GateRV3") is gaterv3_adapter
    # Channels-last is only a measured win for FDAT; GateRV3 starts at the contiguous layout.
    assert layout_for("GateRV3") is False


def test_the_loader_contract_is_implemented():
    assert gaterv3_adapter.ARCHITECTURE == "GateRV3"
    for hook in ("build_fast_model", "cross_check", "info_from_header", "unshuffle_of"):
        assert callable(getattr(gaterv3_adapter, hook))

    # The header hook declines every argument it is handed, so the loader owns the description.
    assert (
        gaterv3_adapter.info_from_header(
            GATERV3,
            {},
            architecture="GateRV3",
            stored_dtypes=(),
            checkpoint_sha256="",
            loader_version=None,
        )
        is None
    )


def test_an_unsupported_architecture_is_rejected_with_a_hint():
    info = ModelInfo(architecture="HAT", scale=4, in_channels=3, out_channels=3)

    with pytest.raises(ValueError, match="--preset original"):
        gaterv3_adapter.build_fast_model(torch.nn.Identity(), info)


@requires_gaterv3
def test_the_header_hook_declines_so_the_loader_owns_the_description(checkpoint: tuple[torch.nn.Module, ModelInfo]):
    _model, info = checkpoint

    assert read_model_info(GATERV3) == info
    assert info.architecture == "GateRV3"
    assert info.scale == 1
    assert info.in_channels == 3
    assert info.unshuffle == 1


@requires_gaterv3
def test_unshuffle_of_the_checkpoint_is_one(checkpoint: tuple[torch.nn.Module, ModelInfo]):
    model, _info = checkpoint

    assert gaterv3_adapter.unshuffle_of(model, GATERV3) == 1

    # Only the encoder's internal downsampling may unshuffle pixels; an input-level one is refused.
    broken = copy.deepcopy(model)
    broken.in_to_dim = torch.nn.Sequential(torch.nn.PixelUnshuffle(2), broken.in_to_dim)

    with pytest.raises(ValueError, match="PixelUnshuffle"):
        gaterv3_adapter.unshuffle_of(broken, GATERV3)


@requires_gaterv3
def test_the_contract_is_cross_checked(checkpoint: tuple[torch.nn.Module, ModelInfo]):
    model, info = checkpoint

    gaterv3_adapter.cross_check(model, info, GATERV3)

    wrong_scale = ModelInfo(architecture="GateRV3", scale=2, in_channels=3, out_channels=3, loader_version="1.4.1")
    with pytest.raises(ValueError, match="does not match the constructed model"):
        gaterv3_adapter.cross_check(model, wrong_scale, GATERV3)

    wrong_channels = ModelInfo(architecture="GateRV3", scale=1, in_channels=4, out_channels=3, loader_version="1.4.1")
    with pytest.raises(ValueError, match="in_to_dim takes 3 channels, expected 4"):
        gaterv3_adapter.cross_check(model, wrong_channels, GATERV3)


@requires_gaterv3
def test_build_fast_model_requires_an_eval_model(checkpoint: tuple[torch.nn.Module, ModelInfo]):
    model, info = checkpoint
    training = copy.deepcopy(model).train()

    with pytest.raises(ValueError, match="evaluation forward pass"):
        build_fast_model(training, info)


@requires_gaterv3
def test_build_fast_model_requires_fp32(checkpoint: tuple[torch.nn.Module, ModelInfo]):
    model, info = checkpoint
    half = copy.deepcopy(model).half()

    with pytest.raises(ValueError, match="FP32"):
        build_fast_model(half, info)


@requires_gaterv3
def test_build_fast_model_does_not_mutate_the_source(checkpoint: tuple[torch.nn.Module, ModelInfo]):
    model, info = checkpoint
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    shapes = sorted(type(module).__name__ for module in model.modules())

    rebuilt = build_fast_model(model, info)

    assert rebuilt is not model
    for name, tensor in model.state_dict().items():
        assert torch.equal(state[name], tensor), name
    # The source keeps every normalization, residual scale and training branch it was loaded with.
    assert sorted(type(module).__name__ for module in model.modules()) == shapes

    source_pointers = {parameter.data_ptr() for parameter in model.parameters()}
    candidate_pointers = {parameter.data_ptr() for parameter in rebuilt.parameters()}
    assert not source_pointers & candidate_pointers
    assert rebuilt.training is False
    assert all(parameter.requires_grad is False for parameter in rebuilt.parameters())


@requires_gaterv3
def test_a_broken_structure_is_reported(checkpoint: tuple[torch.nn.Module, ModelInfo]):
    model, info = checkpoint

    broken = copy.deepcopy(model)
    broken.gater_encode[0] = torch.nn.Identity()
    with pytest.raises(ValueError, match="encoder Block"):
        build_fast_model(broken, info)

    broken = copy.deepcopy(model)
    broken.latent[3] = torch.nn.Identity()
    with pytest.raises(ValueError, match="GatedCNNBlock"):
        build_fast_model(broken, info)

    broken = copy.deepcopy(model)
    broken.decode[2].gated[0] = torch.nn.Identity()
    with pytest.raises(ValueError, match="MetaGated"):
        build_fast_model(broken, info)

    broken = copy.deepcopy(model)
    broken.decode[0].shor = torch.nn.Conv2d(512, 256, 3, 1, 1)
    with pytest.raises(ValueError, match="--preset original"):
        build_fast_model(broken, info)


@requires_gaterv3
def test_a_stale_inference_kernel_is_recomputed(checkpoint: tuple[torch.nn.Module, ModelInfo], fast: torch.nn.Module):
    """
    GateRV3 fuses on `train(False)`, so the stored kernel is a snapshot of the branches.

    Corrupting it must change neither side of the comparison: the rewrite recomputes the kernel
    from the live 1x1/3x3/1x1 branch and the side kernel instead of copying the stored one.
    """
    model, info = checkpoint
    broken = copy.deepcopy(model)
    with torch.inference_mode():
        broken.span_n_b[0].c1_r.eval_conv.weight.zero_()
    assert not broken.span_n_b[0].c1_r.eval_conv.weight.any()

    rebuilt = build_fast_model(broken, info)
    fused = rebuilt.span_n_b[0].c1_r

    assert isinstance(fused, torch.nn.Conv2d)
    assert fused.weight.abs().sum() > 0
    assert torch.equal(fused.weight, fast.span_n_b[0].c1_r.weight)


@requires_gaterv3
def test_the_norm_fold_is_the_same_function(checkpoint: tuple[torch.nn.Module, ModelInfo], fast: torch.nn.Module):
    """
    Compare the folded normalization against the algebra it stands for, by hand.

    `conv(scale * (x / (RMS(x) + eps)) + offset)` has to equal a convolution whose input-channel
    columns carry the scale and whose bias was computed from the *old* weight.
    """
    model, _info = checkpoint
    source = model.gater_encode[0].gated[0]
    folded = fast.gater_encode[0].gated[0]
    norm, conv = source.local[0], source.local[1]
    sample = torch.rand((1, norm.scale.numel(), 8, 8), generator=torch.Generator().manual_seed(7))

    # The checkpoint really does train the two affine vectors, so the fold is not a no-op.
    assert not torch.equal(norm.scale, torch.ones_like(norm.scale))
    assert not torch.equal(norm.offset, torch.zeros_like(norm.offset))

    with torch.inference_mode():
        rms = sample.norm(2, dim=1, keepdim=True) * (sample.size(1) ** -0.5)
        normalized = sample / (rms + norm.eps)
        weight = conv.weight * norm.scale[None, :, None, None]
        bias = conv.bias + conv.weight[:, :, 0, 0] @ norm.offset
        expected = torch.nn.functional.conv2d(normalized, weight, bias)
        actual = folded.local[0:2](sample)

    assert type(folded.local[0]).__name__ == "BareRMSNorm"
    assert sum(1 for _parameter in folded.local[0].parameters()) == 0
    assert folded.local[0].eps == norm.eps
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0.0)
    assert _psnr(source.local(sample), folded.local(sample)) > 120.0


def test_the_rms_norm_epsilon_stays_outside_the_root():
    """`RMSNorm` adds epsilon to the root, so a bare normalization cannot move it inside."""
    bare_norm, _folded_meta_gated, _projected_decoder = gaterv3_adapter._lean_classes()
    epsilon = 0.5
    module = bare_norm(epsilon)
    sample = torch.rand((1, 16, 8, 8), generator=torch.Generator().manual_seed(11))

    with torch.inference_mode():
        rms = sample.norm(2, dim=1, keepdim=True) * (sample.size(1) ** -0.5)
        outside = sample / (rms + epsilon)
        inside = sample / torch.sqrt(sample.square().mean(dim=1, keepdim=True) + epsilon)
        actual = module(sample)

    assert tuple(module.parameters()) == ()
    assert module.eps == epsilon
    assert torch.equal(actual, outside)
    assert not torch.allclose(actual, inside)


@requires_gaterv3
def test_the_residual_scale_fold_keeps_the_residual_order(
    checkpoint: tuple[torch.nn.Module, ModelInfo], fast: torch.nn.Module
):
    """
    `local * gamma0 * sca(local) + short` and `local * (gamma0 * sca)(local) + short` are equal.

    The check is on the module, so the broadcast shape of `gamma0`, the attention bias and the
    order of the residual addition are all exercised together.
    """
    model, _info = checkpoint
    source = model.gater_encode[1].gated[0]
    folded = fast.gater_encode[1].gated[0]
    gamma = source.gamma0.reshape(-1)

    assert type(source).__name__ == "MetaGated"
    assert type(folded).__name__ == "FoldedMetaGated"
    assert not hasattr(folded, "gamma0")
    assert not torch.equal(source.gamma0, torch.ones_like(source.gamma0))
    assert torch.equal(folded.gamma1, source.gamma1)
    assert torch.equal(folded.sca[1].weight, source.sca[1].weight * gamma[:, None, None, None])
    assert torch.equal(folded.sca[1].bias, source.sca[1].bias * gamma)

    sample = torch.rand((1, gamma.numel(), 8, 8), generator=torch.Generator().manual_seed(13))
    with torch.inference_mode():
        expected = source(sample)
        actual = folded(sample)

    assert actual.shape == expected.shape
    assert _psnr(expected, actual) > 120.0


@requires_gaterv3
def test_the_decoder_projection_keeps_the_pixel_shuffle_order(
    checkpoint: tuple[torch.nn.Module, ModelInfo], fast: torch.nn.Module
):
    """
    `W . cat(pixel_shuffle(K x), short) + b` and `pixel_shuffle((W_up K) x) + W_skip short + b`.

    A transposed channel order would still produce a finite tensor, so every decoder depth is
    compared against the block it replaces instead of relying on the aggregate parity test.
    """
    model, _info = checkpoint
    generator = torch.Generator().manual_seed(17)

    for index, source in enumerate(model.decode):
        folded = fast.decode[index]
        channels = source.shor.out_channels
        sample = torch.rand((1, source.scale[0].in_channels, 4, 4), generator=generator)
        short = torch.rand((1, channels, 4 * folded.factor, 4 * folded.factor), generator=generator)

        assert type(folded).__name__ == "ProjectedDecoder"
        assert folded.up.out_channels == channels * folded.factor**2
        with torch.inference_mode():
            expected = source(sample, short)
            actual = folded(sample, short)

        assert actual.shape == expected.shape, f"decode[{index}]"
        # The projection reassociates the 3x3 kernel product, so this is looser than the folds; a
        # transposed channel order would be nowhere near either bound.
        assert _psnr(expected, actual) > 100.0, f"decode[{index}]"


@requires_gaterv3
def test_the_full_depth_invariants_survive(checkpoint: tuple[torch.nn.Module, ModelInfo], fast: torch.nn.Module):
    _gaterv3, conv3xc, rms_norm, meta_gated, gate_block, _block, spab, attention, inception = _adapter_classes()
    model, _info = checkpoint

    # Depth, width and the reparameterizable branches are the ones the supplied checkpoint has.
    assert _count(model, conv3xc) == 19
    assert _count(model, rms_norm) == 60
    assert _count(model, meta_gated) == 24
    assert _count(model, gate_block) == 36
    assert _count(model, spab) == 6
    assert len(model.gater_encode) == len(model.decode) == 4
    assert len(model.latent) == 12

    assert _count(fast, conv3xc) == 0
    assert _count(fast, rms_norm) == 0
    assert _count(fast, meta_gated) == 0
    assert _count(fast, gate_block) == 36
    assert len(fast.latent) == 12
    assert len(fast.gater_encode) == len(fast.decode) == 4
    # No block is pruned: the same number of gated stages, normalizations and decoder depths.
    for kind, expected in (("BareRMSNorm", 60), ("FoldedMetaGated", 24), ("ProjectedDecoder", 4)):
        assert sum(1 for module in fast.modules() if type(module).__name__ == kind) == expected, kind

    # SPAB, the attention blocks and the inception branches are carried over untouched.
    assert _count(fast, spab) == _count(model, spab) == 6
    assert _count(fast, attention) == _count(model, attention) == 12
    assert _count(fast, inception) == _count(model, inception) == 24

    learned = {
        name: tensor
        for name, tensor in model.named_parameters()
        if name.startswith("latent.") and ".token_mix." in name
    }
    # temperature, qkv, qkv_dwconv weight and bias, project_out - per latent block.
    assert len(learned) == 12 * 5
    written = dict(fast.named_parameters())
    for name, tensor in learned.items():
        assert torch.equal(tensor, written[name]), name
    assert torch.equal(model.gamma, fast.gamma)


@requires_gaterv3
def test_the_reflect_padding_keeps_the_output_shape(
    checkpoint: tuple[torch.nn.Module, ModelInfo], fast: torch.nn.Module
):
    model, info = checkpoint
    sample = torch.rand((1, 3, 63, 65), generator=torch.Generator().manual_seed(19))

    with torch.inference_mode():
        reference = model(sample)
        optimized = fast(sample)

    assert tuple(reference.shape) == (1, 3, 63 * info.scale, 65 * info.scale)
    assert optimized.shape == reference.shape
    assert torch.isfinite(optimized).all()
    assert _psnr(reference, optimized) > 90.0


@requires_gaterv3
def test_repeated_predictions_leave_the_weights_stable(
    checkpoint: tuple[torch.nn.Module, ModelInfo], fast: torch.nn.Module
):
    _model, _info = checkpoint
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


@requires_gaterv3
def test_parity_on_the_supplied_checkpoint(checkpoint: tuple[torch.nn.Module, ModelInfo], fast: torch.nn.Module):
    model, _info = checkpoint
    cases = _reference_cases()

    reference = _predict_all(model, cases)
    optimized = _predict_all(fast, cases)

    assert optimized.keys() == reference.keys()
    for name in reference:
        assert optimized[name].shape == reference[name].shape, name
        assert torch.isfinite(optimized[name]).all(), name
        assert _psnr(reference[name], optimized[name]) > 90.0, name

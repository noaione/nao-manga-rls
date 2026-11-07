"""
Tests for the JaNai style tensor tiler in :mod:`nmanga.ml.tiling`.

Everything here runs on CPU with a fake runner, so no checkpoint, no CUDA device and none of the
legacy ONNX overlap tiling is involved. The reference for these expectations is the validated
JaNai implementation this module ports: stride ``tile_size - tile_overlap``, replicate padded edge
tiles and linear feather ramps whose weights add up to one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from nmanga.ml.contracts import TilingConfig
from nmanga.ml.tiling import infer_tensor

REPO_ROOT = Path(__file__).resolve().parents[1]


def _noise(height: int, width: int, seed: int = 0) -> torch.Tensor:
    return torch.rand(1, 3, height, width, generator=torch.Generator().manual_seed(seed))


def _ramp(height: int, width: int) -> torch.Tensor:
    """A linear ramp: every tile predicts the same value for a shared pixel."""
    ys = torch.arange(height, dtype=torch.float32)[:, None] / height
    xs = torch.arange(width, dtype=torch.float32)[None, :] / width
    return ((ys + xs) / 2).expand(1, 3, height, width).clone()


def _repeat_runner(scale: int):
    """A nearest neighbour upsampler: deterministic, exactly shift equivariant."""

    def run(tile: torch.Tensor) -> torch.Tensor:
        return tile.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)

    return run


def _identity_runner(tile: torch.Tensor) -> torch.Tensor:
    return tile


def test_full_tile_matches_a_single_prediction():
    tensor = _noise(64, 64)
    runner = _repeat_runner(2)

    output = infer_tensor(tensor, runner, scale=2, config=TilingConfig(tile_size=64, tile_overlap=0))

    torch.testing.assert_close(output, runner(tensor), rtol=0, atol=0)


def test_no_overlap_covers_every_pixel_exactly():
    tensor = _noise(200, 150)

    output = infer_tensor(tensor, _identity_runner, scale=1, config=TilingConfig(tile_size=64, tile_overlap=0))

    torch.testing.assert_close(output, tensor, rtol=0, atol=0)


@pytest.mark.parametrize("overlap", [0, 8, 32])
def test_overlap_blending_reproduces_a_linear_ramp(overlap: int):
    tensor = _ramp(300, 220)

    output = infer_tensor(tensor, _identity_runner, scale=1, config=TilingConfig(tile_size=64, tile_overlap=overlap))

    # The complementary ramps add up to one, so the seam is invisible: no banding anywhere.
    torch.testing.assert_close(output, tensor, rtol=0, atol=1e-6)


def test_overlap_blending_scales_a_linear_ramp():
    tensor = _ramp(300, 220)
    runner = _repeat_runner(2)

    output = infer_tensor(tensor, runner, scale=2, config=TilingConfig(tile_size=64, tile_overlap=16))

    torch.testing.assert_close(output, runner(tensor), rtol=0, atol=1e-6)


@pytest.mark.parametrize(("height", "width"), [(1, 1), (5, 7), (63, 65), (64, 64), (65, 65), (100, 37)])
def test_partial_edge_tiles_are_padded_and_cropped(height: int, width: int):
    tensor = _noise(height, width, seed=height + width)
    runner = _repeat_runner(2)

    output = infer_tensor(tensor, runner, scale=2, config=TilingConfig(tile_size=64, tile_overlap=16))

    assert output.shape == (1, 3, height * 2, width * 2)
    torch.testing.assert_close(output, runner(tensor), rtol=0, atol=1e-6)


def test_edge_tiles_are_padded_with_the_real_edge_not_a_canvas():
    # A constant image must stay constant: black/white canvas padding would show up here.
    tensor = torch.full((1, 3, 70, 66), 0.5)

    output = infer_tensor(tensor, _identity_runner, scale=1, config=TilingConfig(tile_size=64, tile_overlap=16))

    torch.testing.assert_close(output, tensor, rtol=0, atol=0)


def test_input_tensor_is_not_mutated():
    tensor = _noise(70, 70)
    before = tensor.clone()

    infer_tensor(tensor, _repeat_runner(2), scale=2, config=TilingConfig(tile_size=32, tile_overlap=8))

    assert torch.equal(tensor, before)


def test_results_are_float32_on_the_cpu():
    tensor = _noise(40, 40).to(torch.float64)

    output = infer_tensor(tensor, _repeat_runner(2), scale=2, config=TilingConfig(tile_size=32, tile_overlap=8))

    assert output.dtype == torch.float32
    assert output.device.type == "cpu"


@pytest.mark.parametrize(
    "broken",
    [
        lambda tile: tile,  # not scaled
        lambda tile: tile.repeat(1, 2, 1, 1),  # wrong channel count
        lambda tile: tile[..., :16, :16],  # wrong spatial size
    ],
)
def test_runner_contract_violations_raise(broken):
    tensor = _noise(40, 40)

    with pytest.raises(ValueError, match="unexpected output shape"):
        infer_tensor(tensor, broken, scale=2, config=TilingConfig(tile_size=32, tile_overlap=8))


def test_runner_fixed_input_size_is_checked():
    def runner(tile: torch.Tensor) -> torch.Tensor:
        return tile

    runner.input_size = (32, 32)

    with pytest.raises(ValueError, match="fixed"):
        infer_tensor(_noise(40, 40), runner, scale=1, config=TilingConfig(tile_size=64, tile_overlap=8))


def test_invalid_scale_is_rejected():
    with pytest.raises(ValueError, match="Scale"):
        infer_tensor(_noise(8, 8), _identity_runner, scale=0, config=TilingConfig(tile_size=8, tile_overlap=0))


@pytest.mark.parametrize("shape", [(3, 8, 8), (1, 1, 8, 8), (2, 3, 8, 8)])
def test_invalid_tensor_shape_is_rejected(shape: tuple[int, ...]):
    with pytest.raises(ValueError, match="NCHW"):
        infer_tensor(torch.zeros(*shape), _identity_runner, scale=1, config=TilingConfig(tile_size=8, tile_overlap=0))


def test_stride_counts_the_total_overlap():
    # The legacy ONNX tiler subtracts the overlap twice; this one must subtract it once.
    assert TilingConfig(tile_size=256, tile_overlap=32).stride == 224


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_size": 16, "tile_overlap": 16},
        {"tile_size": 16, "tile_overlap": 17},
        {"tile_size": 16, "tile_overlap": -1},
        {"tile_size": 0, "tile_overlap": 0},
    ],
)
def test_tiling_config_rejects_invalid_grids(kwargs: dict[str, int]):
    with pytest.raises(ValueError):
        TilingConfig(**kwargs)


def test_importing_the_tiler_does_not_import_torch():
    script = "\n".join([
        "import sys",
        "import nmanga.ml.tiling",
        "import nmanga.ml.images",
        "print('torch' in sys.modules)",
    ])

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=REPO_ROOT)

    assert result.stdout.strip() == "False"

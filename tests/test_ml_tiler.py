"""Tests for the shared tiler planner, variable edge tiling, and Umzi session pool."""

from __future__ import annotations

import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from nmanga.ml.onnx_backend import (
    _get_fp16_to_uint8_lut,
    _quantize_tile_to_uint8,
    denoise_single_image_with_overlap,
)
from nmanga.ml.tiler import (
    TilingPolicy,
    default_expand_tolerance,
    feather_ramp_values,
    group_tiles_by_shape,
    plan_axis,
    plan_tiles,
)
from nmanga.ml.umzi import UmziPackageInfo, UmziSessionPool, get_umzi_native, umzi_tiling_constraints

# --------------------------------------------------------------------------------------
# Axis & 2D grid planning
# --------------------------------------------------------------------------------------


def test_fixed_axis_pads_every_tile_to_tile_size():
    spans = plan_axis(100, tile_size=64, tile_overlap=16, policy=TilingPolicy(mode="fixed"))

    assert [(s.dst_start, s.valid_len, s.run_len, s.has_prev, s.has_next) for s in spans] == [
        (0, 64, 64, False, True),
        (48, 52, 64, True, False),
    ]


def test_variable_axis_shrinks_remainder_to_alignment_and_minimum():
    policy = TilingPolicy(mode="variable", alignment=8, min_tile_size=16)

    spans = plan_axis(133, tile_size=64, tile_overlap=0, policy=policy)
    assert [(s.dst_start, s.valid_len, s.run_len) for s in spans] == [
        (0, 64, 64),
        (64, 64, 64),
        (128, 5, 16),
    ]

    tiny = plan_axis(3, tile_size=64, tile_overlap=0, policy=policy)
    assert [(s.dst_start, s.valid_len, s.run_len) for s in tiny] == [(0, 3, 16)]


def test_variable_axis_absorbs_small_trailing_sliver():
    # 1040 = 2 * 512 + 16; with expand_tolerance >= 16 the 16 px tail merges into tile 2.
    policy = TilingPolicy(mode="variable", expand_tolerance=64, alignment=8)

    spans = plan_axis(1040, tile_size=512, tile_overlap=0, policy=policy)

    assert [(s.dst_start, s.valid_len, s.run_len, s.has_prev, s.has_next) for s in spans] == [
        (0, 512, 512, False, True),
        (512, 528, 528, True, False),
    ]


def test_variable_axis_absorbs_sliver_with_overlap():
    # 120 with tile=64, overlap=16 (step=48) would take 3 tiles; 8 px tail merges into tile 2.
    policy = TilingPolicy(mode="variable", expand_tolerance=16, alignment=2)

    spans = plan_axis(120, tile_size=64, tile_overlap=16, policy=policy)

    assert [(s.dst_start, s.valid_len, s.run_len, s.has_prev, s.has_next) for s in spans] == [
        (0, 64, 64, False, True),
        (48, 72, 72, True, False),
    ]


def test_variable_axis_buckets_edge_sizes():
    policy = TilingPolicy(mode="variable", bucket_step=32, alignment=8)

    # 197 and 204 both snap up to 224 so neighbouring page widths share one shape.
    assert plan_axis(512 + 197, tile_size=512, policy=policy)[-1].run_len == 224
    assert plan_axis(512 + 204, tile_size=512, policy=policy)[-1].run_len == 224
    # A near-full remainder snaps up to tile_size without exceeding it.
    assert plan_axis(512 + 500, tile_size=512, policy=policy)[-1].run_len == 512


def test_context_pads_symmetric_input_window():
    tiles = plan_tiles(100, 100, tile_size=32, tile_overlap=0, context=4, policy=TilingPolicy(mode="variable"))

    # Core tile is 24x24 -> 5x5 grid; top-left tile gets 4 px left/top replication padding.
    first = tiles[0]
    assert (first.dst_x, first.dst_y, first.valid_w, first.valid_h) == (0, 0, 24, 24)
    assert (first.pad_left, first.pad_top, first.src_x, first.src_y, first.src_w, first.src_h) == (4, 4, 0, 0, 28, 28)
    assert (first.run_w, first.run_h) == (32, 32)


def test_grouping_puts_primary_shape_first():
    tiles = plan_tiles(150, 100, tile_size=64, policy=TilingPolicy(mode="variable", alignment=2))
    groups = group_tiles_by_shape(tiles, primary_shape=(64, 64))

    assert [shape for shape, _ in groups] == [(64, 64), (64, 22), (36, 64), (36, 22)]
    assert [len(group) for _, group in groups] == [2, 1, 2, 1]


def test_feather_ramps_sum_to_one_across_overlap():
    left = feather_ramp_values(64, overlap=16, has_prev=False, has_next=True)
    right = feather_ramp_values(72, overlap=16, has_prev=True, has_next=False)

    sums = [a + b for a, b in zip(left[48:], right[:16], strict=True)]
    assert sums == pytest.approx([1.0] * 16)


def test_default_expand_tolerance_scales_with_tile_size():
    assert default_expand_tolerance(128) == 16
    assert default_expand_tolerance(512) == 64
    assert default_expand_tolerance(1024) == 64


# --------------------------------------------------------------------------------------
# Variable tiling through the batched NumPy tiler
# --------------------------------------------------------------------------------------


class _VariableSession:
    """Variable-shape x2 nearest-neighbour session recording every batch shape it saw."""

    scale_factor = 2
    context = 0

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, int]] = []

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=("batch", 3, "height", "width"), type="tensor(float32)")]

    def get_outputs(self):
        return [SimpleNamespace(name="output", shape=("batch", 3, "height", "width"), type="tensor(float32)")]

    def run(self, names, feeds):
        batch = feeds["input"]
        self.calls.append(tuple(int(d) for d in batch.shape))
        return [np.ascontiguousarray(batch.repeat(2, axis=2).repeat(2, axis=3))]


@pytest.mark.parametrize(("width", "height"), [(173, 260), (140, 140), (37, 25)])
def test_variable_tiling_reconstructs_exact_scaled_image(width: int, height: int):
    rng = np.random.default_rng(width + height)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    page = Image.fromarray(pixels)
    session = _VariableSession()
    policy = TilingPolicy(mode="variable", expand_tolerance=16, bucket_step=16, alignment=4, min_tile_size=8)

    try:
        result = denoise_single_image_with_overlap(
            page,
            session,  # type: ignore[arg-type]
            batch_size=2,
            tile_size=64,
            tile_overlap=16,
            use_fp32=True,
            pad_final_batch=True,
            tiling_policy=policy,
        )

        expected = pixels.repeat(2, axis=0).repeat(2, axis=1)
        assert result.size == (width * 2, height * 2)
        np.testing.assert_array_equal(np.asarray(result), expected)
    finally:
        page.close()


def test_variable_tiling_absorbs_slivers_and_avoids_over_padding_minority_batches():
    # 136x136 with tile=64, overlap=0, expand=16 -> 2x2 grid of (64, 64), (64, 72), (72, 64), (72, 72)
    # instead of 3x3=9 fixed tiles; each 1-tile minority group runs batch=1 even with batch_size=8.
    page = Image.new("RGB", (136, 136), (12, 34, 56))
    session = _VariableSession()
    policy = TilingPolicy(mode="variable", expand_tolerance=16, alignment=2)

    try:
        result = denoise_single_image_with_overlap(
            page,
            session,  # type: ignore[arg-type]
            batch_size=8,
            tile_size=64,
            tile_overlap=0,
            use_fp32=True,
            pad_final_batch=True,
            tiling_policy=policy,
        )

        assert result.size == (272, 272)
        assert session.calls == [
            (1, 3, 64, 64),
            (1, 3, 64, 72),
            (1, 3, 72, 64),
            (1, 3, 72, 72),
        ]
    finally:
        page.close()


def test_variable_tiling_rejects_fixed_spatial_model():
    class _FixedSpatialSession(_VariableSession):
        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=(1, 3, 64, 64), type="tensor(float32)")]

    page = Image.new("RGB", (80, 80))
    try:
        with pytest.raises(ValueError, match="dynamic spatial"):
            denoise_single_image_with_overlap(
                page,
                _FixedSpatialSession(),  # type: ignore[arg-type]
                tile_size=64,
                tile_overlap=0,
                tiling_policy=TilingPolicy(mode="variable"),
            )
    finally:
        page.close()


# --------------------------------------------------------------------------------------
# Umzi constraints and session pool
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "alignment", "expected"),
    [
        ("gaterv3", 16, (16, 1)),
        ("gaterv2", 16, (16, 1)),
        ("moesr", 2, (2, 2)),
        ("gfisrv2", None, (1, 1)),
        ("smosr", None, (1, 3)),
        ("figsr", 2, (2, 6)),
    ],
)
def test_umzi_tiling_constraints_match_family_requirements(
    family: str, alignment: int | None, expected: tuple[int, int]
):
    info = UmziPackageInfo(
        path=Path("/models/m.ummodel"),
        version=1,
        family_id=1,
        family=family,
        schema=1,
        size=100,
        digest="00",
        alignment=alignment,
    )

    assert umzi_tiling_constraints(info) == expected


def test_umzi_session_pool_reuses_shapes_and_evicts_lru(monkeypatch: pytest.MonkeyPatch):
    created: list[object] = []

    class _CountingSession:
        scale_factor = 2
        context = 0

        def __init__(self, model_path: Path, options: object) -> None:
            self.prepared: list[tuple[int, int, int, int]] = []
            self.closed = False
            created.append(self)

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=("batch", 3, "height", "width"), type="tensor(float16)")]

        def get_outputs(self):
            return [SimpleNamespace(name="output", shape=("batch", 3, "height", "width"), type="tensor(float16)")]

        def run(self, names, feeds):
            arr = feeds["input"]
            shape = tuple(int(d) for d in arr.shape)
            if not self.prepared or self.prepared[-1] != shape:
                self.prepared.append(shape)
            return [np.zeros((shape[0], 3, shape[2] * 2, shape[3] * 2), dtype=arr.dtype)]

        def close(self) -> None:
            self.closed = True

    module = types.ModuleType("umzi_native")
    module.SessionOptions = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    module.InferenceSession = _CountingSession  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "umzi_native", module)
    monkeypatch.setattr("nmanga.ml.umzi._load_torch_cuda_libraries", lambda: None)
    get_umzi_native.cache_clear()

    try:
        with UmziSessionPool("/models/smosr.ummodel", family="smosr", max_sessions=2) as pool:
            s1 = np.zeros((1, 3, 64, 64), dtype=np.float16)
            s2 = np.zeros((1, 3, 64, 32), dtype=np.float16)
            s3 = np.zeros((1, 3, 32, 64), dtype=np.float16)

            pool.run(None, {"input": s1})
            pool.run(None, {"input": s2})
            assert pool.session_count == 2
            # Repeating s1 hits cache and marks s2 as LRU.
            pool.run(None, {"input": s1})
            assert len(created) == 2

            # Introducing a 3rd shape reuses the LRU session (s2) in-place without opening a 3rd session.
            pool.run(None, {"input": s3})
            assert pool.session_count == 2
            assert len(created) == 2
            assert created[1].prepared == [(1, 3, 64, 32), (1, 3, 32, 64)]  # type: ignore[attr-defined]
            assert created[0].prepared == [(1, 3, 64, 64)]  # type: ignore[attr-defined]

        assert all(s.closed for s in created)  # type: ignore[attr-defined]
    finally:
        get_umzi_native.cache_clear()


# --------------------------------------------------------------------------------------
# Direct uint8 quantization & CHW reconstruction parity
# --------------------------------------------------------------------------------------


def test_fp16_to_uint8_lut_matches_float32_rounding_for_all_finite_half_values():
    lut = _get_fp16_to_uint8_lut()
    raw = np.arange(65536, dtype=np.uint16).view(np.float16)
    finite_mask = np.isfinite(raw)

    f32 = np.clip(raw[finite_mask].astype(np.float32), 0.0, 1.0) * 255.0
    np.round(f32, out=f32)
    expected = f32.astype(np.uint8)

    np.testing.assert_array_equal(lut[finite_mask], expected)
    # NaN maps to 0, +/-inf clamp to [0, 255].
    nan_idx = np.uint16(0x7E00)
    assert lut[nan_idx] == 0
    assert lut[np.uint16(0x7C00)] == 255
    assert lut[np.uint16(0xFC00)] == 0


def test_quantize_tile_handles_fp16_and_fp32_without_mutating_input():
    f32_tile = np.array([[[-0.2, 0.5], [1.2, 0.0]]], dtype=np.float32)
    f32_before = f32_tile.copy()
    f16_tile = f32_tile.astype(np.float16)

    np.testing.assert_array_equal(_quantize_tile_to_uint8(f32_tile), np.array([[[0, 128], [255, 0]]], dtype=np.uint8))
    np.testing.assert_array_equal(_quantize_tile_to_uint8(f16_tile), np.array([[[0, 128], [255, 0]]], dtype=np.uint8))
    np.testing.assert_array_equal(f32_tile, f32_before)


def test_grayscale_and_contrast_stretch_roundtrip_through_chw_canvas():
    class _GrayscaleSession:
        scale_factor = 1
        context = 0

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=("batch", 1, "height", "width"), type="tensor(float16)")]

        def get_outputs(self):
            return [SimpleNamespace(name="output", shape=("batch", 1, "height", "width"), type="tensor(float16)")]

        def run(self, names, feeds):
            return [np.ascontiguousarray(feeds["input"])]

    pixels = np.linspace(40, 200, 48 * 32, dtype=np.uint8).reshape(32, 48)
    page = Image.fromarray(pixels, mode="L")
    try:
        plain = denoise_single_image_with_overlap(
            page,
            _GrayscaleSession(),  # type: ignore[arg-type]
            tile_size=16,
            tile_overlap=0,
            tiling_policy=TilingPolicy(mode="variable"),
        )
        stretched = denoise_single_image_with_overlap(
            page,
            _GrayscaleSession(),  # type: ignore[arg-type]
            tile_size=16,
            tile_overlap=4,
            contrast_stretch=True,
            tiling_policy=TilingPolicy(mode="variable"),
        )
        assert plain.mode == "L"
        np.testing.assert_array_equal(np.asarray(plain), pixels)
        assert stretched.mode == "L"
        assert np.asarray(stretched).min() == 0
        assert np.asarray(stretched).max() == 255
    finally:
        page.close()

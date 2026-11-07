"""
Tests for the ONNX runtime cache key and optimization profile helpers.

Nobody here builds a TensorRT engine: the goal is to pin down the pieces that decide
*which* compiled engine an invocation may reuse, and which optimization profiles get
built for a graph that declares fixed dimensions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nmanga.ml.contracts import MLDataType
from nmanga.ml.onnx_cache import build_compilation_cache_key, get_model_fingerprint, get_tf32_tag
from nmanga.ml.onnx_session import get_model_static_input, resolve_model_static_shapes

GRAPH = {"ir_version": 8, "producer_name": "nmanga-test", "graph": {"node": [], "name": "g"}}


def _write_graph(path: Path, marker: str = "first") -> Path:
    graph = dict(GRAPH)
    graph["producer_version"] = marker
    path.write_text(json.dumps(graph), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# Graph fingerprinting
# --------------------------------------------------------------------------------------


def test_fingerprint_follows_the_file_contents(tmp_path: Path):
    model = _write_graph(tmp_path / "model.onnx")

    first = get_model_fingerprint(model)
    _write_graph(model, marker="second")
    second = get_model_fingerprint(model)

    assert first != second


def test_fingerprint_matches_the_same_graph_at_another_path(tmp_path: Path):
    model = _write_graph(tmp_path / "model.onnx")
    copy = _write_graph(tmp_path / "copy.onnx")

    assert get_model_fingerprint(model) == get_model_fingerprint(copy)


def test_fingerprint_is_a_hex_digest(tmp_path: Path):
    model = _write_graph(tmp_path / "model.onnx")

    fingerprint = get_model_fingerprint(model)

    assert len(fingerprint) == 64
    int(fingerprint, 16)


# --------------------------------------------------------------------------------------
# Cache keys
# --------------------------------------------------------------------------------------


def test_cache_key_is_stable_for_the_same_settings(tmp_path: Path):
    model = _write_graph(tmp_path / "model.onnx")

    first = build_compilation_cache_key(model, tile_size=512, batch_size=1)
    second = build_compilation_cache_key(model, tile_size=512, batch_size=1)

    assert first == second


def test_cache_key_changes_when_the_weights_change_in_place(tmp_path: Path):
    model = _write_graph(tmp_path / "model.onnx")

    before = build_compilation_cache_key(model, tile_size=512, batch_size=1)
    _write_graph(model, marker="other weights")
    after = build_compilation_cache_key(model, tile_size=512, batch_size=1)

    assert before != after


@pytest.mark.parametrize(
    "changed",
    [
        {"tile_size": 256},
        {"batch_size": 8},
        {"workspace_size": 512 * 1024 * 1024},
        {"enable_cuda_graph": False},
        {"with_nvrtx": True},
        {"data_type": MLDataType.FP32},
    ],
)
def test_cache_key_changes_with_the_build_options(tmp_path: Path, changed: dict):
    model = _write_graph(tmp_path / "model.onnx")
    defaults: dict = {
        "data_type": MLDataType.FP16,
        "tile_size": 512,
        "batch_size": 1,
        "workspace_size": None,
        "enable_cuda_graph": True,
        "with_nvrtx": False,
    }

    base = build_compilation_cache_key(model, **defaults)
    tweaked = build_compilation_cache_key(model, **{**defaults, **changed})

    assert base != tweaked


def test_cache_key_carries_the_tf32_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = _write_graph(tmp_path / "model.onnx")
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)

    default_key = build_compilation_cache_key(model, tile_size=512, batch_size=1)
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    tf32_off_key = build_compilation_cache_key(model, tile_size=512, batch_size=1)

    assert get_tf32_tag() == "tf32off"
    assert default_key != tf32_off_key


def test_cache_key_is_filesystem_safe(tmp_path: Path):
    model = _write_graph(tmp_path / "model.onnx")

    key = build_compilation_cache_key(model, tile_size=512, batch_size=1)

    assert not set(key) & set('<>:"/\\|?*')


# --------------------------------------------------------------------------------------
# Optimization profiles for a graph that declares fixed dimensions
# --------------------------------------------------------------------------------------


def test_fixed_batch_is_adopted_for_the_profile():
    batch_size, tile_size = resolve_model_static_shapes((1, 3, 512, 512), batch_size=64, tile_size=512)

    assert batch_size == 1
    assert tile_size == 512


def test_dynamic_axes_keep_the_requested_settings():
    assert resolve_model_static_shapes((None, 3, None, None), batch_size=8, tile_size=256) == (8, 256)


def test_unknown_shape_keeps_the_requested_settings():
    assert resolve_model_static_shapes(None, batch_size=8, tile_size=256) == (8, 256)


def test_fixed_tile_size_mismatch_is_rejected():
    with pytest.raises(ValueError, match="fixed tile size of 512x512"):
        resolve_model_static_shapes((1, 3, 512, 512), batch_size=1, tile_size=128)


def test_matching_fixed_tile_size_is_accepted():
    assert resolve_model_static_shapes((1, 3, 512, 512), batch_size=1, tile_size=512) == (1, 512)


def test_unrequested_tile_size_is_not_pinned_to_the_graph():
    assert resolve_model_static_shapes((1, 3, 512, 512), batch_size=1, tile_size=None) == (1, None)


# --------------------------------------------------------------------------------------
# Reading the declared input shape off a real graph
# --------------------------------------------------------------------------------------


def _make_onnx_model(path: Path, spatial: int | None) -> Path:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    height = spatial if spatial is not None else "height"
    width = spatial if spatial is not None else "width"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "static-test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, height, width])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, height, width])],
    )
    onnx.save_model(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))
    return path


def test_static_input_of_an_onnx_graph(tmp_path: Path):
    model = _make_onnx_model(tmp_path / "fixed.onnx", spatial=512)

    assert get_model_static_input(model) == ("input", (1, 3, 512, 512))


def test_dynamic_axes_are_reported_as_unknown(tmp_path: Path):
    model = _make_onnx_model(tmp_path / "dynamic.onnx", spatial=None)

    assert get_model_static_input(model) == ("input", (1, 3, None, None))


def test_unparsable_graph_reports_no_static_input(tmp_path: Path):
    model = _write_graph(tmp_path / "broken.onnx")

    assert get_model_static_input(model) is None


# --------------------------------------------------------------------------------------
# The compatibility facade still exposes the exact canonical objects
# --------------------------------------------------------------------------------------


def test_compat_denoiser_imports_are_the_canonical_objects():
    from nmanga import denoiser

    assert denoiser.MLDataType is MLDataType
    assert denoiser.build_compilation_cache_key is build_compilation_cache_key
    assert denoiser.get_model_fingerprint is get_model_fingerprint
    assert denoiser.get_model_static_input is get_model_static_input
    assert denoiser.get_tf32_tag is get_tf32_tag
    assert denoiser.resolve_model_static_shapes is resolve_model_static_shapes

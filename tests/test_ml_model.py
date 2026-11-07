"""Tests for the resselt checkpoint loader and the normalized ModelInfo contract."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nmanga.ml import model as model_module  # ruff: ignore[module-import-not-at-top-of-file] - after importorskip
from nmanga.ml.contracts import ModelInfo  # ruff: ignore[module-import-not-at-top-of-file]
from nmanga.ml.model import load_checkpoint, read_model_info  # ruff: ignore[module-import-not-at-top-of-file]

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("NMANGA_TEST_MODELS", REPO_ROOT.parent / "umzi-bench" / "models"))
FDAT_M = MODELS_DIR / "2x_IllustrationJaNai_V3detail_FDAT_M_unshuffle_40k_fp16.safetensors"
FDAT_XL = MODELS_DIR / "4x_IllustrationJaNai_V3detail_FDAT_XL_27k_bf16.safetensors"
FDAT_M_4X = MODELS_DIR / "4x_IllustrationJaNai_V3detail_FDAT_M_40k_fp16.safetensors"
SPAN_S = MODELS_DIR / "2x_IllustrationJaNai_V3detail_SPAN_S_40k_fp16.safetensors"
GATERV3_PTH = MODELS_DIR / "1x_umzi_digital_decompress_gaterv3_1.pth"
requires_span = pytest.mark.skipif(not SPAN_S.is_file(), reason=f"missing test checkpoint {SPAN_S.name}")
requires_gaterv3 = pytest.mark.skipif(not GATERV3_PTH.is_file(), reason=f"missing test checkpoint {GATERV3_PTH.name}")

requires_m = pytest.mark.skipif(not FDAT_M.is_file(), reason=f"missing test checkpoint {FDAT_M.name}")
requires_xl = pytest.mark.skipif(not FDAT_XL.is_file(), reason=f"missing test checkpoint {FDAT_XL.name}")


def test_importing_the_module_stays_lightweight():
    script = "import sys, nmanga.ml.model; print('torch' in sys.modules, 'resselt' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    )

    assert result.stdout.strip() == "False False"


def test_missing_checkpoint_is_reported(tmp_path: Path):
    with pytest.raises(ValueError, match="Checkpoint not found"):
        read_model_info(tmp_path / "nope.safetensors")
    with pytest.raises(ValueError, match="Checkpoint not found"):
        load_checkpoint(tmp_path / "nope.safetensors")


def test_directory_is_not_a_checkpoint(tmp_path: Path):
    folder = tmp_path / "model.safetensors"
    folder.mkdir()

    with pytest.raises(ValueError, match="not a file"):
        read_model_info(folder)


def test_non_safetensors_format_is_rejected(tmp_path: Path):
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"\x00")

    with pytest.raises(ValueError, match="safetensors"):
        read_model_info(onnx)
    with pytest.raises(ValueError, match="safetensors"):
        load_checkpoint(onnx)


@requires_m
def test_read_model_info_describes_the_unshuffle_fdat():
    info = read_model_info(FDAT_M)

    assert isinstance(info, ModelInfo)
    assert info.architecture == "FDAT"
    assert info.scale == 2
    assert info.unshuffle == 2
    assert info.in_channels == 3
    assert info.out_channels == 3
    assert info.stored_dtypes == ("fp32",)
    assert info.loader == "resselt"
    assert info.loader_version is not None and info.loader_version.startswith("1.4")
    assert info.checkpoint_sha256 is not None and len(info.checkpoint_sha256) == 64


@requires_xl
def test_read_model_info_describes_the_plain_fdat():
    info = read_model_info(FDAT_XL)

    assert info.architecture == "FDAT"
    assert info.scale == 4
    assert info.unshuffle == 1
    assert info.in_channels == 3
    assert info.out_channels == 3
    # The file name says bf16, the tensors store FP32.
    assert info.stored_dtypes == ("fp32",)


@requires_m
def test_header_and_loaded_metadata_agree():
    assert read_model_info(FDAT_M) == load_checkpoint(FDAT_M)[1]


@requires_xl
def test_header_and_loaded_metadata_agree_for_the_xl_checkpoint():
    assert read_model_info(FDAT_XL) == load_checkpoint(FDAT_XL)[1]


@requires_m
def test_load_checkpoint_returns_an_fp32_eval_model():
    model, info = load_checkpoint(FDAT_M)

    assert type(model).__name__ == "FDAT"
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    assert {str(parameter.dtype) for parameter in model.parameters()} == {"torch.float32"}
    assert info.unshuffle == 2


@requires_m
def test_the_effective_scale_is_the_external_one():
    model, info = load_checkpoint(FDAT_M)

    # resselt builds the unshuffle model with a 4x upsampler and crops back to the effective 2x.
    assert int(model.upsampler.MetaUpsample[2].item()) == 4
    assert model.upscale == 2 == info.scale
    assert model.pad == 2 == info.unshuffle


@requires_xl
def test_a_plain_fdat_does_not_unshuffle():
    model, info = load_checkpoint(FDAT_XL)

    assert info.unshuffle == 1
    assert model.pad == 0
    assert int(model.upsampler.MetaUpsample[2].item()) == 4


@pytest.mark.skipif(not FDAT_M_4X.is_file(), reason=f"missing test checkpoint {FDAT_M_4X.name}")
def test_a_4x_fdat_without_unshuffle_is_detected():
    info = read_model_info(FDAT_M_4X)

    assert info.architecture == "FDAT"
    assert info.scale == 4
    assert info.unshuffle == 1


@requires_m
def test_renaming_a_checkpoint_does_not_change_detection(tmp_path: Path):
    disguised = tmp_path / "4x_IllustrationJaNai_V3detail_SPAN_XL_bf16.safetensors"
    shutil.copyfile(FDAT_M, disguised)

    original = read_model_info(FDAT_M)
    renamed = read_model_info(disguised)

    assert renamed.architecture == original.architecture == "FDAT"
    assert renamed.scale == original.scale == 2
    assert renamed.unshuffle == original.unshuffle == 2
    assert (renamed.in_channels, renamed.out_channels) == (original.in_channels, original.out_channels)
    assert renamed.stored_dtypes == original.stored_dtypes == ("fp32",)
    assert renamed.checkpoint_sha256 == original.checkpoint_sha256
    assert load_checkpoint(disguised)[1] == original


def _metadata_model(**overrides: object) -> torch.nn.Module:
    from resselt.factory.arch import ModelMetadata

    module = torch.nn.Module()
    module.parameters_info = ModelMetadata(
        in_channels=overrides.get("in_channels", 3),
        out_channels=overrides.get("out_channels", 3),
        upscale=overrides.get("upscale", 2),
        name=overrides.get("name", "HAT"),
    )
    return module


def test_normalize_accepts_an_isotropic_scale_sequence():
    info = model_module._normalize_metadata(_metadata_model(upscale=(2, 2)))

    assert info.scale == 2
    assert info.unshuffle == 1
    assert info.architecture == "HAT"


def test_normalize_rejects_an_anisotropic_scale():
    with pytest.raises(ValueError, match="anisotropic"):
        model_module._normalize_metadata(_metadata_model(upscale=(2, 3)))


@pytest.mark.parametrize("scale", [0, -2, (), "2"])
def test_normalize_rejects_an_invalid_scale(scale: object):
    with pytest.raises(ValueError):
        model_module._normalize_metadata(_metadata_model(upscale=scale))


def test_normalize_rejects_a_non_integer_channel_count():
    with pytest.raises(ValueError, match="non-integer in_channels"):
        model_module._normalize_metadata(_metadata_model(in_channels=3.5))


@pytest.mark.parametrize("channels", [0, -1, True])
def test_normalize_rejects_an_invalid_channel_count(channels: object):
    with pytest.raises(ValueError):
        model_module._normalize_metadata(_metadata_model(out_channels=channels))


def test_normalize_rejects_a_missing_metadata_record():
    with pytest.raises(ValueError, match="parameters_info"):
        model_module._normalize_metadata(torch.nn.Module())


def test_normalize_rejects_an_empty_architecture_name():
    with pytest.raises(ValueError, match="architecture name"):
        model_module._normalize_metadata(_metadata_model(name="  "))


@requires_m
def test_fdat_structure_mismatch_is_reported():
    model, _info = load_checkpoint(FDAT_M)
    model.upscale = 3

    with pytest.raises(ValueError, match="does not match the constructed model"):
        model_module._normalize_metadata(model, path=FDAT_M)


@requires_span
def test_a_non_fdat_architecture_is_still_described():
    info = read_model_info(SPAN_S)

    assert info.architecture == "SPAN"
    assert info.scale == 2
    assert info.unshuffle == 1


@requires_gaterv3
def test_a_pth_checkpoint_is_loaded_through_resselt():
    model, info = load_checkpoint(GATERV3_PTH)

    assert type(model).__name__ == "GateRV3"
    assert info.architecture == "GateRV3"
    assert info.scale == 1
    # GateRV3 downscales and re-upsamples internally; the external contract is still 1:1.
    assert info.unshuffle == 1
    assert (info.in_channels, info.out_channels) == (3, 3)
    assert info.checkpoint_sha256 is not None and len(info.checkpoint_sha256) == 64
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    assert {str(parameter.dtype) for parameter in model.parameters()} == {"torch.float32"}


@requires_gaterv3
def test_a_pth_checkpoint_reports_an_unknown_storage_dtype():
    """A `.pth` archive has no safetensors header, so the stored precision stays unknown
    rather than being read back from the FP32 parameters the loader promoted."""
    assert read_model_info(GATERV3_PTH).stored_dtypes == ()


@requires_gaterv3
def test_read_model_info_matches_the_loaded_pth():
    assert read_model_info(GATERV3_PTH) == load_checkpoint(GATERV3_PTH)[1]


@requires_gaterv3
def test_an_unsupported_pth_format_is_still_rejected(tmp_path: Path):
    false_pth = tmp_path / "model.ckpt"
    false_pth.write_bytes(b"\x00")

    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        read_model_info(false_pth)
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        load_checkpoint(false_pth)


@requires_span
def test_the_span_adapter_hooks_are_reachable_through_the_loader():
    from nmanga.ml.opt_arch import adapter_for

    info = read_model_info(SPAN_S)
    adapter = adapter_for(info.architecture)

    assert adapter is not None
    for hook in ("build_fast_model", "unshuffle_of", "cross_check", "info_from_header"):
        assert callable(getattr(adapter, hook)), hook


@requires_gaterv3
def test_the_gaterv3_adapter_hooks_are_reachable_through_the_loader():
    from nmanga.ml.opt_arch import adapter_for

    info = read_model_info(GATERV3_PTH)
    adapter = adapter_for(info.architecture)

    assert adapter is not None
    for hook in ("build_fast_model", "unshuffle_of", "cross_check", "info_from_header"):
        assert callable(getattr(adapter, hook)), hook

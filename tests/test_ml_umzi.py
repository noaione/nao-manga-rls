"""Tests for the Umzi native runtime wrapper and its commands.

Nothing here needs `umzi-native`, a CUDA device or a model file: the runtime is a fake module in
`sys.modules`, so the checks stay about how nmanga inspects a package and opens a session.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nmanga.ml.umzi import (
    UmziPackageInfo,
    build_kernel_options,
    create_umzi_session,
    describe_package,
    export_ummodel,
    get_umzi_native,
    parse_kernel_options,
    read_ummodel,
    recommended_kernel_options,
    resolve_precision,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeSessionOptions:
    """Captures the keyword arguments the wrapper builds."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeSession:
    """Stands in for an open native session."""

    scale_factor = 1
    context = 0

    def __init__(self, model_path: Path, options: FakeSessionOptions) -> None:
        self.model_path = model_path
        self.options = options
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _fake_info(*, family: str | None, problems: tuple[str, ...] = ()) -> SimpleNamespace:
    """Build what `umzi_native.peek_model` returns, with a GateRV3 style payload."""
    payload = SimpleNamespace(
        scale=1,
        input_channels=3,
        output_channels=3,
        alignment=16,
        checkpoint="1x_umzi_digital_decompress_gaterv3_1.pth",
    )
    return SimpleNamespace(
        path=Path("/models/page.ummodel"),
        family_name=family,
        payload=payload,
        size=2048,
        digest="abc123",
        problems=problems,
    )


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    info: SimpleNamespace | None = None,
) -> SimpleNamespace:
    """Put a fake `umzi_native` in `sys.modules` and reset the import cache."""
    installed = info if info is not None else _fake_info(family="gaterv3")
    module = types.ModuleType("umzi_native")
    module.peek_model = lambda path: installed  # type: ignore[attr-defined]
    module.SessionOptions = FakeSessionOptions  # type: ignore[attr-defined]
    module.InferenceSession = FakeSession  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "umzi_native", module)
    get_umzi_native.cache_clear()
    return installed


@pytest.fixture(autouse=True)
def _reset_runtime_cache():
    """Keep one test's fake runtime from leaking into the next one."""
    get_umzi_native.cache_clear()
    yield
    get_umzi_native.cache_clear()


def test_importing_the_module_stays_lightweight():
    script = "import sys, nmanga.ml.umzi, nmanga.cli.umzi; print('umzi_native' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    )

    assert result.stdout.strip() == "False"


def test_a_missing_runtime_names_the_extra_to_install(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "umzi_native", None)
    get_umzi_native.cache_clear()

    with pytest.raises(ImportError, match=r"nmanga\[denoiser\]"):
        read_ummodel("/models/page.ummodel")


# --------------------------------------------------------------------------------------
# Kernel options
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "switches"),
    [
        ("gaterv3", {"fuse_gate_pool": 1, "tune_projections": 1, "fuse_silu": 1}),
        ("gaterv2", {"tune_plans": 1}),
        ("moesr", {"fused_gate": 1}),
        ("gfisrv2", {"fused_conv": 1}),
    ],
)
def test_each_family_pins_its_recommended_switches(family: str, switches: dict[str, int]):
    assert recommended_kernel_options(family) == switches


def test_an_unknown_family_adds_no_switches():
    assert recommended_kernel_options(None) == {}


def test_an_explicit_switch_overrides_a_recommended_one():
    options = build_kernel_options("gaterv3", {"fuse_gate_pool": 0})

    assert options == {"fuse_gate_pool": 0, "tune_projections": 1, "fuse_silu": 1}


def test_kernel_options_parse_name_value_pairs():
    assert parse_kernel_options(["fuse_gate_pool=1", "workspace_mib = 256"]) == {
        "fuse_gate_pool": 1,
        "workspace_mib": 256,
    }


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("fuse_gate_pool", "is not `name=value`"),
        ("=1", "is not `name=value`"),
        ("fuse_gate_pool=", "is not `name=value`"),
        ("fuse_gate_pool=on", "needs an integer value"),
    ],
)
def test_a_malformed_kernel_option_is_rejected(value: str, message: str):
    with pytest.raises(ValueError, match=message):
        parse_kernel_options([value])


# --------------------------------------------------------------------------------------
# Precision
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["gaterv3", "gaterv2", "moesr"])
def test_only_gfisrv2_has_an_fp32_path(family: str):
    with pytest.raises(ValueError, match="only available for GFISRv2"):
        resolve_precision(family, "fp32")


def test_gfisrv2_accepts_fp32():
    assert resolve_precision("gfisrv2", "FP32") == "fp32"


def test_an_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match="Unknown precision"):
        resolve_precision("gaterv3", "bf16")


# --------------------------------------------------------------------------------------
# Package inspection
# --------------------------------------------------------------------------------------


def test_reading_a_package_normalizes_the_runtime_dataclasses(monkeypatch: pytest.MonkeyPatch):
    _install_fake_runtime(monkeypatch)

    info = read_ummodel("/models/page.ummodel")

    assert info == UmziPackageInfo(
        path=Path("/models/page.ummodel"),
        family="gaterv3",
        scale=1,
        in_channels=3,
        out_channels=3,
        alignment=16,
        size=2048,
        digest="abc123",
        checkpoint="1x_umzi_digital_decompress_gaterv3_1.pth",
    )
    assert info.supported


def test_an_unsupported_package_reports_why(monkeypatch: pytest.MonkeyPatch):
    info = _fake_info(family=None, problems=("unknown kernel family 9",))
    _install_fake_runtime(monkeypatch, info=info)

    package = read_ummodel("/models/other.ummodel")
    lines = describe_package(package)

    assert not package.supported
    assert "unknown family" in lines[0]
    assert "unsupported" in lines[0]
    assert lines[-1] == "  problem: unknown kernel family 9"


def test_a_supported_package_reports_its_alignment(monkeypatch: pytest.MonkeyPatch):
    _install_fake_runtime(monkeypatch)

    lines = describe_package(read_ummodel("/models/page.ummodel"))

    assert lines[0] == "/models/page.ummodel: gaterv3, x1 scale, supported"
    assert "3 -> 3 channels, 16-pixel alignment" in lines[2]


# --------------------------------------------------------------------------------------
# Session creation
# --------------------------------------------------------------------------------------


def _no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nmanga.ml.umzi._load_torch_cuda_libraries", lambda: None)


def test_a_session_opens_with_the_family_switches_and_the_requested_precision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _install_fake_runtime(monkeypatch)
    _no_torch(monkeypatch)
    cache_dir = tmp_path / "umzi-cache"

    session = create_umzi_session(
        "/models/page.ummodel",
        family="gaterv3",
        device_id=1,
        precision="fp16",
        cache_dir=cache_dir,
        enable_cuda_graph=False,
        kernel_options={"fuse_gate_pool": 0},
    )

    assert isinstance(session, FakeSession)
    assert session.model_path == Path("/models/page.ummodel")
    assert session.options.kwargs == {
        "precision": "fp16",
        "device_id": 1,
        "cache_dir": cache_dir,
        "enable_cuda_graph": False,
        "kernel_options": {"fuse_gate_pool": 0, "tune_projections": 1, "fuse_silu": 1},
    }


def test_a_session_can_disable_the_plan_cache(monkeypatch: pytest.MonkeyPatch):
    _install_fake_runtime(monkeypatch)
    _no_torch(monkeypatch)

    session = create_umzi_session("/models/page.ummodel", family="moesr", cache_dir=None)

    assert session.options.kwargs["cache_dir"] is None
    assert session.options.kwargs["kernel_options"] == {"fused_gate": 1}


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------


def test_exporting_without_the_exporter_extra_says_how_to_add_it(monkeypatch: pytest.MonkeyPatch):
    _install_fake_runtime(monkeypatch)
    monkeypatch.setitem(sys.modules, "umzi_native.exporter", None)

    with pytest.raises(ImportError, match=r"umzi-native\[exporter\]"):
        export_ummodel("model.safetensors", "model.ummodel")


def test_exporting_returns_the_written_package(monkeypatch: pytest.MonkeyPatch):
    info = _fake_info(family="gaterv2")
    _install_fake_runtime(monkeypatch, info=info)
    exporter = types.ModuleType("umzi_native.exporter")
    exporter.export_model = lambda checkpoint, output, kernel=None: info  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "umzi_native.exporter", exporter)

    package = export_ummodel("model.safetensors", "model.ummodel", kernel="gaterv2")

    assert package.family == "gaterv2"
    assert package.path == Path("/models/page.ummodel")


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------


def test_peek_fails_on_an_unsupported_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from nmanga.cli.umzi import umzi_group

    model = tmp_path / "page.ummodel"
    model.write_bytes(b"not a package")
    _install_fake_runtime(monkeypatch, info=_fake_info(family=None, problems=("unknown kernel family 9",)))

    peek = umzi_group.commands["peek"].callback

    assert peek((model,)) == 1


def test_peek_accepts_a_folder_of_packages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from nmanga.cli.umzi import umzi_group

    (tmp_path / "page.ummodel").write_bytes(b"package")
    _install_fake_runtime(monkeypatch)

    peek = umzi_group.commands["peek"].callback

    assert peek((tmp_path,)) == 0


def test_infer_rejects_a_model_that_is_not_a_ummodel(tmp_path: Path):
    from click.testing import CliRunner

    from nmanga.cli.umzi import umzi_group

    (tmp_path / "in").mkdir()
    model = tmp_path / "model.onnx"
    model.write_bytes(b"")
    out = tmp_path / "out"

    result = CliRunner().invoke(
        umzi_group,
        ["infer", str(tmp_path / "in"), "-o", str(out), "-m", str(model)],
    )

    assert result.exit_code == 2
    assert "nmanga umzi to-ummodel" in result.output


# --------------------------------------------------------------------------------------
# The shared tiler
# --------------------------------------------------------------------------------------

# How `umzi_native.TensorInfo` names a dynamic dimension.
DYNAMIC_SHAPE = ("batch", "channels", "height", "width")


class _IdentityTilerSession:
    """A x1 native session that returns its input and records the batch sizes it saw."""

    scale_factor = 1
    context = 0

    def __init__(self) -> None:
        self.batches: list[int] = []

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=DYNAMIC_SHAPE, type="tensor(float16)")]

    def get_outputs(self):
        return [SimpleNamespace(name="output", shape=DYNAMIC_SHAPE, type="tensor(float16)")]

    def run(self, names, feeds):
        sample = feeds["input"]
        assert sample.dtype == np.float16
        self.batches.append(len(sample))
        return [np.ascontiguousarray(sample)]


def test_the_shared_tiler_runs_a_native_session():
    """The ONNX tiler only needs the surface `umzi_native.InferenceSession` offers."""
    from PIL import Image

    from nmanga.ml.onnx_backend import denoise_single_image_with_overlap

    page = Image.new("RGB", (96, 64), (10, 20, 30))
    try:
        result = denoise_single_image_with_overlap(
            page, _IdentityTilerSession(), batch_size=2, tile_size=32, tile_overlap=8
        )

        assert result.size == page.size
        assert result.tobytes() == page.tobytes()
    finally:
        page.close()


def test_a_short_final_batch_is_padded_for_a_native_session():
    """The native session prepares one padded shape at a time, so a short last batch can be filled."""
    from PIL import Image

    from nmanga.ml.onnx_backend import denoise_single_image_with_overlap

    page = Image.new("RGB", (96, 64), (10, 20, 30))
    exact = _IdentityTilerSession()
    padded = _IdentityTilerSession()
    try:
        denoise_single_image_with_overlap(page, exact, batch_size=5, tile_size=32, tile_overlap=8)
        result = denoise_single_image_with_overlap(
            page, padded, batch_size=5, tile_size=32, tile_overlap=8, pad_final_batch=True
        )

        assert exact.batches == [5, 5, 2]
        assert padded.batches == [5, 5, 5]
        assert result.size == page.size
    finally:
        page.close()

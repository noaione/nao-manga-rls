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
    FiGSRDetails,
    GateRV3Details,
    SMoSRDetails,
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


def _payload(family: str | None) -> SimpleNamespace | None:
    """Build the family payload `umzi_native.peek_model` attaches, named like the runtime dataclasses."""
    if family == "gaterv3":
        return SimpleNamespace(
            scale=1,
            input_channels=3,
            output_channels=3,
            alignment=16,
            base_channels=32,
            tensors=12,
            operations=9,
            weight_bytes=1024,
            op_kinds={"Conv": 4, "Silu": 5},
            checkpoint="1x_umzi_digital_decompress_gaterv3_1.pth",
        )
    if family == "gaterv2":
        return SimpleNamespace(
            scale=1,
            input_channels=3,
            output_channels=3,
            alignment=16,
            base_channels=32,
            encoder_depths=(1, 1),
            decoder_depths=(1, 2),
            latent_blocks=2,
            attention_eps=0.00001,
            parameters=1000,
            weight_bytes=4000,
            checkpoint="gaterv2.pth",
        )
    if family == "moesr":
        return SimpleNamespace(
            scale=4,
            input_channels=3,
            output_channels=3,
            alignment=8,
            base_channels=64,
            groups=3,
            full_blocks=3,
            reduced_blocks=1,
            full_hidden=96,
            reduced_hidden=64,
            weight_bytes=5000,
            checkpoint="moesr.pth",
        )
    if family == "gfisrv2":
        return SimpleNamespace(
            scale=4,
            input_channels=3,
            output_channels=3,
            base_channels=48,
            blocks=24,
            hidden_channels=128,
            fourier_channels=30,
            weight_bytes=6000,
            checkpoint="gfisrv2.pth",
        )
    if family == "smosr":
        return SimpleNamespace(
            scale=2,
            input_channels=3,
            output_channels=3,
            reflection_padding=2,
            base_channels=48,
            first_blocks=2,
            middle_blocks=3,
            end_blocks=1,
            weight_bytes=7000,
            checkpoint="smosr.safetensors",
        )
    if family == "figsr":
        return SimpleNamespace(
            scale=2,
            input_channels=3,
            output_channels=3,
            reflection_padding=4,
            alignment=2,
            base_channels=32,
            blocks_per_group=12,
            hidden_channels=80,
            fourier_channels=8,
            square_kernel=13,
            band_kernel=17,
            weight_bytes=8000,
            checkpoint="figsr.safetensors",
        )
    return None


def _fake_info(*, family: str | None, problems: tuple[str, ...] = ()) -> SimpleNamespace:
    """Build what `umzi_native.peek_model` returns for that family."""
    sections = (
        SimpleNamespace(id=1, name="architecture", offset=0, length=64),
        SimpleNamespace(id=3, name="weights", offset=64, length=1984),
    )
    return SimpleNamespace(
        path=Path("/models/page.ummodel"),
        size=2048,
        version=1,
        family=1,
        family_name=family,
        schema=1,
        digest="abc123",
        sections=sections,
        payload=_payload(family),
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
    ("family", "precision", "switches"),
    [
        ("gaterv3", "fp16", {"fuse_gate_pool": 1, "tune_projections": 1, "fuse_silu": 1}),
        ("gaterv3", "mixed", {"fuse_gate_pool": 1, "tune_projections": 1, "fuse_silu": 1}),
        ("gaterv2", "fp16", {"tune_plans": 1}),
        ("moesr", "fp16", {"fused_gate": 1}),
        ("gfisrv2", "fp16", {"fused_conv": 1}),
        ("smosr", "fp16", {"fuse_silu": 1}),
        ("smosr", "mixed", {"tune_convolutions": 1, "fuse_silu": 1}),
        ("smosr", "fp32", {}),
        ("figsr", "fp16", {"tune_convolutions": 1, "fuse_expand_gate": 1}),
        ("figsr", "mixed", {"tune_convolutions": 1, "fuse_row_gate": 1, "fuse_expand_gate": 1}),
        ("figsr", "fp32", {"tune_convolutions": 1, "fp32_nchw": 1}),
    ],
)
def test_each_family_pins_its_recommended_switches(family: str, precision: str, switches: dict[str, int]):
    assert recommended_kernel_options(family, precision) == switches


def test_recommended_switches_default_to_fp16():
    assert recommended_kernel_options("smosr") == {"fuse_silu": 1}
    assert recommended_kernel_options("figsr") == {"tune_convolutions": 1, "fuse_expand_gate": 1}


def test_an_unknown_family_adds_no_switches():
    assert recommended_kernel_options(None) == {}


def test_an_explicit_switch_overrides_a_recommended_one():
    options = build_kernel_options("gaterv3", {"fuse_gate_pool": 0})

    assert options == {"fuse_gate_pool": 0, "tune_projections": 1, "fuse_silu": 1}


def test_precision_scoped_overrides_merge_with_that_precisions_recommendations():
    assert build_kernel_options("figsr", {"fuse_row_gate": 0}, precision="mixed") == {
        "tune_convolutions": 1,
        "fuse_row_gate": 0,
        "fuse_expand_gate": 1,
    }


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
def test_only_supported_families_have_an_fp32_path(family: str):
    with pytest.raises(ValueError, match="only available for GFISRv2, SMoSR, and FiGSR"):
        resolve_precision(family, "fp32")


@pytest.mark.parametrize("family", ["gfisrv2", "smosr", "figsr"])
def test_fp32_capable_families_accept_fp32(family: str):
    assert resolve_precision(family, "FP32") == "fp32"


def test_an_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match="Unknown precision"):
        resolve_precision("gaterv3", "bf16")


# --------------------------------------------------------------------------------------
# Package inspection
# --------------------------------------------------------------------------------------


def test_reading_a_package_normalizes_the_runtime_dataclasses(monkeypatch: pytest.MonkeyPatch):
    _install_fake_runtime(monkeypatch)

    info = read_ummodel("/models/page.ummodel")

    assert (info.family, info.schema, info.version, info.family_id) == ("gaterv3", 1, 1, 1)
    assert (info.scale, info.in_channels, info.out_channels) == (1, 3, 3)
    assert (info.alignment, info.base_channels) == (16, 32)
    assert [(section.id, section.name, section.length) for section in info.sections] == [
        (1, "architecture", 64),
        (3, "weights", 1984),
    ]
    assert info.details == GateRV3Details(
        tensors=12,
        operations=9,
        weight_bytes=1024,
        op_kinds=(("Conv", 4), ("Silu", 5)),
    )
    assert info.checkpoint == "1x_umzi_digital_decompress_gaterv3_1.pth"
    assert info.supported


def test_an_unsupported_package_reports_why(monkeypatch: pytest.MonkeyPatch):
    info = _fake_info(family=None, problems=("unknown kernel family 9",))
    _install_fake_runtime(monkeypatch, info=info)

    package = read_ummodel("/models/other.ummodel")
    lines = describe_package(package)

    assert not package.supported
    assert "kernel family 1" in lines[0]
    assert "unsupported" in lines[0]
    assert lines[-1] == "  problem: unknown kernel family 9"


def test_a_gaterv3_report_covers_the_header_and_the_schedule(monkeypatch: pytest.MonkeyPatch):
    _install_fake_runtime(monkeypatch)

    lines = describe_package(read_ummodel("/models/page.ummodel"))

    assert lines[0] == "/models/page.ummodel: gaterv3 schema 1, supported"
    assert lines[1] == "  2.00 KiB (2,048 bytes), sha256 abc123"
    assert lines[2] == "  section 1 architecture: 64 bytes"
    assert lines[3] == "  section 3 weights: 1.94 KiB (1,984 bytes)"
    assert lines[4] == "  3 -> 3 channels, x1 scale, 16-pixel padding, 32 base channels"
    assert lines[5] == "  12 tensors, 9 operations, 1.00 KiB (1,024 bytes) of weights"
    assert lines[6] == "  operations: Conv 4, Silu 5"
    assert lines[7] == "  checkpoint: 1x_umzi_digital_decompress_gaterv3_1.pth"


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        (
            "gaterv2",
            [
                "  3 -> 3 channels, x1 scale, 16-pixel padding, 32 base channels",
                "  encoder (1, 1), decoder (1, 2), 2 latent attention blocks (eps 1e-05)",
                "  1000 FP32 parameters, 3.91 KiB (4,000 bytes) of weights",
            ],
        ),
        (
            "moesr",
            [
                "  3 -> 3 channels, x4 scale, 8-pixel padding, 64 base channels",
                "  3 groups of 3 full-resolution and 1 reduced-resolution blocks",
                "  full 96 and reduced 64 hidden channels, 4.88 KiB (5,000 bytes) of weights",
            ],
        ),
        (
            "gfisrv2",
            [
                "  3 -> 3 channels, x4 scale, 48 base channels",
                "  24 blocks, 128 hidden and 30 Fourier channels",
                "  5.86 KiB (6,000 bytes) of weights",
            ],
        ),
        (
            "smosr",
            [
                "  3 -> 3 channels, x2 scale, 48 base channels",
                "  2 + 3 + 1 blocks, 2-pixel reflection padding",
                "  6.84 KiB (7,000 bytes) of weights",
            ],
        ),
        (
            "figsr",
            [
                "  3 -> 3 channels, x2 scale, 2-pixel padding, 32 base channels",
                "  2 groups of 12 blocks, 80 hidden and 8 Fourier channels",
                "  13x13 and 1x17 spatial kernels, 4-pixel reflection padding",
                "  7.81 KiB (8,000 bytes) of weights",
            ],
        ),
    ],
)
def test_every_family_reports_its_architecture(monkeypatch: pytest.MonkeyPatch, family: str, expected: list[str]):
    _install_fake_runtime(monkeypatch, info=_fake_info(family=family))

    lines = describe_package(read_ummodel("/models/page.ummodel"))

    assert lines[0].startswith(f"/models/page.ummodel: {family} schema 1, ")
    assert lines[4:-1] == expected


def test_smosr_and_figsr_normalize_their_payload_details(monkeypatch: pytest.MonkeyPatch):
    _install_fake_runtime(monkeypatch, info=_fake_info(family="smosr"))
    smosr_info = read_ummodel("/models/page.ummodel")
    assert smosr_info.details == SMoSRDetails(
        reflection_padding=2,
        first_blocks=2,
        middle_blocks=3,
        end_blocks=1,
        weight_bytes=7000,
    )

    _install_fake_runtime(monkeypatch, info=_fake_info(family="figsr"))
    figsr_info = read_ummodel("/models/page.ummodel")
    assert figsr_info.details == FiGSRDetails(
        reflection_padding=4,
        blocks_per_group=12,
        hidden_channels=80,
        fourier_channels=8,
        square_kernel=13,
        band_kernel=17,
        weight_bytes=8000,
    )


def test_an_unreadable_payload_still_reports_the_container(monkeypatch: pytest.MonkeyPatch):
    """A known family with an unsupported payload schema has no payload to report."""
    info = _fake_info(family="gaterv3", problems=("Unsupported GateRV3 model payload version",))
    info.payload = None
    _install_fake_runtime(monkeypatch, info=info)

    lines = describe_package(read_ummodel("/models/page.ummodel"))

    assert lines[0] == "/models/page.ummodel: gaterv3 schema 1, unsupported"
    assert lines[-1] == "  problem: Unsupported GateRV3 model payload version"


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


@pytest.mark.parametrize(
    ("family", "precision", "expected_options"),
    [
        ("smosr", "fp16", {"fuse_silu": 1}),
        ("smosr", "mixed", {"tune_convolutions": 1, "fuse_silu": 1}),
        ("smosr", "fp32", {}),
        ("figsr", "fp16", {"tune_convolutions": 1, "fuse_expand_gate": 1}),
        ("figsr", "mixed", {"tune_convolutions": 1, "fuse_row_gate": 1, "fuse_expand_gate": 1}),
        ("figsr", "fp32", {"tune_convolutions": 1, "fp32_nchw": 1}),
    ],
)
def test_a_session_picks_precision_scoped_recommended_options(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    precision: str,
    expected_options: dict[str, int],
):
    _install_fake_runtime(monkeypatch, info=_fake_info(family=family))
    _no_torch(monkeypatch)

    session = create_umzi_session("/models/page.ummodel", family=family, precision=precision)

    assert session.options.kwargs["precision"] == precision
    assert session.options.kwargs["kernel_options"] == expected_options


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

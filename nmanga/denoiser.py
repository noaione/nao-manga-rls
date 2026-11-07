"""
MIT License

Copyright (c) 2022-present noaione
Copyright (c) 2025-present anon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import time
import warnings
from enum import Enum
from hashlib import md5, sha256
from importlib import util as importutil
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from PIL import Image

from .common import format_elapsed_time
from .lazy import get_einops_rearrange, get_numpy, get_onnxruntime
from .term import get_console

if TYPE_CHECKING:
    import onnxruntime as ort

    class InferenceSessionWithScale(ort.InferenceSession):
        scale_factor: int | None
        use_halfp: bool
        clamped_batch_size_warned: bool


__all__ = (
    "MLDataType",
    "build_compilation_cache_key",
    "denoise_single_image",
    "denoise_single_image_with_overlap",
    "get_model_fingerprint",
    "get_model_information",
    "get_model_static_input",
    "get_tf32_tag",
    "prepare_model_runtime",
    "prepare_model_runtime_builders",
    "resolve_model_static_shapes",
)

logger = logging.getLogger(__name__)
__winml_registered__ = False
__nmanga_trt_extension_hooked__ = False


class MLDataType(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    @classmethod
    def from_str(cls, type_str: str) -> MLDataType:
        type_str = type_str.lower()
        if type_str == "fp32":
            return cls.FP32
        elif type_str == "fp16":
            return cls.FP16
        elif type_str == "bf16":
            return cls.BF16
        else:
            raise ValueError(f"Unknown MLDataType string: {type_str}")

    def to_onv(self) -> str:
        if self == MLDataType.FP32:
            return "f32"
        elif self == MLDataType.FP16:
            return "f16"
        elif self == MLDataType.BF16:
            return "bf16"
        else:
            raise ValueError(f"Unsupported MLDataType for OpenVINO: {self}")


def get_data_dir() -> Path:
    # Should use APPDATA on Windows, XDG_CACHE_HOME on Linux, etc. but for simplicity just use home/.cache
    if sys.platform == "win32":
        cache_dir = Path.home() / "AppData" / "Local" / "nmanga"
    else:
        cache_dir = Path.home() / ".cache" / "nmanga"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_model_scale_factor(
    session: "InferenceSessionWithScale", *, tile_size: int | None = None, batch_size: int | None = None
) -> tuple[int, bool]:
    import numpy as np

    if scale_factor := getattr(session, "scale_factor", None):
        return scale_factor

    # Detect scale using small input and output shapes
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    is_float16 = "float16" in str(inp.type)

    maybe_tile_size = inp.shape[-1]
    maybe_output_size = out.shape[-1]

    if isinstance(maybe_tile_size, int) and isinstance(maybe_output_size, int):
        return maybe_output_size // maybe_tile_size, is_float16  # Fast path if both sizes are known

    real_tile_size = tile_size if tile_size is not None else maybe_tile_size
    if not isinstance(real_tile_size, int):
        real_tile_size = 64  # Fallback tile size

    # Most model would have shape like (batch_size, channels, height, width)
    # A symbolic output forces this probe to actually run inference, so it has to use the batch
    # the session will later be captured with: a TRT-RTX CUDA graph is replayed with the shapes of
    # the first run, and a probe at batch 1 would capture a graph that cannot serve a bigger batch.
    probe_batch_size = batch_size if batch_size is not None else inp.shape[0]
    if not isinstance(probe_batch_size, int):
        probe_batch_size = 1
    shapes = [probe_batch_size, inp.shape[1], real_tile_size, real_tile_size]

    x = np.zeros(shapes, dtype=np.float16 if is_float16 else np.float32)
    input_name = inp.name
    output_name = out.name

    out = session.run([output_name], {input_name: x})
    scale_height = np.array(out[0]).shape[-1]
    return scale_height // real_tile_size, is_float16


def get_model_fixed_batch_size(model: "InferenceSessionWithScale") -> int | None:
    """Get the batch size that is hardcoded into the model input, if there is one.

    Some exports (the chaiNNer/spandrel ones, usually) declare a static batch dimension, most
    commonly `1`. ONNX Runtime rejects any bigger batch on those models, so callers have to stay
    within the declared batch size.

    :param model: The ONNX Runtime inference session
    :return: The fixed batch size, or `None` if the batch dimension is dynamic
    """

    input_shape = model.get_inputs()[0].shape
    batch_dim = input_shape[0] if input_shape else None
    if isinstance(batch_dim, int) and batch_dim > 0:
        return batch_dim
    return None


def resolve_model_input_sizes(model: "InferenceSessionWithScale", *, batch_size: int, tile_size: int) -> int:
    """Resolve the batch size that the model input actually accepts.

    Models with fixed input dimensions reject everything else with an opaque `INVALID_ARGUMENT`
    error, so the batch size is clamped to the declared one (warning once per session), while a
    tile size that does not match a fixed spatial dimension is reported as a proper error.

    :param model: The ONNX Runtime inference session
    :param batch_size: The requested batch size
    :param tile_size: The requested tile size
    :return: The batch size that the model accepts
    :raises ValueError: If the model declares a fixed tile size different from `tile_size`
    """

    input_shape = model.get_inputs()[0].shape
    if len(input_shape) < 4:
        return batch_size

    fixed_batch_size = get_model_fixed_batch_size(model)
    if fixed_batch_size is not None and fixed_batch_size < batch_size:
        if not getattr(model, "clamped_batch_size_warned", False):
            model.clamped_batch_size_warned = True
            get_console().warning(
                f"Model declares a fixed batch size of {fixed_batch_size}; "
                f"clamping the requested batch size of {batch_size} down to {fixed_batch_size}."
            )
        batch_size = fixed_batch_size

    fixed_spatial_dims = [dim for dim in input_shape[-2:] if isinstance(dim, int) and dim > 0]
    if fixed_spatial_dims and any(dim != tile_size for dim in fixed_spatial_dims):
        fixed_tile_size = "x".join(str(dim) for dim in fixed_spatial_dims)
        raise ValueError(
            f"The model only accepts a fixed tile size of {fixed_tile_size}, but {tile_size} was requested. "
            f"Re-run with the tile size set to {fixed_tile_size}."
        )
    return batch_size


def resolve_model_static_shapes(
    input_shape: tuple[int | None, ...] | list[int | None] | None,
    *,
    batch_size: int,
    tile_size: int | None,
) -> tuple[int, int | None]:
    """Clamp the requested optimization profile to the dimensions the graph declares as static.

    The TensorRT optimization profiles are configured *before* any session exists, so they cannot
    rely on :func:`resolve_model_input_sizes`, which only runs once an image is already being
    processed. Building profiles that contradict a fixed-shape graph yields an engine that fails at
    the first inference instead of a clear error here.

    :param input_shape: The declared input dimensions, `None` for a dynamic axis, or `None` when unknown
    :param batch_size: The requested batch size
    :param tile_size: The requested tile size, or `None` when it should stay dynamic
    :return: The batch size and tile size to build the profiles with
    :raises ValueError: If the graph declares a fixed tile size different from `tile_size`
    """

    if not input_shape or len(input_shape) < 4:
        return batch_size, tile_size

    fixed_batch_size = input_shape[0]
    if fixed_batch_size is not None and 0 < fixed_batch_size < batch_size:
        get_console().warning(
            f"Model declares a fixed batch size of {fixed_batch_size}; "
            f"building the optimization profiles for it instead of the requested {batch_size}."
        )
        batch_size = fixed_batch_size

    if tile_size is None:
        return batch_size, tile_size

    fixed_spatial_dims = [dim for dim in input_shape[-2:] if dim is not None and dim > 0]
    if fixed_spatial_dims and any(dim != tile_size for dim in fixed_spatial_dims):
        fixed_tile_size = "x".join(str(dim) for dim in fixed_spatial_dims)
        raise ValueError(
            f"The model only accepts a fixed tile size of {fixed_tile_size}, but {tile_size} was requested. "
            f"Re-run with the tile size set to {fixed_tile_size}."
        )
    return batch_size, tile_size


def prepare_model_runtime(
    model_path: Path,
    *,
    device_id: int = 0,
    is_verbose: bool = False,
) -> "InferenceSessionWithScale":
    # _preload_custom_extensions()

    ort = get_onnxruntime()

    if sys.platform == "darwin":
        CACHE_DIR = Path.home() / ".cache" / "nmanga-denoiser"
        providers = [
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "1",
                    "EnableOnSubgraphs": "1",
                    "ModelCacheDirectory": str(CACHE_DIR),
                    "SpecializationStrategy": "FastPrediction",
                },
            )
        ]
    else:
        available_providers = set(ort.get_available_providers())
        providers = []
        if "TensorrtExecutionProvider" in available_providers:
            providers.append((
                "TensorrtExecutionProvider",
                {
                    "device_id": device_id,
                    "trt_fp16_enable": True,
                    "trt_sparsity_enable": True,
                },
            ))
        if "CUDAExecutionProvider" in available_providers:
            providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
        if not providers:
            raise RuntimeError(
                "No suitable ONNX Runtime GPU execution providers found. "
                "Install the denoiser extra with onnxruntime-gpu CUDA/cuDNN support."
            )

    verb_level = 0 if is_verbose else 3
    ort.set_default_logger_severity(verb_level)
    ort.set_default_logger_verbosity(verb_level)

    sess_opt = ort.SessionOptions()
    session = cast(
        "InferenceSessionWithScale", ort.InferenceSession(model_path, sess_options=sess_opt, providers=providers)
    )
    scale_factor, use_halfp = get_model_scale_factor(session, tile_size=None)
    session.scale_factor = scale_factor
    session.use_halfp = use_halfp
    return session


def get_model_information(model_path: Path) -> tuple[str, int]:
    try:
        import onnx  # pyright: ignore[reportMissingImports]
    except ImportError:
        warnings.warn(
            "onnx is not installed, cannot determine model input name. "
            "Defaulting to 'input'. Install onnx to enable proper model input name detection.",
            ImportWarning,
            stacklevel=2,
        )
        return "input", 3  # Default input name and channel count

    model = onnx.load_model(str(model_path), load_external_data=False)
    # input channel count
    if len(model.graph.input) != 1:
        raise ValueError("Model has multiple inputs, cannot determine input name.")
    model_channel_count = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    return model.graph.input[0].name, model_channel_count


def get_model_static_input(model_path: Path) -> tuple[str, tuple[int | None, ...]] | None:
    """
    Read the input name and the declared size of every input dimension of an ONNX graph.

    Only the graph metadata is parsed (`load_external_data=False`), so this is safe for
    graphs whose weights live in a sidecar file. A dimension is reported as `None` when the
    graph declares it as dynamic/symbolic.

    :param model_path: The ONNX model file
    :return: The input name with its dimensions, or `None` when it cannot be determined
    """

    try:
        import onnx  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None

    try:
        model = onnx.load_model(str(model_path), load_external_data=False)
    except Exception:
        # A model that cannot be parsed will fail later with a better error, but the caller
        # (profile configuration, cache keys) must not die while only reading metadata.
        return None

    if len(model.graph.input) != 1:
        return None

    graph_input = model.graph.input[0]
    dimensions = tuple(
        dimension.dim_value if dimension.dim_value > 0 else None for dimension in graph_input.type.tensor_type.shape.dim
    )
    return graph_input.name, dimensions


# (resolved path, size, mtime_ns): changes whenever the file is replaced or rewritten
_FileStat = tuple[str, int, int]

_HASH_CHUNK_SIZE = 1024 * 1024
_external_data_cache: dict[_FileStat, tuple[Path, ...]] = {}
_model_fingerprint_cache: dict[tuple[_FileStat, tuple[_FileStat, ...]], str] = {}


def _hash_file(path: Path) -> str:
    """
    Stream a file through SHA-256.

    :param path: The file to hash
    :return: The lowercase hexadecimal digest
    """

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_external_data_files(model_path: Path) -> tuple[Path, ...]:
    """
    List the sidecar files that hold a model's external tensor data.

    A graph exported with external data keeps only metadata in the `.onnx` file, so hashing
    the main file alone would not notice changed weights.

    :param model_path: The ONNX model file
    :return: The existing external data files, sorted
    """

    stat = model_path.stat()
    cache_key = (str(model_path.resolve()), stat.st_size, stat.st_mtime_ns)
    cached = _external_data_cache.get(cache_key)
    if cached is not None:
        return cached

    locations: set[Path] = set()
    try:
        import onnx  # pyright: ignore[reportMissingImports]
    except ImportError:
        onnx = None  # type: ignore[assignment]

    if onnx is not None:
        try:
            model = onnx.load_model(str(model_path), load_external_data=False)
        except Exception:
            model = None

        if model is not None:
            for initializer in model.graph.initializer:
                if initializer.data_location != onnx.TensorProto.EXTERNAL:
                    continue
                for entry in initializer.external_data:
                    if entry.key == "location" and entry.value:
                        locations.add((model_path.parent / entry.value))

    external_files = tuple(sorted(path for path in locations if path.is_file()))
    _external_data_cache[cache_key] = external_files
    return external_files


def get_model_fingerprint(model_path: Path) -> str:
    """
    Fingerprint everything that makes an ONNX graph the graph it is.

    This covers the bytes of the model file plus the bytes of any external tensor data, so
    replacing the weights behind an unchanged pathname produces a different fingerprint.
    Results are memoized on the file stats, so repeated calls stay cheap.

    :param model_path: The ONNX model file
    :return: The lowercase hexadecimal SHA-256 digest of the graph contents
    """

    model_path = Path(model_path)
    stat = model_path.stat()
    external_files = _get_external_data_files(model_path)
    external_stats = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in external_files)
    cache_key = ((str(model_path.resolve()), stat.st_size, stat.st_mtime_ns), external_stats)
    cached = _model_fingerprint_cache.get(cache_key)
    if cached is not None:
        return cached

    digest = sha256()
    digest.update(_hash_file(model_path).encode("ascii"))
    for external_file in external_files:
        digest.update(external_file.name.encode("utf-8"))
        digest.update(_hash_file(external_file).encode("ascii"))

    fingerprint = digest.hexdigest()
    _model_fingerprint_cache[cache_key] = fingerprint
    return fingerprint


def get_tf32_tag() -> str:
    """
    Describe the effective TensorRT TF32 policy as a filesystem-safe cache tag.

    TF32 is only controllable through the `NVIDIA_TF32_OVERRIDE` environment variable, which
    TensorRT reads while it builds an engine, so the value belongs in every engine cache key.

    :return: A short tag such as `tf32default`, `tf32off` or `tf321`
    """

    override = os.environ.get("NVIDIA_TF32_OVERRIDE")
    if override is None or not override.strip():
        return "tf32default"

    normalized = override.strip().lower()
    if normalized in {"0", "false", "off", "no"}:
        return "tf32off"
    return f"tf32{normalized}"


def build_compilation_cache_key(
    model_path: Path,
    *,
    device_id: int = 0,
    data_type: MLDataType = MLDataType.FP16,
    tile_size: int | None = None,
    batch_size: int | None = None,
    workspace_size: int | None = None,
    enable_cuda_graph: bool = True,
    with_nvrtx: bool = False,
) -> str:
    """
    Build the cache key shared by every compilation cache entry of one graph.

    The key has to cover everything that changes the compiled artifact: the graph contents
    (not just its path), the requested precision, the device, the optimization profiles and
    the build options. Anything missing here silently reuses an engine built for other
    settings, which is how a TF32-off run can end up loading a TF32-on engine.

    :param model_path: The ONNX model file
    :param device_id: The GPU device ID the engine is built for
    :param data_type: The requested precision
    :param tile_size: The tile size baked into the optimization profiles
    :param batch_size: The batch size baked into the optimization profiles
    :param workspace_size: The workspace limit in bytes the build will use
    :param enable_cuda_graph: Whether CUDA graph replay is requested
    :param with_nvrtx: Whether the TRT-RTX execution provider is requested
    :return: A filesystem-safe cache key
    """

    parts = [
        f"nmodel_t{tile_size}b{batch_size}",
        f"cd{data_type.name}",
        f"d{device_id}",
        f"rtx{int(with_nvrtx)}",
        f"cg{int(enable_cuda_graph)}",
        f"ws{'auto' if workspace_size is None else workspace_size}",
        get_tf32_tag(),
        f"g{get_model_fingerprint(model_path)[:16]}",
        _get_trt_runtime_version_tag(),
    ]
    return "_".join(parts)


def get_torch_memory_limit_and_rtx(device_id: int) -> tuple[int, int] | None:
    try:
        import torch.cuda  # type: ignore

        # Both queries have to target the same device: the capability of the *current* device
        # would answer for another GPU than the one the properties are read from
        cu_major, _ = torch.cuda.get_device_capability(device_id)
        return torch.cuda.get_device_properties(device_id).total_memory, cu_major
    except (ImportError, AssertionError, RuntimeError):
        # CPU-only torch, or a CUDA build on a machine without a usable NVIDIA driver (the
        # AMD/Intel extras install the CPU wheel), so fall back to the default workspace.
        return None


_TRT_RUNTIME_VERSION_MODULES: tuple[tuple[str, str], ...] = (
    ("trt", "tensorrt"),
    ("ep", "onnxruntime_ep_nv_tensorrt_rtx"),
    ("ort", "onnxruntime"),
)


def _get_trt_runtime_version_tag() -> str:
    """
    Build a cache-key tag out of the TRT-RTX runtime component versions.

    TensorRT compiles engines against a fixed engine serialization version and refuses to
    deserialize anything produced by an incompatible runtime (the
    ``stdVersionRead == kSERIALIZATION_VERSION`` failure). Folding the runtime versions into
    the cache key makes sure a stale engine is rebuilt instead of being reused after an upgrade.

    :return: A filesystem-safe version tag, or ``"unknown"`` if nothing could be detected.
    """

    versions: list[str] = []
    for tag, module_name in _TRT_RUNTIME_VERSION_MODULES:
        if importutil.find_spec(module_name) is None:
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        module_version = getattr(module, "__version__", None)
        versions.append(f"{tag}{module_version}" if module_version is not None else f"{tag}unknown")
    if not versions:
        return "unknown"
    return "-".join(versions)


def _get_nvrtx_compiled_model_path(model_path: Path, data_dir: Path, cache_key: str) -> Path:
    cache_hash = md5(cache_key.encode("utf-8")).hexdigest()  # ruff: ignore[hashlib-insecure-hash-function]
    return data_dir / "rtx_compiled" / f"{model_path.stem}_{cache_hash}_ctx.onnx"


def _compile_nvrtx_model(
    model_path: Path,
    data_dir: Path,
    cache_key: str,
    ep_name: str,
    ep_config: dict[str, str],
) -> Path:
    ort = get_onnxruntime()

    cnsl = get_console()
    model_comp = _get_nvrtx_compiled_model_path(model_path, data_dir, cache_key)
    model_comp.parent.mkdir(parents=True, exist_ok=True)
    if model_comp.exists():
        cnsl.info(f"Found pre-compiled model {model_path.stem} for TRT-RTX engine")
        return model_comp

    sess_opt = ort.SessionOptions()
    sess_opt.add_provider(ep_name, ep_config)
    # get file size, if larger than 1.5gb, we don't embed ep context (although the recommended is 2gb)
    model_stat = model_path.stat()
    is_large_size = model_stat.st_size > 1.5 * (1024**3)
    compiler = ort.ModelCompiler(
        sess_opt,
        model_path,
        embed_compiled_data_into_model=not is_large_size,
        graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    )
    cnsl.info(f"Pre-compiling model {model_path.stem} for TRT-RTX engine")
    st_time = time.time()
    compiler.compile_to_file(str(model_comp))
    et_time = time.time()
    elapsed_pretty = format_elapsed_time(et_time - st_time)
    cnsl.info(f"Compiled model {model_path.stem} for TRT-RTX engine (in {elapsed_pretty})")
    return model_comp


def _compile_nvrtx_model_for_ep(
    model_path: Path,
    data_dir: Path,
    cache_key: str,
    ep_device,
    ep_config: dict[str, str],
) -> Path:
    ort = get_onnxruntime()

    cnsl = get_console()
    model_comp = _get_nvrtx_compiled_model_path(model_path, data_dir, cache_key)
    model_comp.parent.mkdir(parents=True, exist_ok=True)
    if model_comp.exists():
        cnsl.info(f"Found pre-compiled model {model_path.stem} for TRT-RTX EP engine")
        return model_comp

    sess_opt = ort.SessionOptions()
    sess_opt.add_provider_for_devices([ep_device], ep_config)
    # get file size, if larger than 1.5gb, we don't embed ep context (although the recommended is 2gb)
    model_stat = model_path.stat()
    is_large_size = model_stat.st_size > 1.5 * (1024**3)
    compiler = ort.ModelCompiler(
        sess_opt,
        model_path,
        embed_compiled_data_into_model=not is_large_size,
        graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    )
    cnsl.info(f"Pre-compiling model {model_path.stem} for TRT-RTX EP engine")
    st_time = time.time()
    compiler.compile_to_file(str(model_comp))
    et_time = time.time()
    elapsed_pretty = format_elapsed_time(et_time - st_time)
    cnsl.info(f"Compiled model {model_path.stem} for TRT-RTX engine (in {elapsed_pretty})")
    return model_comp


def prepare_model_runtime_builders(
    model_path: Path,
    *,
    device_id: int = 0,
    is_verbose: bool = False,
    tile_size: int | None = None,
    batch_size: int = 64,
    data_type: MLDataType = MLDataType.FP16,
    with_nvrtx: bool = False,
    workspace_size: int | None = None,
    enable_cuda_graph: bool = True,
    strict_ep: bool = False,
) -> "InferenceSessionWithScale":
    """Build an ONNX Runtime session, letting the TensorRT RTX builder compile the engine.

    Unlike :func:`prepare_model_runtime`, this never forces a global FP16 optimization on the graph,
    so a mixed precision export keeps its own per-operator types.

    :param model_path: The ONNX model file
    :param device_id: The GPU device ID to build for
    :param is_verbose: Enable the detailed build logs of the execution providers
    :param tile_size: The tile size to build the optimization profiles for
    :param batch_size: The batch size to build the optimization profiles for
    :param data_type: The requested build precision
    :param with_nvrtx: Request the TRT-RTX execution provider
    :param workspace_size: The workspace limit in bytes, or `None` to use 75% of the GPU memory
    :param enable_cuda_graph: Enable CUDA graph replay for the TRT-RTX engine
    :param strict_ep: Fail instead of falling back when the requested provider is unavailable
    :return: The inference session with the detected scale factor attached
    :raises RuntimeError: If `strict_ep` is set and the requested provider cannot be used
    """

    # _preload_custom_extensions()

    ort = get_onnxruntime()

    verb_level = 0 if is_verbose else 3
    ort.set_default_logger_severity(verb_level)
    ort.set_default_logger_verbosity(verb_level)

    data_dir = get_data_dir()
    cache_dir = data_dir / "trt_engines"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_rtx_dir = data_dir / "trtrtx_engines"
    cache_rtx_dir.mkdir(parents=True, exist_ok=True)
    cache_migx_dir = data_dir / "migx_cache"
    cache_migx_dir.mkdir(parents=True, exist_ok=True)
    cache_onv_dir = data_dir / "openvino_cache"
    cache_onv_dir.mkdir(parents=True, exist_ok=True)

    torch_info = get_torch_memory_limit_and_rtx(device_id)
    memory_limit = int(torch_info[0] * 0.75) if torch_info else 2 * (1024**3)  # 2GB or 75% of GPU memory
    if workspace_size is not None:
        memory_limit = workspace_size

    cnsl = get_console()

    has_trt_rtx = torch_info[1] >= 8 if torch_info else False
    if with_nvrtx and not has_trt_rtx:
        if strict_ep:
            compute_capability = f"{torch_info[1]}.x" if torch_info else "unknown"
            raise RuntimeError(
                "TensorRT RTX was requested but it is not supported on this GPU "
                f"(CUDA compute capability {compute_capability}, TRT-RTX needs 8.0 or newer). "
                "Re-run without --with-nvrtx, or stop asking for a strict execution provider."
            )
        cnsl.warning("TensorRT RTX is not supported on this GPU. Falling back to TensorRT.")
        with_nvrtx = False

    # The optimization profiles are configured before any session exists, so the graph itself has
    # to say which shapes it accepts; the image-time clamp in `resolve_model_input_sizes` runs too
    # late to keep a contradictory profile from being built.
    static_input = get_model_static_input(model_path)
    if static_input is not None:
        model_input, static_input_shape = static_input
        model_channel_count = static_input_shape[1] if len(static_input_shape) > 1 else None
        if not model_channel_count:
            model_channel_count = 3
    else:
        static_input_shape = None
        model_input, model_channel_count = get_model_information(model_path)
    batch_size, tile_size = resolve_model_static_shapes(static_input_shape, batch_size=batch_size, tile_size=tile_size)

    # The cache key covers the graph contents and every build option, so an engine compiled for
    # another graph, precision, workspace or TF32 policy is never reused.
    cache_prefix = build_compilation_cache_key(
        model_path,
        device_id=device_id,
        data_type=data_type,
        tile_size=tile_size,
        batch_size=batch_size,
        workspace_size=memory_limit,
        enable_cuda_graph=enable_cuda_graph,
        with_nvrtx=with_nvrtx,
    )

    needs_trt_compat = torch_info[1] >= 12 if torch_info else False

    cnsl.info(f"Memory limit is set to: {memory_limit}")

    trt_ep_config = {
        "device_id": device_id,
        "trt_fp16_enable": data_type == MLDataType.FP16,
        "trt_bf16_enable": data_type == MLDataType.BF16,
        "trt_sparsity_enable": True,
        "trt_max_workspace_size": memory_limit,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "trt_engines",
        "trt_engine_cache_prefix": cache_prefix,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(cache_dir),
        "trt_build_heuristics_enable": True,
        "trt_builder_optimization_level": 3,
        "trt_context_memory_sharing_enable": True,
        "trt_dump_ep_context_model": True,
        "trt_ep_context_file_path": str(data_dir),
        "trt_detailed_build_log": True if is_verbose else False,
        "trt_engine_hw_compatible": needs_trt_compat,
    }
    trtrtx_ep_config = {
        "enable_cuda_graph": "1" if enable_cuda_graph else "0",
        "nv_max_workspace_size": str(memory_limit),
        "nv_detailed_build_log": "1" if is_verbose else "0",
        "nv_runtime_cache_path": str(cache_rtx_dir),
    }
    migraphx_ep_config = {
        "device_id": device_id,
        "migraphx_fp16_enable": data_type == MLDataType.FP16,
        "migraphx_bf16_enable": data_type == MLDataType.BF16,
        "migraphx_exhaustive_tune": 1,
        "migraphx_mem_limit": memory_limit,
        "migraphx_cache_dir": str(cache_migx_dir),
    }
    onv_json_ep_config = {
        "CPU": {
            "PERFORMANCE_HINT": "THROUGHPUT",
            "EXECUTION_MODE_HINT": "ACCURACY",
            "CACHE_DIR": str(cache_onv_dir),
            "CACHE_MODE": "OPTIMIZE_SPEED",
        },
        "GPU": {
            "PERFORMANCE_HINT": "THROUGHPUT",
            "EXECUTION_MODE_HINT": "ACCURACY",
            "INFERENCE_PRECISION_HINT": data_type.to_onv(),
            "NUM_STREAMS": "2",  # A bit more
            "CACHE_DIR": str(cache_onv_dir),
            "CACHE_MODE": "OPTIMIZE_SPEED",
        },
    }
    onv_ep_config = {
        "device_type": "HETERO:GPU,CPU",
        "load_config": json.dumps(onv_json_ep_config),
    }

    if tile_size is not None:
        min_shapes = f"{model_input}:1x{model_channel_count}x{tile_size}x{tile_size}"
        max_shapes = f"{model_input}:{batch_size}x{model_channel_count}x{tile_size}x{tile_size}"
        opt_shapes = max_shapes
        trt_ep_config["trt_profile_min_shapes"] = min_shapes
        trt_ep_config["trt_profile_max_shapes"] = max_shapes
        trt_ep_config["trt_profile_opt_shapes"] = opt_shapes

        trtrtx_ep_config["nv_profile_min_shapes"] = min_shapes
        trtrtx_ep_config["nv_profile_max_shapes"] = max_shapes
        trtrtx_ep_config["nv_profile_opt_shapes"] = opt_shapes

    # Several devices can share one EP name (OpenVINO exposes a CPU, a GPU and a NPU), so keep the
    # list around instead of only the last device seen per name.
    all_ep_devices = ort.get_ep_devices()
    ep_devices = {ep_device.ep_name: ep_device for ep_device in all_ep_devices}
    has_good_ep_devices = False
    has_nvrtx = False
    raw_providers = set(ort.get_available_providers())
    if sys.platform != "darwin":
        providers = []
        ep_providers = []
        has_nvrtx = False
        if "NvTensorRTRTXExecutionProvider" in ep_devices and with_nvrtx:
            ep_providers.append((ep_devices["NvTensorRTRTXExecutionProvider"], trtrtx_ep_config))
            has_nvrtx = True
            has_good_ep_devices = True
        if "TensorrtExecutionProvider" in raw_providers and not has_nvrtx:
            providers.append(("TensorrtExecutionProvider", trt_ep_config))
        if "MIGraphXExecutionProvider" in raw_providers:
            providers.append(("MIGraphXExecutionProvider", migraphx_ep_config))
        elif "MIGraphXExecutionProvider" in ep_devices:
            ep_providers.append((ep_devices["MIGraphXExecutionProvider"], migraphx_ep_config))
            has_good_ep_devices = True
        if "CUDAExecutionProvider" in raw_providers:
            providers.append((
                "CUDAExecutionProvider",
                {
                    "device_id": device_id,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": memory_limit,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                    "cudnn_conv_use_max_workspace": True,
                    "prefer_nhwc": True,
                },
            ))
        ov_plugin_devices = [
            dev
            for dev in all_ep_devices
            if dev.ep_name == "OpenVINOExecutionProvider" and dev.ep_metadata.get("ov_device")
        ]
        if ov_plugin_devices and not providers and not ep_providers:
            # Nothing else was picked, so use the plugin's device API: the device itself is what
            # selects CPU, GPU or NPU. Keep the old `HETERO:GPU,CPU` preference, minus the hetero.
            by_ov_device = {dev.ep_metadata["ov_device"]: dev for dev in ov_plugin_devices}
            chosen_ov_device = by_ov_device.get("GPU")
            if chosen_ov_device is None:
                chosen_ov_device = by_ov_device.get("CPU", ov_plugin_devices[0])
            ep_providers.append((chosen_ov_device, {}))
            has_good_ep_devices = True
        elif ov_plugin_devices or "OpenVINOExecutionProvider" in raw_providers:
            # A plugin device cannot be mixed with a provider list, so when an accelerator already
            # comes first (CUDA or TensorRT RTX) add OpenVINO as a name only fallback. The in-tree
            # options only apply to a runtime that ships the provider itself.
            providers.append(("OpenVINOExecutionProvider", {} if ov_plugin_devices else onv_ep_config))
    else:
        ep_providers = []
        providers = []
        if "MLXExecutionProvider" in raw_providers:
            # Apple Silicon only; unsupported nodes fall through to CoreML and then the CPU
            providers.append(("MLXExecutionProvider", {}))
        providers.append(
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "1",
                    "EnableOnSubgraphs": "1",
                    "ModelCacheDirectory": str(cache_dir),
                    "SpecializationStrategy": "FastPrediction",
                },
            ),
        )

    if not providers and not ep_providers:
        raise RuntimeError(
            "No suitable ONNX Runtime execution providers found. "
            "Ensure you have compatible hardware and the necessary dependencies installed."
        )

    # Resolve what a strict run has to see in the finished session. TRT-RTX answers to two names
    # depending on whether it is the built-in provider or the standalone plugin, so both count.
    required_providers: set[str] = set()
    if strict_ep:
        if with_nvrtx:
            required_providers = {"NvTensorRTRTXExecutionProvider"}
            if not has_nvrtx:
                raise RuntimeError(
                    "TensorRT RTX was requested but this ONNX Runtime exposes no RTX execution provider. "
                    f"Available providers: {', '.join(sorted(raw_providers))}. "
                    "Install the TRT-RTX plugin, or drop the strict execution provider check."
                )
        elif providers:
            required_providers = {providers[0][0]}

    cnsl.info(f"Running with ONNX Runtime v{ort.version}")
    cnsl.info("Available providers:", raw_providers)
    cnsl.info("Available EP devices:", list(ep_devices.keys()))

    st_time = time.time()

    sess_opt = ort.SessionOptions()
    for ep_devices, ep_config in ep_providers:
        sess_opt.add_provider_for_devices([ep_devices], ep_config)

    # Check if nvrtx exist, if yes we should recompile
    selected_model_path = model_path

    if has_good_ep_devices:
        providers = None  # Use EP devices for NvRTX

    for ep_devices, ep_config in ep_providers:
        if ep_devices.ep_name == "NvTensorRTRTXExecutionProvider":
            selected_model_path = _compile_nvrtx_model_for_ep(
                model_path,
                data_dir,
                cache_prefix,
                ep_devices,
                ep_config,
            )
            break

    session = cast(
        "InferenceSessionWithScale",
        ort.InferenceSession(selected_model_path, sess_options=sess_opt, providers=providers, enable_fallback=0),
    )
    end_time = time.time()

    elapsed = format_elapsed_time(end_time - st_time)
    active_providers = session.get_providers()
    cnsl.info("Active providers:", active_providers)
    cnsl.info(f"Loaded model: {model_path.stem} (in {elapsed})")
    if required_providers and not required_providers.intersection(active_providers):
        raise RuntimeError(
            f"The {', '.join(sorted(required_providers))} execution provider was requested, but the session "
            f"only reports {', '.join(active_providers) or 'no providers'}. "
            "A fallback provider was chosen instead, so the run would not match the requested setup."
        )
    scale_factor, use_halfp = get_model_scale_factor(session, tile_size=tile_size, batch_size=batch_size)
    session.scale_factor = scale_factor
    session.use_halfp = use_halfp
    return session


def denoise_single_image(
    input_image: Image.Image,
    model: "InferenceSessionWithScale",
    *,
    batch_size: int = 64,
    tile_size: int = 128,
    contrast_stretch: bool = False,
    background: Literal["white", "black"] = "black",
    use_fp32: bool = False,
) -> Image.Image:
    np = get_numpy()
    rearrange = get_einops_rearrange()

    # Calculate upscale scale from model input and output shapes
    scale_factor = getattr(model, "scale_factor", None)
    if not isinstance(scale_factor, int):
        raise ValueError("Model scale factor is not set. Ensure the model was prepared correctly.")
    input_channel_count: int = model.get_inputs()[0].shape[1]
    is_grayscale = bool(input_channel_count == 1)
    # Some models declare fixed input dimensions, only feed them what they accept.
    batch_size = resolve_model_input_sizes(model, batch_size=batch_size, tile_size=tile_size)

    orig_img_mode = input_image.mode
    orig_img_palette = input_image.palette
    img = input_image.convert("RGB") if not is_grayscale else input_image.convert("L")
    image_width, image_height = img.size
    # compute the new canvas size
    tile_count_height = int(np.ceil((image_height / tile_size)))
    tile_count_width = int(np.ceil((image_width / tile_size)))

    padded_height = int(tile_count_height * tile_size)
    padded_width = int(tile_count_width * tile_size)
    # Create a new image with the instructed background and paste the old one into it.
    # A white background helps with contrast stretching (can lead to reversing CMYK shift).
    background_tuple = (0, 0, 0) if background == "black" else (255, 255, 255)
    new_padded_image = Image.new(
        ("RGB" if not is_grayscale else "L"),
        (padded_width, padded_height),
        background_tuple if not is_grayscale else background_tuple[0],
    )
    new_padded_image.paste(img, (0, 0))  # Paste at (0, 0) position
    # Pre-Processing
    image_array = np.array(new_padded_image)
    # For grayscale images, add channel dimension so rearrange operations work correctly
    if is_grayscale:
        image_array = np.expand_dims(image_array, axis=-1)
    # contrast stretching: scaling the image based on its own darkest and brightest pixels.
    # For example, a very dark photo (e.g., pixel values from 10 to 50) and a very bright photo (e.g., values
    # from 200 to 250) will both be stretched to the full [0.0, 1.0] range.
    # The model loses all information about the image's absolute brightness. Can be desirable with a white
    # background to reverse CMYK shift in rare (color) images.
    # Not desirable in general so the alternative path should be taken in the vast majority of cases.
    # See also: https://en.wikipedia.org/wiki/Normalization_(image_processing)#Contrast_Stretching_for_Image_Enhancement
    image_array = (
        image_array / 255.0 if not contrast_stretch else (image_array - np.min(image_array)) / np.ptp(image_array)
    )
    # The model's engine is being created expecting FP32 input if it is using TensorRT, therefore adjusting accordingly.
    image_array = image_array.astype(np.float32) if use_fp32 else image_array.astype(np.float16)
    # Rearranging (Width, Height, Channel) -> (Channel, Width, Height) to match the expected input shape
    image_array = rearrange(image_array, "w h c -> c w h")

    # Cut into tiles
    padded_image_tiles = rearrange(
        image_array,
        "c (h th) (w tw) -> (h w) c th tw",
        th=tile_size,
        tw=tile_size,
    )
    # Defining maximum batch size and getting input and output names
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    # Creating an empty list to store all chunks
    all_output_chunks = []
    num_chunks = padded_image_tiles.shape[0]

    # Looping through all chunks in batches of max_batch_size
    for i in range(0, num_chunks, batch_size):
        # Create a batch of chunks, ensuring not to go out of bounds
        batch_of_chunks = padded_image_tiles[i : min(i + batch_size, num_chunks)]

        # Run inference on the current batch
        model_output = model.run([output_name], {input_name: batch_of_chunks})

        # Store the output of the batch
        all_output_chunks.append(model_output[0])

    # Concatenate the results from all batches into a single array
    tiled_output_image = np.concatenate(all_output_chunks, axis=0)

    if scale_factor > 1:
        image_height = image_height * scale_factor
        image_width = image_width * scale_factor

    # Reshape it back to the input
    reconstructed_image_with_pad = rearrange(
        tiled_output_image,
        "(h w) c th tw -> (h th) (w tw) c",
        h=tile_count_height,
        w=tile_count_width,
    )

    # Post-processing
    # See an explanation for image-specific normalization in the pre-processing section.
    if contrast_stretch:
        postprocessed_array = (
            reconstructed_image_with_pad.astype(np.float32)
            if use_fp32
            else reconstructed_image_with_pad.astype(np.float16)
        )
        postprocessed_array = (postprocessed_array - np.min(postprocessed_array)) / np.ptp(postprocessed_array)
    else:
        postprocessed_array = np.clip(reconstructed_image_with_pad, 0.0, 1.0)
    # Scaling back to [0,255] from [0.0, 1.0]
    postprocessed_array = postprocessed_array * 255.0
    # Rounding the array for better casting to uint8.
    postprocessed_array = np.round(postprocessed_array)
    postprocessed_array = postprocessed_array.astype(np.uint8)
    # For grayscale images, remove channel dimension before creating PIL Image
    if is_grayscale:
        postprocessed_array = np.squeeze(postprocessed_array, axis=-1)
    output_image = Image.fromarray(postprocessed_array)

    # Cropping the image from the overall canvas.
    output_image = output_image.crop((0, 0, image_width, image_height))
    if not is_grayscale:
        if orig_img_palette:
            palette_image = Image.new("P", (1, 1))
            palette_image.putpalette(orig_img_palette)
            output_image = output_image.convert(mode="RGB").quantize(palette=palette_image, dither=Image.Dither.NONE)
        else:
            output_image = output_image.convert(mode=orig_img_mode)
    else:
        output_image = output_image.convert(mode="L")

    return output_image


def denoise_single_image_with_overlap(
    input_image: Image.Image,
    model: "InferenceSessionWithScale",
    *,
    batch_size: int = 64,
    tile_size: int = 128,
    tile_overlap: int = 16,
    contrast_stretch: bool = False,
    background: Literal["white", "black"] = "black",
    use_fp32: bool = False,
) -> Image.Image:
    """Denoise an image using fixed-size overlapping tiles and feathered seams.

    The following tile overlap function has been extracted from:
    - https://github.com/rewaifu/resr - (c) rewaifu team - MIT License
    """
    if tile_overlap == 0:
        return denoise_single_image(
            input_image,
            model,
            batch_size=batch_size,
            tile_size=tile_size,
            contrast_stretch=contrast_stretch,
            background=background,
            use_fp32=use_fp32,
        )

    import numpy as np  # type: ignore

    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    if tile_size < 1:
        raise ValueError("Tile size must be at least 1.")
    if tile_overlap < 0 or tile_overlap * 2 >= tile_size:
        raise ValueError("Tile overlap must be non-negative and less than half the tile size.")

    scale_factor = getattr(model, "scale_factor", None)
    if not isinstance(scale_factor, int):
        raise ValueError("Model scale factor is not set. Ensure the model was prepared correctly.")
    input_channel_count: int = model.get_inputs()[0].shape[1]
    is_grayscale = input_channel_count == 1
    # Some models declare fixed input dimensions, only feed them what they accept.
    batch_size = resolve_model_input_sizes(model, batch_size=batch_size, tile_size=tile_size)

    orig_img_mode = input_image.mode
    orig_img_palette = input_image.palette
    img = input_image.convert("L" if is_grayscale else "RGB")
    image_width, image_height = img.size

    tile_stride = tile_size - tile_overlap * 2
    tile_count_height = max(1, int(np.ceil(max(0, image_height - tile_size) / tile_stride)) + 1)
    tile_count_width = max(1, int(np.ceil(max(0, image_width - tile_size) / tile_stride)) + 1)
    padded_height = (tile_count_height - 1) * tile_stride + tile_size
    padded_width = (tile_count_width - 1) * tile_stride + tile_size

    background_tuple = (0, 0, 0) if background == "black" else (255, 255, 255)
    padded_image = Image.new(
        "L" if is_grayscale else "RGB",
        (padded_width, padded_height),
        background_tuple[0] if is_grayscale else background_tuple,
    )
    padded_image.paste(img, (0, 0))

    image_array = np.asarray(padded_image)
    if is_grayscale:
        image_array = np.expand_dims(image_array, axis=-1)
    image_array = (
        image_array / 255.0 if not contrast_stretch else (image_array - np.min(image_array)) / np.ptp(image_array)
    )
    image_array = image_array.astype(np.float32 if use_fp32 else np.float16)
    image_array = np.transpose(image_array, (2, 0, 1))

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    output_tile_size = tile_size * scale_factor
    output_stride = tile_stride * scale_factor
    output_overlap = tile_overlap * 2 * scale_factor
    output_accumulator = np.zeros(
        (padded_height * scale_factor, padded_width * scale_factor, input_channel_count),
        dtype=np.float32,
    )
    output_weights = np.zeros((*output_accumulator.shape[:2], 1), dtype=np.float32)

    blend_position = np.arange(output_overlap, dtype=np.float32) / (output_overlap - 1)
    blend_position = np.clip(blend_position * 2.0 - 0.5, 0.0, 1.0)
    blend_ramp = (np.sin(blend_position * np.pi - np.pi / 2.0) + 1.0) / 2.0

    tile_locations = [(tile_y, tile_x) for tile_y in range(tile_count_height) for tile_x in range(tile_count_width)]
    for batch_start in range(0, len(tile_locations), batch_size):
        batch_locations = tile_locations[batch_start : batch_start + batch_size]
        batch_of_tiles = np.stack([
            image_array[
                :,
                tile_y * tile_stride : tile_y * tile_stride + tile_size,
                tile_x * tile_stride : tile_x * tile_stride + tile_size,
            ]
            for tile_y, tile_x in batch_locations
        ])
        model_output = np.asarray(model.run([output_name], {input_name: batch_of_tiles})[0])
        expected_shape = (len(batch_locations), input_channel_count, output_tile_size, output_tile_size)
        if model_output.shape != expected_shape:
            raise ValueError(f"Unexpected model output shape {model_output.shape}; expected {expected_shape}.")

        for tile_output, (tile_y, tile_x) in zip(model_output, batch_locations, strict=True):
            tile_output = np.transpose(tile_output, (1, 2, 0)).astype(np.float32, copy=False)
            tile_weights = np.ones((output_tile_size, output_tile_size), dtype=np.float32)
            if tile_x > 0:
                tile_weights[:, :output_overlap] *= blend_ramp[None, :]
            if tile_x + 1 < tile_count_width:
                tile_weights[:, -output_overlap:] *= 1.0 - blend_ramp[None, :]
            if tile_y > 0:
                tile_weights[:output_overlap, :] *= blend_ramp[:, None]
            if tile_y + 1 < tile_count_height:
                tile_weights[-output_overlap:, :] *= 1.0 - blend_ramp[:, None]

            output_y = tile_y * output_stride
            output_x = tile_x * output_stride
            output_slice = np.s_[
                output_y : output_y + output_tile_size,
                output_x : output_x + output_tile_size,
            ]
            output_accumulator[output_slice] += tile_output * tile_weights[..., None]
            output_weights[output_slice] += tile_weights[..., None]

    reconstructed_image = output_accumulator / np.maximum(output_weights, np.finfo(np.float32).eps)
    if contrast_stretch:
        output_array = reconstructed_image.astype(np.float32 if use_fp32 else np.float16)
        output_array = (output_array - np.min(output_array)) / np.ptp(output_array)
    else:
        output_array = np.clip(reconstructed_image, 0.0, 1.0)
    output_array = np.round(output_array * 255.0).astype(np.uint8)
    if is_grayscale:
        output_array = np.squeeze(output_array, axis=-1)

    output_image = Image.fromarray(output_array)
    output_image = output_image.crop((0, 0, image_width * scale_factor, image_height * scale_factor))
    if is_grayscale:
        return output_image.convert(mode="L")
    if orig_img_palette:
        palette_image = Image.new("P", (1, 1))
        palette_image.putpalette(orig_img_palette)
        return output_image.convert(mode="RGB").quantize(palette=palette_image, dither=Image.Dither.NONE)
    return output_image.convert(mode=orig_img_mode)

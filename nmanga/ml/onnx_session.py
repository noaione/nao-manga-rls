"""
MIT License

Copyright (c) 2022-present noaione

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

# ONNX session builders and provider configuration.

from __future__ import annotations

import importlib
import json
import logging
import sys
import time
import warnings
from hashlib import md5
from importlib import util as importutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ..common import format_elapsed_time
from ..term import get_console
from .contracts import MLDataType
from .onnx_cache import build_compilation_cache_key, get_data_dir
from .ort import get_onnxruntime

if TYPE_CHECKING:
    import onnxruntime as ort

    class InferenceSessionWithScale(ort.InferenceSession):
        scale_factor: int | None
        use_halfp: bool
        clamped_batch_size_warned: bool


__all__ = (
    "get_model_fixed_batch_size",
    "get_model_information",
    "get_model_scale_factor",
    "get_model_static_input",
    "get_torch_memory_limit_and_rtx",
    "prepare_model_runtime",
    "prepare_model_runtime_builders",
    "resolve_model_input_sizes",
    "resolve_model_static_shapes",
)

logger = logging.getLogger(__name__)


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
    disable_cpu: bool = False,
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
        "migraphx_hip_graph_enable": 1,
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
    coreml_ep_config = {
        "ModelFormat": "MLProgram",
        "MLComputeUnits": "CPUAndGPU",
        "RequireStaticInputShapes": "1",
        "EnableOnSubgraphs": "1",
        "ModelCacheDirectory": str(cache_dir),
        "SpecializationStrategy": "FastPrediction",
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
    config_entry: dict[str, str] = {
        "session.disable_cpu_ep_fallback": "1" if disable_cpu else "0",
    }
    has_good_ep_devices = False
    has_nvrtx = False
    raw_providers = set(ort.get_available_providers())

    # Selected providers
    providers = []
    ep_providers = []

    if sys.platform != "darwin":
        has_nvrtx = False
        if "NvTensorRTRTXExecutionProvider" in ep_devices and with_nvrtx:
            ep_providers.append((ep_devices["NvTensorRTRTXExecutionProvider"], trtrtx_ep_config))
            # disable CPU fallback
            config_entry["session.disable_cpu_ep_fallback"] = "1"
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
        # Apple Silicon only; unsupported MLX nodes fall through to CoreML and then the CPU
        if "MLXExecutionProvider" in raw_providers:
            providers.append(("MLXExecutionProvider", {}))
            config_entry["ep.mlx.enable_compile"] = "1"
        providers.append(("CoreMLExecutionProvider", coreml_ep_config))

    if not providers and not ep_providers:
        if "WebGpuExecutionProvider" not in ep_devices:
            raise RuntimeError(
                "No suitable ONNX Runtime execution providers found. "
                "Ensure you have compatible hardware and the necessary dependencies installed."
            )

        # Has webgpu, add it
        ep_providers.append((ep_devices["WebGpuExecutionProvider"], {}))
        has_good_ep_devices = True

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

    for key_c, val_c in config_entry.items():
        sess_opt.add_session_config_entry(key_c, val_c)

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

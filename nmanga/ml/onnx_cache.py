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

# ONNX graph fingerprints and the compilation cache key.
#
# The cache identity and the cache-directory layout are part of the on-disk contract: an engine
# compiled before this module existed has to be found again afterwards.

from __future__ import annotations

import os
import sys
from hashlib import sha256
from pathlib import Path

from .contracts import MLDataType

__all__ = (
    "build_compilation_cache_key",
    "get_data_dir",
    "get_model_fingerprint",
    "get_tf32_tag",
)


def get_data_dir() -> Path:
    # Should use APPDATA on Windows, XDG_CACHE_HOME on Linux, etc. but for simplicity just use home/.cache
    if sys.platform == "win32":
        cache_dir = Path.home() / "AppData" / "Local" / "nmanga"
    else:
        cache_dir = Path.home() / ".cache" / "nmanga"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


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

    # Imported here because `nmanga.ml.onnx_session` imports this module: the runtime version tag
    # belongs to the session builders, but the key needs it.
    from .onnx_session import _get_trt_runtime_version_tag

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

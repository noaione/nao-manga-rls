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

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator, model_validator
from pydantic_core import PydanticCustomError
from typing_extensions import Self

from ... import file_handler
from ...common import AsyncImageWriter, RegexCollection, default_cpu_threads
from ...ml.backends import InferenceBackend
from ...ml.contracts import MLDataType, TilingConfig, TorchPreset, TorchRuntimeConfig
from ...ml.model import CHECKPOINT_SUFFIXES
from ...ml.onnx_backend import denoise_single_image_with_overlap
from ...ml.onnx_cache import get_data_dir, get_model_fingerprint, get_tf32_tag
from ...ml.onnx_session import prepare_model_runtime_builders
from ...ml.tiler import TilingPolicy, default_expand_tolerance
from ...ml.umzi import (
    UmziPrecision,
    UmziSession,
    UmziSessionPool,
    build_kernel_options,
    create_umzi_session,
    read_ummodel,
    resolve_precision,
    umzi_tiling_constraints,
)
from ...resizer import ResizeKernel, rescale_image
from ..common import perform_skip_action
from ._base import ActionKind, BaseAction, ToolsKind, WorkerContext
from .rescale import RescaleTarget

if TYPE_CHECKING:
    from ...ml.onnx_session import InferenceSessionWithScale
    from ...ml.torch_session import TorchRuntime
    from ..models import OrchestratorConfig, VolumeConfig

_STATIC_ORT_BACKENDS = frozenset({InferenceBackend.TRT, InferenceBackend.TRT_RTX, InferenceBackend.COREML})
_STATIC_ORT_EPS = frozenset({"TensorrtExecutionProvider", "NvTensorRTRTXExecutionProvider", "CoreMLExecutionProvider"})
_UMZI_CONSTRAINTS_CACHE: dict[str, tuple[int, int]] = {}

__all__ = (
    "ActionDenoise",
    "ActionUpscale",
)


@dataclass
class MLActionName:
    load: str
    run: str
    finish: str


class ActionML(BaseAction):
    """
    Base class for ML-based actions, mainly denoising and upscaling.
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - ML-based Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    model: FilePath = Field(..., title="The ONNX, Umzi, or PyTorch Model Path")
    """The path to the `.onnx`, `.ummodel`, `.safetensors`, or `.pth` model"""
    device_id: int | str = Field(0, title="Device ID")
    """The device to use (`0`, `1`, ..., or for PyTorch/auto selection: `auto`, `cpu`, `cuda`, `cuda:N`)"""
    batch_size: int = Field(64, ge=1, title="Batch Size")
    """The batch size to use for denoising"""
    tile_size: int = Field(128, ge=64, title="Tile Size")
    """The tile size to use for denoising"""
    tile_overlap: int = Field(0, ge=0, title="Tile Overlap")
    """The tile overlap to use for denoising"""
    variable_tiling: bool = Field(False, title="Variable Edge Tiling")
    """Shrink or expand edge tiles instead of padding every tile to `tile_size`"""
    tile_expand: int | None = Field(None, ge=0, title="Variable Tile Expand Tolerance")
    """Max extra pixels to absorb a small edge sliver into the previous tile under variable tiling"""
    tile_bucket: int = Field(32, ge=0, title="Variable Tile Bucket Step")
    """Snap variable edge dimensions up to this multiple across pages (`0` disables)"""
    background: Literal["white", "black"] = Field("black", title="Padding Background Color")
    """The background color to use for padding"""
    contrast_strectch: bool = Field(False, title="Contrast Stretch After Denoising")
    """Whether to apply contrast stretch after denoising"""
    tta: bool = Field(False, title="Test-Time Augmentation")
    """Average each batch over its 8 flips/rotations, about 8x slower"""
    backend: InferenceBackend = Field(InferenceBackend.AUTO, title="Inference Backend")
    """The ONNX Runtime backend, `auto` picks the best one this machine offers"""
    data_type: MLDataType = Field(MLDataType.FP16, title="The data type to use for TensorRT model")
    """The data type to build the TensorRT engine with"""
    precision: UmziPrecision | None = Field(None, title="Umzi Precision")
    """Precision override for `.ummodel` models (`mixed`, `fp16`, or `fp32`); defaults from `data_type`"""
    kernel_options: dict[str, int] = Field(default_factory=dict, title="Umzi Kernel Options")
    """Per-family kernel option overrides for `.ummodel` models"""
    use_plan_cache: bool = Field(True, title="Use Umzi Plan Cache")
    """Persist tuned GPU plans under nmanga's data directory for `.ummodel` models"""
    preset: TorchPreset = Field("fast", title="PyTorch Recipe")
    """Inference recipe for PyTorch checkpoint models (`fast` or `original`)"""
    enable_compile: bool = Field(False, title="Enable torch.compile")
    """Wrap a PyTorch checkpoint model with `torch.compile` (CUDA `fast` recipe only)"""
    use_channels_last: bool | None = Field(None, title="PyTorch Channels Last Override")
    """Override channels-last memory format for PyTorch checkpoint models (`None` follows adapter)"""
    workspace_size: int | None = Field(None, ge=1, title="TensorRT Workspace Size")
    """The TensorRT workspace limit in MiB, defaults to 75% of the total GPU memory"""
    enable_cuda_graph: bool = Field(True, title="Enable CUDA Graph Replay")
    """Whether to enable CUDA graph replay for the TRT-RTX or Umzi engine"""
    strict_ep: bool = Field(False, title="Fail if the execution provider is unavailable")
    """Whether to fail instead of falling back when the requested execution provider is unavailable"""
    threads: int = Field(default_factory=default_cpu_threads, ge=1, title="Image Write Threads")
    """Number of worker threads used to encode and write output PNG pages"""
    base_path: str = Field("output", title="Output Base Path")
    """The base path to save the resulting images to"""

    @property
    def is_umzi_model(self) -> bool:
        """Whether `model` points to a native `.ummodel` package."""
        return self.model.suffix.lower() == ".ummodel"

    @property
    def is_torch_model(self) -> bool:
        """Whether `model` points to a PyTorch checkpoint."""
        return self.model.suffix.lower() in CHECKPOINT_SUFFIXES

    @field_validator("device_id")
    @classmethod
    def validate_device_id(cls, value: int | str) -> int | str:
        if isinstance(value, bool):
            raise PydanticCustomError("invalid_device_id", "device_id cannot be a boolean.")
        if isinstance(value, int):
            if value < 0:
                raise PydanticCustomError("invalid_device_id", "device_id must be non-negative.")
            return value
        normalized = value.strip().lower()
        if normalized in {"auto", "cpu"}:
            return normalized
        if normalized == "cuda":
            return "cuda:0"
        if normalized.startswith("cuda:") and normalized.partition(":")[2].isdigit():
            return f"cuda:{int(normalized.partition(':')[2])}"
        raise PydanticCustomError(
            "invalid_device_id",
            "Invalid device_id; expected a non-negative integer, 'auto', 'cpu', or 'cuda:N'.",
        )

    def _resolve_cuda_device_id(self) -> int:
        if isinstance(self.device_id, int):
            return self.device_id
        if self.device_id == "auto":
            return 0
        if self.device_id.startswith("cuda:"):
            return int(self.device_id.partition(":")[2])
        raise ValueError("`device_id: cpu` is only supported for PyTorch checkpoint models.")

    def _resolve_torch_device(self) -> str:
        if isinstance(self.device_id, int):
            return f"cuda:{self.device_id}"
        return self.device_id

    @model_validator(mode="after")
    def validate_tile_overlap(self) -> Self:
        if self.tile_overlap * 2 >= self.tile_size:
            raise PydanticCustomError(
                "tile_overlap_too_large",
                "tile_overlap must be less than half the tile_size.",
            )
        if self.is_umzi_model:
            if self.device_id == "cpu":
                raise PydanticCustomError(
                    "invalid_umzi_device",
                    "Umzi `.ummodel` models run on CUDA and do not support `device_id: cpu`.",
                )
            if self.precision is None and self.data_type == MLDataType.BF16:
                raise PydanticCustomError(
                    "invalid_umzi_precision",
                    "Umzi `.ummodel` models do not support bf16; set `precision` to mixed, fp16, or fp32.",
                )
        elif self.is_torch_model:
            if self.enable_compile and self.preset != "fast":
                raise PydanticCustomError(
                    "torch_compile_requires_fast",
                    "enable_compile is only supported for the PyTorch `fast` preset.",
                )
            if self.variable_tiling:
                raise PydanticCustomError(
                    "variable_tiling_unsupported_torch",
                    "variable_tiling is not supported for PyTorch checkpoint models.",
                )
            if self.precision == "mixed":
                raise PydanticCustomError(
                    "invalid_torch_precision",
                    "`precision: mixed` is only supported for Umzi `.ummodel` models.",
                )
        else:
            if self.device_id == "cpu":
                raise PydanticCustomError(
                    "invalid_onnx_device",
                    "ONNX models select the execution provider via `backend`, not `device_id: cpu`.",
                )
            if self.variable_tiling and self.backend in _STATIC_ORT_BACKENDS:
                raise PydanticCustomError(
                    "variable_tiling_static_backend",
                    "variable_tiling cannot be used with the static-shape {backend} backend.",
                    {"backend": self.backend.label},
                )
            if self.precision == "mixed":
                raise PydanticCustomError(
                    "invalid_onnx_precision",
                    "`precision: mixed` is only supported for Umzi `.ummodel` models.",
                )
        return self

    def _resolve_umzi_precision(self) -> UmziPrecision:
        if self.precision is not None:
            return self.precision
        if self.data_type == MLDataType.FP32:
            return "fp32"
        if self.data_type == MLDataType.FP16:
            return "fp16"
        raise ValueError("Umzi `.ummodel` models only support mixed, fp16, or fp32 precision.")

    @property
    def action_name(self) -> MLActionName:
        return MLActionName(
            load="generic",
            run="Processing",
            finish="Processed",
        )

    def post_process(
        self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig", image: Image.Image
    ) -> Image.Image:
        """
        Post-process the image after denoising/upscaling

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        :param image: The image to post-process
        :return: The post-processed image
        """

        return image

    def make_cache_key(self) -> str:
        """
        Make a cache key for the model session

        The key covers the model contents (not just its path) and the compilation settings, so
        swapping the weights behind an unchanged pathname, or changing the precision or TF32
        policy, builds a fresh session instead of reusing one for other settings.

        :return: The cache key
        """

        fqdn_model = Path(self.model).resolve(True).as_posix()
        model_hash = sha256(fqdn_model.encode("utf-8")).hexdigest()
        graph_hash = get_model_fingerprint(self.model)[:16]

        if self.is_umzi_model:
            cuda_device = self._resolve_cuda_device_id()
            opts = ",".join(f"{k}={v}" for k, v in sorted(self.kernel_options.items()))
            other_key = (
                f"umzi-d{cuda_device}-p{self._resolve_umzi_precision()}-g{graph_hash}"
                f"-cg{int(self.enable_cuda_graph)}-pc{int(self.use_plan_cache)}"
                f"-vt{int(self.variable_tiling)}-ko[{opts}]"
            )
            return f"{model_hash}_{other_key}"

        if self.is_torch_model:
            dev = self._resolve_torch_device()
            other_key = (
                f"torch-dev{dev}-pr{self.preset}-cmp{int(self.enable_compile)}"
                f"-cl{self.use_channels_last}-t{self.tile_size}-to{self.tile_overlap}-g{graph_hash}"
            )
            return f"{model_hash}_{other_key}"

        cuda_device = self._resolve_cuda_device_id()
        other_key = (
            f"d{cuda_device}-b{self.batch_size}-t{self.tile_size}-be{self.backend.value}"
            f"-dt{self.data_type.value}-g{graph_hash}-{get_tf32_tag()}"
            f"-ws{self.workspace_size or 'auto'}-cg{int(self.enable_cuda_graph)}"
        )
        if self.strict_ep:
            other_key += "-strict"

        return f"{model_hash}_{other_key}"

    def _torch_runtime_config(self) -> TorchRuntimeConfig:
        return TorchRuntimeConfig(
            device=self._resolve_torch_device(),
            preset=self.preset,
            use_channels_last=self.use_channels_last,
            enable_compile=self.enable_compile,
            tiling=TilingConfig(tile_size=self.tile_size, tile_overlap=self.tile_overlap),
        )

    def build_or_load_model(
        self, context: WorkerContext
    ) -> "InferenceSessionWithScale | TorchRuntime | UmziSession | UmziSessionPool":
        """
        Build or load the ML model for the action

        :param context: The worker context
        :return: The ONNX, PyTorch, or Umzi inference session
        """

        cache_key = self.make_cache_key()
        cached_session = context.ml_model_session.get(cache_key)
        if cached_session is not None:
            context.terminal.log(f"Using cached model session for key {cache_key}.")
            return cached_session

        if self.is_torch_model:
            from ...ml.torch_session import create_runtime

            runtime_config = self._torch_runtime_config()
            context.terminal.info(
                f"Loading PyTorch checkpoint {self.model.name} "
                f"(device={runtime_config.device}, preset={runtime_config.preset}, "
                f"compile={runtime_config.enable_compile})..."
            )
            torch_runtime = create_runtime(self.model, runtime_config)
            context.ml_model_session[cache_key] = torch_runtime
            context.terminal.log(f"Cached PyTorch runtime for key {cache_key}.")
            return torch_runtime

        if self.is_umzi_model:
            package = read_ummodel(self.model)
            if not package.supported:
                problems = "\n".join(f"- {problem}" for problem in package.problems)
                raise ValueError(f"{self.model.name} cannot run on this umzi-native build:\n{problems}")

            selected_precision = resolve_precision(package.family, self._resolve_umzi_precision())
            resolved_cache: Path | None = None
            if self.use_plan_cache:
                resolved_cache = get_data_dir() / "umzi"
                resolved_cache.mkdir(parents=True, exist_ok=True)

            used_options = build_kernel_options(package.family, self.kernel_options, precision=selected_precision)
            rendered_options = ", ".join(f"{name}={value}" for name, value in sorted(used_options.items()))
            context.terminal.info(
                f"Opening Umzi {package.family} session ({selected_precision}, "
                f"kernel options: {rendered_options or 'family defaults'})..."
            )
            cuda_device = self._resolve_cuda_device_id()
            session: UmziSession | UmziSessionPool
            if self.variable_tiling:
                session = UmziSessionPool(
                    self.model,
                    family=package.family,
                    device_id=cuda_device,
                    precision=selected_precision,
                    cache_dir=resolved_cache,
                    enable_cuda_graph=self.enable_cuda_graph,
                    kernel_options=self.kernel_options,
                )
            else:
                session = create_umzi_session(
                    self.model,
                    family=package.family,
                    device_id=cuda_device,
                    precision=selected_precision,
                    cache_dir=resolved_cache,
                    enable_cuda_graph=self.enable_cuda_graph,
                    kernel_options=self.kernel_options,
                )
            _UMZI_CONSTRAINTS_CACHE[cache_key] = umzi_tiling_constraints(package)
            context.ml_model_session[cache_key] = session
            context.terminal.log(f"Cached Umzi model session for key {cache_key}.")
            return session

        context.terminal.info(
            f"Building/loading the engine for {self.model.name}. "
            + "If it is building, you may need to wait up to 20-25 minutes for the first time."
        )
        ort_session = prepare_model_runtime_builders(
            self.model,
            device_id=self._resolve_cuda_device_id(),
            is_verbose=context.terminal.debugged,
            tile_size=self.tile_size,
            batch_size=self.batch_size,
            data_type=self.data_type,
            backend=self.backend,
            workspace_size=None if self.workspace_size is None else self.workspace_size * 1024 * 1024,
            enable_cuda_graph=self.enable_cuda_graph,
            strict_ep=self.strict_ep,
        )

        context.ml_model_session[cache_key] = ort_session
        context.terminal.log(f"Cached model session for key {cache_key}.")

        return ort_session

    def _build_tiling_policy(
        self, session: "InferenceSessionWithScale | UmziSession | UmziSessionPool"
    ) -> TilingPolicy:
        if not self.variable_tiling:
            return TilingPolicy(mode="fixed")

        expand = default_expand_tolerance(self.tile_size) if self.tile_expand is None else self.tile_expand
        if self.is_umzi_model:
            alignment, min_tile = _UMZI_CONSTRAINTS_CACHE.get(self.make_cache_key(), (1, 1))
            return TilingPolicy(
                mode="variable",
                expand_tolerance=expand,
                bucket_step=self.tile_bucket,
                alignment=alignment,
                min_tile_size=min_tile,
            )

        get_providers = getattr(session, "get_providers", None)
        if callable(get_providers) and set(get_providers()).intersection(_STATIC_ORT_EPS):
            raise ValueError("variable_tiling cannot be used when TensorRT, TRT-RTX, or CoreML is active.")
        input_shape = getattr(session.get_inputs()[0], "shape", None)
        if input_shape and len(input_shape) >= 4 and any(isinstance(dim, int) and dim > 0 for dim in input_shape[-2:]):
            raise ValueError(
                "variable_tiling requires dynamic spatial input dimensions, but this model has a fixed input shape."
            )
        return TilingPolicy(mode="variable", expand_tolerance=expand, bucket_step=self.tile_bucket)

    def run(self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig") -> None:
        """
        Run the action on a volume

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        """

        # Prepare
        output_dir = context.root_dir / Path(self.base_path) / Path(volume.path)

        if context.dry_run:
            kind_label = "Umzi" if self.is_umzi_model else ("PyTorch" if self.is_torch_model else "ONNX")
            context.terminal.info(f"- Model Path: {self.model} ({kind_label})")
            context.terminal.info(f"- Output Base Path: {self.base_path}")
            if self.is_torch_model:
                context.terminal.info(f"- Device: {self._resolve_torch_device()}")
                context.terminal.info(f"- Preset: {self.preset}")
                context.terminal.info(f"- Compile: {'Yes' if self.enable_compile else 'No'}")
                if self.use_channels_last is not None:
                    context.terminal.info(f"- Channels Last: {'Yes' if self.use_channels_last else 'No'}")
            else:
                context.terminal.info(f"- Device ID: {self._resolve_cuda_device_id()}")
                context.terminal.info(f"- Batch Size: {self.batch_size}")
            context.terminal.info(f"- Tile Size: {self.tile_size}")
            context.terminal.info(f"- Tile Overlap: {self.tile_overlap}")
            if not self.is_torch_model:
                context.terminal.info(f"- Variable Tiling: {'Yes' if self.variable_tiling else 'No'}")
                if self.variable_tiling:
                    expand = default_expand_tolerance(self.tile_size) if self.tile_expand is None else self.tile_expand
                    context.terminal.info(f"- Variable Tile Expand / Bucket: {expand}px / {self.tile_bucket}px")
                context.terminal.info(f"- Background Color: {self.background}")
                context.terminal.info(f"- Contrast Stretch: {self.contrast_strectch}")
                context.terminal.info(f"- Test-Time Augmentation: {self.tta}")
            if self.is_umzi_model:
                context.terminal.info(f"- Precision: {self._resolve_umzi_precision()}")
                context.terminal.info(f"- Plan Cache: {'Yes' if self.use_plan_cache else 'No'}")
                if self.kernel_options:
                    opts = ", ".join(f"{k}={v}" for k, v in sorted(self.kernel_options.items()))
                    context.terminal.info(f"- Kernel Options: {opts}")
                context.terminal.info(f"- CUDA Graph: {'Yes' if self.enable_cuda_graph else 'No'}")
            elif not self.is_torch_model:
                context.terminal.info(f"- Backend: {self.backend.value}")
                context.terminal.info(f"- Data Type: {self.data_type.value}")
                context.terminal.info(f"- Workspace Size: {self.workspace_size or 'auto'} MiB")
                context.terminal.info(f"- CUDA Graph: {'Yes' if self.enable_cuda_graph else 'No'}")
            context.terminal.info(f"- Write Threads: {self.threads}")
            context.update_cwd(output_dir)
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        if not context.current_dir.exists():
            context.terminal.warning(
                f"Current directory {context.current_dir} does not exist, skipping {self.action_name.run.lower()}."
            )
            context.update_cwd(output_dir)  # We still need to update CWD
            return

        context.terminal.info(f"Loading {self.action_name.load} model from {self.model.name}...")

        page_re = RegexCollection.page_re()
        session = self.build_or_load_model(context)
        if self.is_torch_model:
            from ...ml.images import image_to_tensor, tensor_to_image
            from ...ml.torch_backend import infer_tensor

            torch_runtime = cast("TorchRuntime", session)
            torch_tiling = TilingConfig(tile_size=self.tile_size, tile_overlap=self.tile_overlap)

            def infer_image(img_file: Image.Image) -> Image.Image:
                input_tensor = image_to_tensor(img_file)
                prediction = infer_tensor(
                    input_tensor,
                    torch_runtime.predict,
                    scale=torch_runtime.info.scale,
                    config=torch_tiling,
                )
                return tensor_to_image(prediction, img_file)

        else:
            onnx_session = cast("InferenceSessionWithScale | UmziSession | UmziSessionPool", session)
            tiling_policy = self._build_tiling_policy(onnx_session)
            use_fp32 = (
                self._resolve_umzi_precision() == "fp32"
                if self.is_umzi_model
                else not getattr(onnx_session, "use_halfp", False)
            )

            def infer_image(img_file: Image.Image) -> Image.Image:
                return denoise_single_image_with_overlap(
                    img_file,
                    onnx_session,  # type: ignore[arg-type]  # UmziSession and UmziSessionPool satisfy the ONNX tiler surface
                    batch_size=self.batch_size,
                    tile_size=self.tile_size,
                    contrast_stretch=self.contrast_strectch,
                    tta=self.tta,
                    background=self.background,
                    tile_overlap=self.tile_overlap,
                    use_fp32=use_fp32,
                    pad_final_batch=self.is_umzi_model,
                    tiling_policy=tiling_policy,
                )

        current_index = 1
        total_image = 0
        context.terminal.info(f"{self.action_name.run} images in {context.current_dir}...")
        progress = context.terminal.make_progress()
        task = context.terminal.make_task(
            progress,
            f"{self.action_name.run} images...",
            finished_text=f"{self.action_name.finish} images",
            total=None,
        )
        with AsyncImageWriter(workers=self.threads) as writer:
            for file_path, _, total_image, _ in file_handler.collect_image_from_folder(context.current_dir):
                progress.update(task, total=total_image, completed=current_index - 1)

                if title_match := page_re.match(file_path.stem):
                    first_part = int(title_match.group("a"))  # We only care about this
                    if context.skip_action is not None and first_part in context.skip_action.pages:
                        perform_skip_action(file_path, output_dir, context.skip_action.action, context.terminal)
                        current_index += 1
                        continue

                with Image.open(file_path) as img_file:
                    img_file.load()
                    output_image = infer_image(img_file)
                    output_image = self.post_process(context, volume, orchestrator, output_image)

                output_path = output_dir / f"{file_path.stem}.png"
                writer.submit(output_image, output_path)
                current_index += 1
        progress.update(task, advance=1)
        context.terminal.stop_progress(progress, f"{self.action_name.finish} {total_image} images.")

        # Update CWD
        context.update_cwd(output_dir)

    def get_tools(self, orchestrator: "OrchestratorConfig"):
        """
        Get the required tools for the action

        :param orchestrator: The orchestrator configuration
        :return: A dictionary of tool names and their kinds
        """

        base = {"numpy": ToolsKind.PACKAGE}

        if self.is_umzi_model:
            return {
                **base,
                "umzi_native": ToolsKind.PACKAGE,
            }
        if self.is_torch_model:
            return {
                **base,
                "torch": ToolsKind.PACKAGE,
                "resselt": ToolsKind.PACKAGE,
            }
        return {
            **base,
            "onnxruntime": ToolsKind.PACKAGE,
        }


class ActionDenoise(ActionML):
    """
    Action to denoise all images in a volume with `nmanga ort infer`
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Denoise Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.DENOISE] = Field(ActionKind.DENOISE, title="Denoise Images Action")
    """The kind of action"""
    base_path: str = Field("denoised", title="Output Base Path")
    """The base path to save the denoised images to"""

    @property
    def action_name(self) -> MLActionName:
        return MLActionName(
            load="denoise",
            run="Denoising",
            finish="Denoised",
        )


class UpscaleRescaleConfig(BaseModel):
    """
    Configuration for rescaling after upscaling.
    """

    target: RescaleTarget = Field(..., title="Rescale Target")
    """The rescale target configuration"""
    kernel: ResizeKernel = Field(..., title="Rescale Kernel")
    """The rescale kernel to use"""


class ActionUpscale(ActionML):
    """
    Action to upscale all images in a volume with `nmanga ort infer`
    """

    model_config = ConfigDict(
        title="nmanga Orchestrator - Upscale Images Action",
        strict=True,
        extra="forbid",
        validate_default=True,
    )

    kind: Literal[ActionKind.UPSCALE] = Field(ActionKind.UPSCALE, title="Upscale Images Action")
    """The kind of action"""
    base_path: str = Field("upscaled", title="Output Base Path")
    """The base path to save the upscaled images to"""

    rescale: UpscaleRescaleConfig | None = Field(None, title="Rescale Configuration")
    """The rescale configuration after upscaling"""

    @property
    def action_name(self) -> MLActionName:
        return MLActionName(
            load="upscale",
            run="Upscaling",
            finish="Upscaled",
        )

    def post_process(
        self, context: WorkerContext, volume: "VolumeConfig", orchestrator: "OrchestratorConfig", image: Image.Image
    ) -> Image.Image:
        """
        Post-process the image after denoising/upscaling

        :param context: The worker context
        :param volume: The volume configuration
        :param orchestrator: The orchestrator configuration
        :param image: The image to post-process
        :return: The post-processed image
        """

        if self.rescale is None:
            return image

        rescaler = self.rescale.target
        if metafield := volume.metafields:
            if metafield.rescale is not None:
                rescaler = metafield.rescale

        target = rescaler.to_params()
        output_img = rescale_image(
            image,
            target=target,
            kernel=self.rescale.kernel,
        )
        image.close()
        return output_img

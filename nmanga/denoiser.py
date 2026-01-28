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

import importlib.util
import logging
import sys
import warnings
from enum import Enum
from hashlib import md5
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from PIL import Image

if TYPE_CHECKING:
    from onnxruntime import InferenceSession  # type: ignore

    class InferenceSessionWithScale(InferenceSession):
        scale_factor: int | None


__all__ = (
    "MLDataType",
    "denoise_single_image",
    "prepare_model_runtime",
    "prepare_model_runtime_builders",
)

logger = logging.getLogger(__name__)


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


def get_data_dir() -> Path:
    # Should use APPDATA on Windows, XDG_CACHE_HOME on Linux, etc. but for simplicity just use home/.cache
    if sys.platform == "win32":
        cache_dir = Path.home() / "AppData" / "Local" / "nmanga"
    else:
        cache_dir = Path.home() / ".cache" / "nmanga"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_model_scale_factor(session: "InferenceSessionWithScale", *, tile_size: int | None = None) -> int:
    import numpy as np

    if scale_factor := getattr(session, "scale_factor", None):
        return scale_factor

    # Detect scale using small input and output shapes
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]

    maybe_tile_size = inp.shape[-1]
    maybe_output_size = out.shape[-1]

    if isinstance(maybe_tile_size, int) and isinstance(maybe_output_size, int):
        return maybe_output_size // maybe_tile_size  # Fast path if both sizes are known

    real_tile_size = tile_size if tile_size is not None else maybe_tile_size
    if not isinstance(real_tile_size, int):
        real_tile_size = 64  # Fallback tile size

    # Most model would have shape like (batch_size, channels, height, width)
    batch_size = inp.shape[0] if isinstance(inp.shape[0], int) else 1
    shapes = [batch_size, inp.shape[1], real_tile_size, real_tile_size]

    x = np.zeros(shapes, dtype=np.float32)
    input_name = inp.name
    output_name = out.name

    out = session.run([output_name], {input_name: x})
    scale_height = np.array(out[0]).shape[-1]
    return scale_height // real_tile_size


def prepare_model_runtime(
    model_path: Path,
    *,
    device_id: int = 0,
    is_verbose: bool = False,
) -> "InferenceSessionWithScale":
    import onnxruntime as ort  # type: ignore

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
        providers = [
            (
                # Force use of TensorRT if available
                "TensorrtExecutionProvider",
                {
                    "device_id": device_id,
                    "trt_fp16_enable": True,
                    "trt_sparsity_enable": True,
                },
            ),
        ]

    verb_level = 0 if is_verbose else 3
    ort.set_default_logger_severity(verb_level)
    ort.set_default_logger_verbosity(verb_level)

    sess_opt = ort.SessionOptions()
    session = cast(
        "InferenceSessionWithScale", ort.InferenceSession(model_path, sess_options=sess_opt, providers=providers)
    )
    scale_factor = get_model_scale_factor(session, tile_size=None)
    session.scale_factor = scale_factor
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

    model = onnx.load(str(model_path))
    # input channel count
    if len(model.graph.input) != 1:
        raise ValueError("Model has multiple inputs, cannot determine input name.")
    model_channel_count = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    return model.graph.input[0].name, model_channel_count


def prepare_model_runtime_builders(
    model_path: Path,
    *,
    device_id: int = 0,
    is_verbose: bool = False,
    tile_size: int | None = None,
    batch_size: int = 64,
    data_type: MLDataType = MLDataType.FP16,
) -> "InferenceSessionWithScale":
    import onnxruntime as ort  # type: ignore

    is_torch_available = importlib.util.find_spec("torch")
    if is_torch_available is not None:
        import torch  # type: ignore
    is_tensorrt_available = importlib.util.find_spec("tensorrt")
    if is_tensorrt_available is not None:
        import tensorrt  # type: ignore  # noqa: F401

    data_dir = get_data_dir()
    cache_dir = data_dir / "trt_engines"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_rtx_dir = data_dir / "trtrtx_engines"
    cache_rtx_dir.mkdir(parents=True, exist_ok=True)

    hashed_path = md5(str(model_path.resolve()).encode("utf-8")).hexdigest()  # noqa: S324
    cache_prefix = f"nmodel_t{tile_size}b{batch_size}cd{data_type.name}_{hashed_path}"

    memory_limit = (
        int(
            torch.cuda.get_device_properties(device_id).total_memory * 0.75  # type: ignore
        )
        if is_torch_available
        else 2 * (1024**3)
    )  # 2GB or 75% of GPU memory

    trt_ep_config = {
        "device_id": device_id,
        "trt_fp16_enable": data_type == MLDataType.FP16,
        "trt_bf16_enable": data_type == MLDataType.BF16,
        "trt_sparsity_enable": True,
        "trt_max_workspace_size": memory_limit,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "trt_engines",
        "trt_engine_cache_prefix": cache_prefix,
        "trt_build_heuristics_enable": True,
        "trt_builder_optimization_level": 3,
        "trt_context_memory_sharing_enable": True,
        "trt_dump_ep_context_model": True,
        "trt_ep_context_file_path": str(data_dir),
        "trt_detailed_build_log": True if is_verbose else False,
    }
    trtrtx_ep_config = {
        "device_id": device_id,
        "nv_max_workspace_size": memory_limit,
        "nv_detailed_build_log": True if is_verbose else False,
        "nv_runtime_cache_path": str(cache_rtx_dir),
    }

    model_input, model_channel_count = get_model_information(model_path)
    if tile_size is not None:
        min_shapes = f"{model_input}:1x{model_channel_count}x{tile_size}x{tile_size}"
        max_shapes = f"{model_input}:{batch_size}x{model_channel_count}x{tile_size}x{tile_size}"
        opt_shapes = max_shapes
        trt_ep_config["trt_profile_min_shapes"] = min_shapes
        trt_ep_config["trt_profile_max_shapes"] = max_shapes
        trt_ep_config["trt_profile_opt_shapes"] = opt_shapes

        trtrtx_ep_config["nv_profile_min_shapes"] = model_input
        trtrtx_ep_config["nv_profile_max_shapes"] = min_shapes
        trtrtx_ep_config["nv_profile_opt_shapes"] = max_shapes
    if sys.platform != "darwin":
        providers = [
            ("NvTensorRTRTXExecutionProvider", trtrtx_ep_config),
            ("TensorrtExecutionProvider", trt_ep_config),
            (
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
            ),
        ]
    else:
        providers = [
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
        ]

    verb_level = 0 if is_verbose else 3
    ort.set_default_logger_severity(verb_level)
    ort.set_default_logger_verbosity(verb_level)

    sess_opt = ort.SessionOptions()
    session = cast(
        "InferenceSessionWithScale", ort.InferenceSession(model_path, sess_options=sess_opt, providers=providers)
    )
    scale_factor = get_model_scale_factor(session, tile_size=tile_size)
    session.scale_factor = scale_factor
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
    import numpy as np  # type: ignore
    from einops import rearrange  # type: ignore

    # Calculate upscale scale from model input and output shapes
    scale_factor = getattr(model, "scale_factor", None)
    if not isinstance(scale_factor, int):
        raise ValueError("Model scale factor is not set. Ensure the model was prepared correctly.")
    input_channel_count: int = model.get_inputs()[0].shape[1]
    is_grayscale = bool(input_channel_count == 1)

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

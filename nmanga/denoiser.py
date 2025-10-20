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

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PIL import Image

if TYPE_CHECKING:
    from onnxruntime import InferenceSession  # type: ignore

__all__ = (
    "denoise_single_image",
    "prepare_model_runtime",
)


def prepare_model_runtime(model_path: Path, device_id: int = 0, is_verbose: bool = False) -> "InferenceSession":
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
    return ort.InferenceSession(
        model_path,
        sess_options=sess_opt,
        providers=providers,
    )


def denoise_single_image(
    input_image: Image.Image,
    model: "InferenceSession",
    *,
    batch_size: int = 64,
    tile_size: int = 132,
    contrast_stretch: bool = False,
    background: Literal["white", "black"] = "black",
) -> Image.Image:
    import numpy as np  # type: ignore
    from einops import rearrange  # type: ignore

    orig_img_mode = input_image.mode
    orig_palette = input_image.palette

    img = input_image.convert("RGB")
    source_width, source_height = img.size
    # compute the new canvas size
    tile_count_height = int(np.ceil((source_height / tile_size)))
    tile_count_width = int(np.ceil((source_width / tile_size)))

    padded_height = int(tile_count_height * tile_size)
    padded_width = int(tile_count_width * tile_size)
    background_tuple = (0, 0, 0) if background == "black" else (255, 255, 255)

    new_padded_image = Image.new("RGB", (padded_width, padded_height), background_tuple)
    new_padded_image.paste(img, (0, 0))

    # Pre-process
    image_array = np.array(new_padded_image)
    # contrast stretching: scaling the image based on its own darkest and brightest pixels.
    # For example, a very dark photo (e.g., pixel values from 10 to 50) and a very bright photo (e.g., values
    # from 200 to 250) will both be stretched to the full [0.0, 1.0] range.
    # The model loses all information about the image's absolute brightness. Can be desirable with a white
    # background to reverse CMYK shift in rare (color) images.
    # Not desirable in general so the alternative path should be taken in the vast majority of cases.
    # See also: https://en.wikipedia.org/wiki/Normalization_(image_processing)#Contrast_Stretching_for_Image_Enhancement
    if contrast_stretch:
        image_array = (image_array - np.min(image_array)) / np.ptp(image_array)
    else:
        image_array = image_array / 255.0
    # Casting to FP16 since we use FP16 quantized models wherever possible
    image_array = image_array.astype(np.float16)
    # Rearranging (Width, Height, Channel) -> (Channel, Width, Height) to match the expected input shape
    image_array = rearrange(image_array, "w h c -> c w h")

    padded_tiles = rearrange(
        image_array,
        "c (h th) (w tw) -> (h w) c th tw",
        th=tile_size,
        tw=tile_size,
    )

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    all_output_chunks = []
    num_chunks = padded_tiles.shape[0]

    for i in range(0, num_chunks, batch_size):
        batch_of_chunks = padded_tiles[i : (i + batch_size if i + batch_size < num_chunks else num_chunks)]

        # infer
        model_output = model.run([output_name], {input_name: batch_of_chunks})
        # append the output
        all_output_chunks.append(model_output[0])

    # concat all chunks
    tiled_output_image = np.concatenate(all_output_chunks, axis=0)

    reconstructed_image_with_pad = rearrange(
        tiled_output_image,
        "(h w) c th tw -> (h th) (w tw) c",
        h=tile_count_height,
        w=tile_count_width,
    )

    # post process
    if contrast_stretch:
        postprocessed_array = reconstructed_image_with_pad.astype(np.float16)
        postprocessed_array = (postprocessed_array - np.min(postprocessed_array)) / np.ptp(postprocessed_array)
    else:
        postprocessed_array = np.clip(reconstructed_image_with_pad, 0.0, 1.0)

    # scaling back to [0, 255] from [0.0, 1.0]
    postprocessed_array = postprocessed_array * 255.0

    # rounding and cast back to uint8
    postprocessed_array = np.round(postprocessed_array)
    postprocessed_array = postprocessed_array.astype(np.uint8)
    output_image = Image.fromarray(postprocessed_array)

    output_image = output_image.crop((0, 0, source_width, source_height))
    if orig_palette:
        palette_image = Image.new("P", (1, 1))
        palette_image.putpalette(orig_palette)
        output_image = output_image.convert("RGB").quantize(palette=palette_image, dither=Image.Dither.NONE)
    else:
        output_image = output_image.convert(orig_img_mode)

    return output_image

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

# The ONNX inference entry points.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from PIL import Image

from ..lazy import get_einops_rearrange, get_numpy
from .onnx_session import resolve_model_input_sizes

if TYPE_CHECKING:
    from .onnx_session import InferenceSessionWithScale

__all__ = (
    "denoise_single_image",
    "denoise_single_image_with_overlap",
)


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

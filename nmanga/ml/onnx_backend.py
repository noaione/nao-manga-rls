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

from typing import TYPE_CHECKING, Any, Literal

from PIL import Image

from ..lazy import get_einops_rearrange, get_numpy
from .onnx_session import resolve_model_input_sizes

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from .onnx_session import InferenceSessionWithScale

__all__ = (
    "denoise_single_image",
    "denoise_single_image_with_overlap",
)


def _resolve_model_dtype(model: "InferenceSessionWithScale", *, use_fp32: bool) -> Any:
    """The dtype the model's input declares, so an fp16 and an fp32 file are both fed as they expect."""
    np = get_numpy()
    input_type = getattr(model.get_inputs()[0], "type", "")
    if input_type == "tensor(float16)":
        return np.float16
    if input_type in ("tensor(float)", "tensor(float32)"):
        return np.float32
    # No declared type to read: fall back to the caller's precision hint.
    return np.float32 if use_fp32 else np.float16


def _run_tta(infer: "Callable[[np.ndarray], np.ndarray]", batch: "np.ndarray") -> "np.ndarray":
    """Average `infer` over the batch's 8 flips/rotations, each output turned back first.

    Test-time augmentation, as waifu2x runs it: 8 runs of the whole batch.
    """
    np = get_numpy()
    output = np.zeros(1, np.float32)
    for k in range(8):
        x = np.rot90(batch, k % 4, axes=(2, 3))
        y = infer(np.ascontiguousarray(x[..., ::-1] if k >= 4 else x))
        y = y[..., ::-1] if k >= 4 else y
        output = output + np.rot90(y, -(k % 4), axes=(2, 3))
    return output / 8


def _denoise_image(
    input_image: Image.Image,
    model: "InferenceSessionWithScale",
    *,
    batch_size: int,
    tile_size: int,
    tile_overlap: int,
    contrast_stretch: bool,
    background: Literal["white", "black"],
    use_fp32: bool,
    tta: bool,
) -> Image.Image:
    np = get_numpy()
    rearrange = get_einops_rearrange()

    scale_factor = getattr(model, "scale_factor", None)
    if not isinstance(scale_factor, int):
        raise ValueError("Model scale factor is not set. Ensure the model was prepared correctly.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    if tile_size < 1:
        raise ValueError("Tile size must be at least 1.")
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError("Tile overlap must be non-negative and smaller than the tile size.")

    input_spec = model.get_inputs()[0]
    input_channel_count: int = input_spec.shape[1]
    is_grayscale = input_channel_count == 1
    # Some models declare fixed input dimensions, only feed them what they accept.
    batch_size = resolve_model_input_sizes(model, batch_size=batch_size, tile_size=tile_size)
    model_dtype = _resolve_model_dtype(model, use_fp32=use_fp32)

    orig_img_mode = input_image.mode
    orig_img_palette = input_image.palette
    img = input_image.convert("L" if is_grayscale else "RGB")
    image_width, image_height = img.size

    overlapping = tile_overlap > 0
    step = tile_size - tile_overlap
    if overlapping:
        tile_count_height = max(1, -(-(image_height - tile_overlap) // step))
        tile_count_width = max(1, -(-(image_width - tile_overlap) // step))
        padded_height = (tile_count_height - 1) * step + tile_size
        padded_width = (tile_count_width - 1) * step + tile_size

        image_array = np.asarray(img)
        if is_grayscale:
            image_array = image_array[..., None]
        # Replicate the real edge instead of a flat canvas: with overlapping tiles that padding is
        # one tile's share of the seam, and a hard black/white border there would show up as one.
        image_array = np.pad(
            image_array,
            ((0, padded_height - image_height), (0, padded_width - image_width), (0, 0)),
            mode="edge",
        )
    else:
        tile_count_height = int(np.ceil(image_height / tile_size))
        tile_count_width = int(np.ceil(image_width / tile_size))
        padded_height = tile_count_height * tile_size
        padded_width = tile_count_width * tile_size
        # A white background helps with contrast stretching (can lead to reversing CMYK shift).
        pad_value = 255 if background == "white" else 0
        canvas = Image.new("L" if is_grayscale else "RGB", (padded_width, padded_height), pad_value)
        canvas.paste(img, (0, 0))
        image_array = np.asarray(canvas)
        if is_grayscale:
            image_array = image_array[..., None]

    # Pre-Processing
    # contrast stretching: scaling the image based on its own darkest and brightest pixels.
    # For example, a very dark photo (e.g., pixel values from 10 to 50) and a very bright photo (e.g., values
    # from 200 to 250) will both be stretched to the full [0.0, 1.0] range.
    # The model loses all information about the image's absolute brightness. Can be desirable with a white
    # background to reverse CMYK shift in rare (color) images.
    # Not desirable in general so the alternative path should be taken in the vast majority of cases.
    # See also: https://en.wikipedia.org/wiki/Normalization_(image_processing)#Contrast_Stretching_for_Image_Enhancement
    if not contrast_stretch and image_array.dtype == np.uint8:
        # Same x / 255.0 then cast, done once per shade instead of on a full-page float copy.
        image_array = (np.arange(256) / 255.0).astype(model_dtype)[image_array]
    else:
        # A page of one colour has nothing to stretch, and would divide by zero.
        span = np.ptp(image_array) if contrast_stretch else 0
        image_array = (image_array - np.min(image_array)) / span if span else image_array / 255.0
        image_array = image_array.astype(model_dtype)
    # Rearranging (Height, Width, Channel) -> (Channel, Height, Width) to match the expected input shape
    image_array = rearrange(image_array, "w h c -> c w h")

    # Cut into tiles
    if overlapping:
        # A view: tiles get copied a batch at a time, not the whole set at once.
        windows = np.lib.stride_tricks.sliding_window_view(image_array, (tile_size, tile_size), axis=(1, 2))[
            :, ::step, ::step
        ].transpose(1, 2, 0, 3, 4)
        num_chunks = tile_count_height * tile_count_width

        def tiles_of(start: int, stop: int) -> "np.ndarray":
            ys, xs = np.divmod(np.arange(start, stop), tile_count_width)
            return windows[ys, xs]

    else:
        padded_image_tiles = rearrange(image_array, "c (h th) (w tw) -> (h w) c th tw", th=tile_size, tw=tile_size)
        num_chunks = padded_image_tiles.shape[0]

        def tiles_of(start: int, stop: int) -> "np.ndarray":
            return padded_image_tiles[start:stop]

    input_name = input_spec.name
    # An int on a static input (e.g. a TensorRT-RTX engine): short batches get padded up to it below.
    input_shape = getattr(input_spec, "shape", None)
    fixed_batch = input_shape[0] if input_shape else None
    output_name = model.get_outputs()[0].name

    all_output_chunks: list["np.ndarray"] = []

    def infer(batch: "np.ndarray") -> "np.ndarray":
        return np.asarray(model.run([output_name], {input_name: batch})[0])

    for i in range(0, num_chunks, batch_size):
        batch_of_chunks = tiles_of(i, min(i + batch_size, num_chunks))

        real_tiles = len(batch_of_chunks)
        if isinstance(fixed_batch, int) and real_tiles < fixed_batch:
            # Repeat a real tile: tiles don't affect each other, and blank ones could give NaNs.
            batch_of_chunks = np.pad(
                batch_of_chunks,
                ((0, fixed_batch - real_tiles), (0, 0), (0, 0), (0, 0)),
                mode="edge",
            )

        model_output = _run_tta(infer, batch_of_chunks) if tta else infer(batch_of_chunks)
        all_output_chunks.append(model_output[:real_tiles])

    tiled_output_image = np.concatenate(all_output_chunks, axis=0)

    if scale_factor > 1:
        image_height = image_height * scale_factor
        image_width = image_width * scale_factor

    if overlapping:
        out_tile = tile_size * scale_factor
        out_step = step * scale_factor
        out_overlap = tile_overlap * scale_factor
        # Overlap-add: linear ramps over the shared pixels (1 elsewhere), divided by their sum.
        ramp = np.minimum(
            1.0,
            np.minimum(np.arange(1, out_tile + 1), np.arange(out_tile, 0, -1)) / (out_overlap + 1),
        ).astype(np.float32)
        weight = np.outer(ramp, ramp)[..., None]
        reconstructed_image = np.zeros(
            (padded_height * scale_factor, padded_width * scale_factor, input_channel_count), np.float32
        )
        weight_sum = np.zeros((*reconstructed_image.shape[:2], 1), np.float32)
        for n, out in enumerate(tiled_output_image):
            y, x = (v * out_step for v in divmod(n, tile_count_width))
            reconstructed_image[y : y + out_tile, x : x + out_tile] += out.transpose(1, 2, 0) * weight
            weight_sum[y : y + out_tile, x : x + out_tile] += weight
        reconstructed_image /= np.maximum(weight_sum, np.finfo(np.float32).eps)
    else:
        reconstructed_image = rearrange(
            tiled_output_image, "(h w) c th tw -> (h th) (w tw) c", h=tile_count_height, w=tile_count_width
        )

    # Post-processing
    # See an explanation for image-specific normalization in the pre-processing section.
    if contrast_stretch:
        postprocessed_array = reconstructed_image.astype(np.float32)
        span = np.ptp(postprocessed_array)
        postprocessed_array = (
            (postprocessed_array - np.min(postprocessed_array)) / span
            if span
            else np.clip(postprocessed_array, 0.0, 1.0)
        )
    else:
        postprocessed_array = np.clip(reconstructed_image, 0.0, 1.0)
    # Scaling back to [0,255] from [0.0, 1.0], then rounding for a clean cast to uint8.
    postprocessed_array = np.round(postprocessed_array * 255.0).astype(np.uint8)
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


def denoise_single_image(
    input_image: Image.Image,
    model: "InferenceSessionWithScale",
    *,
    batch_size: int = 64,
    tile_size: int = 128,
    contrast_stretch: bool = False,
    background: Literal["white", "black"] = "black",
    use_fp32: bool = False,
    tta: bool = False,
) -> Image.Image:
    """Process an image tile by tile, without blending neighbouring tiles.

    A x1 model removes noise; anything bigger enlarges the page.

    :param input_image: The page to process
    :param model: The ONNX Runtime session, with its scale factor attached
    :param batch_size: How many tiles to run per inference call
    :param tile_size: The side of the square tiles the model is fed
    :param contrast_stretch: Normalise each page by its own darkest and brightest pixels
    :param background: The colour the last partial tile row/column is padded with
    :param use_fp32: Fall back to fp32 input when the model declares no precision
    :param tta: Average each batch over its 8 flips/rotations (test-time augmentation)
    :return: The processed page, in its original mode (palette pages keep their palette)
    """

    return _denoise_image(
        input_image,
        model,
        batch_size=batch_size,
        tile_size=tile_size,
        tile_overlap=0,
        contrast_stretch=contrast_stretch,
        background=background,
        use_fp32=use_fp32,
        tta=tta,
    )


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
    tta: bool = False,
) -> Image.Image:
    """Process an image with overlapping tiles and feathered seams.

    Neighbouring tiles share `tile_overlap` output pixels, blended with complementary linear ramps
    so the shared pixels add up to one. The seam handling follows waifu2x's overlap-add scheme.

    :param input_image: The page to process
    :param model: The ONNX Runtime session, with its scale factor attached
    :param batch_size: How many tiles to run per inference call
    :param tile_size: The side of the square tiles the model is fed
    :param tile_overlap: The total output overlap between two neighbouring tiles
    :param contrast_stretch: Normalise each page by its own darkest and brightest pixels
    :param background: The colour a non-overlapping run pads the last partial tile with
    :param use_fp32: Fall back to fp32 input when the model declares no precision
    :param tta: Average each batch over its 8 flips/rotations (test-time augmentation)
    :return: The processed page, in its original mode (palette pages keep their palette)
    """

    return _denoise_image(
        input_image,
        model,
        batch_size=batch_size,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        contrast_stretch=contrast_stretch,
        background=background,
        use_fp32=use_fp32,
        tta=tta,
    )

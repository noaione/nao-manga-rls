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

from ..lazy import get_numpy
from .onnx_session import resolve_model_input_sizes
from .tiler import TileBox, TilingPolicy, feather_ramp_values, group_tiles_by_shape, plan_tiles

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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


def _build_tile_batch(
    image_array: "np.ndarray",
    group: "Sequence[TileBox]",
    *,
    run_h: int,
    run_w: int,
    model_dtype: Any,
    lookup: "np.ndarray | None",
    pad_value: int | None,
) -> "np.ndarray":
    """Extract a batch of tiles of uniform ``(run_h, run_w)`` shape and convert dtype."""
    np = get_numpy()
    channels = image_array.shape[0]
    batch = np.empty(
        (len(group), channels, run_h, run_w), dtype=image_array.dtype if lookup is not None else model_dtype
    )
    for idx, tile in enumerate(group):
        patch = image_array[:, tile.src_y : tile.src_y + tile.src_h, tile.src_x : tile.src_x + tile.src_w]
        if tile.pad_left or tile.pad_top or tile.pad_right or tile.pad_bottom:
            pad_width = ((0, 0), (tile.pad_top, tile.pad_bottom), (tile.pad_left, tile.pad_right))
            if pad_value is not None:
                patch = np.pad(patch, pad_width, mode="constant", constant_values=pad_value)
            else:
                patch = np.pad(patch, pad_width, mode="edge")
        batch[idx] = patch
    if lookup is not None:
        return lookup[batch]
    return np.ascontiguousarray(batch, dtype=model_dtype)


def _denoise_image(
    input_image: Image.Image,
    model: "InferenceSessionWithScale",
    *,
    batch_size: int,
    tile_size: int,
    tile_overlap: int,
    context: int | None,
    contrast_stretch: bool,
    background: Literal["white", "black"],
    use_fp32: bool,
    tta: bool,
    pad_final_batch: bool,
    tiling_policy: TilingPolicy | None = None,
) -> Image.Image:
    np = get_numpy()

    if context is None:
        context = getattr(model, "context", 0) or 0
    if not isinstance(context, int):
        raise ValueError("Model context must be an integer.")
    scale_factor = getattr(model, "scale_factor", None)
    if not isinstance(scale_factor, int):
        raise ValueError("Model scale factor is not set. Ensure the model was prepared correctly.")
    if scale_factor < 1:
        if context:
            # Shape-based scale detection sees a valid-convolution x1 model as shrinking.
            scale_factor = 1
        else:
            raise ValueError("Model scale factor must be at least 1.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    if context and scale_factor != 1:
        raise ValueError("Context trimming is only supported for x1 models.")

    resolved_policy = tiling_policy or TilingPolicy()
    input_spec = model.get_inputs()[0]
    input_channel_count: int = input_spec.shape[1]
    is_grayscale = input_channel_count == 1
    # Some models declare fixed input dimensions, only feed them what they accept.
    batch_size = resolve_model_input_sizes(model, batch_size=batch_size, tile_size=tile_size)
    input_shape = getattr(input_spec, "shape", None)
    if resolved_policy.mode == "variable" and input_shape and len(input_shape) >= 4:
        fixed_spatial = [dim for dim in input_shape[-2:] if isinstance(dim, int) and dim > 0]
        if fixed_spatial:
            raise ValueError(
                "Variable tiling requires dynamic spatial input dimensions, "
                f"but this model declares fixed spatial dimensions {fixed_spatial}."
            )
    model_dtype = _resolve_model_dtype(model, use_fp32=use_fp32)

    orig_img_mode = input_image.mode
    has_alpha = "A" in input_image.getbands() or (
        input_image.mode == "P" and input_image.info.get("transparency") is not None
    )
    alpha = None
    if has_alpha:
        if "A" in input_image.getbands():
            alpha = input_image.getchannel("A")
        else:
            with input_image.convert("RGBA") as rgba:
                alpha = rgba.getchannel("A")

    img = input_image.convert("L" if is_grayscale else "RGB")
    image_width, image_height = img.size

    tiles = plan_tiles(
        image_width,
        image_height,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        context=context,
        policy=resolved_policy,
    )
    grouped_tiles = group_tiles_by_shape(tiles, primary_shape=(tile_size, tile_size))

    overlapping = tile_overlap > 0 or context > 0
    pad_value: int | None = None
    if resolved_policy.mode == "fixed" and not overlapping:
        # A white background helps with contrast stretching (can lead to reversing CMYK shift).
        pad_value = 255 if background == "white" else 0

    image_array = np.asarray(img)
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
        # Convert tiles as they are batched so a large page does not need a full float copy.
        lookup = (np.arange(256) / 255.0).astype(model_dtype)
        if overlapping:
            # Shared pixels would otherwise be converted once for every tile that contains them.
            image_array = lookup[image_array]
            lookup = None
    else:
        lookup = None
        if contrast_stretch:
            img_min = float(np.min(image_array))
            img_max = float(np.max(image_array))
            if pad_value is not None and any(t.pad_right > 0 or t.pad_bottom > 0 for t in tiles):
                img_min = min(img_min, float(pad_value))
                img_max = max(img_max, float(pad_value))
            # A page of one colour has nothing to stretch, and would divide by zero.
            span = img_max - img_min
            image_array = (image_array - img_min) / span if span else image_array / 255.0
        else:
            image_array = image_array / 255.0
        image_array = image_array.astype(model_dtype)
    # Shape the page as C x H x W so tile extraction only slices contiguous channel planes.
    image_array = image_array.transpose(2, 0, 1)

    input_name = input_spec.name
    # An int on a static input (e.g. a TensorRT-RTX engine): short batches get padded up to it below.
    fixed_batch = input_shape[0] if input_shape and isinstance(input_shape[0], int) else None
    output_name = model.get_outputs()[0].name

    def infer(batch: "np.ndarray") -> "np.ndarray":
        return np.asarray(model.run([output_name], {input_name: batch})[0])

    out_height = image_height * scale_factor
    out_width = image_width * scale_factor
    output_overlap = tile_overlap * scale_factor

    reconstructed_image: "np.ndarray | None" = None
    weight_sum = np.zeros((out_height, out_width, 1), np.float32) if tile_overlap else None
    blend_cache: dict[tuple[int, int, bool, bool, bool, bool], "np.ndarray"] = {}

    def _blend_mask(tile: TileBox, eh: int, ew: int) -> "np.ndarray":
        key = (eh, ew, tile.has_top, tile.has_bottom, tile.has_left, tile.has_right)
        cached = blend_cache.get(key)
        if cached is not None:
            return cached
        wy = np.asarray(
            feather_ramp_values(eh, overlap=output_overlap, has_prev=tile.has_top, has_next=tile.has_bottom),
            dtype=np.float32,
        )
        wx = np.asarray(
            feather_ramp_values(ew, overlap=output_overlap, has_prev=tile.has_left, has_next=tile.has_right),
            dtype=np.float32,
        )
        mask = np.outer(wy, wx)[..., None]
        blend_cache[key] = mask
        return mask

    def place(batch_tiles: "Sequence[TileBox]", outputs: "np.ndarray") -> None:
        nonlocal reconstructed_image
        if reconstructed_image is None:
            reconstructed_image = np.zeros((out_height, out_width, outputs.shape[1]), np.float32)
        for tile, output in zip(batch_tiles, outputs, strict=True):
            eh = tile.valid_h * scale_factor
            ew = tile.valid_w * scale_factor
            sy = tile.dst_y * scale_factor
            sx = tile.dst_x * scale_factor
            region = (slice(sy, sy + eh), slice(sx, sx + ew))
            output_image = output[:, :eh, :ew].transpose(1, 2, 0)
            if weight_sum is not None:
                weight = _blend_mask(tile, eh, ew)
                reconstructed_image[region] += output_image * weight
                weight_sum[region] += weight
            else:
                reconstructed_image[region] = output_image

    for (run_h, run_w), group in grouped_tiles:
        # In variable mode, avoid padding a tiny minority edge group (e.g. 1 corner tile) up to
        # `batch_size`; still pad trailing batches when a shape runs multiple batches.
        pad_target = fixed_batch
        if pad_target is None and pad_final_batch and (resolved_policy.mode == "fixed" or len(group) > batch_size):
            pad_target = batch_size

        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            batch_of_tiles = _build_tile_batch(
                image_array,
                chunk,
                run_h=run_h,
                run_w=run_w,
                model_dtype=model_dtype,
                lookup=lookup,
                pad_value=pad_value,
            )
            real_tiles = len(chunk)
            if pad_target is not None and real_tiles < pad_target:
                # Repeat a real tile: tiles do not affect each other, and blank ones could give NaNs.
                batch_of_tiles = np.pad(
                    batch_of_tiles,
                    ((0, pad_target - real_tiles), (0, 0), (0, 0), (0, 0)),
                    mode="edge",
                )

            model_output = _run_tta(infer, batch_of_tiles) if tta else infer(batch_of_tiles)
            place(chunk, model_output[:real_tiles])

    if reconstructed_image is None:
        raise RuntimeError("The image produced no tiles.")
    if weight_sum is not None:
        reconstructed_image /= np.maximum(weight_sum, np.finfo(np.float32).eps)

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
        postprocessed_array = np.clip(reconstructed_image, 0.0, 1.0, out=reconstructed_image)
    # Scaling back to [0,255] from [0.0, 1.0], then rounding for a clean cast to uint8.
    postprocessed_array *= 255.0
    np.round(postprocessed_array, out=postprocessed_array)
    postprocessed_array = postprocessed_array.astype(np.uint8)
    if is_grayscale:
        postprocessed_array = np.squeeze(postprocessed_array, axis=-1)

    output_image = Image.fromarray(postprocessed_array)
    if alpha is not None:
        alpha_mode = "LA" if is_grayscale or orig_img_mode == "LA" else "RGBA"
        output_image = output_image.convert(mode="L" if alpha_mode == "LA" else "RGB")
        output_image = output_image.convert(mode=alpha_mode)
        resized_alpha = alpha.resize(output_image.size, Image.Resampling.BICUBIC)
        output_image.putalpha(resized_alpha)
        resized_alpha.close()
        alpha.close()
    elif not is_grayscale:
        if input_image.palette:
            palette_image = Image.new("P", (1, 1))
            palette_image.putpalette(input_image.palette)
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
    context: int | None = None,
    contrast_stretch: bool = False,
    background: Literal["white", "black"] = "black",
    use_fp32: bool = False,
    tta: bool = False,
    pad_final_batch: bool = False,
    tiling_policy: TilingPolicy | None = None,
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
    :param pad_final_batch: Repeat tiles so the last batch is full, which keeps one model shape
    :param context: Input pixels the model trims off each tile; defaults to the model's detected context
    :param tiling_policy: Fixed or variable edge-tiling policy
    :return: The processed page with alpha and palette preserved
    """

    return _denoise_image(
        input_image,
        model,
        batch_size=batch_size,
        tile_size=tile_size,
        tile_overlap=0,
        context=context,
        contrast_stretch=contrast_stretch,
        background=background,
        use_fp32=use_fp32,
        tta=tta,
        pad_final_batch=pad_final_batch,
        tiling_policy=tiling_policy,
    )


def denoise_single_image_with_overlap(
    input_image: Image.Image,
    model: "InferenceSessionWithScale",
    *,
    batch_size: int = 64,
    tile_size: int = 128,
    tile_overlap: int = 16,
    context: int | None = None,
    contrast_stretch: bool = False,
    background: Literal["white", "black"] = "black",
    use_fp32: bool = False,
    tta: bool = False,
    pad_final_batch: bool = False,
    tiling_policy: TilingPolicy | None = None,
) -> Image.Image:
    """Process an image with overlapping tiles and feathered seams.

    Neighbouring tiles share `tile_overlap` output pixels, blended with complementary linear ramps
    so the shared pixels add up to one. The seam handling follows waifu2x's overlap-add scheme.

    :param input_image: The page to process
    :param model: The ONNX Runtime session, with its scale factor attached
    :param batch_size: How many tiles to run per inference call
    :param tile_size: The side of the square tiles the model is fed
    :param tile_overlap: The total output overlap between two neighbouring tiles
    :param context: Input pixels the model trims off each tile; defaults to the model's detected context
    :param contrast_stretch: Normalise each page by its own darkest and brightest pixels
    :param background: The colour a non-overlapping run pads the last partial tile with
    :param use_fp32: Fall back to fp32 input when the model declares no precision
    :param tta: Average each batch over its 8 flips/rotations (test-time augmentation)
    :param pad_final_batch: Repeat tiles so the last batch is full, which keeps one model shape
    :param tiling_policy: Fixed or variable edge-tiling policy
    :return: The processed page with alpha and palette preserved
    """

    return _denoise_image(
        input_image,
        model,
        batch_size=batch_size,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        context=context,
        contrast_stretch=contrast_stretch,
        background=background,
        use_fp32=use_fp32,
        tta=tta,
        pad_final_batch=pad_final_batch,
        tiling_policy=tiling_policy,
    )

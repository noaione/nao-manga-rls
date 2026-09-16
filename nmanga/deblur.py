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

from typing import TYPE_CHECKING

from PIL import Image, ImageOps

from .lazy import get_numpy, get_scipy_ndimage

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

__all__ = (
    "deblur_deconv",
    "deblur_edge_sharp",
)


def smoothstep(x: "NDArray[np.float64]") -> "NDArray[np.float64]":
    """
    Apply the Hermite smoothstep curve to an array.

    Parameters
    ----------
    x: :class:`numpy.ndarray`
        Input values, which are clipped to the ``[0, 1]`` range before the curve is applied.

    Returns
    -------
    :class:`numpy.ndarray`
        Array with the same shape as ``x``, scaled by ``x * x * (3 - 2 * x)``, giving a smooth ``0``
        to ``1`` transition with a zero first derivative at both ends.
    """

    np = get_numpy()

    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def edge_mask(y: "NDArray[np.float64]", threshold: float) -> "NDArray[np.float64]":
    """
    Build a soft edge mask from the gradient magnitude of a luma plane.

    The mask is ``0`` in flat areas and ramps to ``1`` on strong edges, so it can be used to restrict a
    sharpening operator to edge regions only. The gradient is measured with a Sobel operator on a lightly
    prefiltered copy of ``y``, so the prefilter only affects the edge detector and never the source
    artwork.

    Parameters
    ----------
    y: :class:`numpy.ndarray`
        2D luma plane normalized to ``[0, 1]``.
    threshold: :class:`float`
        Gradient magnitude, roughly in 8-bit gray levels per pixel, at which the mask starts to open,
        with the ramp spanning ``3 * threshold`` above it. The gradient is the Sobel response of the
        normalized luma, rescaled so a hard black to white edge measures around 115. Raising the value
        restricts the mask to stronger edges and leaves more of the artwork alone, lowering it lets
        fainter edges and fine texture through.

    Returns
    -------
    :class:`numpy.ndarray`
        Float mask in ``[0, 1]`` with the same shape as ``y``, slightly blurred to avoid hard mask
        transitions.
    """

    np = get_numpy()
    ndimage = get_scipy_ndimage()
    # Gently prefilter only the edge detector, not the source artwork.
    base = ndimage.gaussian_filter(y, 0.5, mode="reflect")
    gradient = np.hypot(
        ndimage.sobel(base, axis=0, mode="reflect"),
        ndimage.sobel(base, axis=1, mode="reflect"),
    )

    # Normalize the 2D Sobel response and express it in 8-bit levels per pixel.
    magnitude = gradient * (255.0 / 8.0)
    mask = smoothstep((magnitude - threshold) / max(3 * threshold, 1e-6))
    return ndimage.gaussian_filter(mask, 0.45, mode="reflect")


def simple_blur(y: "NDArray[np.float64]", radius: float) -> "NDArray[np.float64]":
    """
    Blur an array with a reflected Gaussian kernel.

    Parameters
    ----------
    y: :class:`numpy.ndarray`
        Input array, usually a normalized 2D luma plane.
    radius: :class:`float`
        Gaussian standard deviation in pixels. Raising it widens the blur so the comparison happens
        over a larger neighbourhood, lowering it keeps the blur tight and limits the effect to the
        finest detail.

    Returns
    -------
    :class:`numpy.ndarray`
        Blurred array with the same shape as ``y``, using a kernel truncated at ``4 * radius`` and
        reflected edge handling.
    """

    ndimage = get_scipy_ndimage()
    return ndimage.gaussian_filter(y, radius, mode="reflect", truncate=4)


def restore_limited_edge_sharpen(y: "NDArray[np.float64]", radius: float, strength: float) -> "NDArray[np.float64]":
    """
    Sharpen an array with a classic unsharp mask.

    Parameters
    ----------
    y: :class:`numpy.ndarray`
        Input array, usually a normalized 2D luma plane.
    radius: :class:`float`
        Gaussian standard deviation in pixels used for the blurred reference. Raising it sharpens
        coarser features and spreads any halo wider, lowering it keeps the effect on the finest detail.
    strength: :class:`float`
        Amount of the high-frequency residual which is added back. Raising it gives a harder and
        crisper edge, lowering it gives a gentler result that is less likely to ring.

    Returns
    -------
    :class:`numpy.ndarray`
        Sharpened array with the same shape as ``y``. Neither the overshoot nor the ringing is limited
        here, so the result is expected to be passed through :func:`blend_limited_sharpening`.
    """

    return y + strength * (y - simple_blur(y, radius))


def restore_deconv(y: "NDArray[np.float64]", radius: float, strength: float, iterations: int) -> "NDArray[np.float64]":
    """
    Sharpen an array with a Richardson-Lucy style multiplicative deconvolution.

    The Gaussian blur is used as the point-spread function, and each iteration applies a multiplicative
    correction to the current estimate. Only the difference between the estimate and the input is added
    back, scaled by ``strength``, so this stays a sharpening step instead of a full deconvolution.

    Parameters
    ----------
    y: :class:`numpy.ndarray`
        Input array, usually a normalized 2D luma plane.
    radius: :class:`float`
        Gaussian standard deviation in pixels of the assumed blur kernel. Raising it makes the
        correction chase wider and softer detail, lowering it keeps the correction on the sharpest
        lines.
    strength: :class:`float`
        Amount of the estimated correction which is added back to ``y``. Raising it recovers more
        detail but amplifies ringing, lowering it stays closer to the input.
    iterations: :class:`int`
        Number of multiplicative refinement passes over the estimate. Raising it pushes the estimate
        further from the input and costs more time, lowering it stops earlier and stays conservative.

    Returns
    -------
    :class:`numpy.ndarray`
        Sharpened array with the same shape as ``y``, with no clipping applied.
    """

    np = get_numpy()

    # Positive pedestal avoids zero-locking and numerical division problems
    # at pure black. The reflected Gaussian operator is symmetric.
    pedestal = 1 / 255
    observed = y + pedestal
    estimate = observed.copy()
    for _ in range(iterations):
        estimate *= simple_blur(observed / np.maximum(simple_blur(estimate, radius), 1e-7), radius)
    return y + strength * (estimate - pedestal - y)


def blend_limited_sharpening(
    y: "NDArray[np.float64]", candidate: "NDArray[np.float64]", threshold: float, overshoot: float
) -> "NDArray[np.float64]":
    """
    Blend a sharpened candidate back into the original, limited and edge masked.

    The candidate is first clamped between the local minimum and maximum of ``y`` (plus the optional
    overshoot) to suppress halos and ringing, then mixed into ``y`` weighted by :func:`edge_mask`, so
    flat regions are left untouched.

    Parameters
    ----------
    y: :class:`numpy.ndarray`
        Original array, usually a normalized 2D luma plane.
    candidate: :class:`numpy.ndarray`
        Sharpened or deconvolved proposal, with the same shape as ``y``.
    threshold: :class:`float`
        Edge threshold in 8-bit gray levels, forwarded to :func:`edge_mask`. Raising it protects flat
        areas and fine texture, lowering it lets the blend reach weaker edges.
    overshoot: :class:`float`
        Extra excursion allowed beyond the local extremes, expressed in 8-bit steps and divided by
        ``255`` internally. ``0`` means no overshoot and is the most conservative setting, raising it
        allows more ringing around strong edges for crisper transitions.

    Returns
    -------
    :class:`numpy.ndarray`
        Blended array in ``[0, 1]`` with the same shape as ``y``.
    """

    np = get_numpy()
    ndimage = get_scipy_ndimage()

    lo = ndimage.minimum_filter(y, size=3, mode="reflect") - overshoot / 255
    hi = ndimage.maximum_filter(y, size=3, mode="reflect") + overshoot / 255
    candidate = np.clip(candidate, lo, hi)
    return np.clip(y + edge_mask(y, threshold) * (candidate - y), 0, 1)


def _prepare_image(
    img: Image.Image,
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]"]:
    """
    Split an image into the planes the deblurring kernels operate on.

    The image is EXIF-transposed and converted to RGBA first, so the rotation and the alpha channel are
    preserved and handed back to the caller untouched.

    Parameters
    ----------
    img: :class:`PIL.Image.Image`
        Source image in any Pillow-supported mode.

    Returns
    -------
    :class:`tuple` of :class:`numpy.ndarray`
        A ``(y, rgb, rgba)`` tuple, where ``y`` is the encoded-value luma plane in ``[0, 1]`` as
        :class:`numpy.float32`, ``rgb`` is the color plane in ``[0, 1]``, and ``rgba`` is the raw
        transposed 8-bit RGBA array. The luma is computed from gamma-encoded values on purpose, since
        the input is digitally resampled artwork and not a calibrated linear-light capture.
    """

    np = get_numpy()

    rgba = np.asarray(ImageOps.exif_transpose(img).convert("RGBA")).copy()
    rgb = rgba[..., :3].astype(np.float32) / 255

    # Encoded-value luma is intentional for digitally resampled artwork.
    # This is not a calibrated linear-light optical restoration pipeline.
    y = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return y, rgb, rgba


def deblur_deconv(
    img: Image.Image,
    *,
    radius: float = 0.8,
    strength: float = 0.65,
    iterations: int = 6,
    threshold: float = 2,
    overshoot: float = 0,
) -> Image.Image:
    """
    Sharpen an image with a Richardson-Lucy style deconvolution.

    Only the luma plane is deconvolved. The resulting change is applied to all three color channels as
    an equal offset, which preserves the original channel differences, and the offset is limited to the
    available gamut instead of clipping each channel separately.

    Parameters
    ----------
    img: :class:`PIL.Image.Image`
        Source image in any Pillow-supported mode.
    radius: :class:`float`, optional
        Gaussian standard deviation in pixels of the assumed blur kernel (default is 0.8). Raising it
        targets coarser softness, lowering it targets the finest lines.
    strength: :class:`float`, optional
        Amount of the deconvolution correction which is added back (default is 0.65). Raising it
        sharpens harder and recovers more detail, lowering it is gentler and less prone to halos.
    iterations: :class:`int`, optional
        Number of multiplicative refinement passes (default is 6). Raising it keeps refining the
        estimate and costs more time, lowering it stops sooner with a milder result.
    threshold: :class:`float`, optional
        Edge threshold in 8-bit gray levels, forwarded to :func:`edge_mask` (default is 2). Raising it
        limits the effect to stronger edges, lowering it also sharpens fainter edges and texture.
    overshoot: :class:`float`, optional
        Extra excursion allowed beyond the local extremes, in 8-bit steps (default is 0). Raising it
        allows more ringing around strong edges for crisper transitions.

    Returns
    -------
    :class:`PIL.Image.Image`
        A new image with the same mode as ``img``.
    """

    np = get_numpy()

    y, rgb, rgba = _prepare_image(img)
    deblur_candidate = restore_deconv(y, radius, strength, iterations)
    restored = blend_limited_sharpening(y, deblur_candidate, threshold, overshoot)

    delta = restored - y
    # Add equal RGB offsets to preserve channel differences, and limit the
    # offset to the available gamut instead of clipping channels separately.
    delta = np.clip(delta, -rgb.min(axis=2), 1 - rgb.max(axis=2))
    result = rgba.copy()
    result[..., :3] = np.rint(np.clip(rgb + delta[..., None], 0, 1) * 255).astype(np.uint8)

    target_img = Image.fromarray(result)
    return target_img.convert(img.mode)


def deblur_edge_sharp(
    img: Image.Image, *, radius: float = 0.8, strength: float = 0.85, threshold: float = 2, overshoot: float = 0
) -> Image.Image:
    """
    Sharpen an image with an edge-masked unsharp mask.

    Only the luma plane is sharpened. The resulting change is applied to all three color channels as an
    equal offset, which preserves the original channel differences, and the offset is limited to the
    available gamut instead of clipping each channel separately.

    This is the cheaper and more conservative counterpart to :func:`deblur_deconv`.

    Parameters
    ----------
    img: :class:`PIL.Image.Image`
        Source image in any Pillow-supported mode.
    radius: :class:`float`, optional
        Gaussian standard deviation in pixels used for the blurred reference (default is 0.8). Raising
        it sharpens coarser features, lowering it confines the effect to fine detail.
    strength: :class:`float`, optional
        Amount of the high-frequency residual which is added back (default is 0.85). Raising it gives
        a harder and crisper edge, lowering it gives a gentler result that is less likely to ring.
    threshold: :class:`float`, optional
        Edge threshold in 8-bit gray levels, forwarded to :func:`edge_mask` (default is 2). Raising it
        limits the effect to stronger edges, lowering it also sharpens fainter edges and texture.
    overshoot: :class:`float`, optional
        Extra excursion allowed beyond the local extremes, in 8-bit steps (default is 0). Raising it
        allows more ringing around strong edges for crisper transitions.

    Returns
    -------
    :class:`PIL.Image.Image`
        A new image with the same mode as ``img``.
    """

    np = get_numpy()

    y, rgb, rgba = _prepare_image(img)  # pyright: ignore[reportUnusedVariable]
    deblur_candidate = restore_limited_edge_sharpen(y, radius, strength)
    restored = blend_limited_sharpening(y, deblur_candidate, threshold, overshoot)

    delta = restored - y
    # Add equal RGB offsets to preserve channel differences, and limit the
    # offset to the available gamut instead of clipping channels separately.
    delta = np.clip(delta, -rgb.min(axis=2), 1 - rgb.max(axis=2))
    result = rgba.copy()
    result[..., :3] = np.rint(np.clip(rgb + delta[..., None], 0, 1) * 255).astype(np.uint8)

    target_img = Image.fromarray(result)
    return target_img.convert(img.mode)

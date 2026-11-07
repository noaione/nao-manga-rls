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

# PIL <-> tensor conversion for the PyTorch inference backend.
#
# The model contract is always RGB FP32 NCHW in [0, 1], independent of the source mode: a
# grayscale page is expanded to RGB for inference and converted back afterwards. The palette
# choice is deliberate: a palette source is decoded to RGB (or RGBA when it carries
# transparency) and the prediction is returned in that decoded mode, not requantized back onto
# the original 256 entry palette. The model changes colours, so requantizing would throw the
# prediction away and leave PIL picking an approximate palette entry per pixel.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ..lazy import get_torch

if TYPE_CHECKING:
    import torch

__all__ = (
    "image_to_tensor",
    "tensor_to_image",
)

_CHANNELS = 3
"""The channel count of the inference contract."""

_LUMINANCE_MODES = frozenset({"1", "L", "I", "F", "I;16"})
"""Source modes that are folded back into one luminance band after inference."""


def image_to_tensor(image: Image.Image) -> "torch.Tensor":
    """
    Convert a PIL image into the model input contract.

    The image is decoded to RGB, scaled to ``[0, 1]`` and rearranged into NCHW. Grayscale and
    palette sources therefore reach the model as RGB; their original mode is needed again only
    when the prediction is written back, which is what :func:`tensor_to_image` uses its template
    for. Alpha is not fed to the model.

    Parameters
    ----------
    image:
        The source page. It is not modified.

    Returns
    -------
    torch.Tensor
        A contiguous FP32 CPU tensor of ``1 x 3 x H x W`` with values in ``[0, 1]``.
    """
    torch = get_torch()

    with image.convert("RGB") as decoded:
        data = np.asarray(decoded, dtype=np.float32) / 255.0
    return torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).contiguous()


def tensor_to_image(tensor: "torch.Tensor", template: Image.Image) -> Image.Image:
    """
    Convert one prediction back into a PIL image that follows `template`.

    The values are clipped to ``[0, 1]``, multiplied by 255, rounded and cast to ``uint8``; the
    template decides the output mode and supplies the preserved alpha channel, which is resized
    to the prediction with bicubic sampling:

    - ``1``/``L``/``I``/``F``/``I;16`` templates come back as ``L``.
    - A template with an alpha band comes back as ``LA`` (for ``LA``) or ``RGBA``, with the
      template alpha resized: alpha is never upscaled by the model.
    - A palette template is decoded: ``RGB`` normally, ``RGBA`` when it carries transparency.
      The palette is not reused, see the module note.
    - Any other template mode is converted back to that mode.

    Parameters
    ----------
    tensor:
        One RGB prediction in NCHW format, ``1 x 3 x H x W``. Raw and unclipped values are
        accepted; nothing is stretched.
    template:
        The source page the prediction belongs to. It is not modified.

    Returns
    -------
    Image.Image
        A new image of the prediction size, owned by the caller.

    Raises
    ------
    ValueError
        When `tensor` is not one RGB image in NCHW format.
    """
    torch = get_torch()

    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != _CHANNELS:
        raise ValueError("Expected one RGB image in NCHW format")

    data = tensor[0].clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    image = Image.fromarray(data)

    if template.mode in _LUMINANCE_MODES:
        with image as decoded:
            return decoded.convert("L")

    if _has_alpha(template):
        target_mode = "LA" if template.mode == "LA" else "RGBA"
        with image as decoded:
            converted = decoded.convert(target_mode)
        with _alpha_band(template) as alpha:
            with alpha.resize(converted.size, Image.Resampling.BICUBIC) as resized:
                converted.putalpha(resized)
        return converted

    if template.mode == "P":
        # Decoded to RGB above; a palette page keeps its colours instead of its palette.
        return image

    if template.mode == "RGB":
        return image

    with image as decoded:
        return decoded.convert(template.mode)


def _has_alpha(template: Image.Image) -> bool:
    """Whether `template` carries alpha, either as a band or as palette transparency."""
    if "A" in template.getbands():
        return True
    # A palette image stores transparency in `info` instead of as an alpha band.
    return template.mode == "P" and template.info.get("transparency") is not None


def _alpha_band(template: Image.Image) -> Image.Image:
    """Return the template alpha as a single band, decoding palette transparency if needed."""
    if "A" in template.getbands():
        return template.getchannel("A")
    with template.convert("RGBA") as decoded:
        return decoded.getchannel("A")

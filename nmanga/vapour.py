"""
MIT License

Copyright (c) 2022-present noaione, anon

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

# Some feature that utilize vapoursynth

from __future__ import annotations

from functools import partial
from os import PathLike
from typing import TYPE_CHECKING, Any, cast

from PIL import Image

from .lazy import get_numpy, get_vapoursynth

if TYPE_CHECKING:
    import numpy as np
    from vapoursynth import PresetVideoFormat, VideoFrame, VideoNode


def fill_frame_rgb24(n: int, f: "VideoFrame | list[VideoFrame]", *, array: "np.ndarray[Any]") -> "VideoFrame":
    if isinstance(f, list):
        # get on n
        fout = f[n].copy()
    else:
        fout = f.copy()

    np = get_numpy()

    for plane in range(3):
        np.asarray(fout[plane])[:] = array[:, :, plane]

    return fout


def fill_frame_gray8(n: int, f: "VideoFrame | list[VideoFrame]", *, array: "np.ndarray[Any]") -> "VideoFrame":
    if isinstance(f, list):
        # get on n
        fout = f[n].copy()
    else:
        fout = f.copy()

    np = get_numpy()

    np.asarray(fout[0])[:] = array
    return fout


def get_pil_image_with_callback(img: Image.Image) -> tuple[partial["VideoFrame"], "PresetVideoFormat"]:
    vs = get_vapoursynth()

    np = get_numpy()

    match img.mode:
        case "RGB":
            # use partial function with array pre-allocated
            return partial(fill_frame_rgb24, array=np.asarray(img)), vs.RGB24
        case "L":
            # use partial function with array pre-allocated
            return partial(fill_frame_gray8, array=np.asarray(img)), vs.GRAY8
        case "P":
            # palette!, use RGB
            return partial(fill_frame_rgb24, array=np.asarray(img.convert("RGB"))), vs.RGB24
        case _:
            raise ValueError(f"Unsupported image mode: {img.mode}")


def vs_prepare_image(img: str | PathLike | Image.Image) -> "VideoNode":
    vs = get_vapoursynth()
    core = vs.core

    if isinstance(img, Image.Image):
        callback, vs_fmt = get_pil_image_with_callback(img)
        clip = core.std.BlankClip(width=img.width, height=img.height, format=vs_fmt, length=1)
        clip = core.std.ModifyFrame(clip=clip, clips=clip, selector=callback)

        # we only support two now
        match vs_fmt:
            case vs.RGB24:
                # We need to add color information
                clip = core.resize.Bicubic(clip, format=vs.RGBS)
            case vs.GRAY8:
                # We need to add color information
                clip = core.resize.Bicubic(clip, format=vs.GRAYS)
            case _:
                raise ValueError(f"Unsupported image mode: {img.mode}")
    else:
        clip = core.bs.VideoSource(str(img))
    return clip


def vs_ssimulacra2(reference: "VideoNode", distorted: "VideoNode") -> float:
    result = reference.vship.SSIMULACRA2(distorted, numStream=1)
    with result.get_frame(0) as f:
        ssim_score = cast(float, f.props["_SSIMULACRA2"])
    return ssim_score


def vs_find_missing_plugins(plugins: str | list[str]) -> list[str]:
    plugins_set = set(plugins) if isinstance(plugins, list) else {plugins}

    core = get_vapoursynth().core
    identifiers = {p.identifier for p in core.plugins()}

    # return missing plugins
    return list(plugins_set - identifiers)

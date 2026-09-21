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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, cast

from PIL import Image

from .lazy import get_numpy, get_vapoursynth
from .ogsov import DetectedColor

if TYPE_CHECKING:
    import numpy as np
    from vapoursynth import MessageType, PresetVideoFormat, VideoFrame, VideoNode

    from .term import Console


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


def vs_attach_logger(console: "Console"):
    if not console.debugged:
        return

    vs = get_vapoursynth()
    type_to_str = {
        vs.MESSAGE_TYPE_DEBUG: "DEBUG",
        vs.MESSAGE_TYPE_INFORMATION: "INFO",
        vs.MESSAGE_TYPE_WARNING: "WARNING",
        vs.MESSAGE_TYPE_CRITICAL: "CRITICAL",
        vs.MESSAGE_TYPE_FATAL: "FATAL",
    }

    def _log_message(msg_type: "MessageType", message: str):
        console.log(f"[VapourSynth {type_to_str[msg_type]}]", message)

    vs.core.add_log_handler(_log_message)


def vs_prepare_image(img: str | PathLike | Image.Image) -> "VideoNode":
    vs = get_vapoursynth()
    core = vs.core

    if isinstance(img, Image.Image):
        callback, vs_fmt = get_pil_image_with_callback(img)
        clip = core.std.BlankClip(width=img.width, height=img.height, format=vs_fmt, length=1)  # type: ignore
        clip = core.std.ModifyFrame(clip=clip, clips=clip, selector=callback)  # type: ignore

        # we only support two now
        match vs_fmt:
            case vs.RGB24:
                # We need to add color information
                clip = cast(Any, core).resize.Bicubic(clip, format=vs.RGBS)
            case vs.GRAY8:
                # We need to add color information
                clip = cast(Any, core).resize.Bicubic(clip, format=vs.GRAYS)
            case _:
                raise ValueError(f"Unsupported image mode: {img.mode}")
    else:
        clip = core.bs.VideoSource(str(img))  # type: ignore
    return clip


def vs_fix_odd_size_chain(n: int, *, source: "VideoNode", plain_rgb: "VideoNode") -> "VideoNode":
    vs = get_vapoursynth()
    frame = source.get_frame(n)  # this frame's format and size
    right = frame.width % 2 if frame.format.subsampling_w else 0
    bottom = frame.height % 2 if frame.format.subsampling_h else 0
    if not right and not bottom:
        return plain_rgb
    even = vs.core.std.CropAbs(source, width=frame.width - right, height=frame.height - bottom)
    rgb = vs.core.resize.Bicubic(even, format=vs.RGB24, matrix_in_s="470bg", range_in_s="limited")
    return vs.core.std.AddBorders(rgb, right=right, bottom=bottom)


def vs_prepare_image_bulk(images: Sequence[PathLike], *, debug: bool = False, prefetch: int = 6) -> "VideoNode":
    vs = get_vapoursynth()
    core = vs.core
    source = core.imgseqs.Read(files=[str(path) for path in images], mismatch=True, prefetch=prefetch, debug=debug)
    plain_rgb = core.resize.Bicubic(source, format=vs.RGB24, matrix_in_s="470bg", range_in_s="limited")
    rgb = core.std.FrameEval(source, partial(vs_fix_odd_size_chain, source=source, plain_rgb=plain_rgb))
    rgb = core.resize.Bicubic(rgb, format=vs.RGB24)  # FrameEval cannot promise a format
    return rgb


def vs_analyze_ogsov_frame(frame: "VideoFrame") -> tuple[DetectedColor, Path]:
    props = frame.props
    img_path = Path(str(props["ImgSeqPath"]))
    confidence = round(cast(float, props["OGSOVConfidence"]))
    is_color = props["OGSOVIsColor"] == 1
    is_gray_already = str(props["ImgSeqOriginalColorType"]).upper().startswith("L")
    reasoning = DetectedColor(
        is_color,
        confidence,
        reason="ML-based detection via VapourSynth",
        should_convert=not is_color and not is_gray_already,
    )
    return reasoning, img_path


def vs_ssimulacra2(reference: "VideoNode", distorted: "VideoNode") -> float:
    result = reference.vship.SSIMULACRA2(distorted, numStream=1)  # type: ignore
    with result.get_frame(0) as f:
        ssim_score = cast(float, f.props["_SSIMULACRA2"])
    return ssim_score


def vs_find_missing_plugins(plugins: str | list[str]) -> list[str]:
    plugins_set = set(plugins) if isinstance(plugins, list) else {plugins}

    core = get_vapoursynth().core
    identifiers = {p.identifier for p in core.plugins()}

    # return missing plugins
    return list(plugins_set - identifiers)

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

from dataclasses import dataclass
from functools import partial
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence, cast

from PIL import Image

from .lazy import get_numpy, get_vapoursynth
from .ogsov import DetectedColor

if TYPE_CHECKING:
    import numpy as np
    from vapoursynth import Core, MessageType, PresetVideoFormat, VideoFrame, VideoNode

    from .term import Console

    class AutolevelOptions(Protocol):
        """The subset of the `autolevel3` config that reaches the plugin."""

        upper_limit: int
        peak_min_pct: float | None
        peak_prom_pct: float | None
        peak_offset: int
        no_white: bool
        cache_mb: int


def fill_frame_from_frame(n: int, f: "VideoFrame | list[VideoFrame]", *, frame: "VideoFrame") -> "VideoFrame":
    """Copy every plane of `frame` into the blank clip this is a selector for."""
    fout = (f[n] if isinstance(f, list) else f).copy()
    np = get_numpy()
    for plane in range(fout.format.num_planes):
        np.asarray(fout[plane])[:] = np.asarray(frame[plane])
    return fout


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


def _trim_and_convert(clip: "VideoNode", width: int, height: int, core: "Core") -> "VideoNode":
    """
    Convert a YUV clip to a constant `GRAY8` through `RGB24`.

    An odd width or height cannot be 4:2:0, and every filter but the decoder refuses it,
    so an odd edge is trimmed before the conversion and padded back after it.
    """
    vs = get_vapoursynth()

    right = width % 2 if clip.format.subsampling_w else 0
    bottom = height % 2 if clip.format.subsampling_h else 0
    region = clip
    if right or bottom:
        region = core.std.CropAbs(clip, width=width - right, height=height - bottom)

    # No matrix or range arguments: the frame properties carry both and are what makes
    # this round trip agree with a Pillow convert("L").
    rgb = cast(Any, core).resize.Bicubic(region, format=vs.RGB24)
    gray = cast(Any, core).resize.Bicubic(rgb, format=vs.GRAY8, matrix_s="470bg", range_s="full")
    if right or bottom:
        gray = core.std.AddBorders(gray, right=right, bottom=bottom)
    return gray


def vs_to_gray8(source: "VideoNode", core: "Core | None" = None) -> "VideoNode":
    """
    Normalise a sequence to a constant `GRAY8`.

    `imgseqs` hands out whatever the container holds: `RGB24` for jpeg and png, `YUV420P8`
    for lossy webp, heif and avif. A YUV clip cannot be converted by taking its luma plane,
    because that is the decoder's limited range Y and it does not hold the same values a
    Pillow `convert("L")` of the same page does. Routing it through RGB makes both sides
    analyse the same samples.
    """
    vs = get_vapoursynth()
    if core is None:
        core = vs.core

    if source.format.color_family == vs.RGB:
        return cast(Any, core).resize.Bicubic(source, format=vs.GRAY8, matrix_s="470bg", range_s="full")
    if source.format.color_family == vs.YUV:
        return _trim_and_convert(source, source.width, source.height, core)

    # The clip varies, so this node reports an undefined format and cannot say whether its
    # frames are RGB. One probe answers it.
    probe = source.get_frame(0)
    if probe.format.color_family == vs.RGB:
        return cast(Any, core).resize.Bicubic(source, format=vs.GRAY8, matrix_s="470bg", range_s="full")

    # A sequence mixing both has to be converted one frame at a time.
    plain_gray = cast(Any, core).resize.Bicubic(source, format=vs.GRAY8, matrix_s="470bg", range_s="full")

    def convert(n: int = 0, **_):
        frame = source.get_frame(n)
        if frame.format.color_family == vs.RGB:
            return plain_gray
        return _trim_and_convert(source, frame.width, frame.height, core)

    gray = core.std.FrameEval(source, convert)
    return cast(Any, core).resize.Bicubic(gray, format=vs.GRAY8)


@dataclass
class AutolevelChain:
    """The nodes `vs_autolevel_clip` built, kept so a caller can pull frames from them."""

    source: "VideoNode"
    gray: "VideoNode"
    stats: "VideoNode"
    leveled: "VideoNode"


def vs_autolevel_clip(
    files: Sequence[PathLike],
    *,
    options: "AutolevelOptions",
    prefetch: int = 0,
    debug: bool = False,
    core: "Core | None" = None,
) -> AutolevelChain:
    """
    Read `files` and build the `PeakStats` to `Levels` chain over their gray form.

    `gray` is the analysed clip and `leveled` is the same pages levelled with the levels
    found on them. Pull the frame you want from `leveled` and read the level properties off
    it, rather than pulling from `stats` and then `leveled`, which analyses every page twice.

    `prefetch` is how many pages `imgseqs` decodes ahead of the frame being asked for, and
    `debug` turns on the plugin's log lines.
    """
    vs = get_vapoursynth()
    if core is None:
        core = vs.core

    core.max_cache_size = options.cache_mb
    source = core.imgseqs.Read(
        files=[str(path) for path in files],
        mismatch=True,
        prefetch=prefetch,
        debug=int(debug),
    )
    gray = vs_to_gray8(source, core=core)

    stats_args: dict[str, Any] = {
        "upper_limit": options.upper_limit,
        "peak_percentage": options.peak_min_pct,
        "skip_white": 1 if options.no_white else 0,
        "debug": int(debug),
    }
    # `None` is not a value a VapourSynth argument list can carry, so the argument has to
    # be left off entirely to leave prominence disabled.
    if options.peak_prom_pct is not None:
        stats_args["peak_prominence"] = options.peak_prom_pct

    stats = core.nimages.PeakStats(gray, **stats_args)
    leveled = core.nimages.Levels(
        stats,
        use_props=True,
        peak_offset=options.peak_offset,
        auto_gamma=True,
        debug=int(debug),
    )
    return AutolevelChain(source=source, gray=gray, stats=stats, leveled=leveled)


def vs_carry_levels(rgb: "VideoNode | VideoFrame", stats: "VideoNode", core: "Core | None" = None) -> "VideoNode":
    """
    Copy the level properties from the analysed gray clip onto `rgb`.

    `Levels(use_props=True)` reads the levels from the frame properties of *its* input, so
    levelling a colour page with the levels detected on its luma needs the two properties on
    the colour clip. Feed the result to `Levels(..., use_props=True)` to rewrite every plane
    with the same curve.

    `rgb` may be a clip or a single frame. A frame is wrapped into a one frame clip of the
    same size and format, which is what a caller holding a page pulled out of the chain has;
    `ModifyFrame` needs a clip, not a frame.
    """
    vs = get_vapoursynth()
    if core is None:
        core = vs.core

    if isinstance(rgb, vs.VideoFrame):
        frame = rgb
        blank = core.std.BlankClip(
            width=frame.width,
            height=frame.height,
            format=frame.format.id,
            length=1,
        )
        rgb = core.std.ModifyFrame(blank, [blank], partial(fill_frame_from_frame, frame=frame))

    def carry(n: int, f: "list[VideoFrame]"):
        # The second parameter has to be called `f`: VapourSynth passes the frames as a
        # keyword argument.
        out = f[0].copy()
        measured = f[1].props
        out.props["NImagesBlackLevel"] = measured["NImagesBlackLevel"]
        out.props["NImagesWhiteLevel"] = measured["NImagesWhiteLevel"]
        return out

    return core.std.ModifyFrame(rgb, [rgb, stats], carry)


def level_color_page(
    chain: AutolevelChain,
    index: int,
    config: "AutolevelOptions",
    *,
    debug: bool = False,
    core: "Core | None" = None,
) -> Image.Image:
    """
    Level one page's own colour planes with the levels found on its luma.

    `PeakStats` only ever sees the gray normalisation of a page, so levelling the colour page
    with the same curve means moving the two detected level properties onto the colour frame
    and letting `Levels(use_props=True)` rewrite every plane. `Levels` reads the properties
    from the frame of *its* input, so the colour frame has to be the one :func:`vs_carry_levels`
    produced; handing it `chain.stats` would read the gray page's own levels and level with a
    curve derived from a different page.

    Only `peak_offset` is read from `config`, so anything satisfying :class:`AutolevelOptions`
    works.
    """

    vs = get_vapoursynth()
    if core is None:
        core = vs.core

    with chain.source.get_frame(index) as source_frame:
        # `vs_carry_levels` takes a clip; a bare frame is wrapped into a one frame clip of its
        # own size and format, so the properties land on a single frame with index 0.
        carried = vs_carry_levels(source_frame, chain.stats, core=core)
        with core.nimages.Levels(
            carried,
            use_props=True,
            auto_gamma=True,
            peak_offset=config.peak_offset,
            debug=int(debug),
        ).get_frame(0) as result:
            planes = [vs_frame_to_image(result, plane) for plane in range(result.format.num_planes)]

    if len(planes) == 1:
        return planes[0]
    return Image.merge("RGB", planes[:3])


def vs_frame_to_image(frame: "VideoFrame", plane: int = 0) -> Image.Image:
    """
    Return plane `plane` of `frame` as a Pillow image.

    `np.asarray` gives a `(height, width)` `uint8` view for every accepted format, so the
    mode is `L` for a gray frame and the caller picks the plane for a colour one. The copy
    matters: the array aliases the frame and the frame is released when the `with` block ends.
    """
    np = get_numpy()
    return Image.fromarray(np.asarray(frame[plane]).copy())


@dataclass
class PosterizeChain:
    """The nodes `vs_posterize_clip` built, keeping `gray` for the quality gate."""

    source: "VideoNode"
    gray: "VideoNode"
    posterized: "VideoNode"


def vs_posterize_clip(
    files: Sequence[PathLike],
    *,
    bits: int,
    method: int,
    prefetch: int = 0,
    debug: bool = False,
    cache_mb: int = 512,
    core: "Core | None" = None,
) -> PosterizeChain:
    """
    Read `files` and posterize them to `bits`, keeping the gray clip the gate compares to.

    `prefetch` is how many pages `imgseqs` decodes ahead of the frame being asked for, and
    `debug` turns on the plugin's log lines.
    """
    vs = get_vapoursynth()
    if core is None:
        core = vs.core

    core.max_cache_size = cache_mb
    source = core.imgseqs.Read(
        files=[str(path) for path in files],
        mismatch=True,
        prefetch=prefetch,
        debug=int(debug),
    )
    gray = vs_to_gray8(source, core=core)
    posterized = core.nimages.Posterize(gray, bits=bits, method=method, debug=int(debug))
    return PosterizeChain(source=source, gray=gray, posterized=posterized)


def vs_frame_to_grays(frame: "VideoFrame", core: "Core | None" = None) -> "VideoNode":
    """
    Return one frame as a one frame long float gray clip, which is what `vship` accepts.

    The clip is rebuilt at one page's own size because `vship` takes a constant size and one
    page at a time; handing it a clip holding pages of differing sizes fails inside its GPU
    backend. The plane is scaled by `1/255` because a VapourSynth float format holds `0..1`,
    not `0..255`.
    """
    vs = get_vapoursynth()
    np = get_numpy()
    if core is None:
        core = vs.core

    blank = core.std.BlankClip(width=frame.width, height=frame.height, format=vs.GRAYS, length=1)
    scaled = np.asarray(frame[0]).astype(np.float32) / 255.0

    def fill(n: int, f: "VideoFrame | list[VideoFrame]"):
        out = (f[n] if isinstance(f, list) else f).copy()
        np.asarray(out[0])[:] = scaled
        return out

    return core.std.ModifyFrame(blank, [blank], fill)

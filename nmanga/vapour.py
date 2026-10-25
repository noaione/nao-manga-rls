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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, cast

from PIL import Image

from .lazy import get_numpy, get_vapoursynth
from .ogsov import DetectedColor

if TYPE_CHECKING:
    import numpy as np
    from vapoursynth import Core, MessageType, PresetVideoFormat, VideoFormat, VideoFrame, VideoNode

    from .term import Console

    class AutolevelOptions(Protocol):
        """The subset of the `autolevel3` config that reaches the plugin."""

        upper_limit: int
        peak_min_pct: float | None
        peak_prom_pct: float | None
        peak_offset: int
        no_white: bool
        cache_mb: int


# `metric -> bits -> one score per page`, the shape the selection reduces over. Defined at run
# time rather than under `TYPE_CHECKING` because functions use it in their signatures.
ScoreMap = dict[str, dict[int, "np.ndarray[Any]"]]


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


def _subsampling_trim(width: int, height: int, frame_format: "VideoFormat") -> tuple[int, int]:
    """
    How many columns and rows to drop so both edges divide evenly by the subsampling.

    The factor is the format's own `1 << subsampling`, not a fixed two: 4:1:1 divides the
    width by four, so an even width that is not a multiple of four is still refused.
    """
    return width % (1 << frame_format.subsampling_w), height % (1 << frame_format.subsampling_h)


def _replicate_missing_edges(clip: "VideoNode", right: int, bottom: int, core: "Core") -> "VideoNode":
    """
    Grow `clip` by `right` columns and `bottom` rows, repeating its last column and row.

    `std.AddBorders` fills the added area with a colour, which for a trimmed page is a black
    line where the decoder could not hold the sample: the pixel that was cut is nearer to its
    neighbour than to black. The clip's own size has to be known, which it is once the
    subsampled frame has been cropped to edges the filters accept.
    """
    if right:
        column = core.std.CropAbs(clip, width=1, height=clip.height, left=clip.width - 1)
        clip = core.std.StackHorizontal([clip, column])
    if bottom:
        row = core.std.CropAbs(clip, width=clip.width, height=1, top=clip.height - 1)
        clip = core.std.StackVertical([clip, row])
    return clip


def vs_fix_odd_size_chain(n: int, *, source: "VideoNode", plain_rgb: "VideoNode") -> "VideoNode":
    vs = get_vapoursynth()
    frame = source.get_frame(n)  # this frame's format and size
    right, bottom = _subsampling_trim(frame.width, frame.height, frame.format)
    if not right and not bottom:
        return plain_rgb
    even = vs.core.std.CropAbs(source, width=frame.width - right, height=frame.height - bottom)
    rgb = vs.core.resize.Bicubic(even, format=vs.RGB24, matrix_in_s="470bg", range_in_s="limited")
    return _replicate_missing_edges(rgb, right, bottom, vs.core)


def vs_prepare_image_bulk_raw(images: Sequence[PathLike], *, debug: bool = False, prefetch: int = 6) -> "VideoNode":
    vs = get_vapoursynth()
    core = vs.core
    source = core.imgseqs.Read(files=[str(path) for path in images], mismatch=True, prefetch=prefetch, debug=debug)
    return source


def vs_prepare_image_bulk(images: Sequence[PathLike], *, debug: bool = False, prefetch: int = 6) -> "VideoNode":
    vs = get_vapoursynth()
    core = vs.core
    source = vs_prepare_image_bulk_raw(images, debug=debug, prefetch=prefetch)
    plain_rgb = core.resize.Bicubic(source, format=vs.RGB24, matrix_in_s="470bg", range_in_s="limited")
    rgb = core.std.FrameEval(source, partial(vs_fix_odd_size_chain, source=source, plain_rgb=plain_rgb))
    rgb = core.resize.Bicubic(rgb, format=vs.RGB24)  # FrameEval cannot promise a format
    return rgb


def vs_analyze_ogsov_frame(frame: "VideoFrame") -> tuple[DetectedColor, Path]:
    props = frame.props
    img_path = Path(str(props["ImgSeqPath"]))
    confidence = round(cast(float, props["_MangaColorConfidence"]))
    is_color = props["_MangaColor"] == 1
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


# The metrics `com.lumen.vship` exposes, the frame property each one reports, and how to read
# its number. `higher_is_better` is the direction of the number, not of the quality:
# `SSIMULACRA2` is a 0 to 100 score where 100 is identical, `BUTTERAUGLI` is a distance where 0
# is identical.
METRIC_NAMES: tuple[str, ...] = ("ssimulacra2", "butteraugli", "cvvdp")

# The display `CVVDP` models. The plugin's own default is `standard_fhd`; a scanned page is not
# a fixed display, so this is stated rather than inherited.
CVVDP_MODEL = "standard_4k"


def vs_metric_scores(node: "VideoNode", property_name: str) -> "np.ndarray[Any]":
    """
    Read one score per frame out of `node` in frame order.

    The frames are pulled one at a time, so the clip holding the scores is what stays resident
    and the result is the array the selection reduces over.
    """
    np = get_numpy()
    scores = np.empty(len(node), dtype=np.float64)
    for index in range(len(node)):
        with node.get_frame(index) as frame:
            scores[index] = cast(float, frame.props[property_name])
    return scores


def _read_ssimulacra2(node: "VideoNode") -> "np.ndarray[Any]":
    return vs_metric_scores(node, "_SSIMULACRA2")


def _read_butteraugli(node: "VideoNode") -> "np.ndarray[Any]":
    return vs_metric_scores(node, "_BUTTERAUGLI_QNorm")


def _read_cvvdp(node: "VideoNode") -> "np.ndarray[Any]":
    return vs_metric_scores(node, "_CVVDP")


@dataclass(frozen=True)
class MetricKind:
    """How to call one `vship` metric, read its score, and whether a larger number is better."""

    name: str
    call: Callable[["VideoNode", "VideoNode", "Core"], "VideoNode"]
    read: Callable[["VideoNode"], "np.ndarray[Any]"]
    higher_is_better: bool
    default_minimum: float | None


def _call_butteraugli(reference: "VideoNode", distorted: "VideoNode", core: "Core") -> "VideoNode":
    # The quality norm, which is the one that scales with the amount of degradation rather than
    # with the worst single pixel. `numStream=1` because several metric nodes are alive at once
    # during a chunked run and each stream costs several frame buffers of VRAM.
    return reference.vship.BUTTERAUGLI(distorted, numStream=1)


def _call_cvvdp(reference: "VideoNode", distorted: "VideoNode", core: "Core") -> "VideoNode":
    # CVVDP accumulates its score along the clip by default, so frame `i` would report the
    # sequence from frame 0 to frame `i` rather than the page. `disableTemporal` is what turns
    # it into a per frame sensitivity score, which is what independent manga pages can use.
    return reference.vship.CVVDP(distorted, model_name=CVVDP_MODEL, disableTemporal=1)


METRIC_KINDS: Mapping[str, MetricKind] = {
    "ssimulacra2": MetricKind(
        name="ssimulacra2",
        call=lambda reference, distorted, core: reference.vship.SSIMULACRA2(distorted, numStream=1),
        read=_read_ssimulacra2,
        higher_is_better=True,
        default_minimum=80.0,
    ),
    "butteraugli": MetricKind(
        name="butteraugli",
        call=_call_butteraugli,
        read=_read_butteraugli,
        higher_is_better=False,
        default_minimum=1.0,
    ),
    # CVVDP is a 0 to 10 quality score and higher is better: a page scored against itself is
    # exactly 10, and degrading it lowers the score. That anchor settles the direction, and it is
    # the opposite of what the first probe of this metric suggested -- which is why `<=` was
    # wired here once and a page then chose 1 bit through a floor of 9.9.
    #
    # 9.9 is the depth at which the score stops being meaningfully short of a perfect 10: on
    # generated line art the ladder is monotone and 9.9 asks for five bits (9.81 at four, 9.95 at
    # five). A real page's ladder can sit lower throughout, so a floor every candidate fails is
    # not a constraint at all -- check `--metric-min cvvdp=<value>` against a volume before
    # trusting the default on it.
    "cvvdp": MetricKind(
        name="cvvdp",
        call=_call_cvvdp,
        read=_read_cvvdp,
        higher_is_better=True,
        default_minimum=9.9,
    ),
}
METRIC_HIGHER_IS_BETTER: Mapping[str, bool] = {name: kind.higher_is_better for name, kind in METRIC_KINDS.items()}
# The floors a metric is allowed to fail a page with when the caller names no threshold.
METRIC_DEFAULT_MINIMUM: Mapping[str, float] = {
    "ssimulacra2": 80.0,
    "butteraugli": 1.0,
    "cvvdp": 9.9,
}


def is_known_metric(metric: str) -> bool:
    """Whether `metric` is one of the metrics `com.lumen.vship` exposes."""
    return metric in METRIC_KINDS


def vs_metric_clip(
    reference: "VideoNode",
    distorted: "VideoNode",
    metric: str,
    *,
    core: "Core | None" = None,
) -> "VideoNode":
    """
    Score `distorted` against `reference` with `metric`, for every frame in one call.

    The returned clip holds one frame per input frame, carrying the score as a frame property.
    This is the whole point of the fan-out: a volume's scores arrive in one dispatch rather than
    one dispatch per page.
    """
    vs = get_vapoursynth()
    if core is None:
        core = vs.core

    kind = METRIC_KINDS.get(metric)
    if kind is None:
        raise ValueError(f"Unknown metric: {metric}")
    return kind.call(reference, distorted, core)


def vs_find_missing_plugins(plugins: str | list[str]) -> list[str]:
    plugins_set = set(plugins) if isinstance(plugins, list) else {plugins}

    core = get_vapoursynth().core
    identifiers = {p.identifier for p in core.plugins()}

    # return missing plugins
    return list(plugins_set - identifiers)


def _trim_and_convert(
    clip: "VideoNode", width: int, height: int, frame_format: "VideoFormat", core: "Core"
) -> "VideoNode":
    """
    Convert a subsampled clip to a constant `GRAY8` through `RGB24`.

    An odd width or height cannot be 4:2:0, and every filter but the decoder refuses it, so an
    odd edge is trimmed before the conversion and padded back after it.

    `imgseqs` 0.3 cannot hand an odd subsampled frame out at all: it refuses the size when
    `Read` builds its video info, and a mixed clip aborts at frame creation. The trim is
    therefore a guard for a decoder that can hand one out, not a path a webp takes today;
    the per-frame probe in :func:`vs_to_gray8` still has to ask each frame because a mixed
    clip's format is undefined.

    `frame_format` is the format of the frame at `width`/`height`, not `clip.format`: a
    `mismatch` `imgseqs` clip reports an undefined format whose subsampling is always zero,
    so reading the clip's format there trims nothing and hands the odd frame to `resize`.
    """
    vs = get_vapoursynth()

    right, bottom = _subsampling_trim(width, height, frame_format)
    region = clip
    if right or bottom:
        region = core.std.CropAbs(clip, width=width - right, height=height - bottom)

    # No matrix or range arguments: the frame properties carry both and are what makes
    # this round trip agree with a Pillow convert("L").
    rgb = cast(Any, core).resize.Bicubic(region, format=vs.RGB24)
    gray = cast(Any, core).resize.Bicubic(rgb, format=vs.GRAY8, matrix_s="470bg", range_s="full")
    if right or bottom:
        gray = _replicate_missing_edges(gray, right, bottom, core)
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
        return _trim_and_convert(source, source.width, source.height, source.format, core)

    # The clip varies, so this node reports an undefined format and cannot say whether its
    # frames are RGB. One frame cannot answer that for the rest either: a colour page first
    # would send a later webp down the plain resize, which refuses its odd 4:2:0 size.
    # Every frame is asked what it is, and the plain resize is only ever returned for RGB.
    plain_gray = cast(Any, core).resize.Bicubic(source, format=vs.GRAY8, matrix_s="470bg", range_s="full")

    def convert(n: int = 0, **_):
        frame = source.get_frame(n)
        if frame.format.color_family == vs.RGB:
            return plain_gray
        return _trim_and_convert(source, frame.width, frame.height, frame.format, core)

    gray = core.std.FrameEval(source, convert)
    return cast(Any, core).resize.Bicubic(gray, format=vs.GRAY8)


@dataclass
class AutolevelChain:
    """The nodes `vs_autolevel_clip` built, kept so a caller can pull frames from them."""

    source: "VideoNode"
    analysis: "VideoNode"
    stats: "VideoNode"
    leveled: "VideoNode"


def vs_autolevel_clip(
    files: Sequence[PathLike],
    *,
    options: "AutolevelOptions",
    gray: bool = False,
    gpu: bool = False,
    prefetch: int = 0,
    debug: bool = False,
    core: "Core | None" = None,
) -> AutolevelChain:
    """
    Read `files` and build the `PeakStats` to `Levels` chain over their analysis form.

    `analysis` is what `PeakStats` reads, and it is the source's own frames unless `gray` is
    set: `nimages` takes any colour family on a clip whose format varies per frame, reads an
    `RGB24` page through the reference's fixed-point luma (Pillow's `convert("L")` exactly),
    and counts plane 0 of a gray or YUV page. No normalisation is needed before the counter.
    `gray=True` is the `--gray` output mode, where the levelled clip itself has to be one plane
    per page and the pages are therefore converted first.

    `leveled` is the page the write stage stores: the analysed page with the curve applied, in
    each frame's own colour family. A colour page is levelled on its own planes and a gray page
    stays gray, which is the whole of the old `--keep-colorspace` behaviour, by construction;
    `PNGWrite` then stores an `RGB24` still as RGB and converts a YUV still to RGB with the
    frame's matrix and range. Pull the frame you want from `leveled` and read the level
    properties off it, rather than pulling from `stats` and then `leveled`, which analyses
    every page twice.

    `gpu` passes `gpu=1` to the counter and the map, and then downloads the levelled frame
    because every consumer of it takes host frames. It needs `vapoursynth-nimages` 0.2.1 or
    `vapoursynth-nimages` 0.2.1 or newer and a Vulkan 1.4 device; the plugin refuses at
    node creation otherwise rather than quietly running the host path.

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

    stats_args: dict[str, Any] = {
        "upper_limit": options.upper_limit,
        "peak_percentage": options.peak_min_pct,
        "skip_white": 1 if options.no_white else 0,
        "debug": int(debug),
        "gpu": int(gpu),
    }
    # `None` is not a value a VapourSynth argument list can carry, so the argument has to
    # be left off entirely to leave prominence disabled.
    if options.peak_prom_pct is not None:
        stats_args["peak_prominence"] = options.peak_prom_pct
    analysis = vs_to_gray8(source, core=core) if gray else source

    stats = core.nimages.PeakStats(analysis, **stats_args)
    leveled = core.nimages.Levels(
        stats,
        use_props=True,
        peak_offset=options.peak_offset,
        auto_gamma=True,
        debug=int(debug),
        gpu=int(gpu),
    )
    if gpu:
        # One download at the end of the device chain: every consumer of `leveled` is host.
        leveled = core.std.GPUDownload(leveled)
    return AutolevelChain(source=source, analysis=analysis, stats=stats, leveled=leveled)


def vs_frame_to_image(frame: "VideoFrame") -> Image.Image:
    """
    Return one frame as a Pillow image.

    A single plane frame becomes an `L` image and a three plane frame has its planes
    stacked back into `RGB`, so the caller has no family branch of its own. The frame has
    to be an `RGB24` one by the time it gets here, which is the caller's job: stacking a YUV
    frame would label its luma and chroma planes as red, green and blue.

    `np.stack` copies, so the frame can be released at the end of the caller's `with`
    block. The one plane path aliases the frame with `np.asarray` and copies for the same
    reason.
    """
    np = get_numpy()
    if frame.format.num_planes == 1:
        return Image.fromarray(np.asarray(frame[0]).copy())
    planes = [np.asarray(frame[plane]) for plane in range(3)]
    return Image.fromarray(np.stack(planes, axis=-1), "RGB")


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
    gpu: bool = False,
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
    posterized = core.nimages.Posterize(gray, bits=bits, method=method, debug=int(debug), gpu=int(gpu))
    if gpu:
        # The fixed depth path writes with `PNGWrite`, and the orchestrator gate reads the
        # frame with `vship`; both take host frames.
        posterized = core.std.GPUDownload(posterized)
    return PosterizeChain(source=source, gray=gray, posterized=posterized)


@dataclass
class DeblurChain:
    """The nodes `vs_deblur_clip` built, kept at the source's own colour."""

    source: "VideoNode"
    deblurred: "VideoNode"


def vs_deblur_clip(
    files: Sequence[PathLike],
    *,
    method: int,
    radius: float,
    strength: float | None,
    iterations: int,
    threshold: float,
    overshoot: float,
    prefetch: int = 0,
    debug: bool = False,
    cache_mb: int = 512,
    core: "Core | None" = None,
) -> DeblurChain:
    """
    Read `files` and sharpen them in place, keeping the clip at the source's own colour.

    This is the one chain here that does not normalise to gray. `Deblur` moves the luma alone and
    carries the change onto R, G and B as an equal offset, which is what `deblur_deconv` does;
    converting first would discard the colour the filter exists to preserve.

    `strength` is left off the argument list when it is `None`, because the plugin resolves the
    per method default (0.65 for the deconvolution, 0.85 for the unsharp mask) and a VapourSynth
    argument list cannot carry `None`.

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

    args: dict[str, Any] = {
        "method": method,
        "radius": radius,
        "iterations": iterations,
        "threshold": threshold,
        "overshoot": overshoot,
        "debug": int(debug),
    }
    if strength is not None:
        args["strength"] = strength

    deblurred = core.nimages.Deblur(source, **args)
    return DeblurChain(source=source, deblurred=deblurred)


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


def vs_gray_plane(grays: "VideoNode", core: "Core | None" = None) -> "VideoNode":
    """
    Take a float gray clip down to the single plane `vship` expects.

    A one page gray clip from `imgseqs` is already one plane, so the common case costs nothing
    and the clip is handed straight back. A clip whose format is undefined (a sequence mixing
    a gray container with a colour one) is resolved to `GRAY` so the metric node has a plane
    to read, which is the same thing `imgseqs` would have handed out for the gray pages.
    """
    vs = get_vapoursynth()
    if core is None:
        core = vs.core

    if grays.format.color_family == vs.GRAY and grays.format.num_planes == 1:
        return grays

    return core.std.ShufflePlanes(grays, planes=0, colorfamily=vs.GRAY)


def vs_clip_to_grays(
    grays: "VideoNode", *, size: tuple[int, int] | None = None, core: "Core | None" = None
) -> "VideoNode":
    """
    Return a whole constant `GRAY8` clip in the float form `vship` accepts.

    This is the clip equivalent of :func:`vs_frame_to_grays`: the same bytes, the same `1/255`
    scaling into `0..1`, and the same per page size, but for every page in one node. That
    equivalence is what makes a clip score comparable to the per page score the quality gate
    already computes, so it is the one property worth pinning down with a test.

    `range_s="full"` is not optional. `resize` without it treats the input as limited range and
    rewrites the values, which produces a score in the thousands rather than a real one.

    `size` pins the output dimensions. A clip mixing containers reports a size of zero, which
    `vship` rejects with an invalid argument, so a caller that knows the page size states it
    and the clip's video info carries it.
    """
    vs = get_vapoursynth()
    if core is None:
        core = vs.core

    plane = vs_gray_plane(grays, core=core)
    if size is None:
        return cast(Any, core).resize.Point(plane, format=vs.GRAYS, range_s="full")
    width, height = size
    return cast(Any, core).resize.Point(plane, width=width, height=height, format=vs.GRAYS, range_s="full")


def vs_grays_frame_to_image(frame: "VideoFrame") -> Image.Image:
    """
    Return one page of a float gray clip as an 8 bit Pillow image.

    The write path wants `GRAY8` bytes, and the candidate the metric scored is a float clip, so
    the `1/255` scaling has to be undone here rather than by rebuilding the page's clip.
    """
    np = get_numpy()
    scaled = np.asarray(frame[0])
    return Image.fromarray(np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8))


def vs_page_sizes(files: Sequence[PathLike], *, core: "Core | None" = None) -> list[tuple[int, int]]:
    """
    Return `(width, height)` for every page in `files`.

    `vship` takes one size per call, so a volume whose pages are not all the same size has to
    be grouped before any candidate exists. The sizes are read once here rather than discovered
    inside the chunk loop, and each frame is released again so nothing is pinned.
    """
    vs = get_vapoursynth()
    if core is None:
        core = vs.core

    source = core.imgseqs.Read(files=[str(path) for path in files], mismatch=True, prefetch=0)
    sizes: list[tuple[int, int]] = []
    for index in range(len(files)):
        with source.get_frame(index) as frame:
            sizes.append((frame.width, frame.height))
    return sizes


@dataclass
class PosterizeGroup:
    """
    The candidate branches for one page size, and the reference they are scored against.

    `branches` are the float gray forms `vship` scores; `raw` are the eight bit `Posterize`
    outputs behind them, which is the frame the write stage stores. Keeping both means the
    selected page is written from the plugin's own bytes instead of a float round trip.
    """

    size: tuple[int, int]
    gray: "VideoNode"
    reference: "VideoNode"
    branches: dict[int, "VideoNode"]
    raw: dict[int, "VideoNode"]


def vs_posterize_group(
    files: Sequence[PathLike],
    *,
    bits_candidates: Sequence[int],
    method: int,
    prefetch: int = 0,
    debug: bool = False,
    cache_mb: int = 512,
    gpu: bool = False,
    core: "Core | None" = None,
) -> PosterizeGroup:
    """
    Read `files` once and hang one `Posterize` branch per candidate bit depth off the gray clip.

    Every branch shares `files`' single decode: the reference and each candidate are branches of
    the same `imgseqs.Read`, so the posterize passes are the only extra work per candidate.

    `files` has to hold pages of one size. `vship` takes a constant size, so a volume of mixed
    sizes is split by the caller and each group is built on its own. `gray` is kept because it
    is the posterize input, and the size is read off its first page: a clip mixing containers
    reports no size of its own. The scored clips are the two float forms, which is why the
    value written back is the branch's raw `GRAY8` frame.
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
    # `gray.width` is zero when the group mixes containers, so the size is read from the page
    # itself. The caller groups by size, so its first page speaks for the whole group, and the
    # float clips are built at that size because `vship` cannot take an undefined one.
    with gray.get_frame(0) as first:
        size = (first.width, first.height)
    reference = vs_clip_to_grays(gray, size=size, core=core)
    raw = {
        bits: core.nimages.Posterize(gray, bits=bits, method=method, debug=int(debug), gpu=int(gpu))
        for bits in bits_candidates
    }
    if gpu:
        # One download per candidate per page: the metric and the writer both read the host
        # frame, and the writer refuses a device one.
        raw = {bits: core.std.GPUDownload(node) for bits, node in raw.items()}
    branches = {bits: vs_clip_to_grays(node, size=size, core=core) for bits, node in raw.items()}
    return PosterizeGroup(size=size, gray=gray, reference=reference, branches=branches, raw=raw)


def vs_score_candidates(
    group: PosterizeGroup,
    *,
    metrics: Sequence[str],
    core: "Core | None" = None,
) -> ScoreMap:
    """
    Score every candidate in `group` with every metric, one dispatch per candidate per metric.

    Returns `metric -> bits -> one score per page`, with the page axis in the order the pages
    were handed to :func:`vs_posterize_group`. A 200 page volume of 3 candidates and 2 metrics
    costs 6 dispatches here rather than 200.
    """
    vs = get_vapoursynth()
    if core is None:
        core = vs.core

    scores: ScoreMap = {}
    for metric in metrics:
        kind = METRIC_KINDS[metric]
        per_bits: dict[int, "np.ndarray[Any]"] = {}
        for bits, branch in group.branches.items():
            node = vs_metric_clip(group.reference, branch, metric, core=core)
            per_bits[bits] = kind.read(node)
        scores[metric] = per_bits
    return scores


def vs_select_posterize_bits(
    scores: ScoreMap,
    *,
    candidates: Sequence[int],
    minimums: Mapping[str, float],
    order: Sequence[int] | None = None,
) -> "np.ndarray[Any]":
    """
    Reduce the score arrays to one chosen bit depth per page.

    `scores` is `metric -> bits -> pages` and the result is one entry per page. A page whose
    entry is -1 has no candidate that every thresholded metric accepts: the caller writes the
    source page instead, which is the fallback the single depth gate already takes.

    Only metrics with an entry in `minimums` constrain the choice. A metric that was measured
    without a settled threshold is reported but never picks a depth, so the outcome for a page
    is `min(bits that pass)` over the metrics that do decide.

    `order` is the caller's preference order and only breaks ties: of the candidates that pass,
    the one that comes first in `order` wins. Every candidate is scored either way, so the
    order can never hide a passing depth.
    """
    np = get_numpy()

    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("At least one candidate bit depth is required")

    pages = len(next(iter(next(iter(scores.values())).values())))
    order_list = list(order) if order is not None else candidate_list
    ranking = {bits: position for position, bits in enumerate(order_list)}
    # Preference first, then ascending bits, so a candidate the caller ranked and one they did
    # not are still ordered against each other.
    ranked = sorted(candidate_list, key=lambda bits: (ranking.get(bits, len(ranking)), bits))

    passed = np.ones((len(candidate_list), pages), dtype=bool)
    for metric, minimum in minimums.items():
        per_bits = scores.get(metric)
        if per_bits is None:
            raise ValueError(f"No scores for the {metric} metric")
        higher_is_better = METRIC_HIGHER_IS_BETTER[metric]
        for row, bits in enumerate(candidate_list):
            values = per_bits[bits]
            passed[row] &= values >= minimum if higher_is_better else values <= minimum

    chosen = np.full(pages, -1, dtype=np.int64)
    # Least preferred first, so the final write on a passing page is the most preferred.
    for bits in reversed(ranked):
        chosen[passed[candidate_list.index(bits)]] = bits
    return chosen


T = TypeVar("T")


def vs_png_writer(
    clip: "VideoNode",
    output_dir: Path,
    *,
    compression: int = 6,
    overwrite: bool = True,
    depth: int | None = None,
    debug: bool = False,
    core: "Core | None" = None,
) -> "VideoNode":
    """
    One `PNGWrite` node over `clip`, naming each still from its `ImgSeqPath`.

    The node has the same frame count as `clip`; its output frame `n` is the input frame `n`
    plus the write. Build it once over the chain's final clip, never once per page: a
    per page writer forbids the plugin's `Parallel` completion and puts the encode back on a
    Python thread.

    `output_dir` is the destination, not a name template: a still is written as
    `<stem>.png` from the frame's own `ImgSeqPath`, so the destination is known before the
    request and no rename follows. The directory is created when the node is built, and no
    PNG is written until a frame is requested. Points it at the real output directory and
    the encode and the publish both leave Python.

    `overwrite` is a permission, not a skip: `False` refuses an existing destination and
    raises, so a caller that skips does the existence check before submitting instead of
    asking the writer to leave the file alone.

    `depth` is left to the plugin when it is `None` (eight bits for an eight bit frame and
    sixteen for anything wider).
    """
    vs = get_vapoursynth()
    if core is None:
        core = vs.core
    options = {"depth": depth} if depth is not None else {}
    return core.imgseqs.PNGWrite(
        clip,
        output_dir=str(output_dir),
        compression=compression,
        overwrite=overwrite,
        debug=int(debug),
        **options,  # pyright: ignore[reportArgumentType]
    )


@dataclass
class PageWrite(Generic[T]):
    """One open `PNGWrite` request and the result to report once it has landed."""

    index: int
    outcome: T
    future: Any

    def result(self) -> T:
        # Opening the future is what surfaces the writer's path-qualified error; the frame it
        # returns is not needed, so it is closed immediately. `PNGWrite` has already resolved
        # the destination from the frame and published it by the time this returns.
        with self.future.result():
            pass
        return self.outcome

    def discard(self) -> None:
        """Wait for an open request without reporting it; the file may still have landed."""
        try:
            with self.future.result():
                pass
        except Exception:  # ruff: ignore[try-except-pass]
            # The original exception is about to win, and a request cannot be cancelled; this
            # wait only exists so no request outlives the window.
            pass


class PngWriteWindow:
    """The bounded replacement for `BoundedWritePool` on a `PNGWrite` node."""

    def __init__(self, depth: int):
        self.depth = max(depth, 1)
        self._pending: list[PageWrite[Any]] = []

    def submit(self, writer: "VideoNode", index: int, outcome: T) -> PageWrite[T] | None:
        """Request one page; hand back the oldest when the window is full."""
        page: PageWrite[T] = PageWrite(
            index=index,
            outcome=outcome,
            future=writer.get_frame_async(index),
        )
        self._pending.append(page)
        if len(self._pending) >= self.depth:
            return self._pending.pop(0)
        return None

    def pending(self) -> list[PageWrite[Any]]:
        """Take the open requests, oldest first, leaving the window empty."""
        pending, self._pending = self._pending, []
        return pending

    def __enter__(self) -> "PngWriteWindow":
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            # A request cannot be cancelled. Wait for the outstanding ones without reporting
            # them, so the original exception is what reaches the user and no native call
            # outlives the window.
            for page in self._pending:
                page.discard()
            self._pending.clear()
        elif self._pending:
            # A clean exit must have drained the window; a leftover is a bug, and reporting it
            # here would advance the progress bar after the loop that owns it.
            raise RuntimeError("PngWriteWindow exited with requests still open")
        return False


def vs_duplicate_png_stems(files: Sequence[PathLike]) -> dict[str, list[Path]]:
    """
    Return the source stems that would resolve to one `{stem}.png` destination.

    `output_dir` names a still from its source's stem, and the writer only refuses two frames
    that land on one file from inside its own instance, after half the volume may already be
    written. Two pages like `page.jpg` and `page.png` in one folder are the realistic case, and
    one writer per bit depth (the `posterize2` fan-out) could not see across instances even if it
    did, so the check is made once for the whole volume before anything is built.
    """
    by_stem: dict[str, list[Path]] = {}
    for path in files:
        page = Path(path)
        by_stem.setdefault(page.stem, []).append(page)
    return {stem: pages for stem, pages in by_stem.items() if len(pages) > 1}

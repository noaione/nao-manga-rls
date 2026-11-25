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

# Shared tile grid planning, variable edge sizing, and feather-ramp calculation.

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = (
    "AxisSpan",
    "TileBox",
    "TilingMode",
    "TilingPolicy",
    "default_expand_tolerance",
    "feather_ramp_values",
    "group_tiles_by_shape",
    "plan_axis",
    "plan_tiles",
)

TilingMode = Literal["fixed", "variable"]
"""Whether partial edge tiles are padded to `tile_size` or shrunk/expanded to fit."""


@dataclass(frozen=True)
class TilingPolicy:
    """How a tile grid handles partial edge tiles and alignment."""

    mode: TilingMode = "fixed"
    """`fixed` pads every tile to `tile_size`; `variable` shrinks or expands edge tiles."""

    expand_tolerance: int = 0
    """Max extra pixels beyond `tile_size` a penultimate tile may absorb from a small sliver."""

    bucket_step: int = 0
    """Round variable edge sizes up to a multiple of this step (`0` disables bucketing)."""

    alignment: int = 1
    """Spatial multiple the model requires for variable tile sizes."""

    min_tile_size: int = 1
    """Minimum spatial input dimension the model accepts."""

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "variable"):
            raise ValueError(f"Unknown tiling mode {self.mode!r}; expected 'fixed' or 'variable'.")
        if self.expand_tolerance < 0:
            raise ValueError("Expand tolerance must be non-negative.")
        if self.bucket_step < 0:
            raise ValueError("Bucket step must be non-negative.")
        if self.alignment < 1:
            raise ValueError("Alignment must be at least 1.")
        if self.min_tile_size < 1:
            raise ValueError("Minimum tile size must be at least 1.")


@dataclass(frozen=True)
class AxisSpan:
    """One tile's 1D span along a single image axis in input-pixel units."""

    dst_start: int
    """Start coordinate of the valid output region (before multiplying by scale)."""

    valid_len: int
    """Length of the valid output region (before multiplying by scale)."""

    in_start: int
    """Start coordinate of the model input window (shifted left by `context`)."""

    run_len: int
    """Input length fed to the model (`>= valid_len + 2 * context`)."""

    has_prev: bool
    """Whether an earlier tile overlaps this span on the leading edge."""

    has_next: bool
    """Whether a later tile overlaps this span on the trailing edge."""


@dataclass(frozen=True)
class TileBox:
    """One 2D tile's source crop, model input padding, destination slice, and overlap flags."""

    src_x: int
    src_y: int
    src_w: int
    src_h: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    run_w: int
    run_h: int
    dst_x: int
    dst_y: int
    valid_w: int
    valid_h: int
    has_left: bool
    has_right: bool
    has_top: bool
    has_bottom: bool


def default_expand_tolerance(tile_size: int) -> int:
    """Choose a conservative sliver-absorption limit relative to `tile_size`."""
    return min(64, max(0, tile_size // 8))


def _align_up(value: int, step: int) -> int:
    """Round `value` up to the nearest positive multiple of `step`."""
    if step <= 1:
        return value
    return ((value + step - 1) // step) * step


def _snap_variable_length(needed: int, *, tile_size: int, policy: TilingPolicy) -> int:
    """Compute the aligned/bucketed model input size for a variable edge tile."""
    target = max(policy.min_tile_size, needed)
    if needed <= tile_size and policy.bucket_step > 1:
        snapped = _align_up(target, policy.bucket_step)
        if tile_size % policy.alignment == 0:
            snapped = min(snapped, tile_size)
        target = snapped
    elif needed > tile_size and policy.bucket_step > 1:
        target = _align_up(target, policy.bucket_step)
    return _align_up(target, policy.alignment)


def plan_axis(
    length: int,
    *,
    tile_size: int,
    tile_overlap: int = 0,
    context: int = 0,
    policy: TilingPolicy | None = None,
) -> list[AxisSpan]:
    """
    Partition a 1D axis of `length` pixels into tile spans.

    :param length: Image width or height in pixels
    :param tile_size: Nominal model input tile size
    :param tile_overlap: Total overlap shared between neighbouring output tiles
    :param context: Symmetric border trimmed from each output tile
    :param policy: Fixed or variable tiling policy, defaults to fixed
    :return: Ordered spans covering ``[0, length)``
    :raises ValueError: If the grid geometry is invalid
    """
    if length < 1:
        raise ValueError("Axis length must be at least 1.")
    if tile_size < 1:
        raise ValueError("Tile size must be at least 1.")
    if context < 0 or context * 2 >= tile_size:
        raise ValueError("Context must be non-negative and smaller than half the tile size.")

    core_tile_size = tile_size - context * 2
    if tile_overlap < 0 or tile_overlap >= core_tile_size:
        raise ValueError("Tile overlap must be non-negative and smaller than the output tile size.")

    resolved = policy or TilingPolicy()
    step = core_tile_size - tile_overlap

    if length <= core_tile_size:
        needed = length + 2 * context
        if resolved.mode == "fixed":
            run_len = tile_size
        else:
            run_len = _snap_variable_length(needed, tile_size=tile_size, policy=resolved)
        return [
            AxisSpan(
                dst_start=0,
                valid_len=length,
                in_start=-context,
                run_len=run_len,
                has_prev=False,
                has_next=False,
            )
        ]

    count = max(1, -(-(length - tile_overlap) // step))
    if count >= 2 and resolved.mode == "variable" and resolved.expand_tolerance > 0:
        penult_end = (count - 2) * step + core_tile_size
        tail_new = length - penult_end
        if 0 < tail_new <= resolved.expand_tolerance:
            count -= 1

    spans: list[AxisSpan] = []
    for index in range(count):
        dst_start = index * step
        valid_len = (length - dst_start) if (index == count - 1) else core_tile_size
        needed = valid_len + 2 * context
        if resolved.mode == "fixed":
            run_len = tile_size
        elif needed == tile_size:
            run_len = _align_up(max(resolved.min_tile_size, tile_size), resolved.alignment)
        else:
            run_len = _snap_variable_length(needed, tile_size=tile_size, policy=resolved)
        spans.append(
            AxisSpan(
                dst_start=dst_start,
                valid_len=valid_len,
                in_start=dst_start - context,
                run_len=run_len,
                has_prev=index > 0,
                has_next=index + 1 < count,
            )
        )
    return spans


def plan_tiles(
    width: int,
    height: int,
    *,
    tile_size: int,
    tile_overlap: int = 0,
    context: int = 0,
    policy: TilingPolicy | None = None,
) -> list[TileBox]:
    """
    Build the 2D tile grid in row-major order for a ``width x height`` image.

    :param width: Source image width in pixels
    :param height: Source image height in pixels
    :param tile_size: Nominal model input tile size
    :param tile_overlap: Total overlap shared between neighbouring output tiles
    :param context: Symmetric border trimmed from each output tile
    :param policy: Fixed or variable tiling policy, defaults to fixed
    :return: Ordered tile boxes covering the full image
    """
    x_spans = plan_axis(width, tile_size=tile_size, tile_overlap=tile_overlap, context=context, policy=policy)
    y_spans = plan_axis(height, tile_size=tile_size, tile_overlap=tile_overlap, context=context, policy=policy)

    tiles: list[TileBox] = []
    for y_span in y_spans:
        src_y = max(0, y_span.in_start)
        src_y_end = min(height, y_span.in_start + y_span.run_len)
        src_h = src_y_end - src_y
        pad_top = max(0, -y_span.in_start)
        pad_bottom = y_span.run_len - pad_top - src_h

        for x_span in x_spans:
            src_x = max(0, x_span.in_start)
            src_x_end = min(width, x_span.in_start + x_span.run_len)
            src_w = src_x_end - src_x
            pad_left = max(0, -x_span.in_start)
            pad_right = x_span.run_len - pad_left - src_w

            tiles.append(
                TileBox(
                    src_x=src_x,
                    src_y=src_y,
                    src_w=src_w,
                    src_h=src_h,
                    pad_left=pad_left,
                    pad_top=pad_top,
                    pad_right=pad_right,
                    pad_bottom=pad_bottom,
                    run_w=x_span.run_len,
                    run_h=y_span.run_len,
                    dst_x=x_span.dst_start,
                    dst_y=y_span.dst_start,
                    valid_w=x_span.valid_len,
                    valid_h=y_span.valid_len,
                    has_left=x_span.has_prev,
                    has_right=x_span.has_next,
                    has_top=y_span.has_prev,
                    has_bottom=y_span.has_next,
                )
            )
    return tiles


def group_tiles_by_shape(
    tiles: Sequence[TileBox],
    *,
    primary_shape: tuple[int, int] | None = None,
) -> list[tuple[tuple[int, int], list[TileBox]]]:
    """
    Group tiles by their ``(run_h, run_w)`` model input shape, keeping the primary shape first.

    :param tiles: Planned tile boxes
    :param primary_shape: Optional ``(tile_size, tile_size)`` shape to place first when present
    :return: Ordered ``((run_h, run_w), group)`` pairs
    """
    buckets: dict[tuple[int, int], list[TileBox]] = defaultdict(list)
    for tile in tiles:
        buckets[tile.run_h, tile.run_w].append(tile)

    ordered: list[tuple[tuple[int, int], list[TileBox]]] = []
    if primary_shape is not None and primary_shape in buckets:
        ordered.append((primary_shape, buckets.pop(primary_shape)))
    for shape, group in buckets.items():
        ordered.append((shape, group))
    return ordered


def feather_ramp_values(
    valid_len: int,
    *,
    overlap: int,
    has_prev: bool,
    has_next: bool,
) -> list[float]:
    """
    Compute 1D linear feather weights so overlapping neighbours sum to one.

    :param valid_len: Output length along this axis in output pixels
    :param overlap: Overlap width in output pixels
    :param has_prev: Taper up along the leading edge
    :param has_next: Taper down along the trailing edge
    :return: Weights of length `valid_len`
    """
    weights = [1.0] * valid_len
    if overlap <= 0 or (not has_prev and not has_next):
        return weights
    ramp_len = min(overlap, valid_len)
    denom = float(ramp_len + 1)
    if has_prev:
        for i in range(ramp_len):
            weights[i] *= (i + 1) / denom
    if has_next:
        offset = valid_len - ramp_len
        for i in range(ramp_len):
            weights[offset + i] *= (ramp_len - i) / denom
    return weights

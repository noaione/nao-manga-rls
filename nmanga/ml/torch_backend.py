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

# Fixed-tile inference and CPU stitching for the PyTorch backend.

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..lazy import get_torch
from .contracts import TilingConfig

if TYPE_CHECKING:
    import torch

__all__ = (
    "TileRunner",
    "infer_tensor",
)

TileRunner = Callable[["torch.Tensor"], "torch.Tensor"]
"""One prediction for a fixed-size tile: a CPU NCHW RGB tile in, a scaled CPU NCHW RGB tile out."""

_CHANNELS = 3
"""The public channel contract of the models this tiler runs: RGB."""


def infer_tensor(
    tensor: "torch.Tensor",
    runner: TileRunner,
    *,
    scale: int,
    config: TilingConfig,
) -> "torch.Tensor":
    """
    Run `runner` over the torch tile grid and stitch the predictions in host memory.

    Every tile is replicate-padded to the configured fixed tile size before it is handed to
    `runner`, so a model that depends on the whole tile always sees the same shape and context.
    The scaled prediction is cropped back to the part of the tile that exists in the source
    image, feathered with linear ramps towards an interior neighbour, and accumulated as FP32 in
    CPU RAM. Only one prediction is alive at a time: nothing is retained on the device, so GPU
    memory does not grow with the page size.

    Parameters
    ----------
    tensor:
        One RGB image in NCHW format, ``1 x 3 x H x W``. It is detached and copied to CPU, and is
        never modified.
    runner:
        Any callable that takes a CPU NCHW tile of ``1 x 3 x tile_size x tile_size`` and returns
        a CPU NCHW tensor of ``1 x 3 x (tile_size * scale) x (tile_size * scale)``. It does not
        have to be a torch module or an ONNX session; :meth:`nmanga.ml.torch_session.TorchRuntime.predict`
        satisfies it. A callable that exposes a different ``input_size`` attribute is rejected,
        because the grid can only pad to the configured tile size.
    scale:
        The integer upscale factor of the model. Every prediction is expected to be scaled by it.
    config:
        The tile grid. The stride is ``tile_size - tile_overlap``: `tile_overlap` is the *total*
        overlap between two neighbouring tiles, not the amount added to each side.

    Returns
    -------
    torch.Tensor
        A FP32 CPU tensor of ``1 x 3 x (H * scale) x (W * scale)`` holding raw, unclipped
        values. Clipping and quantisation belong to the image pipeline, not to the stitcher.

    Raises
    ------
    ValueError
        When `scale`, the tensor shape, or a prediction violates the contract.

    Notes
    -----
    The tile size and overlap are part of the output contract of a tile-context model. Changing
    them changes the prediction itself and must not be reported as a pure speedup.
    """
    torch_module = get_torch()

    if scale < 1:
        raise ValueError("Scale must be at least 1.")
    if config.tile_size < 1:
        raise ValueError("Tile size must be at least 1.")
    if not 0 <= config.tile_overlap < config.tile_size:
        raise ValueError("Tile overlap must be non-negative and smaller than the tile size.")
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != _CHANNELS:
        raise ValueError("Expected one RGB image in NCHW format")

    tile = config.tile_size
    overlap = config.tile_overlap
    step = config.stride

    source = tensor.detach().cpu()
    height, width = source.shape[-2:]
    scaled_height, scaled_width = height * scale, width * scale

    with torch_module.no_grad():
        # Host accumulators: the division happens once, at the very end.
        out = torch_module.zeros((1, _CHANNELS, scaled_height, scaled_width), dtype=torch_module.float32, device="cpu")
        weights = (
            torch_module.zeros((1, 1, scaled_height, scaled_width), dtype=torch_module.float32, device="cpu")
            if overlap
            else None
        )

        # A regular grid keeps the model's window/unshuffle phase identical for every tile.
        for y in range(0, height, step):
            for x in range(0, width, step):
                patch = source[:, :, y : min(y + tile, height), x : min(x + tile, width)]
                ph, pw = patch.shape[-2:]

                fixed = getattr(runner, "input_size", None)
                if fixed is not None and tuple(fixed) != (tile, tile):
                    raise ValueError(
                        f"Runner requires a fixed {tuple(fixed)} input, but the tile grid pads every tile "
                        f"to {tile}x{tile}."
                    )

                # Match the model contract at the image edges instead of leaving a short tile.
                padded = torch_module.nn.functional.pad(patch, (0, tile - pw, 0, tile - ph), mode="replicate")
                prediction = runner(padded).detach().cpu()
                expected = (1, _CHANNELS, tile * scale, tile * scale)
                if tuple(prediction.shape) != expected:
                    raise ValueError(
                        f"Runner returned unexpected output shape {tuple(prediction.shape)}; expected {expected}."
                    )
                prediction = prediction[:, :, : ph * scale, : pw * scale]
                if prediction.dtype != torch_module.float32:
                    prediction = prediction.float()

                sy, sx = y * scale, x * scale
                eh, ew = ph * scale, pw * scale
                target = (slice(None), slice(None), slice(sy, sy + eh), slice(sx, sx + ew))
                if weights is None:
                    out[target] = prediction
                    continue

                wy = torch_module.ones(eh, dtype=torch_module.float32, device="cpu")
                wx = torch_module.ones(ew, dtype=torch_module.float32, device="cpu")
                n = min(overlap * scale, eh)
                if y:
                    wy[:n] *= torch_module.arange(1, n + 1, dtype=torch_module.float32, device="cpu") / (n + 1)
                if y + ph < height:
                    wy[-n:] *= torch_module.arange(n, 0, -1, dtype=torch_module.float32, device="cpu") / (n + 1)
                n = min(overlap * scale, ew)
                if x:
                    wx[:n] *= torch_module.arange(1, n + 1, dtype=torch_module.float32, device="cpu") / (n + 1)
                if x + pw < width:
                    wx[-n:] *= torch_module.arange(n, 0, -1, dtype=torch_module.float32, device="cpu") / (n + 1)

                blend = wy[:, None] * wx[None, :]
                out[target] += prediction * blend
                weights[:, :, sy : sy + eh, sx : sx + ew] += blend

        if weights is not None:
            out.div_(weights)

    return out

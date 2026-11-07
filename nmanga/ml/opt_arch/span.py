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

# The `fast` inference recipe for resselt's SPAN architecture.

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, cast

from ..contracts import ModelInfo
from ._conv3xc import check_preserved, conv3xc_paths, outside_conv3xc, replace_conv3xc

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

__all__ = ("ARCHITECTURE", "build_fast_model", "cross_check", "info_from_header", "unshuffle_of")

ARCHITECTURE = "SPAN"

_LEAN_CACHE: dict[str, tuple[type, ...]] = {}


def _lean_classes() -> tuple[type]:
    """
    Create (once) the lean rewrite module for the installed torch build.

    The class has to be built after torch is imported, so it is defined inside this factory and
    cached for the lifetime of the process.

    Returns
    -------
    tuple[type]
        The `LeanSPAB` class.
    """
    cached = _LEAN_CACHE.get("classes")
    if cached is not None:
        return cached  # type: ignore[return-value]

    from ...lazy import get_torch

    torch = get_torch()
    nn = torch.nn

    class LeanSPAB(nn.Module):
        """resselt's attention block with the residual and the gate combined in place."""

        def __init__(self, source) -> None:
            super().__init__()
            self.c1_r, self.c2_r, self.c3_r = source.c1_r, source.c2_r, source.c3_r
            self.act1, self.end = source.act1, source.end

        def forward(self, x):
            # resselt activates `out1` through an in-place SiLU and then returns that same
            # tensor, so the terminal block's second value is the *activated* first feature.
            first = self.act1(self.c1_r(x))
            out = self.c3_r(self.act1(self.c2_r(first)))
            # The gate is derived from the third convolution before the residual is added, and
            # both writes reuse buffers nothing else holds.
            attention = out.sigmoid().sub_(0.5)
            out.add_(x).mul_(attention)
            return (out, first) if self.end else out

    classes = (LeanSPAB,)
    _LEAN_CACHE["classes"] = classes
    return classes


def _span_classes() -> tuple[type, type, type]:
    """Import the resselt SPAN classes this adapter is written against."""
    from resselt.archs.span.arch import SPAB, SPAN, Conv3XC

    return (SPAN, SPAB, Conv3XC)


def _structure_error(message: str, info: ModelInfo) -> ValueError:
    """Build the actionable error for a structure the adapter does not know."""
    return ValueError(
        f"{message} The `fast` preset is written against resselt "
        f"{info.loader_version or '(unknown version)'} for {info.architecture} checkpoints; re-run "
        "with --preset original if this checkpoint is not the validated structure."
    )


def _check_activation(block: Any, where: str, info: ModelInfo) -> None:
    """
    Validate the activation the lean block relies on.

    resselt writes the first branch's activation into the branch's own buffer, and the terminal
    block hands that buffer back as its second return value. A non-in-place activation would
    leave the returned tensor unactivated, which is a different model, so the rewrite refuses it.
    """
    import torch

    act = getattr(block, "act1", None)
    if not isinstance(act, torch.nn.SiLU):
        raise _structure_error(f"{where}.act1 is a {type(act).__name__}, expected an nn.SiLU.", info)
    if not act.inplace:
        raise _structure_error(
            f"{where}.act1 is an out-of-place SiLU; the terminal block returns the first branch's "
            "activation buffer as its second value, which only holds for the in-place form.",
            info,
        )


def _check_span_structure(model: Any, classes: tuple[type, ...], info: ModelInfo) -> list[Any]:
    """
    Validate the SPAN structure the adapter rewrites and return its attention blocks.

    The counts are derived from the loaded model rather than assumed, so a checkpoint with a
    different feature width or block count is still rewritten correctly; only the shapes that
    the rewrite itself depends on are asserted.
    """
    import torch

    span, spab, conv3xc = classes
    if not isinstance(model, span):
        raise _structure_error(f"The loaded model is a {type(model).__name__}, not resselt's SPAN.", info)

    conv_1 = getattr(model, "conv_1", None)
    conv_2 = getattr(model, "conv_2", None)
    for name, module in (("conv_1", conv_1), ("conv_2", conv_2)):
        if not isinstance(module, conv3xc):
            raise _structure_error(f"{name} is a {type(module).__name__}, expected resselt's Conv3XC.", info)

    blocks = [child for child in model.children() if isinstance(child, spab)]
    if not blocks:
        raise _structure_error("The SPAN model has no SPAB attention blocks to rewrite.", info)
    for index, block in enumerate(blocks, start=1):
        _check_activation(block, f"block_{index}", info)

    terminal = [index for index, block in enumerate(blocks) if block.end]
    if terminal != [len(blocks) - 1]:
        raise _structure_error(
            f"The SPAN model reports end=True on blocks {terminal}, expected only the last of its "
            f"{len(blocks)} blocks.",
            info,
        )

    cat = getattr(model, "conv_cat", None)
    if not isinstance(cat, torch.nn.Conv2d) or tuple(cat.kernel_size) != (1, 1):
        raise _structure_error(f"conv_cat is a {type(cat).__name__}, expected a 1x1 nn.Conv2d.", info)
    feature = cast(Any, conv_1).sk.out_channels
    if cat.in_channels != 4 * feature:
        raise _structure_error(
            f"conv_cat takes {cat.in_channels} channels, expected the four concatenated {feature}-channel features.",
            info,
        )

    upsampler = getattr(model, "upsampler", None)
    if not isinstance(upsampler, torch.nn.Sequential) or len(upsampler) != 2:
        raise _structure_error(
            f"upsampler is a {type(upsampler).__name__}, expected a convolution and a PixelShuffle.", info
        )
    upsample_conv, shuffle = upsampler[0], upsampler[1]
    if not isinstance(upsample_conv, torch.nn.Conv2d) or not isinstance(shuffle, torch.nn.PixelShuffle):
        raise _structure_error(
            f"upsampler is ({type(upsample_conv).__name__}, {type(shuffle).__name__}), expected "
            "(nn.Conv2d, nn.PixelShuffle).",
            info,
        )
    if upsample_conv.in_channels != feature:
        raise _structure_error(
            f"upsampler.0 takes {upsample_conv.in_channels} channels, expected the {feature} feature "
            "channels the concatenation produces.",
            info,
        )
    if upsample_conv.out_channels != shuffle.upscale_factor**2 * info.out_channels:
        raise _structure_error(
            f"upsampler.0 returns {upsample_conv.out_channels} channels, expected "
            f"{shuffle.upscale_factor**2 * info.out_channels} for a {shuffle.upscale_factor}x "
            f"pixel shuffle to {info.out_channels} channels.",
            info,
        )
    return blocks


def _replace_spab(model: Any, spab: type, lean_spab: type) -> int:
    """Swap every SPAB in the module tree for the lean block; return the count."""
    replaced = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, spab):
                setattr(parent, name, lean_spab(child))
                replaced += 1
    return replaced


def build_fast_model(model: nn.Module, info: ModelInfo) -> nn.Module:
    """
    Build the lean SPAN inference model as an independent copy.

    Two artifacts are removed. Every `Conv3XC` wrapper is replaced by the inference convolution
    its three branches and side kernel reparameterize into, which also stops the per-prediction
    `update_params` call its forward makes; the stored inference kernels a checkpoint carries are
    recomputed rather than trusted, because they go stale whenever the branches are trained
    further. Every `SPAB` is then replaced by a block that fuses the residual addition and the
    gate multiplication in place.

    The source model is never mutated, the parameter tree is compared outside the fused
    convolutions, and the arithmetic order of every remaining operation is unchanged.

    Parameters
    ----------
    model : nn.Module
        The untouched model returned by `load_checkpoint`, in FP32 evaluation mode.
    info : ModelInfo
        Its normalized contract, used for the architecture check and the error messages.

    Returns
    -------
    nn.Module
        An independent, lean FP32 evaluation model.

    Raises
    ------
    ValueError
        If the architecture is not SPAN, the model is not an FP32 evaluation model, or the
        installed resselt structure differs from the one this adapter was written against.
    """
    if info.architecture.strip().upper() != ARCHITECTURE:
        raise ValueError(f"The SPAN `fast` adapter cannot prepare {info.architecture}; re-run with --preset original.")

    import torch

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"build_fast_model expects a torch.nn.Module, got {type(model).__name__}.")
    if model.training:
        raise ValueError(
            "The `fast` preset requires an evaluation model; the loaded model is still training. "
            "Load it with load_checkpoint() or call .eval() first."
        )
    non_fp32 = sorted({str(parameter.dtype) for parameter in model.parameters() if parameter.dtype != torch.float32})
    if non_fp32:
        raise ValueError(
            f"The `fast` preset requires FP32 weights, but the model has {', '.join(non_fp32)} "
            "parameters. FP32 is the precision contract: cast the model to float32 first."
        )

    classes = _span_classes()
    _span, spab, conv3xc = classes
    blocks = _check_span_structure(model, classes, info)
    expected_conv3xc = sum(1 for module in model.modules() if isinstance(module, conv3xc))

    (lean_spab,) = _lean_classes()
    paths = conv3xc_paths(model, conv3xc)
    before = outside_conv3xc(model, paths)

    result = copy.deepcopy(model).eval().requires_grad_(False)
    # 1. Collapse every reparameterizable convolution into its single inference kernel.
    replaced_conv3xc = replace_conv3xc(result, conv3xc, info=info)
    # 2. Lean attention blocks, wrapping the inference convolutions produced by the pass above.
    replaced_blocks = _replace_spab(result, spab, lean_spab)

    fast_model = result.eval().requires_grad_(False)
    counts = {
        "inference convolution": (replaced_conv3xc, expected_conv3xc),
        "lean SPAB": (replaced_blocks, len(blocks)),
    }
    wrong = [
        f"{name}: rewrote {actual} of {expected}" for name, (actual, expected) in counts.items() if actual != expected
    ]
    if wrong:
        raise RuntimeError(
            f"The fast SPAN adapter did not rewrite the whole model ({'; '.join(wrong)}). "
            "This is a bug in the adapter; the untouched model is still usable with --preset original."
        )
    check_preserved(before, outside_conv3xc(fast_model, paths), info)
    return fast_model


def unshuffle_of(model: nn.Module, path: Path) -> int:
    """
    Report the input pixel-unshuffle factor, which is `1` for SPAN.

    SPAN consumes the external image directly: its first convolution takes the checkpoint's RGB
    channels, and the only rearrangement in the model is the pixel-shuffle at the output end. A
    `PixelUnshuffle` anywhere in the tree would mean resselt built something this adapter does
    not describe, so it is rejected rather than reported as `1`.
    """
    import torch

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.PixelUnshuffle):
            raise ValueError(
                f"resselt built {type(model).__name__} with a PixelUnshuffle at {name!r} for "
                f"{path.name}; the SPAN adapter describes a model that consumes pixels directly."
            )
    return 1


def cross_check(model: nn.Module, info: ModelInfo, path: Path) -> None:
    """Verify the normalized SPAN contract against the model resselt actually constructed."""
    import torch

    span, _spab, conv3xc = _span_classes()
    if not isinstance(model, span):
        raise ValueError(
            f"resselt reported SPAN metadata but constructed a {type(model).__name__} for {path.name}; "
            "the installed resselt structure is not the one this backend was validated against."
        )

    conv_1 = getattr(model, "conv_1", None)
    upsampler = getattr(model, "upsampler", None)
    problems: list[str] = []
    if isinstance(conv_1, conv3xc):
        if conv_1.sk.in_channels != info.in_channels:
            problems.append(f"conv_1 takes {conv_1.sk.in_channels} channels, expected {info.in_channels}")
    else:
        problems.append(f"conv_1 is a {type(conv_1).__name__}, expected resselt's Conv3XC")
    if isinstance(upsampler, torch.nn.Sequential) and len(upsampler) == 2:
        shuffle = upsampler[1]
        if shuffle.upscale_factor != info.scale:
            problems.append(f"the upsampler is built for {shuffle.upscale_factor}x, expected {info.scale}x")
    else:
        problems.append(f"upsampler is a {type(upsampler).__name__}, expected a convolution and a PixelShuffle")
    if info.unshuffle != 1:
        problems.append(f"the normalized contract reports a {info.unshuffle}x unshuffle, SPAN does not unshuffle")
    if problems:
        raise ValueError(
            f"resselt reported SPAN metadata for {path.name} that does not match the constructed model: "
            + "; ".join(problems)
            + ". The installed resselt structure is not the one this backend was validated against."
        )


def info_from_header(
    path: Path,
    header: dict[str, tuple[str, tuple[int, ...]]],
    *,
    architecture: str,
    stored_dtypes: tuple[str, ...],
    checkpoint_sha256: str,
    loader_version: str | None,
) -> ModelInfo | None:
    """
    Decline to describe SPAN from the safetensors header alone.

    The scale and the external channels would have to be read from the pixel-shuffle
    convolution and the upsampler, and a wrong guess here is a silently wrong tile grid. SPAN is
    cheap enough to load through resselt, so the loader falls back to the architecture it owns.

    Parameters
    ----------
    path : Path
        The checkpoint being described.
    header : dict[str, tuple[str, tuple[int, ...]]]
        Every tensor name with its stored dtype and shape.
    architecture : str
        The architecture resselt's detectors reported.
    stored_dtypes : tuple[str, ...]
        The floating dtypes actually stored in the checkpoint.
    checkpoint_sha256 : str
        The digest of the checkpoint file.
    loader_version : str | None
        The installed resselt version.

    Returns
    -------
    ModelInfo | None
        Always `None`, so `read_model_info` loads the model instead.
    """
    return None

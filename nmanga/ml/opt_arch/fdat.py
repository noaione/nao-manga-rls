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

# The `fast` inference recipe for resselt's FDAT architecture.

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, cast

from ..contracts import ModelInfo

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

__all__ = ("ARCHITECTURE", "build_fast_model", "cross_check", "info_from_header", "unshuffle_of")

ARCHITECTURE = "FDAT"

METADATA_KEY = "upsampler.MetaUpsample"
"""The upsampler buffer resselt reads to rebuild the architecture arguments."""

UNSHUFFLE_CONV = "conv_first.1.weight"
"""The pixel-unshuffle convolution: present only when the model unshuffles pixels."""

_LEAN_CACHE: dict[str, tuple[type, ...]] = {}


def _lean_classes() -> tuple[type, type, type, type]:
    """
    Create (once) the lean rewrite modules for the installed torch build.

    The classes have to be built after torch is imported, so they are defined inside this factory
    and cached for the lifetime of the process.

    Returns
    -------
    tuple[type, type, type, type]
        The `InplaceGELU`, `LeanSpatialAttention`, `LeanFFN` and `LeanDATBlock` classes.
    """
    cached = _LEAN_CACHE.get("classes")
    if cached is not None:
        return cached  # type: ignore[return-value]

    from ...lazy import get_torch

    torch = get_torch()
    functional = torch.nn.functional
    nn = torch.nn

    class InplaceGELU(nn.Module):
        """Exact GELU written into the activation's own buffer."""

        def __init__(self, approximate: str = "none") -> None:
            super().__init__()
            self.approximate = approximate

        def forward(self, x):
            return torch.ops.aten.gelu.out(x, approximate=self.approximate, out=x)

    class LeanSpatialAttention(nn.Module):
        """Window attention that reuses the score buffer for the bias and the softmax."""

        def __init__(self, source) -> None:
            super().__init__()
            self.ws, self.nh, self.scale = source.ws, source.nh, source.scale
            self.qkv, self.proj, self.bias = source.qkv, source.proj, source.bias

        def forward(self, x, H, W):  # ruff: ignore[invalid-argument-name] - H/W are resselt's block interface
            batch, length, channels = x.shape
            pad_r, pad_b = (-W) % self.ws, (-H) % self.ws
            if pad_r or pad_b:
                x = functional.pad(x.reshape(batch, H, W, channels), (0, 0, 0, pad_r, 0, pad_b))
            height, width = H + pad_b, W + pad_r
            x = (
                x
                .reshape(batch, height // self.ws, self.ws, width // self.ws, self.ws, channels)
                .permute(0, 1, 3, 2, 4, 5)
                .contiguous()
                .view(-1, self.ws**2, channels)
            )
            qkv = self.qkv(x).view(-1, self.ws**2, 3, self.nh, channels // self.nh).permute(2, 0, 3, 1, 4)
            query, key, value = qkv.unbind(0)
            query.mul_(self.scale)
            scores = query @ key.transpose(-2, -1)
            scores.add_(self.bias)
            torch.softmax(scores, dim=-1, out=scores)
            out = (scores @ value).transpose(1, 2).reshape(-1, self.ws**2, channels)
            del qkv, query, key, value, scores
            out = self.proj(out)
            out = (
                out
                .view(batch, height // self.ws, width // self.ws, self.ws, self.ws, channels)
                .permute(0, 1, 3, 2, 4, 5)
                .contiguous()
                .view(batch, height, width, channels)
            )
            return out[:, :H, :W].contiguous().view(batch, length, channels)

    class LeanFFN(nn.Module):
        """Depthwise-mixed feed-forward without the (eval-identity) dropout calls."""

        def __init__(self, source) -> None:
            super().__init__()
            self.fc1, self.act, self.smix, self.fc2 = source.fc1, source.act, source.smix, source.fc2

        def forward(self, x, H, W):  # ruff: ignore[invalid-argument-name] - H/W are resselt's block interface
            batch, length, _channels = x.shape
            x = self.act(self.fc1(x))
            channels = x.shape[-1]
            x = self.smix(x.transpose(1, 2).view(batch, channels, H, W))
            return self.fc2(x.view(batch, channels, length).transpose(1, 2))

    class LeanDATBlock(nn.Module):
        """resselt's DAT block with the same operation order and shorter activation lifetimes."""

        def __init__(self, source) -> None:
            super().__init__()
            self.n1, self.n2, self.attn = source.n1, source.n2, source.attn
            self.conv, self.inter, self.ffn = source.conv, source.inter, LeanFFN(source.ffn)
            self.btype = source.btype

        def forward(self, x, H, W):  # ruff: ignore[invalid-argument-name] - H/W are resselt's block interface
            batch, length, channels = x.shape
            normed = self.n1(x)
            interaction = "channel_modulates_spatial" if self.btype == "spatial" else "spatial_modulates_channel"
            convolved = self.conv(normed.transpose(1, 2).view(batch, channels, H, W))
            fused = self.inter(
                self.attn(normed, H, W),
                convolved.view(batch, channels, length).transpose(1, 2),
                interaction,
                H,
                W,
            )
            del normed, convolved
            x = x + fused
            del fused
            return x + self.ffn(self.n2(x), H, W)

    classes = (InplaceGELU, LeanSpatialAttention, LeanFFN, LeanDATBlock)
    _LEAN_CACHE["classes"] = classes
    return classes


def _fdat_classes() -> tuple[type, ...]:
    """Import the resselt FDAT classes this adapter is written against."""
    from resselt.archs.fdat.arch import (
        FDAT,
        FastChannelAttention,
        FastSpatialWindowAttention,
        SimplifiedAIM,
        SimplifiedDATBlock,
        SimplifiedFFN,
        SimplifiedResidualGroup,
    )

    return (
        FDAT,
        FastChannelAttention,
        FastSpatialWindowAttention,
        SimplifiedAIM,
        SimplifiedDATBlock,
        SimplifiedFFN,
        SimplifiedResidualGroup,
    )


def _structure_error(message: str, info: ModelInfo) -> ValueError:
    """Build the actionable error for a structure the adapter does not know."""
    return ValueError(
        f"{message} The `fast` preset is written against resselt "
        f"{info.loader_version or '(unknown version)'} for {info.architecture} checkpoints; re-run "
        "with --preset original if this checkpoint is not the validated structure."
    )


def _check_attention(source: Any, where: str, info: ModelInfo) -> None:
    """Validate the attributes the lean spatial attention copies out of the source attention."""
    if not isinstance(source.ws, int) or source.ws < 1:
        raise _structure_error(f"{where}.attn.ws is {source.ws!r}, expected a positive window size.", info)
    if not isinstance(source.nh, int) or source.nh < 1:
        raise _structure_error(f"{where}.attn.nh is {source.nh!r}, expected a positive head count.", info)
    if not isinstance(source.scale, float) or source.scale <= 0:
        raise _structure_error(f"{where}.attn.scale is {source.scale!r}, expected a positive float.", info)
    expected_bias = (source.nh, source.ws**2, source.ws**2)
    shape = tuple(getattr(source.bias, "shape", ()))
    if shape != expected_bias:
        raise _structure_error(
            f"{where}.attn.bias has shape {shape}, expected the learned window bias {expected_bias}.", info
        )


def _check_block(block: Any, where: str, classes: tuple[type, ...], info: ModelInfo) -> None:
    """Validate one DAT block against the four rewrites the fast preset applies."""
    import torch

    _, fast_channel_attention, fast_spatial_attention, simplified_aim, *_ = classes
    if block.btype not in {"spatial", "channel"}:
        raise _structure_error(f"{where}.btype is {block.btype!r}, expected 'spatial' or 'channel'.", info)
    expected_attention = fast_spatial_attention if block.btype == "spatial" else fast_channel_attention
    if not isinstance(block.attn, expected_attention):
        raise _structure_error(
            f"{where}.attn is a {type(block.attn).__name__}, expected a {expected_attention.__name__} "
            f"for the {block.btype!r} block.",
            info,
        )
    if not isinstance(block.inter, simplified_aim):
        raise _structure_error(
            f"{where}.inter is a {type(block.inter).__name__}, expected resselt's SimplifiedAIM.", info
        )
    drop_path = block.dp
    if not isinstance(drop_path, torch.nn.Identity) and not hasattr(drop_path, "drop_prob"):
        raise _structure_error(
            f"{where}.dp is a {type(drop_path).__name__}; the lean preset only omits drop-path calls "
            "whose evaluation-mode result is the identity.",
            info,
        )
    if block.btype == "spatial":
        _check_attention(block.attn, where, info)


def _check_eval_only_omissions(model: Any, classes: tuple[type, ...], info: ModelInfo) -> None:
    """
    Verify the calls the lean rewrites omit are identities for this model.

    resselt's `DropPath(...)` factory returns `nn.Identity()`, its `drop_path()` function returns
    its input unchanged whenever `training` is False, and `nn.Dropout` is the identity in
    evaluation mode. The lean block and feed-forward therefore drop those calls instead of paying
    for them - sound only for an evaluation model, which `build_fast_model` requires up front.
    """
    import torch

    if model.training:
        raise ValueError(
            "The `fast` preset omits the drop-path and dropout calls, which is only equivalent to "
            "an evaluation forward pass. Load the model with load_checkpoint() or call .eval() "
            "before build_fast_model()."
        )
    simplified_ffn = classes[5]
    for name, module in model.named_modules():
        if isinstance(module, simplified_ffn) and not isinstance(module.drop, torch.nn.Dropout):
            raise _structure_error(
                f"{name or '<root>'}.drop is a {type(module.drop).__name__}, expected an nn.Dropout.", info
            )


def _check_fdat_structure(model: Any, classes: tuple[type, ...], info: ModelInfo) -> list[Any]:
    """Validate the whole FDAT structure the adapter rewrites and return its DAT blocks."""
    import torch

    fdat, _, _, _, simplified_dat_block, _, simplified_residual_group = classes
    if not isinstance(model, fdat):
        raise _structure_error(f"The loaded model is a {type(model).__name__}, not resselt's FDAT.", info)
    groups = getattr(model, "groups", None)
    if not isinstance(groups, torch.nn.Sequential) or len(groups) == 0:
        raise _structure_error(
            f"The FDAT model has {type(groups).__name__} groups, expected a non-empty nn.Sequential.", info
        )
    blocks: list[Any] = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, simplified_residual_group):
            raise _structure_error(
                f"groups[{group_index}] is a {type(group).__name__}, expected resselt's SimplifiedResidualGroup.",
                info,
            )
        if len(group.blocks) == 0:
            raise _structure_error(f"groups[{group_index}] has no DAT blocks.", info)
        for block_index, block in enumerate(group.blocks):
            where = f"groups[{group_index}].blocks[{block_index}]"
            if not isinstance(block, simplified_dat_block):
                raise _structure_error(
                    f"{where} is a {type(block).__name__}, expected resselt's SimplifiedDATBlock.", info
                )
            _check_block(block, where, classes, info)
            blocks.append(block)
    return blocks


def _parameter_fingerprint(model: Any) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Describe the parameter tree so a rewrite can prove it did not move any weight."""
    return tuple(sorted((name, tuple(tensor.shape)) for name, tensor in model.named_parameters()))


def _replace_gelu(model: Any, inplace_gelu: type) -> int:
    """Swap every `nn.GELU` in the module tree for the in-place activation; return the count."""
    import torch

    replaced = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.GELU):
                setattr(parent, name, inplace_gelu(child.approximate))
                replaced += 1
    return replaced


def build_fast_model(model: nn.Module, info: ModelInfo) -> nn.Module:
    """
    Build the lean FDAT inference model as an independent copy.

    The source model is never mutated: the rewrites are applied to a `copy.deepcopy` that is put
    in evaluation mode with gradients disabled. Every weight tensor, every block and the
    arithmetic order are preserved; only the recalculation of activation buffers, the window
    attention scratch reuse and the omission of evaluation-mode-identity dropout changes.

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
        If the architecture is not FDAT, the model is not an FP32 evaluation model, or the
        installed resselt structure differs from the one this adapter was written against.
    """
    if info.architecture.strip().upper() != ARCHITECTURE:
        raise ValueError(f"The FDAT `fast` adapter cannot prepare {info.architecture}; re-run with --preset original.")

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

    classes = _fdat_classes()
    fast_spatial_attention = classes[2]
    simplified_dat_block = classes[4]
    simplified_ffn = classes[5]
    simplified_residual_group = classes[6]
    blocks = _check_fdat_structure(model, classes, info)
    _check_eval_only_omissions(model, classes, info)

    inplace_gelu, lean_spatial_attention, lean_ffn, lean_dat_block = _lean_classes()
    expected_gelu = sum(1 for module in model.modules() if isinstance(module, torch.nn.GELU))
    expected_spatial = sum(1 for block in blocks if isinstance(block.attn, fast_spatial_attention))
    before = _parameter_fingerprint(model)

    result = copy.deepcopy(model).eval().requires_grad_(False)
    # 1. Exact in-place activation, replacing every nn.GELU (the FFN, the depthwise convolution and
    #    the scale-4 upsampler all use the same activation).
    replaced_gelu = _replace_gelu(result, inplace_gelu)
    # 2. Lean window attention for the spatial blocks; the channel blocks keep resselt's attention.
    replaced_spatial = 0
    for block in [module for module in result.modules() if isinstance(module, simplified_dat_block)]:
        if isinstance(block.attn, fast_spatial_attention):
            block.attn = lean_spatial_attention(block.attn)
            replaced_spatial += 1
    # 3. Lean DAT blocks, wrapping the attention and feed-forward produced by the passes above.
    replaced_blocks = 0
    for group in [module for module in result.modules() if isinstance(module, simplified_residual_group)]:
        for index, block in enumerate(group.blocks):
            if isinstance(block, simplified_dat_block):
                group.blocks[index] = lean_dat_block(block)
                replaced_blocks += 1

    fast_model = result.eval().requires_grad_(False)
    counts = {
        "in-place GELU": (replaced_gelu, expected_gelu),
        "lean window attention": (replaced_spatial, expected_spatial),
        "lean DAT block": (replaced_blocks, len(blocks)),
        "lean feed-forward": (
            sum(1 for module in fast_model.modules() if isinstance(module, lean_ffn)),
            sum(1 for module in model.modules() if isinstance(module, simplified_ffn)),
        ),
    }
    wrong = [
        f"{name}: rewrote {actual} of {expected}" for name, (actual, expected) in counts.items() if actual != expected
    ]
    if wrong:
        raise RuntimeError(
            f"The fast FDAT adapter did not rewrite the whole model ({'; '.join(wrong)}). "
            "This is a bug in the adapter; the untouched model is still usable with --preset original."
        )
    if _parameter_fingerprint(fast_model) != before:
        raise RuntimeError(
            "The fast FDAT adapter changed the parameter tree of the model; the rewrite must only "
            "move references, never replace or drop a weight."
        )
    return fast_model


def _read_integer_tensor(path: Path, key: str) -> tuple[int, ...]:
    """Read one small integer tensor out of a checkpoint without constructing the model."""
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as handle:
        return tuple(int(value) for value in handle.get_tensor(key).tolist())


def unshuffle_of(model: nn.Module, path: Path) -> int:
    """Read the pixel-unshuffle factor out of resselt's constructed FDAT model."""
    import torch

    conv_first = getattr(model, "conv_first", None)
    if isinstance(conv_first, torch.nn.Sequential) and len(conv_first) > 0:
        first = conv_first[0]
        if not isinstance(first, torch.nn.PixelUnshuffle):
            raise ValueError(
                f"resselt built {type(model).__name__} with conv_first={type(conv_first).__name__} whose "
                f"first layer is {type(first).__name__} for {path.name}; the fast-preset adapter expects "
                "a pixel-unshuffle first layer, so this resselt structure is unsupported."
            )
        return int(first.downscale_factor)
    if isinstance(conv_first, torch.nn.Conv2d):
        return 1
    raise ValueError(
        f"resselt built {type(model).__name__} without a recognisable conv_first "
        f"({type(conv_first).__name__}) for {path.name}; the normalized FDAT contract cannot be "
        "cross-checked against this resselt structure."
    )


def _conv_first_in_channels(model: nn.Module, unshuffle: int, path: Path) -> int:
    """Return the channel count entering the first convolution of the unshuffle path."""
    import torch

    # torch's `Module.__getattr__` types every dynamic attribute as `Module | Tensor`.
    conv_first = cast(Any, model.conv_first)
    conv = conv_first[1] if unshuffle > 1 else conv_first
    if not isinstance(conv, torch.nn.Conv2d):
        raise ValueError(
            f"resselt built {type(model).__name__} with a {type(conv).__name__} after its "
            f"pixel-unshuffle layer for {path.name}; the normalized FDAT contract cannot be checked."
        )
    return int(conv.in_channels)


def cross_check(model: nn.Module, info: ModelInfo, path: Path) -> None:
    """Verify the normalized FDAT contract against the model resselt actually constructed."""
    upsampler = cast(Any, model.upsampler)
    internal_scale = int(upsampler.MetaUpsample[2].item())
    upscale = int(cast(Any, model.upscale))
    pads = int(getattr(model, "pad", 0))
    expected_pad = info.unshuffle if info.unshuffle > 1 else 0
    conv_in = _conv_first_in_channels(model, info.unshuffle, path)
    expected_conv_in = info.in_channels * info.unshuffle**2
    problems: list[str] = []
    if conv_in != expected_conv_in:
        problems.append(f"conv_first takes {conv_in} channels, expected {expected_conv_in}")
    if pads != expected_pad:
        problems.append(f"check_img_size pads by {pads}, expected {expected_pad}")
    if upscale != info.scale:
        problems.append(f"the model upscales by {upscale}, including cropping, expected {info.scale}")
    if internal_scale != info.scale * info.unshuffle:
        problems.append(
            f"the upsampler is built for {internal_scale}x, expected {info.scale * info.unshuffle}x "
            f"({info.scale}x effective over a {info.unshuffle}x unshuffle)"
        )
    if problems:
        raise ValueError(
            f"resselt reported FDAT metadata for {path.name} that does not match the constructed model: "
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
    Normalize FDAT metadata straight from the safetensors header, without loading the model.

    The rules mirror resselt's FDAT loader: the upsampler buffer carries the internal scale and
    the external channels, while the first convolution says whether pixels are unshuffled. A
    structure that does not fit these rules returns `None` so the caller can fall back to a real
    load and resselt's own error.
    """
    if METADATA_KEY not in header:
        return None
    meta = _read_integer_tensor(path, METADATA_KEY)
    if len(meta) != 7:
        return None
    _version, _upsampler_index, internal_scale, _embed_dim, out_channels, _mid_dim, _group = meta
    if internal_scale < 1 or out_channels < 1:
        return None
    if UNSHUFFLE_CONV in header:
        conv_in = header[UNSHUFFLE_CONV][1][1]
        if conv_in <= 0 or conv_in % out_channels:
            return None
        unshuffle = math.isqrt(conv_in // out_channels)
    else:
        conv_first = header.get("conv_first.weight")
        if conv_first is None or conv_first[1][1] <= 0:
            return None
        unshuffle = 1
        conv_in = conv_first[1][1]
    if unshuffle < 1 or conv_in != out_channels * unshuffle**2:
        return None
    in_channels = conv_in // unshuffle**2
    if internal_scale % unshuffle:
        return None
    scale = internal_scale // unshuffle
    if scale < 1:
        return None
    return ModelInfo(
        architecture=architecture,
        scale=scale,
        in_channels=in_channels,
        out_channels=out_channels,
        unshuffle=unshuffle,
        stored_dtypes=stored_dtypes,
        checkpoint_sha256=checkpoint_sha256,
        loader_version=loader_version,
    )

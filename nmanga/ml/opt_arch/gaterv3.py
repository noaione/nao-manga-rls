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

# The `fast` inference recipe for resselt's GateRV3 architecture.

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from ..contracts import ModelInfo
from ._conv3xc import check_preserved, conv3xc_paths, outside_conv3xc, replace_conv3xc

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

__all__ = ("ARCHITECTURE", "build_fast_model", "cross_check", "info_from_header", "unshuffle_of")

ARCHITECTURE = "GateRV3"

_LEAN_CACHE: dict[str, tuple[type, type, type]] = {}


def _lean_classes() -> tuple[type, type, type]:
    """
    Create (once) the lean rewrite modules for the installed torch build.

    The classes have to be built after torch is imported, so they are defined inside this factory
    and cached for the lifetime of the process.

    Returns
    -------
    tuple[type, type, type]
        The `BareRMSNorm`, `FoldedMetaGated` and `ProjectedDecoder` classes.
    """
    cached = _LEAN_CACHE.get("classes")
    if cached is not None:
        return cached  # type: ignore[return-value]

    from ...lazy import get_torch

    torch = get_torch()
    functional = torch.nn.functional
    nn = torch.nn

    class BareRMSNorm(nn.Module):
        """
        resselt's normalization with the affine transform folded away.

        `RMSNorm` adds epsilon to the root of the mean square and not to the mean square itself,
        so the denominator cannot be rewritten as a single square root of a shifted mean. Keeping
        the same two operations in the same order is what lets the folded convolution stay
        numerically comparable to the normalization it replaces.
        """

        def __init__(self, eps: float) -> None:
            super().__init__()
            self.eps = eps

        def forward(self, x):
            rms = x.norm(2, dim=1, keepdim=True) * (x.size(1) ** -0.5)
            return x / (rms + self.eps)

    class FoldedMetaGated(nn.Module):
        """
        `MetaGated` with its local residual scale folded into the scale attention.

        `local * (gamma0 * (W . pool(local) + b)) + short` equals
        `local * ((gamma0 W) . pool(local) + gamma0 b) + short`, so the residual keeps the
        broadcast shape of `gamma0`, the attention bias and the order the unscaled gate was
        applied in. `gamma1` stays where it was: it scales the global branch, not the local one.
        """

        def __init__(self, source) -> None:
            super().__init__()
            self.local = source.local
            self.sca = source.sca
            self.glob = source.glob
            self.gamma1 = source.gamma1

        def forward(self, x):
            local = self.local(x)
            x = local * self.sca(local) + x
            return self.glob(x) * self.gamma1 + x

    class ProjectedDecoder(nn.Module):
        """
        The decoder block with the 1x1 projection moved in front of the pixel shuffle.

        `W . cat(pixel_shuffle(K x), short) + b` becomes
        `pixel_shuffle((W_up K) x) + W_skip short + b` when `W_up` takes the columns the shuffled
        branch occupies and `W_skip` the rest. The shuffle reads `(channel, subpixel)` order, so
        the projection is folded into the subpixel groups of the kernel instead of being applied
        to the shuffle's output; the concatenation and half of the full-resolution projection go
        away.
        """

        def __init__(self, source) -> None:
            super().__init__()
            upsample = source.scale[0]
            projection = source.shor
            channels = projection.out_channels
            self.factor = source.scale[1].upscale_factor
            self.up = nn.Conv2d(
                upsample.in_channels,
                channels * self.factor**2,
                upsample.kernel_size,
                stride=upsample.stride,
                padding=upsample.padding,
                bias=False,
                device=upsample.weight.device,
                dtype=upsample.weight.dtype,
            )
            self.skip = nn.Conv2d(
                channels,
                channels,
                1,
                bias=projection.bias is not None,
                device=projection.weight.device,
                dtype=projection.weight.dtype,
            )
            with torch.no_grad():
                kernel = upsample.weight.reshape(channels, self.factor**2, upsample.in_channels, *upsample.kernel_size)
                matrix = projection.weight[:, :channels, 0, 0]
                self.up.weight.copy_(torch.einsum("oc,cpijk->opijk", matrix, kernel).reshape_as(upsample.weight))
                self.skip.weight.copy_(projection.weight[:, channels:])
                # The skip carries the whole bias of the concatenated projection, and it is only
                # present when the projection that was split had one.
                skip_bias = self.skip.bias
                if projection.bias is not None and skip_bias is not None:
                    skip_bias.copy_(projection.bias)
            self.gated = source.gated

        def forward(self, x, short):
            return self.gated(functional.pixel_shuffle(self.up(x), self.factor) + self.skip(short))

    classes = (BareRMSNorm, FoldedMetaGated, ProjectedDecoder)
    _LEAN_CACHE["classes"] = classes
    return classes


def _gaterv3_classes() -> tuple[type, ...]:
    """Import the resselt GateRV3 classes this adapter is written against."""
    from resselt.archs.gaterv3.arch import Block, Conv3XC, GatedCNNBlock, GateRV3, MetaGated, RMSNorm

    return (GateRV3, Conv3XC, RMSNorm, MetaGated, GatedCNNBlock, Block)


def _structure_error(message: str, info: ModelInfo) -> ValueError:
    """Build the actionable error for a structure the adapter does not know."""
    return ValueError(
        f"{message} The `fast` preset is written against resselt "
        f"{info.loader_version or '(unknown version)'} for {info.architecture} checkpoints; re-run "
        "with --preset original if this checkpoint is not the validated structure."
    )


def _module_list(model: Any, name: str, info: ModelInfo) -> nn.ModuleList:
    """
    Return one of the model's staged `nn.ModuleList` attributes, or refuse the structure.

    GateRV3 keeps its encoder and decoder stages in module lists, and every stage the adapter
    walks has to be one: anything else means resselt built a graph this adapter does not describe,
    and a missing attribute would otherwise surface as a confusing iteration error.

    Parameters
    ----------
    model : Any
        The model resselt constructed.
    name : str
        The attribute holding the staged modules.
    info : ModelInfo
        The normalized contract of the checkpoint, used for the error message.

    Returns
    -------
    nn.ModuleList
        The validated module list.

    Raises
    ------
    ValueError
        If the attribute is not a non-empty `nn.ModuleList`.
    """
    import torch

    modules = getattr(model, name, None)
    if not isinstance(modules, torch.nn.ModuleList) or len(modules) == 0:
        raise _structure_error(f"{name} is a {type(modules).__name__}, expected a non-empty nn.ModuleList.", info)
    return modules


def _check_norm_pair(norm: Any, conv: Any, where: str, norm_type: type, info: ModelInfo) -> None:
    """
    Validate one `RMSNorm` and the ungrouped 1x1 convolution its affine transform folds into.

    Folding rewrites the pair as `conv(x / (RMS(x) + eps))` with the scale moved into the
    input-channel columns of the weight and the offset into the bias. That is only the same
    function when the normalization carries one learned value per input channel of a convolution
    that mixes all of them, so anything else is refused instead of being folded approximately.
    """
    import torch

    if not isinstance(norm, norm_type):
        raise _structure_error(f"{where} is a {type(norm).__name__}, expected resselt's RMSNorm.", info)
    if not isinstance(conv, torch.nn.Conv2d):
        raise _structure_error(f"{where} is followed by a {type(conv).__name__}, expected an nn.Conv2d.", info)
    if tuple(conv.kernel_size) != (1, 1) or conv.groups != 1:
        raise _structure_error(
            f"{where} is followed by a {tuple(conv.kernel_size)} convolution with {conv.groups} groups; the "
            "affine fold needs an ungrouped 1x1 convolution.",
            info,
        )
    for attribute in ("scale", "offset"):
        affine = getattr(norm, attribute, None)
        if not isinstance(affine, torch.nn.Parameter) or tuple(affine.shape) != (conv.in_channels,):
            raise _structure_error(
                f"{where}.{attribute} is {getattr(affine, 'shape', None)}, expected one value per "
                f"{conv.in_channels} input channels.",
                info,
            )


def _check_gate_block(block: Any, where: str, norm_type: type, info: ModelInfo) -> None:
    """Validate the `GatedCNNBlock` fields the norm fold reads."""
    _check_norm_pair(getattr(block, "norm", None), getattr(block, "fc1", None), f"{where}.norm", norm_type, info)


def _check_gated(gated: Any, where: str, meta_gated_type: type, info: ModelInfo) -> None:
    """Validate the `MetaGated` stack of one encoder or decoder block."""
    import torch

    if not isinstance(gated, torch.nn.Sequential) or len(gated) == 0:
        raise _structure_error(
            f"{where}.gated is a {type(gated).__name__}, expected a non-empty sequence of MetaGated blocks.", info
        )
    for index, module in enumerate(gated):
        if not isinstance(module, meta_gated_type):
            raise _structure_error(
                f"{where}.gated[{index}] is a {type(module).__name__}, expected resselt's MetaGated.", info
            )


def _check_meta_gated(module: Any, where: str, *, gate_block_type: type, norm_type: type, info: ModelInfo) -> None:
    """
    Validate the `MetaGated` fields the norm and residual-scale folds read.

    The scale attention is a global average pool followed by a 1x1 convolution, so its output is
    a per-channel value that `gamma0` broadcasts over. Folding multiplies that convolution's rows
    and bias by the scale, which needs the scale to have exactly one value per output channel.
    """
    import torch

    local = getattr(module, "local", None)
    if not isinstance(local, torch.nn.Sequential) or len(local) < 2:
        raise _structure_error(
            f"{where}.local is a {type(local).__name__}, expected the normalization and its convolution.", info
        )
    _check_norm_pair(local[0], local[1], f"{where}.local[0]", norm_type, info)

    attention = getattr(module, "sca", None)
    if not isinstance(attention, torch.nn.Sequential) or len(attention) != 2:
        raise _structure_error(
            f"{where}.sca is a {type(attention).__name__}, expected a pooling layer and a 1x1 convolution.", info
        )
    pool, projection = attention[0], attention[1]
    if not isinstance(pool, torch.nn.AdaptiveAvgPool2d) or pool.output_size != 1:
        raise _structure_error(f"{where}.sca[0] is a {type(pool).__name__}, expected an nn.AdaptiveAvgPool2d(1).", info)
    if not isinstance(projection, torch.nn.Conv2d) or tuple(projection.kernel_size) != (1, 1):
        raise _structure_error(f"{where}.sca[1] is a {type(projection).__name__}, expected a 1x1 nn.Conv2d.", info)

    glob = getattr(module, "glob", None)
    if not isinstance(glob, gate_block_type):
        raise _structure_error(f"{where}.glob is a {type(glob).__name__}, expected resselt's GatedCNNBlock.", info)

    for attribute in ("gamma0", "gamma1"):
        gamma = getattr(module, attribute, None)
        if not isinstance(gamma, torch.nn.Parameter) or tuple(gamma.shape) != (1, projection.out_channels, 1, 1):
            raise _structure_error(
                f"{where}.{attribute} is {getattr(gamma, 'shape', None)}, expected one scale per "
                f"{projection.out_channels} channels.",
                info,
            )


def _check_encoder_scale(scale: Any, where: str, info: ModelInfo) -> None:
    """
    Validate one encoder downsampling branch.

    Each encoder block halves the feature grid through a pixel unshuffle, and each decoder block
    has to undo that: the reflection-padded height only comes back to the `H * scale` the crop
    keeps when both directions agree on the factor.
    """
    import torch

    if not isinstance(scale, torch.nn.Sequential) or len(scale) != 2:
        raise _structure_error(
            f"{where}.scale is a {type(scale).__name__}, expected a convolution and a pixel unshuffle.", info
        )
    conv, shuffle = scale[0], scale[1]
    if not isinstance(conv, torch.nn.Conv2d) or tuple(conv.kernel_size) != (3, 3):
        raise _structure_error(f"{where}.scale[0] is a {type(conv).__name__}, expected a 3x3 nn.Conv2d.", info)
    if not isinstance(shuffle, torch.nn.PixelUnshuffle) or shuffle.downscale_factor != 2:
        raise _structure_error(
            f"{where}.scale[1] is a {type(shuffle).__name__}, expected an nn.PixelUnshuffle(2).", info
        )


def _check_decoder_scale(block: Any, where: str, info: ModelInfo) -> None:
    """
    Validate one decoder upsampling branch and the projection it feeds.

    The projection split assumes the concatenation is the pixel-shuffled branch followed by the
    skip, one projection-width half each. The upsampling convolution therefore has to produce
    exactly `factor ** 2` subpixels per output channel and carry no bias, because the shuffle of
    an unbiased convolution is what the skip projection is added to.
    """
    import torch

    scale = getattr(block, "scale", None)
    projection = getattr(block, "shor", None)
    if not isinstance(scale, torch.nn.Sequential) or len(scale) != 2:
        raise _structure_error(
            f"{where}.scale is a {type(scale).__name__}, expected a convolution and a pixel shuffle.", info
        )
    conv, shuffle = scale[0], scale[1]
    if not isinstance(conv, torch.nn.Conv2d) or tuple(conv.kernel_size) != (3, 3):
        raise _structure_error(f"{where}.scale[0] is a {type(conv).__name__}, expected a 3x3 nn.Conv2d.", info)
    if conv.bias is not None:
        raise _structure_error(
            f"{where}.scale[0] carries a bias; the shuffled branch has to stay unbiased for the skip "
            "projection to receive the whole bias of the concatenated projection.",
            info,
        )
    if not isinstance(shuffle, torch.nn.PixelShuffle):
        raise _structure_error(f"{where}.scale[1] is a {type(shuffle).__name__}, expected an nn.PixelShuffle.", info)
    if not isinstance(projection, torch.nn.Conv2d) or tuple(projection.kernel_size) != (1, 1) or projection.groups != 1:
        raise _structure_error(
            f"{where}.shor is a {type(projection).__name__}, expected an ungrouped 1x1 nn.Conv2d.", info
        )
    factor = shuffle.upscale_factor
    expected_channels = factor**2 * projection.out_channels
    if conv.out_channels != expected_channels:
        raise _structure_error(
            f"{where}.scale[0] returns {conv.out_channels} channels, expected {expected_channels} for the "
            f"{projection.out_channels} channels the {factor}x shuffle yields.",
            info,
        )
    if projection.in_channels != 2 * projection.out_channels:
        raise _structure_error(
            f"{where}.shor takes {projection.in_channels} channels, expected the {projection.out_channels}-channel "
            "shuffled branch plus the same-sized skip.",
            info,
        )


def _check_gaterv3_structure(model: Any, classes: tuple[type, ...], info: ModelInfo) -> tuple[list[Any], ...]:
    """
    Validate the whole GateRV3 structure the four stages rewrite.

    The counts are derived from the loaded model, so an architecture variant with a different
    width or depth is still rewritten correctly; only the relationships the rewrites themselves
    depend on are asserted. The returned `MetaGated`, `GatedCNNBlock` and decoder `Block`
    instances are the plan every later stage is measured against.
    """
    import torch

    gaterv3, _conv3xc, norm_type, meta_gated_type, gate_block_type, block_type = classes
    if not isinstance(model, gaterv3):
        raise _structure_error(f"The loaded model is a {type(model).__name__}, not resselt's GateRV3.", info)

    encoder = _module_list(model, "gater_encode", info)
    decoder = _module_list(model, "decode", info)

    for index, module in enumerate(encoder):
        where = f"gater_encode[{index}]"
        if not isinstance(module, block_type) or not module.down:
            raise _structure_error(
                f"{where} is a {type(module).__name__} without a downsampling scale, expected an encoder Block.", info
            )
        _check_encoder_scale(module.scale, where, info)
        _check_gated(module.gated, where, meta_gated_type, info)

    decoders: list[Any] = []
    for index, module in enumerate(decoder):
        where = f"decode[{index}]"
        if not isinstance(module, block_type) or module.down:
            raise _structure_error(
                f"{where} is a {type(module).__name__} without an upsampling scale, expected a decoder Block.", info
            )
        _check_decoder_scale(module, where, info)
        _check_gated(module.gated, where, meta_gated_type, info)
        decoders.append(module)

    latent = getattr(model, "latent", None)
    if not isinstance(latent, torch.nn.Sequential) or len(latent) == 0:
        raise _structure_error(
            f"latent is a {type(latent).__name__}, expected a non-empty sequence of GatedCNNBlock.", info
        )
    for index, module in enumerate(latent):
        if not isinstance(module, gate_block_type):
            raise _structure_error(
                f"latent[{index}] is a {type(module).__name__}, expected resselt's GatedCNNBlock.", info
            )

    gate_blocks = [module for module in model.modules() if isinstance(module, gate_block_type)]
    meta_gated = [module for module in model.modules() if isinstance(module, meta_gated_type)]
    if not gate_blocks or not meta_gated:
        raise _structure_error("The GateRV3 model has no gated blocks to rewrite.", info)

    paths = {id(module): name for name, module in model.named_modules()}
    for module in gate_blocks:
        _check_gate_block(module, paths[id(module)] or "<root>", norm_type, info)
    for module in meta_gated:
        _check_meta_gated(
            module, paths[id(module)] or "<root>", gate_block_type=gate_block_type, norm_type=norm_type, info=info
        )
    return meta_gated, gate_blocks, decoders


def _fold_norm(norm: Any, conv: Any, bare_norm: type) -> nn.Module:
    """
    Fold one `RMSNorm` affine transform into the convolution that follows it.

    `conv(scale * n + offset)` is `conv'(n)` with the scale multiplying the input-channel columns
    of the weight and the offset becoming a bias term. The bias is computed with the *old* weight,
    before its columns are scaled, because the offset is mapped through the original projection.
    The returned module replaces the normalization and only keeps epsilon.
    """
    import torch

    nn = torch.nn
    with torch.no_grad():
        delta = conv.weight[:, :, 0, 0] @ norm.offset
        bias = delta if conv.bias is None else conv.bias + delta
        conv.weight.mul_(norm.scale[None, :, None, None])
        conv.bias = nn.Parameter(bias, requires_grad=False)
    return bare_norm(norm.eps)


def _fold_norms(model: Any, meta_gated_type: type, gate_block_type: type, bare_norm: type) -> int:
    """
    Fold every `RMSNorm` in the module tree into the 1x1 convolution that follows it.

    Both normalization sites of GateRV3 are covered: the one that feeds `MetaGated.local` and the
    one that feeds `GatedCNNBlock.fc1`, which includes the global branch of every `MetaGated`.
    Returns the number of normalizations that were folded.
    """
    folded = 0
    for module in [module for module in model.modules() if isinstance(module, meta_gated_type)]:
        module.local[0] = _fold_norm(module.local[0], module.local[1], bare_norm)
        folded += 1
    for module in [module for module in model.modules() if isinstance(module, gate_block_type)]:
        module.norm = _fold_norm(module.norm, module.fc1, bare_norm)
        folded += 1
    return folded


def _fold_scales(model: Any, meta_gated_type: type, folded_meta_gated: type) -> int:
    """
    Fold each `MetaGated` residual scale into the projection of its scale attention.

    The gate is derived from the pooled feature and then multiplied by a per-channel constant, so
    scaling the attention's rows and bias by that constant is the same product with one multiply
    fewer. The replacement keeps `gamma1`, the global branch and the addition order untouched.
    """
    import torch

    folded = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if not isinstance(child, meta_gated_type):
                continue
            projection = child.sca[1]
            with torch.no_grad():
                gamma = child.gamma0.reshape(-1)
                projection.weight.mul_(gamma[:, None, None, None])
                if projection.bias is not None:
                    projection.bias.mul_(gamma)
            setattr(parent, name, folded_meta_gated(child))
            folded += 1
    return folded


def _project_decoders(model: Any, block_type: type, projected_decoder: type, info: ModelInfo) -> int:
    """
    Replace every decoder block with the branch-split projection and return the count.

    The stages are re-read instead of being reused from the structure check, so the count this
    returns is what the rewrite actually did rather than what the model was expected to hold.
    """
    decoders = _module_list(model, "decode", info)
    projected = 0
    for index, block in enumerate(decoders):
        if not isinstance(block, block_type):
            continue
        decoders[index] = projected_decoder(block)
        projected += 1
    return projected


def _rewrite_scope(model: Any, plan: tuple[list[Any], ...]) -> tuple[set[str], set[str], set[str]]:
    """
    List the parameter names the fold stages modify, drop and introduce.

    Fusion drops the training branches of a `Conv3XC` on purpose, but the folds are expected to be
    the only other stage that touches a weight. Naming those weights up front, from the source
    model, lets the finished rewrite assert that everything else survived bit-exactly and that it
    introduced nothing beyond the decoder's own projections.

    Returns
    -------
    tuple[set[str], set[str], set[str]]
        The names whose values the folds change in place, the names they remove and the names they
        add.
    """
    meta_gated, gate_blocks, decoders = plan
    paths = {id(module): name for name, module in model.named_modules()}

    def prefix(module: Any) -> str:
        name = paths[id(module)]
        return f"{name}." if name else ""

    changed: set[str] = set()
    dropped: set[str] = set()
    for module in meta_gated:
        base = prefix(module)
        changed.update(f"{base}{name}" for name in ("local.1.weight", "local.1.bias"))
        changed.update(f"{base}{name}" for name in ("sca.1.weight", "sca.1.bias"))
        dropped.update(f"{base}{name}" for name in ("local.0.scale", "local.0.offset", "gamma0"))
    for module in gate_blocks:
        base = prefix(module)
        changed.update(f"{base}{name}" for name in ("fc1.weight", "fc1.bias"))
        dropped.update(f"{base}{name}" for name in ("norm.scale", "norm.offset"))

    added: set[str] = set()
    for module in decoders:
        base = prefix(module)
        dropped.update(f"{base}{name}" for name in ("scale.0.weight", "shor.weight"))
        added.update(f"{base}{name}" for name in ("up.weight", "skip.weight"))
        if module.shor.bias is not None:
            dropped.add(f"{base}shor.bias")
            added.add(f"{base}skip.bias")
    return changed, dropped, added


def _check_fold_scope(
    model: Any,
    before: dict[str, Any],
    compact: set[str],
    scope: tuple[set[str], set[str], set[str]],
    info: ModelInfo,
) -> None:
    """
    Assert that the fold stages only rewrote the weights their plan owns.

    The finished model must hold exactly the parameters fusion left behind minus the dropped
    weights plus the decoder projections, and every weight no fold claims has to still match the
    source model bit for bit - the fold stages run on a copy, so this compares against the
    untouched tensors rather than against the copy itself. A fold that silently drops, replaces or
    drifts something outside its plan fails here instead of shipping a different parameter tree.
    """
    changed, dropped, added = scope
    untouched = {name: tensor for name, tensor in before.items() if name not in changed and name not in dropped}
    final = dict(model.named_parameters())
    expected = (compact - dropped) | added
    if set(final) != expected:
        details: list[str] = []
        introduced = sorted(set(final) - expected)
        removed = sorted(expected - set(final))
        if introduced:
            details.append(f"introduced {', '.join(introduced[:4])}")
        if removed:
            details.append(f"dropped {', '.join(removed[:4])}")
        raise RuntimeError(
            f"The fast {info.architecture} adapter changed the parameter tree the folds are allowed "
            f"to touch ({'; '.join(details)}). This is a bug in the adapter; the untouched model is "
            "still usable with --preset original."
        )
    check_preserved(untouched, {name: final[name] for name in untouched}, info)


def build_fast_model(model: nn.Module, info: ModelInfo) -> nn.Module:
    """
    Build the lean GateRV3 inference model as an independent copy.

    Four cumulative rewrites are applied to the copy, and each is an exact algebraic identity on
    an evaluation model:

    1. Every `Conv3XC` wrapper is replaced by the single convolution its 1x1/3x3/1x1 branch and
       side kernel reparameterize into. The stored inference kernel is recomputed from the live
       branches instead of trusted, because it goes stale whenever the branches are trained.
       GateRV3 already fuses on `train(False)`, so this stage is not free arithmetic back; it
       removes the inactive training branches and their wrapper from the parameter tree.
    2. Every `RMSNorm` affine transform is folded into the ungrouped 1x1 convolution that follows
       it, which leaves the same function with two learned vectors fewer per normalization.
    3. Every `MetaGated` residual scale is folded into the projection of its scale attention, so
       the local residual needs one multiply fewer.
    4. Every decoder block projects the upsampled branch and the skip separately, which moves the
       1x1 projection in front of the pixel shuffle and halves its width.

    The source model is never mutated. Fusion deliberately drops the training branches, so the
    parameter tree is compared outside the fused convolutions; the folds then have to leave every
    weight they do not rewrite bit-identical, which is asserted against the plan derived from the
    source model.

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
        If the architecture is not GateRV3, the model is not an FP32 evaluation model, or the
        installed resselt structure differs from the one this adapter was written against.
    """
    if info.architecture.strip().upper() != ARCHITECTURE.upper():
        raise ValueError(
            f"The GateRV3 `fast` adapter cannot prepare {info.architecture}; re-run with --preset original."
        )

    import torch

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"build_fast_model expects a torch.nn.Module, got {type(model).__name__}.")
    if model.training:
        raise ValueError(
            "The `fast` preset folds the normalization and residual scales, which is only equivalent "
            "to an evaluation forward pass. Load the model with load_checkpoint() or call .eval() "
            "before build_fast_model()."
        )
    non_fp32 = sorted({str(parameter.dtype) for parameter in model.parameters() if parameter.dtype != torch.float32})
    if non_fp32:
        raise ValueError(
            f"The `fast` preset requires FP32 weights, but the model has {', '.join(non_fp32)} "
            "parameters. FP32 is the precision contract: cast the model to float32 first."
        )

    classes = _gaterv3_classes()
    _gaterv3, conv3xc, _norm_type, meta_gated_type, gate_block_type, block_type = classes
    plan = _check_gaterv3_structure(model, classes, info)
    meta_gated, gate_blocks, decoders = plan

    bare_norm, folded_meta_gated, projected_decoder = _lean_classes()
    paths = conv3xc_paths(model, conv3xc)
    before = outside_conv3xc(model, paths)
    scope = _rewrite_scope(model, plan)

    result = copy.deepcopy(model).eval().requires_grad_(False)
    # 1. Collapse every reparameterizable convolution into its refreshed inference kernel.
    replaced_conv3xc = replace_conv3xc(result, conv3xc, info=info)
    fused = outside_conv3xc(result, paths)
    compact = set(dict(result.named_parameters()))
    check_preserved(before, fused, info)
    # 2. Fold every normalization affine transform into the convolution that follows it.
    folded_norms = _fold_norms(result, meta_gated_type, gate_block_type, bare_norm)
    # 3. Fold every residual scale into the scale-attention projection it multiplies.
    folded_scales = _fold_scales(result, meta_gated_type, folded_meta_gated)
    # 4. Project the decoder branches separately, in front of the pixel shuffle.
    projected_decoders = _project_decoders(result, block_type, projected_decoder, info)

    fast_model = result.eval().requires_grad_(False)
    counts = {
        "inference convolution": (replaced_conv3xc, len(paths)),
        "folded normalization": (folded_norms, len(meta_gated) + len(gate_blocks)),
        "folded residual scale": (folded_scales, len(meta_gated)),
        "projected decoder": (projected_decoders, len(decoders)),
    }
    wrong = [
        f"{name}: rewrote {actual} of {expected}" for name, (actual, expected) in counts.items() if actual != expected
    ]
    if wrong:
        raise RuntimeError(
            f"The fast GateRV3 adapter did not rewrite the whole model ({'; '.join(wrong)}). "
            "This is a bug in the adapter; the untouched model is still usable with --preset original."
        )
    _check_fold_scope(fast_model, before, compact, scope, info)
    return fast_model


def unshuffle_of(model: nn.Module, path: Path) -> int:
    """
    Report the input pixel-unshuffle factor, which is `1` for GateRV3.

    GateRV3 does contain `PixelUnshuffle` layers, but they sit inside the encoder's downsampling
    blocks, after a convolution that has already mixed the RGB channels: they shrink the feature
    grid, they do not rearrange the input image. The external contract therefore stays one pixel
    per pixel in, at the checkpoint's own channel count. A pixel unshuffle anywhere else in the
    tree would mean resselt built something this adapter does not describe, so it is rejected
    rather than reported as `1`.
    """
    import torch

    encoder = getattr(model, "gater_encode", ())
    internal = {
        f"gater_encode.{index}.scale.{offset}"
        for index, block in enumerate(encoder)
        for offset, layer in enumerate(getattr(block, "scale", ()))
        if isinstance(layer, torch.nn.PixelUnshuffle)
    }
    external = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.PixelUnshuffle) and name not in internal
    ]
    if external:
        raise ValueError(
            f"resselt built {type(model).__name__} with a PixelUnshuffle at {external[0]!r} for {path.name}; "
            "the GateRV3 adapter only describes the encoder's internal downsampling."
        )
    return 1


def cross_check(model: nn.Module, info: ModelInfo, path: Path) -> None:
    """
    Verify the normalized GateRV3 contract against the model resselt actually constructed.

    The effective scale is checked twice over: `model.scale` has to be the contract's scale, and
    the crop in `forward` keeps `H * self.scale` rows, which only holds when the decoder undoes
    every encoder block and the branch that feeds the crop scales by the same factor. The
    padding is checked against the encoder depth because `check_img_size` reflects enough pixels
    to survive the whole downsampling chain.
    """
    import torch

    gaterv3, _conv3xc, *_rest = _gaterv3_classes()
    if not isinstance(model, gaterv3):
        raise ValueError(
            f"resselt reported GateRV3 metadata but constructed a {type(model).__name__} for {path.name}; "
            "the installed resselt structure is not the one this backend was validated against."
        )

    problems: list[str] = []
    in_to_dim = getattr(model, "in_to_dim", None)
    if isinstance(in_to_dim, torch.nn.Conv2d):
        if in_to_dim.in_channels != info.in_channels:
            problems.append(f"in_to_dim takes {in_to_dim.in_channels} channels, expected {info.in_channels}")
    else:
        problems.append(f"in_to_dim is a {type(in_to_dim).__name__}, expected an nn.Conv2d")

    gamma = getattr(model, "gamma", None)
    shape = tuple(getattr(gamma, "shape", ()))
    if shape != (1, info.in_channels, 1, 1):
        problems.append(f"the input residual scale has shape {shape}, expected (1, {info.in_channels}, 1, 1)")

    encoder = getattr(model, "gater_encode", ())
    decoder = getattr(model, "decode", ())
    scale = getattr(model, "scale", None)
    if scale != info.scale:
        problems.append(f"the model upscales by {scale}, expected {info.scale}")
    pads = getattr(model, "pad", None)
    expected_pad = 2 ** len(encoder)
    if pads != expected_pad:
        problems.append(f"check_img_size pads by {pads}, expected {expected_pad} for {len(encoder)} encoder blocks")
    if len(encoder) != len(decoder):
        problems.append(
            f"the model has {len(encoder)} encoder and {len(decoder)} decoder blocks; the crop keeps "
            "H * scale rows, which only adds up when every downsampling block is undone"
        )

    dim_to_in = getattr(model, "dim_to_in", None)
    short_to_dim = getattr(model, "short_to_dim", None)
    if isinstance(scale, int) and scale > 1:
        upsampler = getattr(dim_to_in, "MetaUpsample", None)
        internal = None if upsampler is None else int(upsampler[2])
        if internal != scale:
            problems.append(f"the upsampler is built for {internal}x, expected the {scale}x the crop keeps")
        factor = getattr(short_to_dim, "scale_factor", None)
        if factor != scale:
            problems.append(f"the input residual branch upscales by {factor}, expected {scale}")
    else:
        if not isinstance(dim_to_in, torch.nn.Conv2d):
            problems.append(f"dim_to_in is a {type(dim_to_in).__name__}, expected an nn.Conv2d at 1x")
        if not isinstance(short_to_dim, torch.nn.Identity):
            problems.append(f"short_to_dim is a {type(short_to_dim).__name__}, expected nn.Identity at 1x")
    output = getattr(dim_to_in, "out_channels", None)
    if isinstance(output, int) and output != info.out_channels:
        problems.append(f"dim_to_in returns {output} channels, expected {info.out_channels}")

    if info.unshuffle != 1:
        problems.append(
            f"the normalized contract reports a {info.unshuffle}x unshuffle, GateRV3 consumes pixels directly"
        )
    if problems:
        raise ValueError(
            f"resselt reported GateRV3 metadata for {path.name} that does not match the constructed model: "
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
    Decline to describe GateRV3 from the checkpoint header alone.

    The supplied checkpoints are `.pth` archives, whose tensors the loader only sees after resselt
    has rebuilt the architecture, and a safetensors copy would still have to have its encoder
    depth, feature width and external channels read out of the state dict. A wrong guess here is
    a silently wrong tile grid, so the loader falls back to the architecture resselt owns.

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

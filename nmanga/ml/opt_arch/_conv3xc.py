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

# Conv3XC reparameterization shared by the SPAN and GateRV3 `fast` adapters.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..contracts import ModelInfo

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor, nn

__all__ = (
    "check_preserved",
    "conv3xc_paths",
    "fuse_inference_conv",
    "outside_conv3xc",
    "replace_conv3xc",
)

BRANCHES = 3
"""How many convolutions the 1x1/3x3/1x1 branch of a `Conv3XC` holds."""


def _structure_error(message: str, info: ModelInfo) -> ValueError:
    """Build the actionable error for a `Conv3XC` this adapter does not know."""
    return ValueError(
        f"{message} The `fast` preset is written against resselt "
        f"{info.loader_version or '(unknown version)'} for {info.architecture} checkpoints; re-run "
        "with --preset original if this checkpoint is not the validated structure."
    )


def _check_branch(module: Any, where: str, info: ModelInfo) -> None:
    """Validate the three convolutions and the side kernel a `Conv3XC` fuses."""
    import torch

    conv = getattr(module, "conv", None)
    if not isinstance(conv, torch.nn.Sequential) or len(conv) != BRANCHES:
        raise _structure_error(
            f"{where}.conv is a {type(conv).__name__}, expected a {BRANCHES}-convolution nn.Sequential.", info
        )
    for index, layer in enumerate(conv):
        if not isinstance(layer, torch.nn.Conv2d):
            raise _structure_error(f"{where}.conv[{index}] is a {type(layer).__name__}, expected an nn.Conv2d.", info)
        if layer.weight.requires_grad:
            # The fused kernel is derived from these tensors; a trainable branch means the whole
            # reparameterization is only valid for the current weight values.
            raise _structure_error(
                f"{where}.conv[{index}] is still trainable; the `fast` preset fuses a frozen evaluation model.",
                info,
            )
    side = getattr(module, "sk", None)
    if not isinstance(side, torch.nn.Conv2d):
        raise _structure_error(f"{where}.sk is a {type(side).__name__}, expected an nn.Conv2d.", info)
    if tuple(side.kernel_size) != (1, 1):
        raise _structure_error(f"{where}.sk has kernel {tuple(side.kernel_size)}, expected 1x1.", info)


def fuse_inference_conv(module: Any, *, where: str, info: ModelInfo) -> nn.Conv2d:
    """
    Recompute one `Conv3XC`'s fused kernel and return its inference convolution.

    The stored `eval_conv` tensors are never trusted: the fused weight and bias are recomputed
    from the live 1x1/3x3/1x1 branch and the side kernel with the architecture's own
    `update_params`, which is the same arithmetic its evaluation forward uses. The returned
    convolution keeps the branch's stride, border padding and bias presence, so it replaces the
    wrapper without changing the output.

    Parameters
    ----------
    module : Conv3XC
        The module to reparameterize, already inside the `fast` copy.
    where : str
        The dotted module path, used for the error message.
    info : ModelInfo
        The normalized contract of the checkpoint, used for the error message.

    Returns
    -------
    nn.Conv2d
        The refreshed inference convolution to put in the module's place.

    Raises
    ------
    ValueError
        If the module is not the 1x1/3x3/1x1 plus side-kernel structure the fusion needs.
    """
    import torch

    if not callable(getattr(module, "update_params", None)):
        raise _structure_error(f"{where} has no update_params, expected resselt's Conv3XC.", info)
    _check_branch(module, where, info)

    conv = module.eval_conv
    if not isinstance(conv, torch.nn.Conv2d):
        raise _structure_error(f"{where}.eval_conv is a {type(conv).__name__}, expected an nn.Conv2d.", info)
    if tuple(conv.kernel_size) != (3, 3) or tuple(conv.padding) != (1, 1):
        raise _structure_error(
            f"{where}.eval_conv is {tuple(conv.kernel_size)}x{tuple(conv.kernel_size)} with padding "
            f"{tuple(conv.padding)}, expected a 3x3 border-padded convolution.",
            info,
        )
    if tuple(int(axis) for axis in conv.stride) != (module.stride, module.stride):
        raise _structure_error(
            f"{where}.eval_conv has stride {tuple(conv.stride)} but the module declares {module.stride}.", info
        )

    # GateRV3's Conv3XC supports bias-free branches; SPAN's does not store a flag because its
    # convolutions always carry a bias. `update_params` decides which tensors it derives.
    has_bias = bool(getattr(module, "bias", True))
    if (conv.bias is not None) != has_bias:
        raise _structure_error(
            f"{where}.eval_conv has bias={conv.bias is not None} but the module declares bias={has_bias}.", info
        )

    expected = (module.sk.out_channels, module.sk.in_channels, 3, 3)
    with torch.no_grad():
        module.update_params()
        if tuple(conv.weight.shape) != expected:
            raise _structure_error(f"{where} fused to a {tuple(conv.weight.shape)} kernel, expected {expected}.", info)
        conv.weight.requires_grad_(False)
        if conv.bias is not None:
            conv.bias.requires_grad_(False)
    return conv


def replace_conv3xc(model: nn.Module, conv_type: type, *, info: ModelInfo) -> int:
    """
    Replace every `Conv3XC` in the module tree with its refreshed inference convolution.

    The rewrite runs on the copy passed in, never on the source model, and returns how many
    wrappers it collapsed so the caller can compare that against the count it derived.

    Parameters
    ----------
    model : nn.Module
        The `fast` copy to rewrite in place.
    conv_type : type
        The architecture's concrete `Conv3XC` class, taken from its resselt module.
    info : ModelInfo
        The normalized contract of the checkpoint, used for the error message.

    Returns
    -------
    int
        The number of `Conv3XC` wrappers that were replaced.
    """
    replaced = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, conv_type):
                where = f"{name}" if parent is model else f"{type(parent).__name__}.{name}"
                setattr(parent, name, fuse_inference_conv(child, where=where, info=info))
                replaced += 1
    return replaced


def conv3xc_paths(model: nn.Module, conv_type: type) -> tuple[str, ...]:
    """
    List the dotted paths of every `Conv3XC` in the module tree.

    The rewrite puts the inference convolution back at the same attribute, so the paths are
    identical before and after it. Callers take them from the source model, where the wrappers
    still exist, and reuse them on the rewritten copy.

    Parameters
    ----------
    model : nn.Module
        The model to describe.
    conv_type : type
        The architecture's concrete `Conv3XC` class.

    Returns
    -------
    tuple[str, ...]
        The dotted paths, empty when the model holds no wrapper.
    """
    return tuple(name for name, module in model.named_modules() if isinstance(module, conv_type))


def outside_conv3xc(model: nn.Module, paths: tuple[str, ...]) -> dict[str, Tensor]:
    """
    Fingerprint every parameter that does not live under a `Conv3XC` module.

    Fusion deliberately drops the training branches, so the whole-tree comparison the FDAT
    adapter makes does not apply. Every weight the rewrite is not supposed to touch has to
    survive bit-exactly, which is what this mapping lets the caller assert.

    Parameters
    ----------
    model : nn.Module
        The model to describe.
    paths : tuple[str, ...]
        The wrapper paths from `conv3xc_paths`, taken from the source model.

    Returns
    -------
    dict[str, Tensor]
        Parameter name to tensor, excluding the `Conv3XC` subtrees.
    """
    return {
        name: tensor
        for name, tensor in model.named_parameters()
        if not any(name == path or name.startswith(f"{path}.") for path in paths)
    }


def check_preserved(before: Mapping[str, Tensor], after: Mapping[str, Tensor], info: ModelInfo) -> None:
    """
    Assert that a rewrite left every non-reparameterized weight alone.

    Parameters
    ----------
    before : Mapping[str, Tensor]
        The fingerprint taken from the source model.
    after : Mapping[str, Tensor]
        The fingerprint taken from the rewritten copy.
    info : ModelInfo
        The normalized contract of the checkpoint, used for the error message.

    Raises
    ------
    RuntimeError
        If a weight outside the fused convolutions was dropped, added or changed.
    """
    import torch

    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(name for name, tensor in before.items() if name in after and not torch.equal(tensor, after[name]))
    if missing or added or changed:
        details = []
        if missing:
            details.append(f"dropped {', '.join(missing[:4])}")
        if added:
            details.append(f"added {', '.join(added[:4])}")
        if changed:
            details.append(f"changed {', '.join(changed[:4])}")
        raise RuntimeError(
            f"The fast {info.architecture} adapter modified weights it must only carry over "
            f"({'; '.join(details)}). This is a bug in the adapter; the untouched model is still "
            "usable with --preset original."
        )

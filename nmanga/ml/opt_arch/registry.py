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

# The `fast` preset dispatcher: one adapter module per architecture.

from __future__ import annotations

from typing import TYPE_CHECKING

from ..contracts import ModelInfo

if TYPE_CHECKING:
    from types import ModuleType

    from torch import nn

__all__ = ("FAST_PRESET_ARCHITECTURES", "adapter_for", "build_fast_model", "supports_fast_preset")

FAST_PRESET_ARCHITECTURES: tuple[str, ...] = ("FDAT",)
"""The architectures that have an adapted `fast` recipe."""


def adapter_for(architecture: str) -> ModuleType | None:
    """
    Resolve the rewrite adapter module for an architecture.

    Parameters
    ----------
    architecture : str
        The normalized architecture name reported by the checkpoint loader.

    Returns
    -------
    ModuleType | None
        The adapter module, or `None` when the architecture has no `fast` recipe.
    """
    name = architecture.strip().upper()
    if name == "FDAT":
        from . import fdat

        return fdat
    return None


def supports_fast_preset(info: ModelInfo) -> bool:
    """
    Report whether the loaded architecture has an adapted lean recipe.

    Parameters
    ----------
    info : ModelInfo
        The normalized contract of a loaded checkpoint.

    Returns
    -------
    bool
        `True` when a `fast` adapter exists for the architecture.
    """
    return adapter_for(info.architecture) is not None


def build_fast_model(model: nn.Module, info: ModelInfo) -> nn.Module:
    """
    Build the lean inference model as an independent copy.

    Parameters
    ----------
    model : nn.Module
        The untouched model returned by `load_checkpoint`, in FP32 evaluation mode.
    info : ModelInfo
        Its normalized contract.

    Returns
    -------
    nn.Module
        An independent, lean FP32 evaluation model.

    Raises
    ------
    ValueError
        If the architecture has no `fast` recipe.
    """
    adapter = adapter_for(info.architecture)
    if adapter is None:
        raise ValueError(
            f"{info.architecture} checkpoints are not supported by the `fast` preset; re-run with "
            f"--preset original (only {', '.join(FAST_PRESET_ARCHITECTURES)} has an adapted lean recipe)."
        )
    return adapter.build_fast_model(model, info)

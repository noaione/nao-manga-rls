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

from importlib import import_module
from typing import TYPE_CHECKING

from ..contracts import ModelInfo

if TYPE_CHECKING:
    from types import ModuleType

    from torch import nn

__all__ = (
    "FAST_PRESET_ARCHITECTURES",
    "adapter_for",
    "build_fast_model",
    "layout_for",
    "supports_fast_preset",
)

FAST_PRESET_ARCHITECTURES: tuple[str, ...] = ("FDAT", "SPAN", "GateRV3")
"""The architectures that have an adapted `fast` recipe, by their resselt display name."""

_ADAPTER_MODULES: dict[str, str] = {
    "FDAT": "fdat",
    "SPAN": "span",
    "GATERV3": "gaterv3",
}
"""Normalized architecture key to the adapter module that implements its recipe."""

_CHANNELS_LAST_ARCHITECTURES = frozenset({"FDAT"})
"""The architectures whose `fast` recipe was measured with the channels-last memory format."""


def _key(architecture: str) -> str:
    """Normalize a resselt display name into a dispatch key, so `GateRV3` and `GATERV3` agree."""
    return architecture.strip().upper()


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
    module_name = _ADAPTER_MODULES.get(_key(architecture))
    if module_name is None:
        return None
    # The adapter modules only import torch inside their class factories and their rewrites,
    # so importing one never pulls torch, resselt or CUDA into a process that only asks.
    return import_module(f"{__package__}.{module_name}")


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


def layout_for(architecture: str) -> bool:
    """
    Report the memory format the architecture's `fast` recipe was measured with.

    Channels-last is only a measured win for FDAT, so every other architecture starts at the
    contiguous layout until its own comparison justifies the change. The runtime still gates
    this on the device and the preset; an explicit `TorchRuntimeConfig` override wins.

    Parameters
    ----------
    architecture : str
        The normalized architecture name reported by the checkpoint loader.

    Returns
    -------
    bool
        `True` when the adapter selects the channels-last memory format.
    """
    return _key(architecture) in _CHANNELS_LAST_ARCHITECTURES


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

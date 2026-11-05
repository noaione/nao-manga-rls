"""
Tests for the ``orchestra run-target`` support helpers.

Nobody here runs an actual action chain: the goal is to pin down how a target directory
is turned into a :class:`VolumeConfig`, and which actions are refused up-front because
they need volume metadata that a target run does not have.
"""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from nmanga.cli.orchestrator import _resolve_target_path, _validate_target_actions, orchestractor
from nmanga.orchestrator import (
    ActionColorDetect,
    ActionMoveColor,
    ActionRename,
    ActionShiftName,
    ActionSpreads,
    OrchestratorConfig,
    build_target_volume,
)


def _config(actions: list[dict]) -> OrchestratorConfig:
    return OrchestratorConfig.model_validate({
        "title": "Test Series",
        "publisher": "Test Publisher",
        "credit": "Ripper",
        "email": "ripper@example.com",
        "base_path": "source",
        "volumes": [{"path": "v01", "number": 1}],
        "actions": actions,
    })


def _cwd_action() -> dict:
    return {"kind": "change_cwd", "directory": "denoised"}


# --------------------------------------------------------------------------------------
# Target path resolution
# --------------------------------------------------------------------------------------


def test_resolve_target_inside_base_path(tmp_path: Path):
    base = tmp_path / "source"
    target = base / "v01"
    target.mkdir(parents=True)
    config = _config([_cwd_action()])

    resolved, relative = _resolve_target_path(config, tmp_path, target)

    assert resolved == target.resolve()
    assert relative == Path("v01")
    # The invariant the actions rely on: root_dir / base_path / volume.path == target
    assert (tmp_path / config.base_path / relative).resolve() == resolved


def test_resolve_nested_target(tmp_path: Path):
    target = tmp_path / "source" / "v01" / "denoised"
    target.mkdir(parents=True)
    config = _config([_cwd_action()])

    _, relative = _resolve_target_path(config, tmp_path, target)

    assert relative == Path("v01/denoised")


def test_resolve_target_outside_base_path(tmp_path: Path):
    target = tmp_path / "elsewhere" / "v01"
    target.mkdir(parents=True)
    config = _config([_cwd_action()])

    with pytest.raises(click.ClickException, match="not inside the configured base path"):
        _resolve_target_path(config, tmp_path, target)


def test_resolve_target_is_base_path(tmp_path: Path):
    base = tmp_path / "source"
    base.mkdir()
    config = _config([_cwd_action()])

    with pytest.raises(click.ClickException, match="cannot be the base path itself"):
        _resolve_target_path(config, tmp_path, base)


# --------------------------------------------------------------------------------------
# Volume synthesis
# --------------------------------------------------------------------------------------


def test_build_target_volume_defaults_to_oneshot():
    volume = build_target_volume("v01")

    assert volume.oneshot is True
    assert volume.number == 1
    assert volume.number_display == "1"
    assert volume.colors is None
    assert volume.chapters == []
    assert volume.spreads == []
    assert volume.meta_naming == []


def test_build_target_volume_with_number():
    volume = build_target_volume("v02", number=2, revision=2, extra_text="PNG4")

    assert volume.oneshot is False
    assert volume.number == 2
    assert volume.revision == 2
    assert volume.extra_text == "PNG4"


def test_build_target_volume_with_half_number():
    volume = build_target_volume("v01.5", number=1.5)

    assert volume.number == 1.5  # ruff: ignore[float-equality-comparison]
    assert volume.oneshot is False


def test_build_target_volume_with_omnibus_range():
    volume = build_target_volume("v01-02", number=(2, 1))

    assert volume.number == (1, 2)
    assert volume.is_omnibus is True
    assert volume.number_display == "1-2"


def test_build_target_volume_oneshot_wins_over_number():
    volume = build_target_volume("v02", number=2, oneshot=True)

    assert volume.oneshot is True
    assert volume.number == 2


def test_build_target_volume_keeps_colors_unknown():
    # `None` means "not known yet" so color actions detect them on the fly, instead of
    # `[]` which would mean "this volume has no color pages at all".
    assert build_target_volume("v01").colors is None


@pytest.mark.parametrize("action_cls", [ActionRename, ActionSpreads])
def test_actions_without_target_support(action_cls: type):
    assert action_cls.supported_in_target_run is False


@pytest.mark.parametrize(
    "action_cls",
    [ActionShiftName, ActionColorDetect, ActionMoveColor],
)
def test_actions_with_target_support(action_cls: type):
    assert action_cls.supported_in_target_run is True


def test_validate_target_actions_rejects_rename():
    config = _config([{"kind": "shift_rename"}, {"kind": "rename"}])

    with pytest.raises(click.ClickException, match="rename-1"):
        _validate_target_actions(config)


def test_validate_target_actions_rejects_spreads():
    config = _config([{"kind": "spreads"}])

    with pytest.raises(click.ClickException, match="spreads-1"):
        _validate_target_actions(config)


def test_validate_target_actions_keeps_step_names():
    config = _config([{"kind": "shift_rename"}, {"kind": "shift_rename"}, {"kind": "optimize"}])

    # Step names must match `OrchestratorConfig.action_names` so `skip_actions` keeps working
    assert list(_validate_target_actions(config)) == config.action_names


def test_run_target_command_is_registered():
    assert "run-target" in orchestractor.commands

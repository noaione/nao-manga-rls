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

from __future__ import annotations

import os
import subprocess as sp
from pathlib import Path
from typing import Literal, Mapping, TypeAlias

__all__ = ("SupportedShell", "detect_shell")

SupportedShell: TypeAlias = Literal["pwsh", "bash", "fish", "zsh"]


def _shell_from_click_completion_env(env: Mapping[str, str]) -> SupportedShell | None:
    for key, value in env.items():
        if not key.endswith("_COMPLETE"):
            continue

        normalized = value.lower()
        if normalized.startswith("pwsh_"):
            return "pwsh"
        if normalized.startswith("bash_"):
            return "bash"
        if normalized.startswith("fish_"):
            return "fish"
        if normalized.startswith("zsh_"):
            return "zsh"
    return None


def _shell_from_shell_env(env: Mapping[str, str]) -> SupportedShell | None:
    shell_value = env.get("SHELL") or env.get("STARSHIP_SHELL")
    if not shell_value:
        return None

    shell_name = Path(shell_value).name.lower()
    if shell_name in {"bash", "bash.exe"}:
        return "bash"
    if shell_name in {"fish", "fish.exe"}:
        return "fish"
    if shell_name in {"zsh", "zsh.exe"}:
        return "zsh"
    if shell_name in {"pwsh", "pwsh.exe"}:
        return "pwsh"
    return None


def _is_powershell_7_environment(env: Mapping[str, str]) -> bool:
    if env.get("POWERSHELL_DISTRIBUTION_CHANNEL"):
        return True

    ps_module_path = env.get("PSModulePath", "").lower().replace("\\", "/")
    # Pwsh 7 module paths commonly include /powershell/7/
    return "/powershell/7" in ps_module_path


def detect_shell(env: Mapping[str, str] | None = None) -> SupportedShell | None:
    """
    Detect the active user shell from environment variables.

    Returns one of the currently supported shell identifiers:
    - "pwsh" (PowerShell 7)
    - "bash"
    - "fish"
    - "zsh"

    Returns ``None`` when the shell cannot be determined or is unsupported.
    """
    current_env = env or os.environ

    click_shell = _shell_from_click_completion_env(current_env)
    if click_shell is not None:
        return click_shell

    shell_env = _shell_from_shell_env(current_env)
    if shell_env is not None:
        if shell_env == "pwsh" and not _is_powershell_7_environment(current_env):
            return None
        return shell_env

    if _is_powershell_7_environment(current_env):
        return "pwsh"

    return None


def install_completion_for_pwsh():
    """Install shell completion for PowerShell 7."""

    cmd_name = "nmanga"
    # Get current shell path
    profile = (
        sp.run(["pwsh", "-c", "echo", "$PROFILE"], shell=False, capture_output=True).stdout.decode("utf-8").strip()  # noqa: S607
    )
    profile = Path(profile)

    completion_profile = profile.parent / ".nmanga_completion_profile.ps1"
    completion_varname = "_NMANGA_COMPLETE"

    cmd = "pwsh -c \"$env:{0} = 'pwsh_source'; {1} > '{2}'; $env:{0} = $null\"".format(
        completion_varname, cmd_name, str(completion_profile)
    )
    sp.run(cmd, shell=True)  # noqa: S602

    # read profile path
    if profile.exists():
        with profile.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = []

    # find .nmanga_completion_profile.ps1 line
    has_found = False
    for line in lines:
        if ".nmanga_completion_profile.ps1" in line.strip():
            has_found = True
            break
    if not has_found:
        # not found? add it
        with profile.open("a", encoding="utf-8") as fp:
            fp.write(f"\n& '{completion_profile!s}'\n")


def install_completion_for_bash():
    """Install shell completion for Bash."""

    # eval "$(_FOO_BAR_COMPLETE=bash_source foo-bar)"

    # ~/.bashrc
    home_path = Path.home()
    bashrc_path = home_path / ".bashrc"
    completion_varname = "_NMANGA_COMPLETE"
    cmd_name = "nmanga"

    # read file
    if bashrc_path.exists():
        with bashrc_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = []

    # find _NMANGA_COMPLETE line
    has_found = False
    for line in lines:
        if completion_varname in line.strip():
            has_found = True
            break
    if not has_found:
        # not found? add it
        with bashrc_path.open("a", encoding="utf-8") as fp:
            fp.write(f'\neval "$({completion_varname}=bash_source {cmd_name})"')


def install_completion_for_zsh():
    """Install shell completion for zsh."""

    # eval "$(_FOO_BAR_COMPLETE=zsh_source foo-bar)"

    # ~/.zshrc
    home_path = Path.home()
    bashrc_path = home_path / ".zshrc"
    completion_varname = "_NMANGA_COMPLETE"
    cmd_name = "nmanga"

    # read file
    if bashrc_path.exists():
        with bashrc_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = []

    # find _NMANGA_COMPLETE line
    has_found = False
    for line in lines:
        if completion_varname in line.strip():
            has_found = True
            break
    if not has_found:
        # not found? add it
        with bashrc_path.open("a", encoding="utf-8") as fp:
            fp.write(f'\neval "$({completion_varname}=zsh_source {cmd_name})"')


def install_completion_for_fish():
    """Install shell completion for Fish."""

    # _FOO_BAR_COMPLETE=fish_source foo-bar | source
    # ~/.config/fish/completions/foo-bar.fish

    home_path = Path.home()
    fish_completions_path = home_path / ".config" / "fish" / "completions"
    fish_completions_path.mkdir(parents=True, exist_ok=True)

    completion_varname = "_NMANGA_COMPLETE"
    cmd_name = "nmanga"

    completion_script_path = fish_completions_path / f"{cmd_name}.fish"
    completion_script_path.write_text(
        f"{completion_varname}=fish_source {cmd_name} | source",
        encoding="utf-8",
    )

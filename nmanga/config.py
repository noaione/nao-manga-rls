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

# The config handler, which contains some defaults paramters that can be changed for
# other people using this script.

import json
import os.path
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ._ntypes import BracketTypeT, ConfigT, _ConfigDefaultsT, _ConfigExecutableT

__all__ = (
    "get_config",
    "ConfigHandler",
    "ConfigError",
    "Config",
)

if sys.platform == "win32":
    CONFIG_DIR = Path(os.path.expandvars(r"%APPDATA%\nmanga"))
else:
    CONFIG_DIR = Path.expanduser(Path("~/.config/nmanga"))


@dataclass
class _ConfigExecutable:
    magick_path: str = field(default="magick")
    pingo_path: str = field(default="pingo")
    exiftool_path: str = field(default="exiftool")

    def to_dict(self) -> _ConfigExecutableT:
        return {
            "magick_path": self.magick_path,
            "pingo_path": self.pingo_path,
            "exiftool_path": self.exiftool_path,
        }


@dataclass
class _ConfigDefaults:
    bracket_type: BracketTypeT = field(default="round")
    ripper_credit: str = field(default="nao")
    ripper_email: str = field(default="noaione@protonmail.ch")

    def to_dict(self) -> _ConfigDefaultsT:
        return {
            "bracket_type": self.bracket_type,
            "ripper_credit": self.ripper_credit,
            "ripper_email": self.ripper_email,
        }


@dataclass
class Config:
    defaults: _ConfigDefaults = field(default_factory=_ConfigDefaults)
    executables: _ConfigExecutable = field(default_factory=_ConfigExecutable)

    def to_dict(self) -> ConfigT:
        return {
            "defaults": self.defaults.to_dict(),
            "executables": self.executables.to_dict(),
        }


class ConfigError(TypeError):
    pass


class ConfigHandler:
    def __init__(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        self.__config_file = CONFIG_DIR / "config.json"
        self.__config = Config()
        self._is_first_time_warn = False

        self.read_and_parse()

    def _parse_config(self, json_data: ConfigT) -> Config:
        defaults = json_data.get("defaults", {})
        executables = json_data.get("executables", {})

        if not isinstance(defaults, dict):
            raise ConfigError("`defaults` must be a dict")
        if not isinstance(executables, dict):
            raise ConfigError("`executables` must be a dict")

        config = Config()

        # Parse defaults
        bracket_type = defaults.get("bracket_type", "round")
        if not isinstance(bracket_type, str):
            raise ConfigError("`defaults.bracket_type` must be a string")

        ripper_credit = defaults.get("ripper_credit", "nao")
        if not isinstance(ripper_credit, str):
            raise ConfigError("`defaults.ripper_credit` must be a string")

        ripper_email = defaults.get("ripper_email", "noaione@protonmail.ch")
        if not isinstance(ripper_email, str):
            raise ConfigError("`defaults.ripper_email` must be a string")

        config.defaults = _ConfigDefaults(
            bracket_type=bracket_type, ripper_credit=ripper_credit, ripper_email=ripper_email
        )

        magick_path = executables.get("magick_path", "magick")
        if not isinstance(magick_path, str):
            raise ConfigError("`executables.magick_path` must be a string")

        pingo_path = executables.get("pingo_path", "pingo")
        if not isinstance(pingo_path, str):
            raise ConfigError("`executables.pingo_path` must be a string")

        exiftool_path = executables.get("exiftool_path", "exiftool")
        if not isinstance(exiftool_path, str):
            raise ConfigError("`executables.exiftool_path` must be a string")

        config.executables = _ConfigExecutable(
            magick_path=magick_path, pingo_path=pingo_path, exiftool_path=exiftool_path
        )

        is_first_time = bool(json_data.get("_is_first_time", False))
        self._is_first_time_warn = is_first_time

        return config

    def read_and_parse(self) -> None:
        if self.__config_file.exists():
            with self.__config_file.open("r") as f:
                parsed_config = json.load(f)

            parsed = self._parse_config(parsed_config)
            self.__config = parsed
        else:
            self.save_config(self.__config, True)
            self._is_first_time_warn = True

    def save_config(self, config: Config, mark_first_time: bool = False) -> None:
        as_dict = config.to_dict()
        self.__config = config

        as_dict["_is_first_time"] = mark_first_time

        with self.__config_file.open("w") as f:
            json.dump(as_dict, f, indent=4, ensure_ascii=False)

    @property
    def config(self) -> Config:
        return self.__config

    def is_first_time(self) -> bool:
        return self._is_first_time_warn


_config_handler: Optional[ConfigHandler] = None


def get_config_handler() -> ConfigHandler:
    global _config_handler

    if _config_handler is None:
        # Initialize here to prevent some stupid thing
        _config_handler = ConfigHandler()

    return _config_handler


def get_config() -> Config:
    return get_config_handler().config

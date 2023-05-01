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

import functools
from typing import Callable, Optional, Tuple

import click

from .. import config, term
from .base import CatchAllExceptionsCommand, test_or_find_exiftool, test_or_find_magick, test_or_find_pingo

__all__ = ("cli_config",)
cfhandler = config.get_config_handler()
console = term.get_console()


SAVE_CHOICE = term.ConsoleChoice("save_data", "Save")
CANCEL_CHOICE = term.ConsoleChoice("cancel_data", "Cancel")

# --- Defaults --- #

_default_bracket_index = ["round", "square", "curly"]


def _check_defaults_inquire_validator(text: str):
    if not isinstance(text, str):
        return False

    if len(text.strip()) < 1:
        return False

    return True


def _loop_defaults_bracket_type(config: config.Config) -> config.Config:
    choice_round = term.ConsoleChoice("round", "Round bracket type -> (nao)")
    choice_square = term.ConsoleChoice("square", "Square bracket type -> [nao]")
    choice_curly = term.ConsoleChoice("curly", "Curly bracket type -> {nao}")

    default_idx = _default_bracket_index.index(config.defaults.bracket_type)
    choices = [choice_round, choice_square, choice_curly]

    select_option = console.choice(
        f"Select default bracket type [current: {config.defaults.bracket_type}]",
        choices=choices,
        default=choices[default_idx],
    )

    config.defaults.bracket_type = select_option.name
    return config


def _loop_defaults_ripper_credit(config: config.Config) -> config.Config:
    ripper_name = console.inquire(
        "Ripper credit name",
        validation=_check_defaults_inquire_validator,
        default=config.defaults.ripper_credit,
    )

    config.defaults.ripper_credit = ripper_name
    return config


def _loop_defaults_ripper_email(config: config.Config) -> config.Config:
    ripper_email = console.inquire(
        "Ripper credit email",
        validation=_check_defaults_inquire_validator,
        default=config.defaults.ripper_email,
    )

    config.defaults.ripper_email = ripper_email
    return config


def _loop_defaults_sections(config: config.Config) -> config.Config:
    while True:
        select_option = console.choice(
            "Select what you want to do for defaults section",
            choices=[
                term.ConsoleChoice("bracket_type", "Configure default bracket type"),
                term.ConsoleChoice("ripper_credit", "Configure default ripper credit name"),
                term.ConsoleChoice("ripper_email", "Configure default ripper credit email"),
                SAVE_CHOICE,
            ],
        )

        option = select_option.name
        if option == SAVE_CHOICE.name:
            return config
        elif option == "bracket_type":
            config = _loop_defaults_bracket_type(config)
        elif option == "ripper_credit":
            config = _loop_defaults_ripper_credit(config)
        elif option == "ripper_email":
            config = _loop_defaults_ripper_email(config)
        else:
            console.warning("Invalid option selected")
            console.sleep(2)
        console.enter()


# --- Executables --- #


def _check_executables_inquire_validation(text: str, validator_func: Callable[[str, bool], Optional[str]]):
    if not _check_defaults_inquire_validator(text):
        return False

    if text.lower() == "skip this":
        return True

    validate_res = validator_func(text, False)

    return validate_res is not None


def _loop_executables_check_single(
    executable: str,
    default: str,
    validator_func: Callable[[str, bool], Optional[str]],
) -> Optional[str]:
    console.info(f"Configuring executable `{executable}`")
    console.info("You can skip or cancel this by typing `skip this` (without the backticks)")
    console.info("This section will check if the executable is valid or not")

    validation_wrap = functools.partial(_check_executables_inquire_validation, validator_func=validator_func)

    executable_path = console.inquire(
        f"Executable path for `{executable}`",
        validation=validation_wrap,
        default=default,
    )

    return executable_path if executable_path.lower() != "skip this" else None


def _loop_executables_sections(config: config.Config):
    while True:
        select_option = console.choice(
            "Select what you want to do for executables section",
            choices=[
                term.ConsoleChoice("pingo_path", "Configure `pingo` path"),
                term.ConsoleChoice("exiftool_path", "Configure `exiftool` path"),
                term.ConsoleChoice("magick_path", "Configure `magick` path"),
                SAVE_CHOICE,
            ],
        )

        option = select_option.name
        if option == SAVE_CHOICE.name:
            return config
        elif option == "pingo_path":
            result = _loop_executables_check_single("pingo", config.executables.pingo_path, test_or_find_pingo)
            if result is not None:
                config.executables.pingo_path = result
        elif option == "ripper_credit":
            result = _loop_executables_check_single("exiftool", config.executables.exiftool_path, test_or_find_exiftool)
            if result is not None:
                config.executables.exiftool_path = result
        elif option == "ripper_email":
            result = _loop_executables_check_single("magick", config.executables.magick_path, test_or_find_magick)
            if result is not None:
                config.executables.magick_path = result
        else:
            console.warning("Invalid option selected")
            console.sleep(2)
        console.enter()


# --- Main --- #


def _loop_main_sections(config: config.Config) -> Tuple[bool, config.Config]:
    while True:
        select_option = console.choice(
            "Select what you want to configure",
            choices=[
                term.ConsoleChoice("defaults", "Configure defaults"),
                term.ConsoleChoice("executables", "Configure executables"),
                SAVE_CHOICE,
                CANCEL_CHOICE,
            ],
        )

        option = select_option.name
        if option == SAVE_CHOICE.name:
            return True, config
        elif option == CANCEL_CHOICE.name:
            return False, config
        elif option == "defaults":
            console.enter()
            config = _loop_defaults_sections(config)
        elif option == "executables":
            console.enter()
            config = _loop_executables_sections(config)
        console.enter()


@click.command("config", help="Configure nmanga CLI", cls=CatchAllExceptionsCommand)
def cli_config():
    config = cfhandler.config

    do_save, config = _loop_main_sections(config)

    console.enter()
    if do_save:
        console.info("Saving configuration...")
        cfhandler.save_config(config)
        console.info("Configuration saved!")
    else:
        console.warning("Canceling configuration...")
        # This will just stop the first time warning
        cfhandler.save_config(config)
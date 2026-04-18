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

import atexit
import os
import signal
import sys
from multiprocessing import current_process
from pathlib import Path
from typing import NoReturn

import rich_click as click
from click_pwsh import support_pwsh_shell_completion

from ._metadata import __author__, __name__, __version__
from .cli.archive import pack_releases, pack_releases_comment_archive, pack_releases_epub_mode
from .cli.auto_split import auto_split
from .cli.autolevel import analyze_level, autolevel, autolevel2, force_gray
from .cli.base import NMangaCommandHandler
from .cli.config import cli_config
from .cli.debugging import simulate_progress
from .cli.denoiser import denoiser, denoiser_trt, identify_denoise_candidates
from .cli.extractor import epub_extractor
from .cli.image_optimizer import image_jpegify, image_mixmatch, image_optimizer
from .cli.image_tagging import image_tagging, image_tagging_raw
from .cli.lookup import lookup_group
from .cli.manual_split import manual_split
from .cli.merge_chapters import merge_chapters
from .cli.orchestrator import orchestractor
from .cli.posterize import analyze_shades, auto_posterize, posterize_simple
from .cli.releases import prepare_releases, prepare_releases_chapter
from .cli.renamer import shift_renamer
from .cli.rescaler import rescale_image
from .cli.spreads_manager import spreads
from .cli.timewizard import timewizard_modify
from .cli.upscaler import export_to_onnx, upscale_trt
from .term import detect_shell, get_console
from .term.shell import (
    install_completion_for_bash,
    install_completion_for_fish,
    install_completion_for_pwsh,
    install_completion_for_zsh,
)

support_pwsh_shell_completion()  # Import and register PowerShell 7 shell completion support for Click commands

console = get_console()

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
WORKING_DIR = Path.cwd().absolute()

help_config = click.RichHelpConfiguration(
    theme="quartz2-nu",
    style_option="bold cyan",
    style_argument="bold magenta",
    style_command="bold cyan",
    style_switch="bold italic green",
    style_metavar="bold yellow",
    style_metavar_separator="dim",
    style_usage="bold yellow",
    style_usage_command="bold",
    style_helptext_first_line="",
    style_helptext="dim",
    style_option_default="dim",
    style_required_short="red",
    style_required_long="dim red",
    style_options_panel_border="dim",
    style_commands_panel_border="dim",
    use_markdown=False,
    use_rich_markup=True,
    show_arguments=True,
    show_metavars_column=True,
    options_table_column_types=[
        "required",
        "opt_all",
        "metavar",
        "help",
    ],
    commands_table_column_types=[
        "name_with_aliases",
        "help",
    ],
    options_table_help_sections=[
        "help",
        "required",
        "default",
        "envvar",
    ],
    commands_table_help_sections=["help"],
)


def cleanup(is_root: bool = True):
    if not is_root:
        return
    console.stop_current_progress(skip_total=True)
    console.stop_status()
    console.console.show_cursor()


def abort_signal(*_) -> NoReturn:
    is_root = current_process().name == "MainProcess"
    cleanup(is_root=is_root)
    if is_root:
        console.console.print("[bold red]Aborted![/bold red]")
    sys.exit(130)


def exit_143(*_):
    if current_process().name != "MainProcess":
        return
    sys.exit(143)


atexit.register(cleanup)
signal.signal(signal.SIGINT, abort_signal)
signal.signal(signal.SIGTERM, exit_143)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(
    __version__,
    "--version",
    "-V",
    prog_name=__name__,
    message="%(prog)s v%(version)s - Created by {}".format(__author__),
)
@click.option(
    "-v",
    "--verbose",
    "verbose",
    is_flag=True,
    required=False,
    help="Enable debug/verbose mode",
    default=False,
)
@click.pass_context
@click.rich_config(help_config)
@click.command_panel(
    name="Analyze",
    help="Commands for analyzing and processing images",
    commands=["analyze-peaks", "analyze-shades", "lookup", "identify-quality"],
)
@click.command_panel(
    name="Auto Processing",
    help="Commands for automatically analyzing and processing images",
    commands=["orchestra", "autolevel", "autolevel2", "autoposterize"],
)
@click.command_panel(
    name="ML Processing",
    help="Commands for processing images using Machine Learning models",
    commands=["denoise", "denoise-trt", "upscale-trt", "to-onnx"],
)
@click.command_panel(
    name="Release Management",
    help="Commands for managing manga releases and archives",
    commands=[
        "releases",
        "releasesch",
        "tag",
        "rawtag",
        "shiftname",
        "timewizard",
    ],
)
@click.command_panel(
    name="Packing Utilities",
    help="Commands for packing and managing manga archives",
    commands=[
        "pack",
        "packepub",
        "packcomment",
    ],
)
@click.command_panel(
    name="Extraction Utilities",
    help="Commands for extracting content from manga releases",
    commands=["epub-extract"],
)
@click.command_panel(
    name="Splitting Utilities",
    help="Commands for splitting manga releases",
    commands=["autosplit", "manualsplit", "merge"],
)
@click.command_panel(
    name="Image Utilities",
    help="Commands for optimizing and converting images",
    commands=["forcegray", "posterize", "rescale", "jpegify", "spreads", "optimize", "mixmatch"],
)
@click.command_panel(
    name="Configuration",
    help="Commands for configuring nmanga",
    commands=["config", "install-completion", "version"],
)
def main(ctx: click.Context, verbose: bool):
    """
    Nmanga is a CLI tool for Processing pirated manga.
    """
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE_MODE"] = verbose
    if verbose:
        console.enable_debug()
    else:
        console.disable_debug()


@click.command(
    name="version",
    help="Show nmanga version information",
    cls=NMangaCommandHandler,
)
def show_version():
    """Show nmanga version information."""
    console.console.print(f"[bold]{__name__} version:[/bold] {__version__}")
    console.console.print(f"[bold]Created by:[/bold] {__author__}")
    console.console.print(f"[bold]Working directory:[/bold] {WORKING_DIR}")
    console.console.print(f"[bold]Python version:[/bold] {sys.version.replace(os.linesep, ' ')}")
    console.console.print(f"[bold]Platform:[/bold] {sys.platform}")


@click.command(
    name="install-completion",
    help="Install shell completion for the current user shell",
)
def install_completion():
    """Install shell completion for the current user shell."""
    console.info("Installing shell completion for the current user shell...")
    detected = detect_shell()
    if detected is None:
        console.error("Could not detect a supported shell for completion installation.")
        console.warning(
            "See: https://click.palletsprojects.com/en/stable/shell-completion/ for manual installation instructions"
        )

    match detected:
        case "pwsh":
            console.info("Detected PowerShell 7 environment. Installing completion for PowerShell 7...")
            install_completion_for_pwsh()
        case "bash":
            console.info("Detected Bash shell. Installing completion for Bash...")
            install_completion_for_bash()
        case "fish":
            console.info("Detected Fish shell. Installing completion for Fish...")
            install_completion_for_fish()
        case "zsh":
            console.info("Detected Zsh shell. Installing completion for Zsh...")
            install_completion_for_zsh()
        case _:
            console.error(f"Unsupported shell detected: {detected}. Cannot install completion.")
            console.warning(
                "See: https://click.palletsprojects.com/en/stable/shell-completion/ for manual installation instruction"
            )

    console.success("Shell completion installation complete!")
    if detected is not None and detected != "pwsh":
        console.warning("You can speed-up shell completion by optimizing the script...")
        console.warning("See https://click.palletsprojects.com/en/stable/shell-completion/ for details")


main.add_command(auto_split)
main.add_command(cli_config)
main.add_command(manual_split)
main.add_command(merge_chapters)
main.add_command(pack_releases)
main.add_command(pack_releases_epub_mode)
main.add_command(pack_releases_comment_archive)
main.add_command(prepare_releases)
main.add_command(prepare_releases_chapter)
main.add_command(spreads)
main.add_command(image_tagging)
main.add_command(image_tagging_raw)
main.add_command(image_optimizer)
main.add_command(image_jpegify)
main.add_command(image_mixmatch)
main.add_command(timewizard_modify)
main.add_command(autolevel)
main.add_command(autolevel2)
main.add_command(force_gray)
main.add_command(analyze_level)
main.add_command(posterize_simple)
main.add_command(auto_posterize)
main.add_command(analyze_shades)
main.add_command(denoiser)
main.add_command(denoiser_trt)
main.add_command(upscale_trt)
main.add_command(export_to_onnx)
main.add_command(identify_denoise_candidates)
main.add_command(shift_renamer)
main.add_command(orchestractor)
main.add_command(lookup_group)
main.add_command(rescale_image)
main.add_command(epub_extractor)
main.add_command(show_version)
main.add_command(install_completion)
main.add_command(simulate_progress)


if __name__ == "__main__":
    main()

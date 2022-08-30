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

from pathlib import Path

import click

from .cli.auto_split import auto_split
from .cli.color_level import color_level
from .cli.join_spreads import spreads_join
from .cli.manual_split import manual_split
from .cli.merge_chapters import merge_chapters
from .cli.releases import pack_releases, prepare_releases
from .constants import __author__, __name__, __version__
from .term import get_console

console = get_console()


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
WORKING_DIR = Path.cwd().absolute()


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
def main(ctx: click.Context, verbose: bool):
    """
    nmanga is a CLI tool for Processing pirated manga.
    """
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE_MODE"] = verbose
    if verbose:
        console.enable_debug()
    else:
        console.disable_debug()


main.add_command(auto_split)
main.add_command(color_level)
main.add_command(manual_split)
main.add_command(merge_chapters)
main.add_command(pack_releases)
main.add_command(prepare_releases)
main.add_command(spreads_join)


if __name__ == "__main__":
    main()

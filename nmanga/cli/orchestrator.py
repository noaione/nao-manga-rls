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

# A full blown "automated" manga processing orchestrator

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from time import time
from typing import Literal

import rich_click as click

from .. import term
from ..orchestrator import *
from . import options
from ._deco import check_config_first, time_program
from .base import (
    NMangaCommandHandler,
    test_or_find_cjpegli,
    test_or_find_exiftool,
    test_or_find_magick,
    test_or_find_pingo,
)

console = term.get_console()


@click.group(name="orchestra", help="Orchestrator for manga processing")
def orchestractor():
    pass


@orchestractor.command(
    name="gen",
    help="Generate a default orchestrator configuration file",
    cls=NMangaCommandHandler,
)
@click.argument(
    "output_file",
    metavar="OUTPUT_FILE",
    required=True,
    type=click.Path(resolve_path=True, file_okay=True, dir_okay=False, path_type=Path),
)
@options.manga_title
@options.manga_publisher
@options.rls_credit
@options.rls_email
@options.use_bracket_type
@options.manga_volume
@click.option(
    "-mq",
    "--quality",
    "image_quality",
    type=click.Choice(["LQ", "HQ"]),
    default=None,
    help="Image quality of this release.",
    panel="Release Options",
)
@check_config_first
@time_program
def orchestrator_generate(
    output_file: Path,
    manga_title: str,
    manga_publisher: str,
    rls_credit: str,
    rls_email: str,
    bracket_type: Literal["square", "round", "curly"],
    manga_volume: int | float | None,
    image_quality: Literal["LQ", "HQ"] | None,
):
    if output_file.exists():
        console.error(f"{output_file} already exists, please remove it first")
        raise click.Abort()

    generated_volumes = []
    if manga_volume is not None:
        if isinstance(manga_volume, float):
            raise click.BadParameter("Volume number cannot be a float", param_hint="manga_volume")
        for vol_num in range(1, manga_volume + 1):
            generated_volumes.append(
                VolumeConfig(
                    number=vol_num,
                    path=f"v{vol_num:02d}",
                    chapters=[],
                    quality=image_quality,
                )  # type: ignore
            )
    else:
        generated_volumes.append(
            VolumeConfig(
                number=1,
                path="v01",
                chapters=[],
                quality=image_quality,
            )  # type: ignore
        )

    config = OrchestratorConfig(
        title=manga_title,
        publisher=manga_publisher,
        base_path="source",
        credit=rls_credit,
        email=rls_email,
        bracket_type=bracket_type,
        volumes=generated_volumes,
        actions=[
            # Simple shift name
            ActionShiftName(start=0),  # type: ignore
            # Then use daiz-renamer
            ActionRename(),  # type: ignore
        ],
    )  # type: ignore

    # Pre-serialize to include schema info
    data = config.model_dump(exclude_none=True)
    initial_data = {
        "$schema": "https://raw.githubusercontent.com/noaione/nao-manga-rls/refs/heads/master/orchestrator.jsonschema"
    }
    # Merge with data, so this will make $schema be first
    data = {**initial_data, **data}

    dumped_data = json.dumps(data, indent=4, cls=CustomJSONEncoder)

    with output_file.open("w", encoding="utf-8") as f:
        f.write(dumped_data)

    console.info(f"Generated default orchestrator configuration to {output_file}")
    return 0


@orchestractor.command(
    name="run",
    help="Run the orchestrator with the given configuration file",
    cls=NMangaCommandHandler,
)
@click.argument(
    "input_file",
    metavar="INPUT_FILE",
    required=True,
    type=click.Path(resolve_path=True, file_okay=True, dir_okay=False, path_type=Path),
)
@options.magick_path
@options.pingo_path
@options.exiftool_path
@options.cjpegli_path
@check_config_first
@time_program
def orchestrator_runner(
    input_file: Path,
    magick_path: str,
    pingo_path: str,
    exiftool_path: str,
    cjpegli_path: str,
):
    if not input_file.exists():
        console.error(f"{input_file} does not exist, please provide a valid file")
        raise click.Abort()

    with input_file.open("r", encoding="utf-8") as f:
        config = OrchestratorConfig.model_validate_json(f.read(), strict=True)

    console.info(f"Running orchestrator for {config.title}...")
    full_base = input_file.resolve().parent

    # Tools detection
    toolsets = {}
    tested_pkg = set()
    console.info("Detecting needed tools...")
    for action in config.actions:
        action_name = action.kind.name
        needed_tools = action.get_tools()
        for tool_name, tool_kind in needed_tools.items():
            match tool_kind:
                case ToolsKind.PACKAGE:
                    if tool_name in tested_pkg:
                        continue
                    test_pkg = importlib.util.find_spec(tool_name)
                    if test_pkg is None:
                        console.error(f"Required package '{tool_name}' for action '{action_name}' not found!")
                        raise click.Abort()
                    tested_pkg.add(tool_name)
                case ToolsKind.BINARY:
                    match tool_name:
                        case "magick":
                            tool_path = test_or_find_magick(magick_path)
                        case "pingo":
                            tool_path = test_or_find_pingo(pingo_path)
                        case "exiftool":
                            tool_path = test_or_find_exiftool(exiftool_path)
                        case "cjpegli":
                            tool_path = test_or_find_cjpegli(cjpegli_path)
                        case _:
                            console.error(f"Unknown tool '{tool_name}' for action '{action_name}'!")
                            raise click.Abort()
                    if tool_path is None:
                        console.error(f"Required binary '{tool_name}' for action '{action_name}' not found!")
                        raise click.Abort()
                    toolsets[tool_name] = tool_path
    console.info(f"Detected tools: {', '.join(toolsets.keys()) if toolsets else 'None'}")
    console.info(f"Detected packages: {', '.join(tested_pkg) if tested_pkg else 'None'}")

    input_dir = full_base / Path(config.base_path)
    for volume in config.volumes:
        chapter_path = input_dir / Path(volume.path)
        if not chapter_path.exists():
            console.warning(f"Volume path {chapter_path} does not exist, skipping...")
            continue
        if not chapter_path.is_dir():
            console.warning(f"Volume path {chapter_path} is not a directory, skipping...")
            continue

        skips_mappings: dict[str, SkipActionConfig] = {}
        for skips in volume.skip_actions:
            skips_mappings[skips.step] = skips
        console.info(f"Processing volume {volume.number} {chapter_path}...")

        context = WorkerContext(
            root_dir=full_base,
            current_dir=chapter_path,
            terminal=console,
            toolsets=toolsets,
        )

        volume_start = time()
        for action_name, action in config.actions_maps.items():
            console.info(f" - Running action {action.kind.name}...")
            start_action = time()

            skip_action = skips_mappings.get(action_name)
            context.set_skip_action(skip_action)

            context.terminal.set_space(3)
            action.run(context, volume, config)
            context.terminal.set_space(0)

            end_action = time()
            console.info(f" - Finished action {action.kind.name} in {end_action - start_action:.2f}s")
            console.enter()
        volume_end = time()
        console.info(f"Finished processing volume {volume.number} in {volume_end - volume_start:.2f}s")
    console.info("Orchestrator finished all tasks.")


@orchestractor.command(
    name="validate",
    help="Validate the orchestrator configuration file",
    cls=NMangaCommandHandler,
)
@click.argument(
    "input_file",
    metavar="INPUT_FILE",
    required=True,
    type=click.Path(resolve_path=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--simulate",
    is_flag=True,
    default=False,
    show_default=True,
)
@check_config_first
@time_program
def orchestrator_validate(
    input_file: Path,
    simulate: bool,
):
    if not input_file.exists():
        console.error(f"{input_file} does not exist, please provide a valid file")
        raise click.Abort()

    with input_file.open("r", encoding="utf-8") as f:
        config = OrchestratorConfig.model_validate_json(f.read(), strict=True)

    console.info(f"Orchestrator configuration for {config.title} is valid!")

    if not simulate:
        return

    full_base = input_file.resolve().parent
    console.info(f" - Title: {config.title}")
    console.info(f" - Publisher: {config.publisher}")
    console.info(f" - Credit: {config.credit}")
    console.info(f" - Bracket Type: {config.bracket_type}")
    console.info(f" - Email: {config.email}")
    console.info(f" - Base Path: {config.base_path}")

    # Simulate the volume processing
    console.info(f" - Volumes: {len(config.volumes)}")
    for volume in config.volumes:
        console.info(f"   - Volume {volume.number}:")
        console.info(f"     - Path: {volume.path}")
        if volume.title:
            console.info(f"     - Title: {volume.title}")
        console.info(f"     - Year: {volume.year}")
        console.info(f"     - Publication: {volume.pub_type}")
        console.info(f"     - Quality: {volume.quality}")
        console.info(f"     - Revision: {volume.revision}")
        console.info(f"     - Oneshot: {'Yes' if volume.oneshot else 'No'}")
        console.info(f"     - Colors: Page {', Page '.join(map(str, volume.colors)) if volume.colors else 'None'}")
        console.info(f"     - Spreads: {len(volume.spreads) if volume.spreads else 0} total")
        for first, second in volume.spreads:
            total_spreads = len(range(first, second + 1))
            console.info(f"       - Spread Pages: p{first:03d}-{second:03d} ({total_spreads} pages)")
        console.info(f"     - Chapters: {len(volume.chapters)} total")
        chapter_ranges = volume.to_chapter_ranges()
        for chapter in chapter_ranges:
            console.info(f"       - Chapter {chapter.number}")
            console.info(f"        - Page number: p{chapter.page_num_str}")
            if chapter.name:
                console.info(f"        - Title: {chapter.name}")
        if volume.extra_text:
            console.info(f"     - Extra Text: {volume.extra_text}")
        if volume.skip_actions:
            console.info(f"     - Skip Actions: {len(volume.skip_actions)}")
            for skip in volume.skip_actions:
                console.info(f"       - Step: {skip.step}, Action: {skip.action.name}, Pages: {skip.pages}")

    console.enter()
    console.info(f" - Actions: {len(config.actions)} actions defined")
    input_dir = full_base / Path(config.base_path)
    # Use first volume
    first_vol = config.volumes[0]
    chapter_path = input_dir / Path(first_vol.path)

    context = WorkerContext(
        root_dir=full_base,
        current_dir=chapter_path,
        terminal=console,
        toolsets={},
        dry_run=True,
    )
    for action_name, action in config.actions_maps.items():
        console.info(f'   - Action "{action_name}" ({action.kind.name}):')
        console.info(f"     >> Input Path: {context.current_dir}")
        console.set_space(5)

        action.run(context, first_vol, config)
        console.set_space(0)
        context.terminal = console
        console.enter()

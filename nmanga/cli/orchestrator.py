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
import multiprocessing as mp
import shutil
import signal
import subprocess as sp
from enum import Enum
from pathlib import Path
from time import time
from typing import Literal, TypedDict, cast

import click
from PIL import Image

from nmanga import exporter

from .. import file_handler, term
from ..autolevel import apply_levels, find_local_peak, gamma_correction
from ..common import (
    ChapterRange,
    RegexCollection,
    format_archive_filename,
    format_daiz_like_filename,
    format_volume_text,
    inject_metadata,
    is_pingo_alpha,
    run_pingo_and_verify,
)
from ..denoiser import denoise_single_image, prepare_model_runtime
from ..orchestrator import *
from ..spreads import join_spreads, join_spreads_imagemagick, select_exts
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


def _init_worker():
    """Initialize worker processes to handle keyboard interrupts properly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class ThreadedResult(int, Enum):
    COPIED = 1
    GRAYSCALED = 2
    PROCESSED = 3


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
@check_config_first
@time_program
def orchestrator_generate(
    output_file: Path,
    manga_title: str,
    manga_publisher: str,
    rls_credit: str,
    rls_email: str,
    bracket_type: Literal["square", "round", "curly"],
):
    if output_file.exists():
        console.error(f"{output_file} already exists, please remove it first")
        raise click.Abort()

    config = OrchestratorConfig(
        title=manga_title,
        publisher=manga_publisher,
        base_path=Path("source"),
        credit=rls_credit,
        email=rls_email,
        bracket_type=bracket_type,
        volumes=[
            VolumeConfig(
                number=1,
                path=Path("v01"),
                chapters=[
                    ChapterConfig(number=1, start=0),
                ],
            )
        ],
        actions=[
            # Simple shift name
            ActionShiftName(start=0),
            # Then use daiz-renamer
            ActionRename(),
        ],
    )

    with output_file.open("w", encoding="utf-8") as f:
        f.write(config.model_dump_json(indent=4, exclude_none=True))

    console.info(f"Generated default orchestrator configuration to {output_file}")
    return 0


def detect_needed_tools(actions: list[Actions]) -> set[str]:
    needed_tools = set()
    for action in actions:
        match action.kind:
            case ActionKind.SPREADS:
                if action.pillow:
                    continue  # Pillow is always available
                needed_tools.add("magick")
            case ActionKind.OPTIMIZE:
                needed_tools.add("pingo")
            case ActionKind.COLOR_JPEGIFY:
                needed_tools.add("cjpegli")
            case ActionKind.TAGGING:
                needed_tools.add("exiftool")
            case _:
                # No tools needed
                continue
    return needed_tools


def runner_shift_rename(
    input_folder: Path,
    start_index: int,
    title: str,
    volume: int | float,
) -> None:
    all_images: list[Path] = []
    for image_file, _, _, _ in file_handler.collect_image_from_folder(input_folder):
        all_images.append(image_file.resolve())

    all_images.sort(key=lambda x: x.stem)

    volume_text = format_volume_text(manga_volume=volume, manga_chapter=None)
    total_files = len(all_images) + start_index
    padding = max(3, len(str(total_files + start_index - 1)))
    console.status(f"Renaming {len(all_images)} images in {input_folder}...")
    remapped_names = set()
    should_revert = False
    total_rename = 0
    for idx, image_path in enumerate(all_images):
        console.status(f"Renaming images [{idx + 1}/{total_files}]...")
        new_name = f"{title} - {volume_text} - p{str(start_index + idx).zfill(padding)}"

        img_suffix = image_path.suffix.lower()
        new_path = image_path.with_name(new_name).with_suffix(img_suffix)
        if new_path.name == image_path.name:
            continue
        remapped_names.add((image_path, new_path))
        if new_path.exists():
            console.warning("Conflict detected, reverting all changes...")
            console.log(f"Conflicting file: {new_path}")
            should_revert = True
            break
        image_path.rename(new_path)
        total_rename += 1

    if should_revert:
        console.stop_status(f"Renamed {total_rename} images, reverting...")
        console.log()
        console.status("Reverting all changes...")
        for old_path, new_path in remapped_names:
            if new_path.exists():
                new_path.rename(old_path)
        console.stop_status("Reverted all changes.")
        raise click.Abort()


class _SpreadStuff(TypedDict):
    images: list[Path]
    prefix: str
    postfix: str


def _runner_image_spreads_threaded(
    spread_key: str,
    images: _SpreadStuff,
    imagick: str,
    action: ActionSpreads,
    output_dir: Path,
) -> None:
    final_filename = f"{images['prefix']}p{spread_key}{images['postfix']}"
    if len(images["images"]) < 2:
        console.warning(f"Spread {spread_key} has less than 2 images, skipping...")
        return

    if action.pillow:
        # Load all images
        all_img_paths = [x.path for x in images["images"]]
        loaded_images = [Image.open(p) for p in all_img_paths]

        joined_image = join_spreads(loaded_images, action.direction)
        extension = select_exts(all_img_paths)
        if action.output_fmt != "auto":
            extension = f".{action.output_fmt.lower()}"
        final_filename += extension

        joined_image.save(output_dir / final_filename, quality=int(action.quality))

        # Close all images
        for im in loaded_images:
            im.close()
        joined_image.close()
    else:
        temp_filename = join_spreads_imagemagick(
            images=images["images"],
            output_directory=output_dir,
            quality=action.quality,
            direction=action.direction,
            output_format=action.output_fmt,
            magick_path=imagick,
        )

        input_name = output_dir / temp_filename
        input_name.rename(output_dir / final_filename)


def runner_image_spreads(
    cwd_dir: Path,
    input_folder: Path,
    action: ActionSpreads,
    spreads: list[tuple[int, int]],
    toolsets: dict[str, str],
) -> None:
    if not spreads:
        console.warning("No spreads defined, skipping...")
        return

    imagick = toolsets.get("magick")
    if imagick is None and not action.pillow:
        console.error("ImageMagick is required for spreads joining, but not found!")
        raise click.Abort()
    exported_images: dict[str, _SpreadStuff] = {}
    page_re = RegexCollection.page_re()
    for image_file, _, _, _ in file_handler.collect_image_from_folder(input_folder):
        title_match = page_re.match(image_file.stem)
        if title_match is None:
            console.error(f"Image {image_file} does not match page regex, aborting...")
            raise click.Abort()

        first_part = title_match.group("a")
        second_part = title_match.group("b")
        prefix_text = title_match.group("any")
        postfix_text = title_match.group("anyback")
        if second_part:
            # Already joined spread
            continue
        first_part = int(first_part)
        for spread_start, spread_end in spreads:
            if spread_start <= first_part <= spread_end:
                spread_key = f"{spread_start:03d}-{spread_end:03d}"
                if spread_key not in exported_images:
                    exported_images[spread_key] = {
                        "images": [],
                        "prefix": prefix_text or "",
                        "postfix": postfix_text or "",
                    }
                exported_images[spread_key]["images"].append(image_file)
                break

    if not exported_images:
        console.warning(f"No spreads found in {input_folder}, skipping...")
        return

    current = 1
    total_match_spreads = len(list(exported_images.keys()))
    if action.threads > 1:
        console.status(f"Joining {total_match_spreads} spreads with {action.threads} threads...")
        with mp.Pool(processes=action.threads, initializer=_init_worker) as pool:
            try:
                for spread, images in exported_images.items():
                    pool.apply_async(
                        _runner_image_spreads_threaded,
                        args=(spread, images, imagick, action, input_folder),
                    )
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                console.warning("Keyboard interrupt detected, terminating...")
                pool.terminate()
                pool.join()
                raise click.Abort()
            except Exception as e:
                console.error(f"Error occurred: {e}, terminating...")
                pool.terminate()
                pool.join()
                raise click.Abort()
            finally:
                pool.close()
        console.stop_status("Finished joining spreads.")
    else:
        for spread, images in exported_images.items():
            console.status(f"Joining spreads: {current}/{total_match_spreads} ({spread})...")
            _runner_image_spreads_threaded(spread, images, imagick, action, input_folder)
            current += 1
        console.stop_status("Finished joining spreads.")

    # Make backup folder here
    backup_dir = cwd_dir / "backup" / input_folder.name
    backup_dir.mkdir(exist_ok=True, parents=True)

    console.info(f"Backing up original images to {backup_dir}...")
    for image_data in exported_images.values():
        for img_path in image_data["images"]:
            shutil.move(img_path, backup_dir / img_path.name)


def runner_daiz_rename(
    input_folder: Path,
    volume: VolumeConfig,
    config: OrchestratorConfig,
) -> None:
    # console.status("Renaming with daiz-like format: 1/???")
    cmx_re = RegexCollection.cmx_re()

    packing_extra = dict[int, list[ChapterRange]] = {}
    for chapter in volume.chapters:
        as_range = chapter.to_chapter_range()
        if as_range.base not in packing_extra:
            packing_extra[as_range.base] = []
        packing_extra[as_range.base].append(as_range)

    # Make into chapter range
    meta_namings = volume.meta_name_maps
    renaming_maps: dict[str, Path] = {}  # This is new name -> old path (yeah)
    for image, _, _, _ in file_handler.collect_image_from_folder(input_folder):
        # console.status(f"Renaming with daiz-like format: {image.name}/{total_img}...")
        title_match = cmx_re.match(image.name)
        if title_match is None:
            console.error(f"Image {image} does not match regex, aborting...")
            raise click.Abort()

        page_num = title_match.group("a")
        page_num_int = int(title_match.group("a"))
        page_back = title_match.group("b")
        if page_back is not None:
            page_num = f"{page_num}-{page_back}"

        volume_number = cast(str, format_volume_text(manga_volume=volume.number))
        if volume.oneshot:
            volume_number = ""
        selected_range: ChapterConfig | None = None
        for chapter in volume.chapters:
            if chapter.end is None:
                if page_num_int >= chapter.start:
                    selected_range = chapter
                    break
            else:
                if chapter.start <= page_num_int <= chapter.end:
                    selected_range = chapter
                    break

        if selected_range is None:
            console.error(f"Image {image} does not match any chapter range, aborting...")
            raise click.Abort()

        extra_name: str | None = None
        if page_num_int in meta_namings:
            extra_name = meta_namings[page_num_int]

        image_filename, _ = format_daiz_like_filename(
            manga_title=config.title,
            manga_publisher=config.publisher,
            manga_year=volume.year,
            chapter_info=selected_range.to_chapter_range(),
            page_number=page_num,
            publication_type=volume.pub_type,
            ripper_credit=config.credit,
            bracket_type=config.bracket_type,
            manga_volume=volume_number if not volume.oneshot else None,
            extra_metadata=extra_name,
            image_quality=volume.quality,
            rls_revision=volume.revision,
            chapter_extra_maps=packing_extra,
            extra_archive_metadata=volume.extra_text,
            fallback_volume_name="OShot",
        )
        renaming_maps[image_filename] = image.resolve()

    if not renaming_maps:
        console.warning(f"No images found in {input_folder}, skipping...")
        return

    # Check for conflict
    unique_names = set()
    for new_name in renaming_maps.keys():
        if new_name in unique_names:
            console.error(f"Conflict detected: {new_name} would be duplicated, aborting...")
            raise click.Abort()
        unique_names.add(new_name)

    total_img = len(renaming_maps)
    console.status(f"Renaming {total_img} images in {input_folder}...")
    for idx, (new_name, old_path) in enumerate(renaming_maps.items()):
        console.status(f"Renaming images: {idx + 1}/{total_img} ({new_name})...")
        new_path = old_path.with_stem(new_name)
        if new_path.name == old_path.name:
            continue
        old_path.rename(new_path)
    console.stop_status(f"Renamed {total_img} images.")


def runner_denoiser_trt(
    input_folder: Path,
    output_dir: Path,
    action: ActionDenoise,
) -> None:
    console.info(f"Loading denoising model from {action.model}...")

    session = prepare_model_runtime(
        action.model,
        device_id=action.device_id,
        is_verbose=console.debugged,
    )

    current_index = 1
    for file_path, _, total_image, _ in file_handler.collect_image_from_folder(input_folder):
        console.status(f"Denoising images... [{current_index}/{total_image}]")
        img_file = Image.open(file_path)
        output_image = denoise_single_image(
            img_file,
            session,
            batch_size=action.batch_size,
            tile_size=action.tile_size,
            background=action.background,
        )

        output_path = output_dir / f"{file_path.stem}.png"
        output_image.save(output_path, format="PNG")
        img_file.close()
        output_image.close()
        current_index += 1
    console.stop_status(f"Denoised {total_image} images.")


def _runner_autolevel2_threaded(
    img_path: Path,
    output_dir: Path,
    action: ActionAutolevel,
    is_first: bool,
    is_color: bool,
) -> ThreadedResult:
    if is_first and action.skip_first:
        dest_path = output_dir / img_path.name
        if dest_path.exists():
            console.warning(f"Skipping existing file: {dest_path}")
            return ThreadedResult.COPIED

        shutil.copy2(img_path, dest_path)
        return ThreadedResult.COPIED
    img = Image.open(img_path)
    black_level, white_level, _ = find_local_peak(
        img, upper_limit=action.upper_limit, skip_white_peaks=action.skip_white
    )

    is_black_bad = black_level <= 0
    is_white_bad = white_level >= 255 if not action.skip_white else False

    if (
        (is_black_bad and is_white_bad and not action.skip_white)  # both levels are bad
        or (is_black_bad and action.skip_white)
        or black_level > action.upper_limit
    ):
        dest_path = output_dir / img_path.name
        if not is_color:
            img = img.convert("L")
            img.save(dest_path.with_suffix(".png"), format="PNG")
            img.close()
            return ThreadedResult.GRAYSCALED

        img.close()

        if dest_path.exists():
            console.warning(f"Skipping existing file: {dest_path}")
            return ThreadedResult.COPIED
        shutil.copy2(img_path, dest_path)
        return ThreadedResult.COPIED

    dest_path = output_dir / f"{img_path.stem}.png"
    if dest_path.exists():
        console.warning(f"Skipping existing file: {dest_path}")
        return ThreadedResult.COPIED

    if not is_color:
        img = img.convert("L")
    gamma_correct = gamma_correction(black_level)
    adjusted_img = apply_levels(
        img,
        black_point=black_level + action.peak_offset,
        white_point=255 if action.skip_white else white_level,
        gamma=gamma_correct,
    )

    adjusted_img.save(dest_path, format="PNG")
    img.close()
    adjusted_img.close()
    return ThreadedResult.PROCESSED


def runner_autolevel(
    input_dir: Path,
    output_dir: Path,
    action: ActionAutolevel,
    volume: VolumeConfig,
) -> None:
    cmx_re = RegexCollection.cmx_re()
    console.status("Processing images with autolevel...")

    all_images = [img for img, _, _, _ in file_handler.collect_image_from_folder(input_dir)]
    total_images = len(all_images)
    all_images.sort(key=lambda x: x.stem)

    # Do pre-processing
    images_complete: list[tuple[Path, bool, bool]] = []
    for idx, image in enumerate(all_images):
        img_match = cmx_re.match(image.stem)
        is_color = False
        is_first = idx == 0

        if img_match is not None:
            p01 = int(img_match.group("a"))
            is_color = p01 in volume.colors

        images_complete.append((image, is_first, is_color))

    results: list[ThreadedResult] = []
    if action.threads > 1:
        console.info(f"Using {action.threads} CPU threads for processing.")
        with mp.Pool(processes=action.threads, initializer=_init_worker) as pool:
            try:
                for idx, (image, is_first, is_color) in enumerate(images_complete):
                    # We need to also get the return value here
                    pool.apply_async(
                        _runner_autolevel2_threaded,
                        args=(image, output_dir, action, is_first, is_color),
                        callback=results.append,
                    )
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                console.warning("Keyboard interrupt detected, terminating...")
                pool.terminate()
                pool.join()
                raise click.Abort()
            except Exception as e:
                console.error(f"Error occurred: {e}, terminating...")
                pool.terminate()
                pool.join()
                raise click.Abort()
            finally:
                pool.close()
    else:
        for idx, (image, is_first, is_color) in enumerate(images_complete):
            console.status(f"Processing image with autolevel... [{idx + 1}/{total_images}]")
            results.append(_runner_autolevel2_threaded(image, output_dir, action, is_first, is_color))

    autolevel_count = sum(1 for result in results if result == ThreadedResult.PROCESSED)
    copied_count = sum(1 for result in results if result == ThreadedResult.COPIED)
    grayscaled_count = sum(1 for result in results if result == ThreadedResult.GRAYSCALED)
    console.stop_status(f"Processed {total_images} images with autolevel2.")
    if copied_count > 0:
        console.info(f"Copied {copied_count} images without autolevel.")
    if autolevel_count > 0:
        console.info(f"Autoleveled {autolevel_count} images.")
    if grayscaled_count > 0:
        console.info(f"Grayscaled {grayscaled_count} images.")


def runner_optimize(
    input_dir: Path,
    action: ActionOptimize,
    toolsets: dict[str, str],
) -> None:
    pingo = toolsets.get("pingo")
    if pingo is None:
        console.error("Pingo is required for image optimization, but not found!")
        raise click.Abort()

    all_images = [img for img, _, _, _ in file_handler.collect_image_from_folder(input_dir)]
    total_images = len(all_images)
    console.status(f"Optimizing {total_images} images with pingo...")

    alpha_ver = is_pingo_alpha(pingo)
    base_cmd = [pingo, "-strip", "-sb"] if alpha_ver else [pingo, "-notime", "-lossless", "-notrans", "-s4"]
    full_dir = input_dir.resolve()
    folder_text = str(full_dir / "*")
    if action.limiter:
        folder_text += action.limiter
    if action.aggresive and not alpha_ver:
        base_cmd.append("-grayscale")

    cmd = [*base_cmd, folder_text]
    console.status(f"Optimizing images in: {folder_text}...")
    proc = run_pingo_and_verify(cmd)
    end_msg = "Optimized images files!"
    if proc is not None:
        end_msg += f" [{proc}]"
    console.stop_status(end_msg)


def runner_tagging(
    input_dir: Path,
    volume: VolumeConfig,
    config: OrchestratorConfig,
    toolsets: dict[str, str],
) -> None:
    exiftool = toolsets.get("exiftool")
    if exiftool is None:
        console.error("exiftool is required for image tagging, but not found!")
        raise click.Abort()

    volume_text = cast(str, format_volume_text(manga_volume=volume.number))

    archive_filename = format_archive_filename(
        manga_title=config.title,
        manga_year=volume.year,
        publication_type=volume.pub_type,
        ripper_credit=config.credit,
        bracket_type=config.bracket_type,
        manga_volume_text=volume_text if not volume.oneshot else None,
        extra_metadata=volume.extra_text,
        rls_revision=volume.revision,
    )
    console.status(f"Tagging images in {input_dir} with exif metadata...")
    inject_metadata(exiftool, input_dir, archive_filename, config.email)


def runner_pack(
    input_dir: Path,
    volume: VolumeConfig,
    config: OrchestratorConfig,
    action: ActionPack,
) -> None:
    volume_text = cast(str, format_volume_text(manga_volume=volume.number))

    console.info(f"Packing volume {volume.number} in {input_dir}...")
    archive_filename = format_archive_filename(
        manga_title=config.title,
        manga_year=volume.year,
        publication_type=volume.pub_type,
        ripper_credit=config.credit,
        bracket_type=config.bracket_type,
        manga_volume_text=volume_text if not volume.oneshot else None,
        extra_metadata=volume.extra_text,
        rls_revision=volume.revision,
    )
    parent_dir = input_dir.parent
    arc_target = exporter.exporter_factory(
        archive_filename,
        parent_dir,
        mode=action.output_mode,
        manga_title=config.title,
    )
    if action.output_mode == exporter.ExporterType.epub:
        console.warning("Packing as EPUB, this will be a slower operation because of size checking!")

    arc_target.set_comment(config.email)
    console.status("Packing... (0/???)")
    idx = 1
    for img_file, _, total_img, _ in file_handler.collect_image_from_folder(input_dir):
        arc_target.add_image(img_file.name, img_file)
        console.status(f"Packing... ({idx}/{total_img})")
        idx += 1
    console.stop_status(f"Packed ({idx - 1}/{total_img})")
    arc_target.close()


def runner_move_color_image(
    input_dir: Path,
    output_dir: Path,
    volume: VolumeConfig,
) -> None:
    cmx_re = RegexCollection.cmx_re()
    output_dir.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    for img_file, _, total_img, _ in file_handler.collect_image_from_folder(input_dir):
        title_match = cmx_re.match(img_file.stem)
        if title_match is None:
            console.error(f"Image {img_file} does not match page regex, aborting...")
            continue

        console.status(f"Moving color images... [{moved_count}/{total_img}]")
        p01 = int(title_match.group("a"))
        if p01 in volume.colors:
            dest_path = output_dir / img_file.name
            if dest_path.exists():
                console.warning(f"Skipping existing file: {dest_path}")
                continue
            shutil.move(img_file, dest_path)
            moved_count += 1

    console.stop_status(f"Moved {moved_count} color images to {output_dir}.")


def _runner_jpegify_threaded(
    img_path: Path,
    output_dir: Path,
    cjpegli: str,
    quality: int,
) -> None:
    dest_path = output_dir / f"{img_path.stem}.jpg"
    if dest_path.exists():
        console.warning(f"Skipping existing file: {dest_path}")
        return

    cmd = [cjpegli, "-q", str(quality), str(img_path), str(dest_path)]
    sp.run(cmd, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)


def runner_jpegify(
    input_dir: Path,
    output_dir: Path,
    volume: VolumeConfig,
    action: ActionColorJpegify,
    toolsets: dict[str, str],
) -> None:
    cjpegli = toolsets.get("cjpegli")
    if cjpegli is None:
        console.error("cjpegli is required for JPEG conversion, but not found!")
        raise click.Abort()

    cmx_re = RegexCollection.cmx_re()

    image_candidates: list[Path] = []
    for img_path, _, _, _ in file_handler.collect_image_from_folder(input_dir):
        title_match = cmx_re.match(img_path.stem)
        if title_match is None:
            console.warn(f"Image {img_path} does not match page regex, ignoring...")
            continue
        p01 = int(title_match.group("a"))
        if p01 in volume.colors:
            image_candidates.append(img_path)

    if not image_candidates:
        console.warning(f"No color images found in {input_dir}, skipping...")
        return

    total_images = len(image_candidates)
    console.status(f"Converting {total_images} images to JPEG with cjpegli...")
    quality = max(0, min(100, action.quality))
    if action.threads > 1:
        console.info(f"Using {action.threads} CPU threads for processing.")
        with mp.Pool(processes=action.threads, initializer=_init_worker) as pool:
            try:
                for image in image_candidates:
                    pool.apply_async(
                        _runner_jpegify_threaded,
                        args=(image, output_dir, cjpegli, quality),
                    )
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                console.warning("Keyboard interrupt detected, terminating...")
                pool.terminate()
                pool.join()
                raise click.Abort()
            except Exception as e:
                console.error(f"Error occurred: {e}, terminating...")
                pool.terminate()
                pool.join()
                raise click.Abort()
            finally:
                pool.close()
    else:
        for idx, image in enumerate(image_candidates):
            console.status(f"Converting image to JPEG... [{idx + 1}/{total_images}]")
            _runner_jpegify_threaded(image, output_dir, cjpegli, quality)
    console.stop_status(f"Converted {total_images} images to JPEG.")


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
    console.info("Detecting needed tools...")
    for action in config.actions:
        match action.kind:
            case ActionKind.SPREADS:
                if action.pillow:
                    continue  # Pillow is always available
                # Detect ImageMagick
                imagick = test_or_find_magick(magick_path)
                if imagick is None:
                    console.error("ImageMagick is required for spreads joining, but not found!")
                    raise click.Abort()
                toolsets["magick"] = imagick
            case ActionKind.OPTIMIZE:
                # Detect Pingo
                pingo = test_or_find_pingo(pingo_path)
                if pingo is None:
                    console.error("Pingo is required for image optimization, but not found!")
                    raise click.Abort()
                toolsets["pingo"] = pingo
            case ActionKind.COLOR_JPEGIFY:
                # Detect cjpegli
                cjpegli = test_or_find_cjpegli(cjpegli_path)
                if cjpegli is None:
                    console.error("cjpegli is required for JPEG conversion, but not found!")
                    raise click.Abort()
                toolsets["cjpegli"] = cjpegli
            case ActionKind.TAGGING:
                # Detect ExifTool
                exiftool = test_or_find_exiftool(exiftool_path)
                if exiftool is None:
                    console.error("ExifTool is required for tagging, but not found!")
                    raise click.Abort()
                toolsets["exiftool"] = exiftool_path
            case ActionKind.DENOISE:
                # Check if all the imports are available (onnxruntime, einops, numpy)
                packages = {
                    "onnxruntime": importlib.util.find_spec("onnxruntime"),
                    "einops": importlib.util.find_spec("einops"),
                    "numpy": importlib.util.find_spec("numpy"),
                }
                for pkg_name, pkg in packages.items():
                    if pkg is None:
                        console.error(f"Denoising requires additional dependencies: {pkg_name}, please install them!")
                        raise click.Abort()
            case ActionKind.AUTOLEVEL:
                # Check if all the imports are available (scipy, numpy)
                packages = {
                    "scipy": importlib.util.find_spec("scipy"),
                    "numpy": importlib.util.find_spec("numpy"),
                }
                for pkg_name, pkg in packages.items():
                    if pkg is None:
                        console.error(
                            f"Auto-leveling requires additional dependencies: {pkg_name}, please install them!"
                        )
                        raise click.Abort()
            case _:
                # No tools needed
                continue
    console.info(f"Detected tools: {', '.join(toolsets.keys()) if toolsets else 'None'}")

    input_dir = full_base / config.base_path
    for volume in config.volumes:
        chapter_path = input_dir / volume.path
        if not chapter_path.exists():
            console.warning(f"Volume path {chapter_path} does not exist, skipping...")
            continue
        if not chapter_path.is_dir():
            console.warning(f"Volume path {chapter_path} is not a directory, skipping...")
            continue

        console.info(f"Processing volume {volume.number} {chapter_path}...")
        for action in config.actions:
            console.info(f" - Running action {action.kind.name}...")
            start_action = time()
            match action.kind:
                case ActionKind.SHIFT_RENAME:
                    runner_shift_rename(
                        input_folder=chapter_path,
                        start_index=action.start,
                        title=action.title or config.title,
                        volume=volume.number,
                    )
                case ActionKind.SPREADS:
                    runner_image_spreads(
                        cwd_dir=full_base,
                        input_folder=chapter_path,
                        action=action,
                        spreads=volume.spreads or [],
                        toolsets=toolsets,
                    )
                case ActionKind.RENAME:
                    runner_daiz_rename(
                        input_folder=chapter_path,
                        volume=volume,
                        config=config,
                    )
                case ActionKind.DENOISE:
                    output_dir = full_base / action.base_path / volume.path
                    output_dir.mkdir(parents=True, exist_ok=True)
                    runner_denoiser_trt(
                        input_folder=chapter_path,
                        output_dir=output_dir,
                        action=action,
                    )
                    # New chapter path base
                    chapter_path = output_dir
                case ActionKind.AUTOLEVEL:
                    output_dir = full_base / action.base_path / volume.path
                    output_dir.mkdir(parents=True, exist_ok=True)
                    runner_autolevel(
                        input_dir=chapter_path,
                        output_dir=output_dir,
                        action=action,
                        volume=volume,
                    )
                    # New chapter path base
                    chapter_path = output_dir
                case ActionKind.OPTIMIZE:
                    runner_optimize(
                        input_dir=chapter_path,
                        action=action,
                        toolsets=toolsets,
                    )
                case ActionKind.TAGGING:
                    runner_tagging(
                        input_dir=chapter_path,
                        volume=volume,
                        config=config,
                        toolsets=toolsets,
                    )
                case ActionKind.PACK:
                    source_dir = chapter_path
                    if action.source_dir is not None:
                        source_dir = full_base / action.source_dir / volume.path
                    if not source_dir.exists() or not source_dir.is_dir():
                        console.error(
                            f"Source directory {source_dir} does not exist or is not a directory, skipping packing..."
                        )
                        raise click.Abort()
                    runner_pack(
                        input_dir=source_dir,
                        volume=volume,
                        config=config,
                        action=action,
                    )
                case ActionKind.MOVE_COLOR:
                    target_dir = full_base / action.base_path / volume.path
                    runner_move_color_image(
                        input_dir=chapter_path,
                        output_dir=target_dir,
                        volume=volume,
                    )
                case ActionKind.COLOR_JPEGIFY:
                    output_dir = chapter_path
                    if action.base_path is not None:
                        output_dir = full_base / action.base_path / volume.path
                    output_dir.mkdir(parents=True, exist_ok=True)
                    source_dir = full_base / action.source_path / volume.path
                    if not source_dir.exists() or not source_dir.is_dir():
                        console.error(
                            f"Source directory {source_dir} does not exist or is not a directory, skipping..."
                        )
                        raise click.Abort()
                    runner_jpegify(
                        input_dir=source_dir,
                        output_dir=output_dir,
                        volume=volume,
                        action=action,
                        toolsets=toolsets,
                    )
                case _:
                    console.error(f"Action {action.kind.name} is not implemented yet, skipping...")
                    continue
            end_action = time()
            console.info(f" - Finished action {action.kind.name} in {end_action - start_action:.2f}s")
            console.enter()
    console.info("Orchestrator finished all tasks.")

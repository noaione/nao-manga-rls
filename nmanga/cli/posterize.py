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

# Automatically posterize images based on shade analysis

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

import rich_click as click
from PIL import Image

from .. import file_handler, term
from ..autolevel import (
    PosterizePointMethod,
    analyze_gray_shades,
    detect_nearest_bpc,
    pad_shades_to_bpc,
    posterize_image_by_bits,
    posterize_image_by_shades,
)
from ..common import BoundedWritePool, lowest_or, threaded_worker
from ..lazy import get_vapoursynth
from ..vapour import (
    METRIC_DEFAULT_MINIMUM,
    METRIC_HIGHER_IS_BETTER,
    METRIC_NAMES,
    is_known_metric,
    vs_attach_logger,
    vs_find_missing_plugins,
    vs_frame_to_image,
    vs_grays_frame_to_image,
    vs_page_sizes,
    vs_posterize_clip,
    vs_posterize_group,
    vs_prepare_image,
    vs_score_candidates,
    vs_select_posterize_bits,
    vs_ssimulacra2,
)
from . import options
from ._deco import time_program
from .base import NMangaCommandHandler

console = term.get_console()

# Pages scored per chunk. The candidates for a chunk are held in memory so the winner is still
# warm when it is pulled, so this is sized against the frame cache rather than by taste.
DEFAULT_CHUNK = 16
# The depths automatic degradation offers when the caller gives a range instead of a depth. One
# to six: below one there is nothing to posterize and above six the metric almost never fails.
AUTO_BITS_RANGE = "1-6"


class PosterizedResult(int, Enum):
    PROCESSED = 1
    COPIED = 2


@dataclass
class SsimOption:
    enabled: bool
    minimum: float


def parse_bits(value: str) -> list[int]:
    """
    Parse `--bits` into a list of bit depths, in the order given.

    A single depth is one candidate. Several are a candidate set: `2,3,4` and `2-5` both work,
    and the two mix as `1-2,4`. A range is inclusive and contributes its depths in ascending
    order, which is the only order a range can mean. Duplicates are dropped, keeping the first
    mention, because scoring the same depth twice is a cost with no meaning.

    The order matters: a plain list is a caller's preference order, so `5,4,3` is left as they
    wrote it. "Fewest bits that passes" is applied by :func:`resolve_candidates`, not here.
    """
    candidates: list[int] = []
    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            low, _, high = item.partition("-")
            low, high = low.strip(), high.strip()
            if not low.isdecimal() or not high.isdecimal():
                raise click.BadParameter(f"{item!r} is not a bit depth or a range of them", param_hint="bits")
            start, stop = int(low), int(high)
            if start > stop:
                raise click.BadParameter(f"{item!r} starts above where it ends", param_hint="bits")
            candidates.extend(range(start, stop + 1))
        elif item.isdecimal():
            candidates.append(int(item))
        else:
            raise click.BadParameter(f"{item!r} is not a bit depth or a range of them", param_hint="bits")

    if not candidates:
        raise click.BadParameter("no bit depth was found", param_hint="bits")
    for bits in candidates:
        if not 1 <= bits <= 8:
            raise click.BadParameter(f"{bits} is outside the 1 to 8 bit depth range", param_hint="bits")

    unique: list[int] = []
    for bits in candidates:
        if bits not in unique:
            unique.append(bits)
    return unique


def parse_metrics(value: str | None) -> list[str]:
    """
    Parse `--metrics` into the metric names to score the candidates with.

    `None` means the caller asked for candidates without naming a metric, which is a request to
    score them, so it defaults to `ssimulacra2`: the metric already in use, with the floor it
    already had.
    """
    raw = value if value is not None else "ssimulacra2"
    names = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not names:
        raise click.BadParameter("no metric was named", param_hint="metrics")
    unknown = [name for name in names if not is_known_metric(name)]
    if unknown:
        raise click.BadParameter(
            f"unknown metric(s): {', '.join(unknown)}. Expected one of {', '.join(METRIC_NAMES)}",
            param_hint="metrics",
        )
    return names


def parse_metric_minimums(value: str) -> dict[str, float]:
    """Parse `--metric-min` into `metric -> threshold`. Only the metrics named are constrained."""
    minimums: dict[str, float] = {}
    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise click.BadParameter(f"{item!r} is not a metric=threshold pair")
        name, _, raw = item.partition("=")
        name = name.strip().lower()
        if not is_known_metric(name):
            raise click.BadParameter(f"unknown metric {name!r}, expected one of {', '.join(METRIC_NAMES)}")
        try:
            minimums[name] = float(raw.strip())
        except ValueError as exc:
            raise click.BadParameter(f"{raw!r} is not a number") from exc
    return minimums


def resolve_metric_minimums(
    metrics: Sequence[str],
    *,
    minimums: Mapping[str, float],
) -> dict[str, float]:
    """
    Decide which metrics are allowed to pick a depth, and at what threshold.

    Every shipped metric has a settled direction and a measured default, so naming one is enough.
    `--metric-min` overrides any of them, which is the way to make a threshold stricter or looser
    without touching the others. A metric added later without a measured default is reported but
    cannot fail a page until the caller names a threshold for it.
    """
    resolved: dict[str, float] = {}
    for metric in metrics:
        if metric in minimums:
            resolved[metric] = minimums[metric]
        elif metric in METRIC_DEFAULT_MINIMUM:
            resolved[metric] = METRIC_DEFAULT_MINIMUM[metric]
        else:
            console.warning(
                f"No threshold for the {metric} metric, so it is measured and reported but "
                f"cannot pick a depth. Pass --metric-min {metric}=<value> to let it."
            )
    return resolved


def resolve_candidates(bits: str) -> tuple[list[int], bool]:
    """
    Return the candidate bit depths and whether the default policy is "fewest bits that passes".

    A single depth is one candidate, so there is nothing to choose between and the command is
    the fixed depth posterize it has always been. Several depths are a candidate set: a range is
    not an order, so `2-5` is read as fewest bits first, while a comma list of plain depths is a
    preference order the caller typed and only breaks ties.
    """
    candidates = parse_bits(bits)
    if len(candidates) == 1:
        return candidates, True
    items = [item.strip() for item in bits.split(",") if item.strip()]
    is_plain_list = all(item.isdecimal() for item in items)
    return candidates, not is_plain_list


def _save_png(image: Image.Image, dest_path: Path) -> PosterizedResult:
    """Encode and write one finished page. Runs on the write pool."""
    image.save(dest_path, format="PNG")
    image.close()
    return PosterizedResult.PROCESSED


def _copy_page(img_path: Path, dest_path: Path) -> PosterizedResult:
    """Copy the source page over the output. Runs on the write pool."""
    if dest_path.exists():
        console.warning(f"Skipping existing file: {dest_path}")
        return PosterizedResult.COPIED
    shutil.copy2(img_path, dest_path)
    return PosterizedResult.COPIED


def _submit_fallback(
    pool: BoundedWritePool,
    img_path: Path,
    dest_path: Path,
    image: Image.Image | None,
):
    """
    Write a page whose score failed, or whose depth was not picked, by copying the source.

    The copy keeps the source bytes and costs no encode, and it lands under the source suffix
    rather than the `.png` name the processed pages use, which is the fallback both commands
    already took. `image` is the frame that was never written, or `None` when no candidate frame
    was pulled at all.
    """
    if image is not None:
        image.close()
    return pool.submit(_copy_page, img_path, dest_path.with_suffix(img_path.suffix))


def _posterize_simple_wrapper(
    log_q: term.MessageQueue,
    img_path: Path,
    dest_output: Path,
    num_bits: int,
    method: PosterizePointMethod,
    ssim: SsimOption,
) -> PosterizedResult:
    img = Image.open(img_path)

    posterized = posterize_image_by_bits(img, num_bits, method=method)

    dest_path = dest_output / img_path.with_suffix(".png").name
    if ssim.enabled:
        cnsl = term.with_thread_queue(log_q)
        is_diff = False
        if img.mode != posterized.mode:
            is_diff = True
            distorted_test = posterized.convert(img.mode)  # convert to same mode
        else:
            distorted_test = posterized
        img_reference = vs_prepare_image(img)
        img_distorted = vs_prepare_image(distorted_test)
        if is_diff:
            distorted_test.close()
        cnsl.log(f"Reference ({img.mode}): {img_reference!r}")
        cnsl.log(f"Distorted ({posterized.mode}): {img_distorted!r}")
        score = vs_ssimulacra2(img_reference, img_distorted)
        cnsl.log(f"SSIM score for {img_path.name}: {score}")
        del img_reference, img_distorted
        # if score is lower than minimum, copy file
        if score < ssim.minimum:
            img.close()
            if dest_path.exists():
                cnsl.warning(f"Skipping existing file: {dest_path}")
                return PosterizedResult.COPIED
            shutil.copy2(img_path, dest_path)
            return PosterizedResult.COPIED

    posterized.save(dest_path, format="PNG")
    posterized.close()
    img.close()
    return PosterizedResult.PROCESSED


def _posterize_simple_wrapper_star(args: tuple[term.MessageQueue, Path, Path, int, PosterizePointMethod, SsimOption]):
    return _posterize_simple_wrapper(*args)


@click.command(
    name="posterize",
    help="Force posterize images to a specific bit depth using Pillow",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.dest_output(optional=False)
@click.option(
    "-b",
    "--bits",
    "num_bits",
    type=click.IntRange(1, 8),
    default=4,
    show_default=True,
    help="The number of bits to posterize the image to (1-8)",
)
@click.option(
    "-m",
    "--method",
    "posterize_method",
    type=click.Choice(["even", "lloyd"], case_sensitive=False),
    default="lloyd",
    show_default=True,
    help="The method to use for posterization",
)
@click.option(
    "-ssim",
    "--use-ssimulacra2",
    "use_ssimulacra2",
    is_flag=True,
    default=False,
    help="Try to detect and fix bad posterization, utilize vapoursynth",
)
@click.option(
    "-smin",
    "--ssim-min",
    "ssim_min",
    type=click.FloatRange(0.0, 100.0),
    default=80.0,  # anything above 80% is good
    show_default=True,
    help="The minimum SSIM score for an image to be considered good",
)
@options.threads
@options.recursive
@time_program
def posterize_simple(
    path_or_archive: Path,
    dest_output: Path,
    num_bits: int,
    posterize_method: str,
    use_ssimulacra2: bool,
    ssim_min: float,
    threads: int,
    recursive: bool,
):
    """
    Posterize images in a directory to a specific bit depth using Pillow.

    This will always use PNG as the output format.
    """
    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    candidates: list[Path] = []
    if not recursive:
        candidates.append(path_or_archive)
    else:
        console.info(f"Recursively collecting folder in {path_or_archive}...")
        for comic in file_handler.collect_all_comics(path_or_archive, dir_only=True):
            candidates.append(comic)
        console.info(f"Found {len(candidates)} archives/folders to posterize.")

    if not candidates and recursive:
        console.warning("No valid folders found to posterize.")
        return 1

    post_method = PosterizePointMethod.from_str(posterize_method)

    if use_ssimulacra2:
        vs_attach_logger(console)
        core = get_vapoursynth().core
        core.num_threads = 1
        console.info(f"Using vapoursynth to detect and fix bad posterization... (minimum score {ssim_min}%)")
        missing_plugins = vs_find_missing_plugins(["com.lumen.vship", "com.vapoursynth.bestsource"])

        if missing_plugins:
            console.warning(f"Missing vapoursynth plugins: {', '.join(missing_plugins)}")
            raise click.Abort()

    for path_real in candidates:
        if recursive:
            console.info(f"Processing: {path_real}")
        all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(path_real)]
        total_files = len(all_files)
        if total_files <= 0:
            console.warning(f"No images found in {path_real}, skipping.")
            continue

        console.info(f"Found {total_files} files in the directory.")
        ssim_opt = SsimOption(enabled=use_ssimulacra2, minimum=ssim_min)

        progress = console.make_progress()
        task = progress.add_task("Posterizing images...", finished_text="Posterized images", total=total_files)

        real_output = dest_output
        if recursive:
            real_output = dest_output / path_real.name
        real_output.mkdir(parents=True, exist_ok=True)

        results: list[PosterizedResult] = []
        console.info(f"Using {threads} CPU threads for processing.")
        with threaded_worker(console, lowest_or(threads, all_files)) as (pool, log_q):
            for result in pool.imap_unordered(
                _posterize_simple_wrapper_star,
                [(log_q, img_path, real_output, num_bits, post_method, ssim_opt) for img_path in all_files],
            ):
                results.append(result)
                progress.update(task, advance=1)

        console.stop_progress(progress, f"Posterized {total_files} images to {num_bits} bits.", skip_total=True)

        posterized_count = sum(1 for result in results if result == PosterizedResult.PROCESSED)
        copied_count = sum(1 for result in results if result == PosterizedResult.COPIED)

        if copied_count > 0:
            console.info(f"Copied {copied_count} images without posterization.")
        if posterized_count > 0:
            console.info(f"Posterized {posterized_count} images.")
    if recursive:
        console.info(f"Finished processing {len(candidates)} folders.")


def _drain(results: list[PosterizedResult], future, progress, task):
    """Collect one finished write, advancing the progress bar."""
    if future is not None:
        results.append(future.result())
        progress.update(task, advance=1)


def _submit_selection(
    pool: BoundedWritePool,
    group,
    scores,
    *,
    selected,
    offset: int,
    img_path: Path,
    dest_path: Path,
    debug: bool,
):
    """
    Write the selected candidate for one page, or copy the source when nothing passed.

    The winner is pulled from the same branch the metric just read, while the chunk's frames are
    still resident, so the write costs no second decode.
    """
    bits = int(selected[offset])
    if debug:
        console.log(f"{img_path.name}: {_format_scores(scores, offset, bits)}")
    if bits < 0:
        # No candidate passed every thresholded metric, so the source page is written and no
        # degraded page ever reaches the output.
        return _submit_fallback(pool, img_path, dest_path, None)
    with group.branches[bits].get_frame(offset) as frame:
        image = vs_grays_frame_to_image(frame)
    return pool.submit(_save_png, image, dest_path)


def _format_scores(scores, offset: int, chosen: int) -> str:
    """One page's scores per metric per candidate, for `-v`."""
    parts = []
    for metric, per_bits in scores.items():
        rendered = " ".join(f"{bits}b={float(values[offset]):.3f}" for bits, values in sorted(per_bits.items()))
        parts.append(f"{metric}[{rendered}]")
    picked = "source" if chosen < 0 else f"{chosen}b"
    return f"chose {picked} | " + " ".join(parts)


def _chunk_groups(sizes: Sequence[tuple[int, int]], start: int, stop: int) -> list[tuple[tuple[int, int], list[int]]]:
    """
    Group the absolute page indices in `[start, stop)` by their size.

    Grouping per chunk rather than per volume is what keeps a mixed size volume working without
    building a branch set for every size in the whole directory at once.
    """
    grouped: dict[tuple[int, int], list[int]] = {}
    for index in range(start, stop):
        grouped.setdefault(sizes[index], []).append(index)
    return list(grouped.items())


def _resolve_order(
    *,
    bits_policy: str | None,
    fewest_bits_default: bool,
    candidates: Sequence[int],
) -> list[int] | None:
    """
    Decide whether a caller preference order breaks ties, and if so which one.

    `--bits-policy` wins when it is given. Without it, a plain comma list is a preference order
    the caller typed and a range is not, which is the distinction :func:`resolve_candidates`
    already makes.
    """
    if bits_policy is not None:
        return list(candidates) if bits_policy.lower() == "order" else None
    if not fewest_bits_default:
        return list(candidates)
    return None


def _posterize2_fixed_depth(
    all_files: Sequence[Path],
    real_output: Path,
    *,
    num_bits: int,
    method: int,
    prefetch: int,
    cache_mb: int,
    threads: int,
    debug: bool,
    progress,
    task,
) -> list[PosterizedResult]:
    """
    Posterize every page to one depth, with nothing scored.

    A single candidate has nothing to choose between, so no metric is called and every page is
    written at `num_bits`. This is the command's original path and stays byte identical to it.
    """
    vs = get_vapoursynth()
    chain = vs_posterize_clip(
        all_files,
        bits=num_bits,
        method=method,
        prefetch=prefetch,
        debug=debug,
        cache_mb=cache_mb,
        core=vs.core,
    )

    results: list[PosterizedResult] = []
    with BoundedWritePool(threads) as pool:
        for index, img_path in enumerate(all_files):
            dest_path = real_output / img_path.with_suffix(".png").name
            with chain.posterized.get_frame(index) as frame:
                image = vs_frame_to_image(frame)
            _drain(results, pool.submit(_save_png, image, dest_path), progress, task)

        for future in pool.pending():
            _drain(results, future, progress, task)
    return results


def _posterize2_fan_out(
    all_files: Sequence[Path],
    real_output: Path,
    *,
    bit_candidates: Sequence[int],
    metric_names: Sequence[str],
    minimums: Mapping[str, float],
    order: Sequence[int] | None,
    method: int,
    chunk: int,
    prefetch: int,
    cache_mb: int,
    threads: int,
    debug: bool,
    progress,
    task,
) -> list[PosterizedResult]:
    """
    Score every candidate depth for a chunk of pages and write the least degraded one that passes.

    Pages are grouped by size inside each chunk, because `vship` takes one size per call. Within
    a chunk the frames the metric just read are the frames the winner is pulled from, so the
    winner costs no second decode.

    A page that no candidate passes is copied from the source, which is what the single depth
    gate already does with a failing page.
    """
    vs = get_vapoursynth()
    sizes = vs_page_sizes(all_files, core=vs.core)
    results: list[PosterizedResult] = []

    with BoundedWritePool(threads) as pool:
        for start in range(0, len(all_files), chunk):
            stop = min(start + chunk, len(all_files))
            for size, indices in _chunk_groups(sizes, start, stop):
                group = vs_posterize_group(
                    [all_files[index] for index in indices],
                    bits_candidates=bit_candidates,
                    method=method,
                    prefetch=prefetch,
                    debug=debug,
                    cache_mb=cache_mb,
                    core=vs.core,
                )
                if group.size != size:
                    raise ValueError(f"Built a {group.size} candidate group for a group of {size} pages")

                scores = vs_score_candidates(group, metrics=metric_names, core=vs.core)
                selected = vs_select_posterize_bits(
                    scores,
                    candidates=bit_candidates,
                    minimums=minimums,
                    order=order,
                )
                for offset, index in enumerate(indices):
                    img_path = all_files[index]
                    dest_path = real_output / img_path.with_suffix(".png").name
                    future = _submit_selection(
                        pool,
                        group,
                        scores,
                        selected=selected,
                        offset=offset,
                        img_path=img_path,
                        dest_path=dest_path,
                        debug=debug,
                    )
                    _drain(results, future, progress, task)

        for future in pool.pending():
            _drain(results, future, progress, task)
    return results


@click.command(
    name="posterize2",
    help="Force posterize images to a specific bit depth using VapourSynth (experimental)",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.dest_output(optional=False)
@click.option(
    "-b",
    "--bits",
    "bits",
    type=str,
    default="4",
    show_default=True,
    help=(
        "The bit depth to posterize to (1-8), or several depths to choose between, e.g. 2-5 or "
        "2,3,4. One depth posterizes every page to it; several are scored and the least degraded "
        "one that passes is kept per page."
    ),
)
@click.option(
    "-m",
    "--method",
    "posterize_method",
    type=click.Choice(["even", "lloyd"], case_sensitive=False),
    default="lloyd",
    show_default=True,
    help="The method to use for posterization",
)
@click.option(
    "--metrics",
    "metrics",
    type=str,
    default=None,
    help="Comma separated metrics to score the candidates with, e.g. ssimulacra2,butteraugli",
)
@click.option(
    "--metric-min",
    "metric_min",
    type=str,
    default=None,
    help=(
        "Per metric thresholds, e.g. ssimulacra2=80,butteraugli=1. "
        "A metric with no threshold is reported but never decides."
    ),
)
@click.option(
    "--bits-policy",
    "bits_policy",
    type=click.Choice(["fewest", "order"], case_sensitive=False),
    default=None,
    help="How a passing candidate is picked: fewest bits that passes, or the first one in the order given.",
)
@click.option(
    "--chunk",
    "chunk",
    type=click.IntRange(1, 512),
    default=DEFAULT_CHUNK,
    show_default=True,
    help="Pages whose candidates are scored and written together, sized against the frame cache",
)
@click.option(
    "--cache",
    "cache_mb",
    type=click.IntRange(64, 8192),
    default=512,
    show_default=True,
    help="VapourSynth frame cache in MiB",
)
@options.prefetch
@options.threads
@options.recursive
@time_program
def posterize2(
    path_or_archive: Path,
    dest_output: Path,
    bits: str,
    posterize_method: str,
    metrics: str | None,
    metric_min: str | None,
    bits_policy: str | None,
    chunk: int,
    cache_mb: int,
    prefetch: int,
    threads: int,
    recursive: bool,
):
    """
    Posterize images in a directory using VapourSynth, keeping the least degraded depth per page.

    This will always use PNG as the output format. VapourSynth schedules the pages across its
    own worker threads. `--prefetch` is how many pages the decoder reads ahead, and `--threads`
    is how many pages are encoded and written at once. Pass `-v` for the plugin's per frame log
    lines and the per page scores.

    `--bits` is the whole candidate set. One depth posterizes every page to it and scores
    nothing. Several make a fan-out: every page is posterized at each depth and scored, the least
    degraded depth that every thresholded metric accepts is kept, and a page no candidate passes
    is copied from the source. `--bits 1-6` is automatic degradation.
    """
    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    post_method = PosterizePointMethod.from_str(posterize_method)
    vs_attach_logger(console)

    bit_candidates, fewest_bits_default = resolve_candidates(bits)

    # A single depth is scored exactly like a set of them: the metric reads that one candidate
    # and the page is copied when it fails, which is the same rule as any other candidate. There
    # is simply no choice to make among passing candidates.
    #
    # Scoring is driven by whether a metric was asked for rather than by the candidate count, so
    # `--bits 4` on its own stays the fixed depth posterize it has always been, and `--bits 4
    # --metrics ssimulacra2` becomes the old single depth gate.
    scoring = metrics is not None or metric_min is not None or len(bit_candidates) > 1

    metric_names: list[str] = []
    minimums: dict[str, float] = {}
    if scoring:
        metric_names = parse_metrics(metrics)
        supplied = parse_metric_minimums(metric_min) if metric_min else {}
        minimums = resolve_metric_minimums(metric_names, minimums=supplied)

    # Both paths decode nothing themselves, so unlike `posterize` this never needs bestsource.
    required_plugins = ["xyz.n4o.nimages", "xyz.n4o.imgseqs"]
    if scoring:
        required_plugins.append("com.lumen.vship")
    missing_plugins = vs_find_missing_plugins(required_plugins)
    if missing_plugins:
        console.warning(f"Missing vapoursynth plugins: {', '.join(missing_plugins)}")
        raise click.Abort()

    if scoring:
        thresholds = ", ".join(
            f"{metric}{'<=' if not METRIC_HIGHER_IS_BETTER[metric] else '>='}{minimum:g}"
            for metric, minimum in minimums.items()
        )
        depths = ", ".join(str(bits) for bits in bit_candidates)
        console.info(f"Scoring candidate depth{'s' if len(bit_candidates) > 1 else ''} {depths} with "
                     f"{', '.join(metric_names)} ({thresholds or 'no thresholded metric'})")
        if len(bit_candidates) == 1:
            console.info("A page that fails is copied from the source, so a single depth is a yes or no gate.")

    candidates: list[Path] = []
    if not recursive:
        candidates.append(path_or_archive)
    else:
        console.info(f"Recursively collecting folder in {path_or_archive}...")
        for comic in file_handler.collect_all_comics(path_or_archive, dir_only=True):
            candidates.append(comic)
        console.info(f"Found {len(candidates)} archives/folders to posterize.")

    if not candidates and recursive:
        console.warning("No valid folders found to posterize.")
        return 1

    order = _resolve_order(
        bits_policy=bits_policy,
        fewest_bits_default=fewest_bits_default,
        candidates=bit_candidates,
    )

    for path_real in candidates:
        if recursive:
            console.info(f"Processing: {path_real}")
        all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(path_real)]
        total_files = len(all_files)
        if total_files <= 0:
            console.warning(f"No images found in {path_real}, skipping.")
            continue

        all_files.sort(key=lambda path: path.stem)
        console.info(f"Found {total_files} files in the directory.")

        real_output = dest_output
        if recursive:
            real_output = dest_output / path_real.name
        real_output.mkdir(parents=True, exist_ok=True)

        progress = console.make_progress()
        task = progress.add_task("Posterizing images...", finished_text="Posterized images", total=total_files)

        results: list[PosterizedResult] = []
        if scoring:
            per_page = _posterize2_fan_out(
                all_files,
                real_output,
                bit_candidates=bit_candidates,
                metric_names=metric_names,
                minimums=minimums,
                order=order,
                method=post_method.to_nimages(),
                chunk=chunk,
                prefetch=prefetch,
                cache_mb=cache_mb,
                threads=threads,
                debug=console.debugged,
                progress=progress,
                task=task,
            )
            summary = f"{total_files} images (candidates {', '.join(str(bits) for bits in bit_candidates)})"
        else:
            per_page = _posterize2_fixed_depth(
                all_files,
                real_output,
                num_bits=bit_candidates[0],
                method=post_method.to_nimages(),
                prefetch=prefetch,
                cache_mb=cache_mb,
                threads=threads,
                debug=console.debugged,
                progress=progress,
                task=task,
            )
            summary = f"{total_files} images to {bit_candidates[0]} bits"
        results.extend(per_page)

        console.stop_progress(progress, f"Posterized {summary}.", skip_total=True)

        posterized_count = sum(1 for result in results if result == PosterizedResult.PROCESSED)
        copied_count = sum(1 for result in results if result == PosterizedResult.COPIED)

        if copied_count > 0:
            console.info(f"Copied {copied_count} images without posterization.")
        if posterized_count > 0:
            console.info(f"Posterized {posterized_count} images.")
    if recursive:
        console.info(f"Finished processing {len(candidates)} folders.")


def _autoposterize_wrapper(
    log_q: term.MessageOrInterface, img_path: Path, dest_output: Path, threshold: float, use_palette_mode: bool
) -> PosterizedResult:
    img = Image.open(img_path)
    dest_path = dest_output / img_path.with_suffix(".png").name

    shades = analyze_gray_shades(img, threshold)

    # Pad the shades to the nearest bpc value
    shades_nums = pad_shades_to_bpc(shades)
    bpc_count = max(len(shades_nums).bit_length() - 1, 1)
    if bpc_count >= 8:
        # same 8bpc, just copy the image
        img.close()
        if dest_path.exists():
            cnsl = term.with_thread_queue(log_q)
            cnsl.warning(f"Skipping existing file: {dest_path}")
            return PosterizedResult.COPIED
        shutil.copy2(img_path, dest_path)
        return PosterizedResult.COPIED

    if use_palette_mode:
        posterized = posterize_image_by_bits(img, bpc_count)
    else:
        posterized = posterize_image_by_shades(img, shades_nums)

    posterized.save(dest_path, format="PNG")
    posterized.close()
    img.close()
    return PosterizedResult.PROCESSED


def _autoposterize_wrapper_star(args: tuple[term.MessageQueue, Path, Path, float, bool]) -> PosterizedResult:
    return _autoposterize_wrapper(*args)


@click.command(
    name="autoposterize",
    help="(Experimental) Analyze and posterize images to optimal bit depth using Pillow",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@options.dest_output(optional=False)
@click.option(
    "-th",
    "--threshold",
    "threshold_pct",
    type=click.FloatRange(0.0, 100.0),
    default=0.01,
    show_default=True,
    help="The threshold percentage to consider a shade as significant (0-100%)",
)
@click.option(
    "-pm",
    "--palette-mode",
    "use_palette_mode",
    is_flag=True,
    default=False,
    help="Use palette mode for posterization instead of direct color mapping (may produce better results)",
)
@options.threads
@options.recursive
@time_program
def auto_posterize(
    path_or_archive: Path,
    dest_output: Path,
    threshold_pct: float,
    use_palette_mode: bool,
    threads: int,
    recursive: bool,
):
    """
    Automatically analyze and posterize images in a directory to an optimal bit depth using Pillow.

    This will always use PNG as the output format.
    """
    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    if math.isclose(threshold_pct, 0.0):
        raise click.BadParameter(
            "Threshold percentage cannot be 0.0, as this will consider all shades as significant.",
            param_hint="threshold_pct",
        )

    candidates: list[Path] = []
    if not recursive:
        candidates.append(path_or_archive)
    else:
        console.info(f"Recursively collecting folder in {path_or_archive}...")
        for comic in file_handler.collect_all_comics(path_or_archive, dir_only=True):
            candidates.append(comic)
        console.info(f"Found {len(candidates)} archives/folders to autoposterize.")

    if not candidates and recursive:
        console.warning("No valid folders found to autoposterize.")
        return 1

    for path_real in candidates:
        if recursive:
            console.info(f"Processing: {path_real}")

        all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(path_real)]
        total_files = len(all_files)
        if total_files <= 0:
            console.warning(f"No images found in {path_real}, skipping.")
            continue
        console.info(f"Found {total_files} files in the directory.")

        progress = console.make_progress()
        taks = progress.add_task(
            "Auto-posterizing images...", finished_text="Auto-posterized images", total=total_files
        )

        real_output = dest_output
        if recursive:
            real_output = dest_output / path_real.name
        real_output.mkdir(parents=True, exist_ok=True)

        results: list[PosterizedResult] = []
        console.info(f"Using {threads} CPU threads for processing.")
        with threaded_worker(console, lowest_or(threads, all_files)) as (pool, log_q):
            for result in pool.imap_unordered(
                _autoposterize_wrapper_star,
                [(log_q, img_path, real_output, threshold_pct, use_palette_mode) for img_path in all_files],
            ):
                results.append(result)
                progress.update(taks, advance=1)

        console.stop_progress(progress, f"Auto-posterized {total_files} images.")
        posterized_count = sum(1 for result in results if result == PosterizedResult.PROCESSED)
        copied_count = sum(1 for result in results if result == PosterizedResult.COPIED)

        if copied_count > 0:
            console.info(f"Copied {copied_count} images without autoposterize.")
        if posterized_count > 0:
            console.info(f"Posterized {posterized_count} images.")

    if recursive:
        console.info(f"Finished processing {len(candidates)} folders.")


@click.command(
    name="analyze-shades",
    help="(Experimental) Analyze and show the gray shades in images in a directory using Pillow",
    cls=NMangaCommandHandler,
)
@options.path_or_archive(disable_archive=True)
@click.option(
    "-th",
    "--threshold",
    "threshold_pct",
    type=click.FloatRange(0.0, 100.0),
    default=0.01,
    show_default=True,
    help="The threshold percentage to consider a shade as significant (0-100%)",
)
@time_program
def analyze_shades(
    path_or_archive: Path,
    threshold_pct: float,
):
    """
    Analyze and show the gray shades in images in a directory using Pillow.
    """
    if not path_or_archive.is_dir():
        raise click.BadParameter(
            f"{path_or_archive} is not a directory. Please provide a directory.",
            param_hint="path_or_archive",
        )

    if math.isclose(threshold_pct, 0.0):
        raise click.BadParameter(
            "Threshold percentage cannot be 0.0, as this will consider all shades as significant.",
            param_hint="threshold_pct",
        )

    all_files = [file for file, _, _, _ in file_handler.collect_image_from_folder(path_or_archive)]
    total_files = len(all_files)
    console.info(f"Found {total_files} files in the directory.")

    progress = console.make_progress()
    task = progress.add_task("Analyzing images...", finished_text="Analyzed images", total=total_files)
    for image_path in all_files:
        shades = analyze_gray_shades(Image.open(image_path), threshold_pct)
        if len(shades) == 0:
            console.info(f"No significant shades found in {image_path}")
            continue

        closest_bpc = detect_nearest_bpc(shades)
        total_shades = len(shades)
        console.info(f"Shades found in {image_path}: (Total: {total_shades}, Closest bpc: {closest_bpc}bpp)")
        progress.update(task, advance=1)

    console.stop_progress(progress, f"Analyzed {total_files} images!")

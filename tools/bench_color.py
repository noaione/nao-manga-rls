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

# Benchmark the OGSOV/OGSOX color kernels: the Python kernel on a thread pool versus VapourSynth.

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich import box
from rich.markup import escape
from rich.table import Table

from nmanga import file_handler, term
from nmanga.lazy import get_vapoursynth
from nmanga.ogsov import (
    DetectedColor,
    detect_image_color_ml_bulk,
    register_ogsov_instance,
    register_ogsox_instance,
)
from nmanga.vapour import (
    vs_analyze_ogsov_frame,
    vs_attach_logger,
    vs_find_missing_plugins,
    vs_iter_frames_async,
    vs_prepare_image_bulk_raw,
)

try:
    # pyrefly: ignore[missing-import]
    import pyvips

    logging.getLogger("pyvips").setLevel(logging.WARNING)
except ImportError:  # the pyvips decoder is optional
    pyvips = None

PYTHON_MODES = ("ogsov-base", "ogsov-fast", "ogsox-base", "ogsox-fast")
VAPOURSYNTH_MODES = ("ogsov-vs", "ogsov-vs-gpu", "ogsox-vs", "ogsox-vs-gpu")
ALL_MODES = PYTHON_MODES + VAPOURSYNTH_MODES
VAPOURSYNTH_PLUGINS = ("xyz.n4o.mangacolor", "xyz.n4o.imgseqs")


@dataclass
class BenchResult:
    mode: str
    threads: int
    pages: int
    color_pages: set[str]
    times: list[float]

    @property
    def best(self) -> float:
        return min(self.times)

    @property
    def mean(self) -> float:
        return sum(self.times) / len(self.times)

    @property
    def rate(self) -> float:
        return self.pages / self.best if self.best > 0 else 0.0


def _require_file(parser: argparse.ArgumentParser, value: Path | None, flag: str) -> Path:
    if value is None:
        parser.error(f"{flag} is required for the selected modes")
    if not value.is_file():
        parser.error(f"{flag} does not exist: {value}")
    return value


def _resolve_kernel(
    mode: str,
    *,
    color_model: Path,
    lookup_table: Path | None,
    classifier: Path | None,
) -> Any:
    """Load (or fetch the cached) Python kernel for the mode, outside the timed section."""

    if mode.startswith("ogsov"):
        return register_ogsov_instance(color_model)
    if lookup_table is None or classifier is None:
        raise SystemExit(f"{mode} needs --lookup-table and --classifier")
    return register_ogsox_instance(lookup_table, classifier)


def _detect_image_color_ml_vips(path: Path, *, kernel: Any, fast_method: bool) -> DetectedColor:
    """Classify a page from a libvips decode, without reading the file into Python bytes first."""

    if pyvips is None:
        raise RuntimeError("pyvips is not installed")

    image = pyvips.Image.new_from_file(str(path), access="sequential")
    if image.interpretation == "cmyk":
        image = image.colourspace("srgb")
    array = image.numpy()
    if array.ndim == 2 or array.shape[2] <= 2:
        return DetectedColor(False, 100, "is grayscale mode", False)
    rgb = array[:, :, :3]
    if np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 0], rgb[:, :, 2]):
        return DetectedColor(False, 100, "is grayscale RGB", True)

    is_color, confidence = kernel.predict(rgb[:, :, ::-1], fast_method=fast_method)
    return DetectedColor(is_color, confidence, "ML-based detection", is_color)


def _run_python_vips(images: list[Path], *, kernel: Any, fast_method: bool, threads: int) -> set[str]:
    """Classify every page with a pyvips decode, `threads` pages at a time."""

    if pyvips is not None:
        pyvips.cache_set_max(0)
        pyvips.concurrency_set(max(1, (os.cpu_count() or 1) // max(1, threads)))

    def work(path: Path) -> tuple[str, bool]:
        return path.name, _detect_image_color_ml_vips(path, kernel=kernel, fast_method=fast_method).is_color

    if threads <= 1:
        results = [work(path) for path in images]
    else:
        with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="nmanga-vips") as pool:
            results = list(pool.map(work, images))
    return {name for name, is_color in results if is_color}


def _run_python(
    mode: str, images: list[Path], *, kernel: Any, threads: int, decode: str
) -> tuple[int, set[str]]:
    """Classify every page with the Python kernel, `threads` pages at a time."""

    fast_method = mode.endswith("-fast")
    if decode == "pyvips":
        return len(images), _run_python_vips(images, kernel=kernel, fast_method=fast_method, threads=threads)

    color_pages: set[str] = set()
    for img_path, detected in detect_image_color_ml_bulk(
        images, kernel=kernel, fast_method=fast_method, threads=threads
    ):
        if detected.is_color:
            color_pages.add(img_path.name)
    return len(images), color_pages


def _run_vapoursynth(
    mode: str, images: list[Path], *, threads: int, prefetch: int, debug: bool
) -> tuple[int, set[str]]:
    """Classify every page with the `mangacolor` plugin, `threads` frames in flight."""

    vs = get_vapoursynth()
    core = vs.core
    gpu = mode.endswith("-gpu")
    clip = vs_prepare_image_bulk_raw(images, debug=debug, prefetch=prefetch)
    if mode.startswith("ogsov"):
        analyzed = core.mangacolor.OGSOV(clip, debug=debug, gpu=gpu)
    else:
        analyzed = core.mangacolor.OGSOX(clip, debug=debug, gpu=gpu)

    pages = analyzed.num_frames
    color_pages: set[str] = set()
    for _, frame in vs_iter_frames_async(analyzed, window=threads):
        detected, img_path = vs_analyze_ogsov_frame(frame)
        if detected.is_color:
            color_pages.add(img_path.name)
    del analyzed, clip
    return pages, color_pages


def _references(results: list[BenchResult]) -> dict[str, BenchResult]:
    """The first run of each mode, which every other thread count has to match."""

    references: dict[str, BenchResult] = {}
    for result in results:
        references.setdefault(result.mode, result)
    return references


def _print_header(console: term.Console, images: list[Path], args: argparse.Namespace, threads: list[int]) -> None:
    console.console.print()
    console.console.print("[bold]Color detection benchmark[/]")
    console.console.print(f"  [dim]path[/]     {escape(str(args.path))}")
    console.console.print(f"  [dim]images[/]   {len(images)}")
    console.console.print(f"  [dim]modes[/]    {', '.join(args.modes)}")
    console.console.print(
        f"  [dim]threads[/]  {', '.join(str(value) for value in threads)}"
        f"  [dim]decode[/] {args.decode}  [dim]prefetch[/] {args.prefetch}  [dim]repeat[/] {args.repeat}"
    )
    console.console.print("  [dim]VS threads are frames in flight; one uses the blocking get_frame.[/]")


def _print_results(
    console: term.Console, results: list[BenchResult], references: dict[str, BenchResult]
) -> None:
    table = Table(title="Results", title_justify="left", header_style="bold", box=box.ROUNDED)
    table.add_column("Mode", style="cyan", no_wrap=True)
    table.add_column("Threads", justify="right")
    table.add_column("Pages", justify="right")
    table.add_column("Color", justify="right")
    table.add_column("Best", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Check", justify="right")
    for result in results:
        reference = references[result.mode]
        same = result.pages == reference.pages and result.color_pages == reference.color_pages
        table.add_row(
            result.mode,
            str(result.threads),
            str(result.pages),
            str(len(result.color_pages)),
            f"{result.best:.3f}s",
            f"{result.mean:.3f}s",
            f"{result.rate:.2f}/s",
            "[success]ok[/]" if same else "[error]DIFF[/]",
        )
    console.console.print()
    console.console.print(table)


def _check_correctness(
    console: term.Console, results: list[BenchResult], references: dict[str, BenchResult]
) -> int:
    """
    Every thread count must find the same pages as the lowest one for its mode.

    A different count here is a real bug, not a model difference: the pages are
    classified independently, so changing how many are in flight cannot change which
    ones are color.
    """

    console.console.print()
    failures = [
        result
        for result in results
        if result.pages != references[result.mode].pages
        or result.color_pages != references[result.mode].color_pages
    ]
    if not failures:
        console.console.print("[success]Correctness:[/] every thread count agrees with the lowest one for its mode.")
        return 0

    console.console.print("[error]Correctness: MISMATCH[/] against the lowest thread count for the same mode.")
    for result in failures:
        reference = references[result.mode]
        missing = sorted(reference.color_pages - result.color_pages)
        extra = sorted(result.color_pages - reference.color_pages)
        console.console.print(
            f"  [error]{result.mode}[/] [dim]threads={result.threads}[/]: "
            f"pages={result.pages}/{reference.pages} missing={missing} extra={extra}"
        )
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the OGSOV/OGSOX color kernels over a folder of images.",
        epilog=(
            "Example: bench_color.py -p ./chapter -m ogsov-base ogsov-fast -t 1 4 8 "
            "--color-model ogsov.npz\n\n"
            "The vs modes use the mangacolor plugin's own model configuration, exactly like "
            "`nmanga lookup colorama --method vs-*`; the Python modes need the model files. "
            "Threads are worker threads for the Python modes and frames in flight for the "
            "VapourSynth ones, where one thread uses the blocking get_frame."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--path",
        type=Path,
        required=True,
        help="Folder of images to classify",
    )
    parser.add_argument(
        "-m",
        "--mode",
        dest="modes",
        nargs="+",
        choices=ALL_MODES,
        default=list(ALL_MODES),
        help="Modes to benchmark (default: all of them)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        nargs="+",
        type=int,
        default=[1],
        help="Thread counts to run each mode with (default: 1)",
    )
    parser.add_argument("--color-model", type=Path, help="OGSOV weights table (.npz)")
    parser.add_argument("--lookup-table", type=Path, help="OGSOX lookup table (.npz)")
    parser.add_argument("--classifier", type=Path, help="OGSOX classifier (.onnx)")
    parser.add_argument(
        "--decode",
        choices=["pillow", "pyvips"],
        default="pillow",
        help=("Image decoder for the Python modes (default: pillow); the vs modes decode through the plugin"),
    )
    parser.add_argument("--prefetch", type=int, default=8, help="VapourSynth decoder readahead (default: 8)")
    parser.add_argument("--repeat", type=int, default=1, help="Runs per mode and thread count (default: 1)")
    parser.add_argument("--limit", type=int, default=0, help="Only classify the first N images")
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Classify one page before timing, so model loading is not measured (default: on)",
    )
    parser.add_argument("--debug", action="store_true", help="Print the VapourSynth plugin's messages")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.prefetch < 0:
        parser.error("--prefetch cannot be negative")
    if not args.path.is_dir():
        parser.error(f"{args.path} is not a directory")

    images = [image for image, _, _, _ in file_handler.collect_image_from_folder(args.path)]
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        parser.error(f"No images found in {args.path}")

    threads = sorted({max(1, value) for value in args.threads})
    needs_ogsov_python = any(mode in ("ogsov-base", "ogsov-fast") for mode in args.modes)
    needs_ogsox_python = any(mode in ("ogsox-base", "ogsox-fast") for mode in args.modes)

    if needs_ogsov_python:
        _require_file(parser, args.color_model, "--color-model")
    if needs_ogsox_python:
        _require_file(parser, args.lookup_table, "--lookup-table")
        _require_file(parser, args.classifier, "--classifier")
    if args.decode == "pyvips" and pyvips is None:
        parser.error("--decode pyvips needs pyvips (try: uv run --with pyvips python tools/bench_color.py ...)")
    if any(mode in VAPOURSYNTH_MODES for mode in args.modes):
        missing = vs_find_missing_plugins(list(VAPOURSYNTH_PLUGINS))
        if missing:
            parser.error(f"Missing VapourSynth plugins: {', '.join(missing)}")
        if args.debug:
            vs_attach_logger(term.get_console())

    console = term.get_console()
    _print_header(console, images, args, threads)

    results: list[BenchResult] = []
    for mode in args.modes:
        console.console.print(f"\n[info]▸ {mode}[/]")
        kernel = None
        if mode in PYTHON_MODES:
            kernel = _resolve_kernel(
                mode,
                color_model=args.color_model,
                lookup_table=args.lookup_table,
                classifier=args.classifier,
            )
            if args.warmup:
                _run_python(mode, images[:1], kernel=kernel, threads=1, decode=args.decode)
        elif args.warmup:
            _run_vapoursynth(mode, images[:1], threads=1, prefetch=args.prefetch, debug=args.debug)

        for thread_count in threads:
            times: list[float] = []
            pages = 0
            color_pages: set[str] = set()
            for run in range(args.repeat):
                with console.console.status(
                    f"[dim]running {mode} with {thread_count} thread(s)…[/]",
                    spinner="dots",
                    spinner_style="info",
                ):
                    start = time.perf_counter()
                    if kernel is not None:
                        pages, color_pages = _run_python(
                            mode, images, kernel=kernel, threads=thread_count, decode=args.decode
                        )
                    else:
                        pages, color_pages = _run_vapoursynth(
                            mode, images, threads=thread_count, prefetch=args.prefetch, debug=args.debug
                        )
                    elapsed = time.perf_counter() - start
                times.append(elapsed)
                if args.repeat > 1:
                    console.console.print(f"    [dim]run {run + 1}/{args.repeat}[/]  [bold]{elapsed:.3f}s[/]")
            result = BenchResult(mode, thread_count, pages, color_pages, times)
            results.append(result)
            console.console.print(
                f"  [success]✓[/] [bold]threads={thread_count}[/]  [dim]·[/]  {pages} pages"
                f"  [dim]·[/]  {len(color_pages)} color  [dim]·[/]  best [bold]{result.best:.3f}s[/]"
            )

    references = _references(results)
    _print_results(console, results, references)
    return _check_correctness(console, results, references)


if __name__ == "__main__":
    raise SystemExit(main())

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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nmanga import file_handler, term
from nmanga.lazy import get_vapoursynth
from nmanga.ogsov import detect_image_color_ml_bulk, register_ogsov_instance, register_ogsox_instance
from nmanga.vapour import (
    vs_analyze_ogsov_frame,
    vs_attach_logger,
    vs_find_missing_plugins,
    vs_iter_frames_async,
    vs_prepare_image_bulk_raw,
)

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


def _run_python(mode: str, images: list[Path], *, kernel: Any, threads: int) -> tuple[int, set[str]]:
    """Classify every page with the Python kernel, `threads` pages at a time."""

    fast_method = mode.endswith("-fast")
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


def _check_correctness(results: list[BenchResult], references: dict[str, BenchResult]) -> int:
    """
    Every thread count must find the same pages as the lowest one for its mode.

    A different count here is a real bug, not a model difference: the pages are
    classified independently, so changing how many are in flight cannot change which
    ones are color.
    """

    failures = [
        result
        for result in results
        if result.pages != references[result.mode].pages
        or result.color_pages != references[result.mode].color_pages
    ]
    if not failures:
        print("Correctness: every thread count agrees with the lowest one for its mode.")
        return 0

    print("Correctness: MISMATCH against the lowest thread count for the same mode.")
    for result in failures:
        reference = references[result.mode]
        missing = sorted(reference.color_pages - result.color_pages)
        extra = sorted(result.color_pages - reference.color_pages)
        print(
            f"  {result.mode} threads={result.threads}: "
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
    if any(mode in VAPOURSYNTH_MODES for mode in args.modes):
        missing = vs_find_missing_plugins(list(VAPOURSYNTH_PLUGINS))
        if missing:
            parser.error(f"Missing VapourSynth plugins: {', '.join(missing)}")
        if args.debug:
            vs_attach_logger(term.get_console())

    print(f"Images: {len(images)} in {args.path}")
    print(f"Modes: {', '.join(args.modes)}")
    print(f"Threads: {', '.join(str(value) for value in threads)}  prefetch: {args.prefetch}  repeat: {args.repeat}")
    print("VapourSynth threads are frames in flight; one uses the blocking get_frame.")
    print()

    results: list[BenchResult] = []
    for mode in args.modes:
        kernel = None
        if mode in PYTHON_MODES:
            kernel = _resolve_kernel(
                mode,
                color_model=args.color_model,
                lookup_table=args.lookup_table,
                classifier=args.classifier,
            )
            if args.warmup:
                _run_python(mode, images[:1], kernel=kernel, threads=1)
        elif args.warmup:
            _run_vapoursynth(mode, images[:1], threads=1, prefetch=args.prefetch, debug=args.debug)

        for thread_count in threads:
            times: list[float] = []
            pages = 0
            color_pages: set[str] = set()
            for run in range(args.repeat):
                start = time.perf_counter()
                if kernel is not None:
                    pages, color_pages = _run_python(mode, images, kernel=kernel, threads=thread_count)
                else:
                    pages, color_pages = _run_vapoursynth(
                        mode, images, threads=thread_count, prefetch=args.prefetch, debug=args.debug
                    )
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                if args.repeat > 1:
                    print(f"  {mode} threads={thread_count} run {run + 1}: {elapsed:.3f}s")
            results.append(BenchResult(mode, thread_count, pages, color_pages, times))

    references = _references(results)
    print()
    header = f"{'mode':<15} {'threads':>7} {'pages':>6} {'color':>5} {'best':>9} {'mean':>9} {'rate':>9} {'check':>5}"
    print(header)
    print("-" * len(header))
    for result in results:
        reference = references[result.mode]
        check = "ok" if result.color_pages == reference.color_pages and result.pages == reference.pages else "DIFF"
        print(
            f"{result.mode:<15} {result.threads:>7} {result.pages:>6} {len(result.color_pages):>5} "
            f"{result.best:>8.3f}s {result.mean:>8.3f}s {result.rate:>7.2f}/s {check:>5}"
        )

    print()
    return _check_correctness(results, references)


if __name__ == "__main__":
    raise SystemExit(main())

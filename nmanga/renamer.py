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

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

__all__ = (
    "QualityMapping",
    "determine_quality_suffix",
    "shift_renaming_gen",
)


@dataclass
class QualityMapping:
    """The mapping for image quality based on resolution thresholds, we use height in pixels."""

    LQ: int = 1500
    """Low Quality threshold in height pixels."""
    HQ: int = 2000
    """High Quality threshold in height pixels."""


def shift_renaming_gen(
    all_files: list[Path],
    *,
    start_index: int = 0,
    title: str | None = None,
    volume: str | None = None,
    spreads_aware: bool = False,
    reverse: bool = False,
) -> dict[Path, Path]:
    """
    Rename files in a shifted manner, useful for inserting pages in between existing pages.

    Parameters
    ----------
    all_files: :class:`list` of :class:`pathlib.Path`
        List of all file paths to be renamed.
    start_index: :class:`int`, optional
        The index from which to start renaming (default is 0).
    title: :class:`str` or :class:`None`, optional
        The title to use for renaming files. If :class:`None`, random names will be used (default is :class:`None`).
    spreads_aware: :class:`bool`, optional
        Whether to consider spreads when renaming (default is :class:`False`).
    reverse: :class:`bool`, optional
        Whether to rename files in reverse order (default is :class:`False`).

    Returns
    -------
    :class:`dict` of :class:`pathlib.Path`
        A mapping of original file paths to their new renamed paths.

    Raises
    ------
    :class:`ValueError`
        If the files cannot be renamed due to naming conflicts.
    """

    total_files = len(all_files)
    renaming_map: dict[str, Path] = {}

    # Pre-sort files to ensure consistent ordering
    sorted_files = sorted(all_files, key=lambda path: path.name, reverse=reverse)
    resolution_maps = {}
    # at minimum we do 3 digits, if pages has more we increase accordingly
    digit_count = max(3, len(str(total_files + start_index)))
    current_index = start_index
    if spreads_aware:
        for file_path in sorted_files:
            with Image.open(file_path) as img:
                width, height = img.size
                if width < height:
                    resolution_maps[height] = width
                if width > height:
                    single_page_width = resolution_maps.get(height, 1)
                    width_ratio = float(width) / float(single_page_width)
                    spread_page_count = math.trunc(width_ratio)
                    if spread_page_count == 0:
                        spread_page_count = 1
                    page_index = f"p{current_index:0{digit_count}d}"
                    if spread_page_count > 1:
                        page_index += f"-{current_index + spread_page_count - 1:0{digit_count}d}"
                    renaming_map[page_index] = file_path
                    current_index += spread_page_count
                else:
                    page_index = f"p{current_index:0{digit_count}d}"
                    renaming_map[page_index] = file_path
                    current_index += 1
    else:
        for file_path in sorted_files:
            page_index = f"p{current_index:0{digit_count}d}"
            renaming_map[page_index] = file_path
            current_index += 1

    final_renaming_map: dict[Path, Path] = {}
    for new_page, original_file in renaming_map.items():
        title_and_volume = ""
        if title is not None:
            title_and_volume += f"{title} - "
        if volume is not None:
            title_and_volume += f"{volume} - "
        new_file_name = f"{title_and_volume}{new_page}{original_file.suffix}"
        new_file_path = original_file.with_name(new_file_name)
        if new_file_path in final_renaming_map.values():
            raise ValueError(f"Renaming conflict detected for file: {new_file_path}")
        final_renaming_map[original_file] = new_file_path
    return final_renaming_map


def determine_quality_suffix(*, quality: str, image_path: Path, quality_maps: QualityMapping) -> str:
    """
    Determine the quality suffix based on the base quality and image resolution.

    Parameters
    ----------
    base_quality: :class:`str`
        The base quality setting (e.g., "auto", "LQ", "HQ").
    image_path: :class:`pathlib.Path`
        The path to the image file.
    quality_maps: :class:`QualityMapping`
        The quality mapping thresholds.

    Returns
    -------
    :class:`str`
        The determined quality suffix.
    """

    if quality.lower() in ("auto", "mixed"):
        with Image.open(image_path) as img:
            if img.height >= quality_maps.HQ:
                return "HQ"
            elif img.height <= quality_maps.LQ:
                return "LQ"
    return quality

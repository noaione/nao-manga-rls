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

import ctypes
import errno
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

__all__ = (
    "QualityMapping",
    "RenameConflictError",
    "RenameFailedError",
    "RenameResult",
    "RenameValidationError",
    "apply_rename_order",
    "compute_rename_order",
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

    @classmethod
    def from_config(cls: type["QualityMapping"], lq_threshold: int | None, hq_threshold: int | None) -> QualityMapping:
        """Create a QualityMapping instance from configuration thresholds."""
        defaults = cls()
        if lq_threshold is None:
            lq_threshold = defaults.LQ
        if hq_threshold is None:
            hq_threshold = defaults.HQ
        return cls(LQ=lq_threshold, HQ=hq_threshold)


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
        Whether to number the pages bottom up, so the last file becomes the first page
        (default is :class:`False`). This decides page order only; the order the moves are
        applied in is decided by :func:`apply_rename_order`.

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


class RenameConflictError(Exception):
    """The requested renaming cannot be performed safely, no matter the order."""


class RenameValidationError(Exception):
    """The requested renaming cannot be performed: the mapping or the folder is wrong."""


class RenameFailedError(Exception):
    """A rename failed mid-flight.

    Attributes
    ----------
    error: :class:`OSError`
        The error that stopped the run.
    source: :class:`pathlib.Path` or :class:`None`
        The file that could not be renamed, if the failure came from a move.
    rolled_back: :class:`int`
        How many of the completed moves were successfully undone.
    still_displaced: :class:`list` of :class:`tuple` of :class:`pathlib.Path`
        ``(original, current)`` pairs for moves that could not be undone. Empty when
        the folder was fully restored.
    """

    def __init__(
        self,
        message: str,
        *,
        error: OSError | None = None,
        source: Path | None = None,
        rolled_back: int = 0,
        still_displaced: list[tuple[Path, Path]] | None = None,
    ):
        super().__init__(message)
        self.error = error
        self.source = source
        self.rolled_back = rolled_back
        self.still_displaced: list[tuple[Path, Path]] = still_displaced or []


@dataclass
class RenameResult:
    """What :func:`apply_rename_order` did, or would have done under ``dry_run``."""

    moves: list[tuple[Path, Path]]
    """The full ordered plan, as ``(source, destination)``."""
    applied: list[tuple[Path, Path]]
    """The moves that were actually performed. Always empty under ``dry_run``."""
    rolled_back: int = 0
    """How many moves were undone after a failure."""
    still_displaced: list[tuple[Path, Path]] = field(default_factory=list)
    """``(original, current)`` for moves the rollback could not undo."""


def _detect_case_insensitive(folder: Path) -> bool:
    """
    Whether `folder` resolves names without regard to case.

    A rename that only changes the case is a real rename on such a filesystem, but the
    destination reports as already existing, so the conflict check has to be able to tell
    "this is one of our own files" from "this is somebody else's file".
    """

    probe = folder / "._nmanga_case_probe.TMP"
    try:
        probe.write_bytes(b"")
    except OSError:
        return os.path.normcase("A") == os.path.normcase("a")

    try:
        return (folder / "._nmanga_case_probe.tmp").is_file()
    finally:
        probe.unlink(missing_ok=True)


def _known_destination(
    candidate: Path,
    from_mapping: set[str],
    resolve: Callable[[str], Path | None],
    case_insensitive: bool,
) -> bool:
    """
    Whether `candidate` names a file that is part of this renaming.

    Exact matches are the normal case. A case folding filesystem also reports a
    differently cased name as the same file, so the name is resolved through the directory
    to tell a genuine conflict from a rename that only changes the case.
    """

    if os.fspath(candidate) in from_mapping:
        return True

    resolved = resolve(candidate.name)
    if resolved is None:
        return False
    if os.fspath(resolved) in from_mapping:
        return True
    if not case_insensitive:
        return False
    # The guessed path carries the case the caller asked for, not the case on disk, so
    # fold both sides before comparing.
    folded = os.fspath(resolved).casefold()
    return any(os.fspath(path).casefold() == folded for path in from_mapping)


def _same_path(left: Path, right: Path) -> bool:
    """
    Whether two paths are the same name, exactly.

    :meth:`pathlib.Path.__eq__` folds case on Windows, which would make a rename that
    only changes the case look like a no-op and skip it.
    """

    return os.fspath(left) == os.fspath(right)


def compute_rename_order(mapping: dict[Path, Path]) -> list[tuple[Path, Path]]:
    """
    Work out a safe order to apply `mapping` in.

    Every destination that is also a source has to move before the file that wants its
    name, so the moves form a dependency graph. A depth first post order over that graph
    produces an order where nothing is overwritten: a right shift comes out descending,
    a left shift ascending, with no input from the caller.

    Cycles are refused rather than broken up. A cycle means the requested renaming is not
    expressible as a sequence of in-place renames at all, and silently routing the files
    through a temporary name would hide that.

    Parameters
    ----------
    mapping: :class:`dict` of :class:`pathlib.Path` to :class:`pathlib.Path`
        A mapping of original file paths to their new paths.

    Returns
    -------
    :class:`list` of :class:`tuple` of :class:`pathlib.Path`
        The ``(source, destination)`` moves in a safe order. Files whose name does not
        change are left out.

    Raises
    ------
    :class:`RenameConflictError`
        If the moves contain a cycle.
    """

    moves: list[tuple[Path, Path]] = []

    done: set[str] = set()
    active: set[str] = set()
    path: list[Path] = []

    def visit(source: Path) -> None:
        key = os.fspath(source)
        if key in done:
            return
        if key in active:
            # A back edge: everything from the first sighting of `source` onwards is the
            # cycle. Anything visited before that is a dependency, not part of it.
            cycle = path[path.index(source) :]
            described = "\n".join(f"  {member.name}  ->  {mapping[member].name}" for member in cycle)
            raise RenameConflictError(
                f"cannot rename, no safe order exists:\n{described}\n"
                "These files would overwrite each other in a cycle. Move them out of the "
                "way first, or change the starting index."
            )

        destination = mapping[source]
        if not _same_path(destination, source):
            active.add(key)
            path.append(source)
            if destination in mapping:
                visit(destination)
            path.pop()
            active.discard(key)
            moves.append((source, destination))

        done.add(key)

    for source in mapping:
        visit(source)

    return moves


# The non-replacing rename syscalls are not exposed by Python, so reach for them with
# ctypes. Both are best effort: neither flag works on every filesystem, so a failure to
# even find the symbol leaves the caller with the portable guard instead.
_RENAME_NOREPLACE = 1  # Linux renameat2(2)
_RENAME_EXCL = 0x00000001  # macOS renamex_np(2)
_AT_FDCWD = -100  # Linux AT_FDCWD
_UNSUPPORTED_ERRNOS = frozenset({errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP})


class _NoReplaceUnsupportedError(Exception):
    """Raised internally when the syscall exists but the filesystem refuses the flag."""


def _build_no_replace() -> Callable[[str, str], None] | None:
    """Bind the platform's atomic non-replacing rename, or return :class:`None`."""

    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            renameat2 = libc.renameat2
        except (AttributeError, OSError):
            return None
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int

        def rename_noreplace(source: str, destination: str) -> None:
            ctypes.set_errno(0)
            result = renameat2(
                _AT_FDCWD,
                os.fsencode(source),
                _AT_FDCWD,
                os.fsencode(destination),
                _RENAME_NOREPLACE,
            )
            if result != 0:
                code = ctypes.get_errno()
                if code in _UNSUPPORTED_ERRNOS:
                    raise _NoReplaceUnsupportedError(os.strerror(code))
                raise OSError(code, os.strerror(code), destination)

        return rename_noreplace

    if sys.platform == "darwin":
        try:
            libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
            renamex_np = libsystem.renamex_np
        except (AttributeError, OSError):
            return None
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int

        def rename_excl(source: str, destination: str) -> None:
            ctypes.set_errno(0)
            result = renamex_np(os.fsencode(source), os.fsencode(destination), _RENAME_EXCL)
            if result != 0:
                code = ctypes.get_errno()
                if code in _UNSUPPORTED_ERRNOS:
                    raise _NoReplaceUnsupportedError(os.strerror(code))
                raise OSError(code, os.strerror(code), destination)

        return rename_excl

    return None


def _rename_no_replace(source: Path, destination: Path) -> None:
    """
    Move `source` to `destination`, refusing to replace an unrelated existing file.

    Atomically non-replacing where the platform offers it, an explicit existence check
    otherwise. The ordering from :func:`compute_rename_order` already guarantees the
    destination is free in practice, so this exists to catch an outside writer rather
    than to make the common case safe.

    A rename that only changes the case is allowed through: on a case folding filesystem
    the destination reports as existing, but it is the very same file.
    """

    no_replace = _NO_REPLACE
    if no_replace is not None:
        try:
            no_replace(os.fspath(source), os.fspath(destination))
            return
        except _NoReplaceUnsupportedError:
            pass

    if destination.exists():
        try:
            same_file = source.samefile(destination)
        except OSError:
            same_file = False
        if not same_file:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), os.fspath(destination))

    try:
        source.rename(destination)
    except FileExistsError:
        # Nothing is lost yet, so re-check before giving up: a plain rename refuses to
        # replace on Windows, and on a case folding filesystem it may still have been a
        # case only rename that the check above could not resolve.
        try:
            same_file = source.samefile(destination)
        except OSError:
            same_file = False
        if not same_file:
            raise
        source.replace(destination)


_NO_REPLACE = _build_no_replace()


def _rollback(applied: list[tuple[Path, Path]]) -> tuple[int, list[tuple[Path, Path]]]:
    """Undo `applied` in reverse. Returns how many were restored and what is stuck."""

    rolled_back = 0
    still_displaced: list[tuple[Path, Path]] = []
    for source, destination in reversed(applied):
        try:
            destination.rename(source)
        except OSError:
            still_displaced.append((source, destination))
        else:
            rolled_back += 1
    # Report in application order, which reads better than reverse order.
    still_displaced.reverse()
    return rolled_back, still_displaced


def apply_rename_order(mapping: dict[Path, Path], *, dry_run: bool = False) -> RenameResult:
    """
    Validate and apply `mapping`, undoing everything if a move fails partway.

    Validation runs to completion before the first rename, so a mapping or folder that
    cannot be renamed safely leaves the folder untouched. A failure during the moves
    themselves is rolled back in reverse, which always restores the original names:
    a move only ever runs once its destination is free or was vacated by an earlier move
    in the same run.

    Parameters
    ----------
    mapping: :class:`dict` of :class:`pathlib.Path` to :class:`pathlib.Path`
        A mapping of original file paths to their new paths.
    dry_run: :class:`bool`, optional
        Validate and plan without renaming anything (default is :class:`False`).

    Returns
    -------
    :class:`RenameResult`
        The plan, and what was applied.

    Raises
    ------
    :class:`RenameValidationError`
        If a source is missing, two sources want the same name, or a destination is held
        by a file that is not part of the renaming.
    :class:`RenameConflictError`
        If the moves contain a cycle.
    :class:`RenameFailedError`
        If a move failed while being applied. Carries the rollback outcome.
    """

    moves = compute_rename_order(mapping)

    absent = [path for path in mapping if not path.exists()]
    if absent:
        raise RenameValidationError(
            f"cannot rename, {len(absent)} source file(s) do not exist:\n"
            + "\n".join(f"  {path.name}" for path in absent)
        )

    destinations: dict[Path, Path] = {}
    duplicates: dict[Path, list[Path]] = {}
    for source, destination in moves:
        if destination in destinations:
            duplicates.setdefault(destination, [destinations[destination]]).append(source)
        destinations[destination] = source
    if duplicates:
        described = "\n".join(
            f"  {destination.name}  <-  {', '.join(path.name for path in wanted_by)}"
            for destination, wanted_by in duplicates.items()
        )
        raise RenameValidationError(
            f"cannot rename, {len(duplicates)} name(s) are wanted by more than one file:\n{described}"
        )

    # A destination that exists and is not one of our sources is a file that nobody asked
    # to touch, so refuse rather than rename over it.
    if not mapping:
        return RenameResult(moves=moves, applied=[])

    folder = next(iter(mapping)).parent
    from_mapping = {os.fspath(path) for path in mapping}
    case_insensitive = _detect_case_insensitive(folder)
    resolve_cache: dict[str, Path | None] = {}

    def resolve(name: str) -> Path | None:
        if name not in resolve_cache:
            candidate = folder / name
            resolve_cache[name] = candidate if candidate.is_file() else None
        return resolve_cache[name]

    occupied = [
        destination
        for _, destination in moves
        if destination.exists() and not _known_destination(destination, from_mapping, resolve, case_insensitive)
    ]
    if occupied:
        raise RenameValidationError(
            f"cannot rename, {len(occupied)} destination(s) are already taken by another file:\n"
            + "\n".join(f"  {path.name}" for path in occupied)
        )

    result = RenameResult(moves=moves, applied=[])
    if dry_run:
        return result

    applied: list[tuple[Path, Path]] = []
    # Bound up front so the handler below can always name the move that failed, even if
    # the failure somehow lands before the first iteration binds the loop variables.
    source: Path | None = None
    destination: Path | None = None
    try:
        for current_source, current_destination in moves:
            source, destination = current_source, current_destination
            _rename_no_replace(source, destination)
            applied.append((source, destination))
    except OSError as error:
        rolled_back, still_displaced = _rollback(applied)
        result.applied = applied
        result.rolled_back = rolled_back
        result.still_displaced = still_displaced
        if source is not None and destination is not None:
            message = f"rename failed at {source.name} -> {destination.name} ({error.strerror or error})"
        else:
            message = f"rename failed before any move could be made ({error.strerror or error})"
        if still_displaced:
            message += (
                f"\n       rolled back {rolled_back} of {len(applied)} completed move(s)"
                "\n       STILL DISPLACED, fix by hand:\n"
                + "\n".join(
                    f"         {original.name} is now at {current.name}" for original, current in still_displaced
                )
            )
        else:
            message += f"\n       rolled back all {rolled_back} completed move(s), nothing was changed"
        raise RenameFailedError(
            message,
            error=error,
            source=source,
            rolled_back=rolled_back,
            still_displaced=still_displaced,
        ) from error

    result.applied = applied
    return result


def determine_quality_suffix(*, quality: str, image_path: Path, quality_maps: QualityMapping) -> str | None:
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
    :class:`str` or :class:`None`
        The determined quality suffix, or :class:`None` if it cannot be determined (which means standard quality).
    """

    if quality.lower() in ("auto", "mixed"):
        with Image.open(image_path) as img:
            if img.height >= quality_maps.HQ:
                return "HQ"
            elif img.height <= quality_maps.LQ:
                return "LQ"
            else:
                return None
    return quality

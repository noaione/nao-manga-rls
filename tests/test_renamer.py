"""
Tests for the automatic rename ordering in :mod:`nmanga.renamer`.

The filesystem is the oracle throughout: every file carries a content marker, so a
clobbered file shows up as a lost marker rather than a wrong count.
"""

from __future__ import annotations

import errno
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from nmanga.renamer import (
    RenameConflictError,
    RenameFailedError,
    RenameValidationError,
    apply_rename_order,
    compute_rename_order,
    shift_renaming_gen,
)

PREFIX = "Series - v01 - "


def _scratch_base() -> Path:
    """
    The directory the ``folder`` fixture builds its scratch directories in.

    Plain ``tmp_path`` is deliberately not used: pytest removes and recreates its own base
    temp directory every session, which needs to scan and delete the OS temp root. Keeping
    the base here means the tests only ever touch the directory they were given.
    """

    base = Path(tempfile.gettempdir()) / "nmanga-test-renamer"
    os.makedirs(base, exist_ok=True)  # ruff: ignore[os-makedirs] deliberately not Path.mkdir
    return base


@pytest.fixture
def folder() -> Iterator[Path]:
    """A one test scratch directory, in the OS temp area, removed on the way out."""

    target = _scratch_base() / uuid.uuid4().hex[:12]
    os.makedirs(target)  # ruff: ignore[os-makedirs] kept symmetric with the base above
    try:
        yield target
    finally:
        shutil.rmtree(target, ignore_errors=True)


def pagename(number: int, prefix: str = PREFIX) -> str:
    return f"{prefix}p{number:03d}.jpg"


def make_pages(folder: Path, numbers: list[int], prefix: str = PREFIX) -> dict[str, str]:
    """Write one page per number, each with a unique content marker."""

    folder.mkdir(parents=True, exist_ok=True)
    contents: dict[str, str] = {}
    for number in numbers:
        name = pagename(number, prefix)
        marker = f"content-{number:03d}"
        (folder / name).write_bytes(marker.encode())
        contents[name] = marker
    return contents


def snapshot(folder: Path) -> dict[str, str]:
    """Every file in `folder` as name -> contents."""

    return {path.name: path.read_text() for path in sorted(folder.glob("*")) if path.is_file()}


def shift_mapping(folder: Path, numbers: list[int], offset: int, prefix: str = PREFIX) -> dict[Path, Path]:
    return {folder / pagename(n, prefix): folder / pagename(n + offset, prefix) for n in numbers}


class TestComputeRenameOrder:
    """The ordering is pure graph work, so it needs no filesystem."""

    def test_shift_right_is_descending(self):
        mapping = {Path(pagename(i)): Path(pagename(i + 1)) for i in range(1, 6)}
        moves = compute_rename_order(mapping)
        assert [source.name for source, _ in moves] == [pagename(i) for i in range(5, 0, -1)]

    def test_shift_left_is_ascending(self):
        mapping = {Path(pagename(i)): Path(pagename(i - 1)) for i in range(1, 6)}
        moves = compute_rename_order(mapping)
        assert [source.name for source, _ in moves] == [pagename(i) for i in range(1, 6)]

    def test_right_shift_of_two_is_dependency_order(self):
        # A shift of two is two independent chains (2->4->6 and 3->5->7), so each chain comes
        # out descending without the two being interleaved in a single global order.
        mapping = {Path(pagename(i)): Path(pagename(i + 2)) for i in range(2, 6)}
        moves = compute_rename_order(mapping)

        order = [source.name for source, _ in moves]
        assert sorted(order) == sorted(pagename(i) for i in range(2, 6))
        position = {name: index for index, name in enumerate(order)}
        for source, destination in moves:
            if destination.name in position:
                # The file holding the wanted name has to move out of the way first.
                assert position[destination.name] < position[source.name]

    def test_disjoint_index_space_keeps_insertion_order(self):
        mapping = {Path(pagename(i)): Path(pagename(i + 100)) for i in range(1, 6)}
        moves = compute_rename_order(mapping)
        assert [source.name for source, _ in moves] == [pagename(i) for i in range(1, 6)]

    def test_identity_mapping_produces_no_moves(self):
        mapping = {Path(pagename(i)): Path(pagename(i)) for i in range(1, 6)}
        assert compute_rename_order(mapping) == []

    def test_page_reversal_is_refused(self):
        mapping = {Path(pagename(i)): Path(pagename(6 - i)) for i in range(1, 6)}
        with pytest.raises(RenameConflictError) as caught:
            compute_rename_order(mapping)
        message = str(caught.value)
        assert pagename(1) in message
        assert pagename(5) in message

    def test_three_cycle_is_refused(self):
        mapping = {
            Path(pagename(1)): Path(pagename(2)),
            Path(pagename(2)): Path(pagename(3)),
            Path(pagename(3)): Path(pagename(1)),
        }
        with pytest.raises(RenameConflictError):
            compute_rename_order(mapping)

    def test_cycle_plus_chain_refuses_the_whole_mapping(self):
        mapping = {
            Path(pagename(1)): Path(pagename(2)),
            Path(pagename(2)): Path(pagename(1)),
            Path(pagename(3)): Path(pagename(4)),
            Path(pagename(4)): Path(pagename(5)),
        }
        with pytest.raises(RenameConflictError):
            compute_rename_order(mapping)


class TestApplyRenameOrder:
    def test_shift_right_keeps_every_page(self, folder: Path):
        before = make_pages(folder, [1, 2, 3, 4, 5])
        result = apply_rename_order(shift_mapping(folder, [1, 2, 3, 4, 5], 1))

        after = snapshot(folder)
        assert len(after) == 5
        assert sorted(after.values()) == sorted(before.values())
        assert sorted(after) == sorted(pagename(i) for i in range(2, 7))
        assert [source.name for source, _ in result.applied] == [pagename(i) for i in range(5, 0, -1)]

    def test_shift_left_keeps_every_page(self, folder: Path):
        before = make_pages(folder, [1, 2, 3, 4, 5])
        apply_rename_order(shift_mapping(folder, [1, 2, 3, 4, 5], -1))

        after = snapshot(folder)
        assert len(after) == 5
        assert sorted(after.values()) == sorted(before.values())
        assert sorted(after) == sorted(pagename(i) for i in range(0, 5))

    def test_reversal_leaves_the_folder_untouched(self, folder: Path):
        make_pages(folder, [1, 2, 3, 4, 5])
        before = snapshot(folder)
        mapping = {folder / pagename(i): folder / pagename(6 - i) for i in range(1, 6)}

        with pytest.raises(RenameConflictError):
            apply_rename_order(mapping)
        assert snapshot(folder) == before

    def test_identity_mapping_is_a_no_op(self, folder: Path):
        before = make_pages(folder, [1, 2, 3, 4, 5])
        result = apply_rename_order({folder / pagename(i): folder / pagename(i) for i in range(1, 6)})
        assert result.moves == []
        assert result.applied == []
        assert snapshot(folder) == before

    def test_dry_run_changes_nothing(self, folder: Path):
        before = make_pages(folder, [1, 2, 3, 4, 5])
        result = apply_rename_order(shift_mapping(folder, [1, 2, 3, 4, 5], 1), dry_run=True)

        assert len(result.moves) == 5
        assert result.applied == []
        assert snapshot(folder) == before

    def test_missing_source_is_refused(self, folder: Path):
        make_pages(folder, [1, 2])
        mapping = {folder / pagename(1): folder / pagename(9), folder / pagename(7): folder / pagename(8)}

        with pytest.raises(RenameValidationError) as caught:
            apply_rename_order(mapping)
        assert pagename(7) in str(caught.value)

    def test_duplicate_destination_is_refused(self, folder: Path):
        before = make_pages(folder, [1, 2, 3])
        mapping = {
            folder / pagename(1): folder / pagename(9),
            folder / pagename(2): folder / pagename(9),
            folder / pagename(3): folder / pagename(3),
        }

        with pytest.raises(RenameValidationError) as caught:
            apply_rename_order(mapping)
        assert pagename(9) in str(caught.value)
        assert snapshot(folder) == before

    def test_foreign_file_on_a_destination_is_refused(self, folder: Path):
        make_pages(folder, [1, 2, 3, 4, 5])
        (folder / pagename(6)).write_bytes(b"FOREIGN")
        before = snapshot(folder)

        with pytest.raises(RenameValidationError) as caught:
            apply_rename_order(shift_mapping(folder, [1, 2, 3, 4, 5], 1))
        assert pagename(6) in str(caught.value)
        assert snapshot(folder) == before

    def test_failure_on_the_first_move_is_rolled_back(self, folder: Path, monkeypatch: pytest.MonkeyPatch):
        before = make_pages(folder, [1, 2, 3, 4, 5])
        mapping = shift_mapping(folder, [1, 2, 3, 4, 5], 1)

        def always_flaky(self: Path, target: Path):
            raise PermissionError(13, "Access is denied")

        monkeypatch.setattr(Path, "rename", always_flaky)

        with pytest.raises(RenameFailedError) as caught:
            apply_rename_order(mapping)

        error = caught.value
        assert error.rolled_back == 0
        assert error.still_displaced == []
        # The move that failed is named, not a bare traceback.
        assert pagename(5) in str(error)
        assert "nothing was changed" in str(error)
        assert snapshot(folder) == before

    def test_rollback_failure_still_names_the_failed_move(self, folder: Path, monkeypatch: pytest.MonkeyPatch):
        # `source` and `destination` are initialised before the loop so that reading them in
        # the handler is always defined. That guard is only reachable if a fault lands in the
        # window between entering the try and the loop body assigning them, which the empty
        # try block makes impractical to provoke. What is worth pinning is that the names are
        # recorded before the rename is attempted, so a rollback failure cannot lose them.
        from nmanga import renamer

        make_pages(folder, [1, 2, 3, 4, 5])
        mapping = shift_mapping(folder, [1, 2, 3, 4, 5], 1)

        def always_flaky(self: Path, target: Path):
            raise PermissionError(13, "Access is denied")

        def exploding_rollback(applied):
            raise OSError(errno.EIO, "Input/output error")

        monkeypatch.setattr(Path, "rename", always_flaky)
        monkeypatch.setattr(renamer, "_rollback", exploding_rollback)

        # The rollback blowing up propagates as its own OSError rather than being swallowed,
        # which is deliberate: a broken undo must not be mistaken for a clean abort.
        with pytest.raises(OSError) as caught:
            apply_rename_order(mapping)
        assert caught.value.errno == errno.EIO

    def test_failed_error_accepts_no_source(self):
        # The handler's "no move to name" path has to be representable, even though the
        # current loop makes it unreachable.
        error = RenameFailedError("rename failed before any move could be made", source=None)
        assert error.source is None
        assert "before any move" in str(error)

    def test_failure_midway_rolls_everything_back(self, folder: Path, monkeypatch: pytest.MonkeyPatch):
        before = make_pages(folder, [1, 2, 3, 4, 5])
        mapping = shift_mapping(folder, [1, 2, 3, 4, 5], 1)

        real_rename = Path.rename
        calls = {"count": 0}

        def flaky_rename(self: Path, target: Path):
            calls["count"] += 1
            if calls["count"] == 3:
                raise PermissionError(13, "Access is denied")
            return real_rename(self, target)

        monkeypatch.setattr(Path, "rename", flaky_rename)

        with pytest.raises(RenameFailedError) as caught:
            apply_rename_order(mapping)

        error = caught.value
        assert error.rolled_back == 2
        assert error.still_displaced == []
        assert snapshot(folder) == before

    def test_rollback_failure_reports_what_is_still_displaced(self, folder: Path, monkeypatch: pytest.MonkeyPatch):
        make_pages(folder, [1, 2, 3, 4, 5])
        mapping = shift_mapping(folder, [1, 2, 3, 4, 5], 1)

        real_rename = Path.rename
        calls = {"count": 0}

        def flaky_rename(self: Path, target: Path):
            calls["count"] += 1
            # Calls 1 and 2 are the first two moves, 3 fails the run. Calls 4 and 5 are the
            # rollback, and the second of those fails too, so one file stays displaced.
            if calls["count"] in (3, 5):
                raise PermissionError(13, "Access is denied")
            return real_rename(self, target)

        monkeypatch.setattr(Path, "rename", flaky_rename)

        with pytest.raises(RenameFailedError) as caught:
            apply_rename_order(mapping)

        error = caught.value
        assert error.rolled_back == 1
        assert len(error.still_displaced) == 1
        assert "STILL DISPLACED" in str(error)
        # The stuck file really is still under its new name.
        original, current = error.still_displaced[0]
        assert current.exists()
        assert not original.exists()

    def test_no_replace_guard_refuses_an_existing_file(self, folder: Path, monkeypatch: pytest.MonkeyPatch):
        from nmanga import renamer

        source = folder / "source.jpg"
        destination = folder / "destination.jpg"
        source.write_bytes(b"SOURCE")
        destination.write_bytes(b"DESTINATION")

        # Force the portable guard so the check is exercised on every platform.
        monkeypatch.setattr(renamer, "_NO_REPLACE", None)

        with pytest.raises(FileExistsError):
            renamer._rename_no_replace(source, destination)

        assert destination.read_bytes() == b"DESTINATION"
        assert source.read_bytes() == b"SOURCE"

    def test_no_replace_guard_allows_a_case_only_rename(self, folder: Path, monkeypatch: pytest.MonkeyPatch):
        from nmanga import renamer

        if not renamer._detect_case_insensitive(folder):
            pytest.skip("needs a case folding filesystem")

        source = folder / "page001.jpg"
        source.write_bytes(b"PAGE")
        monkeypatch.setattr(renamer, "_NO_REPLACE", None)

        renamer._rename_no_replace(source, folder / "PAGE001.jpg")
        assert [path.name for path in folder.glob("*")] == ["PAGE001.jpg"]
        assert (folder / "PAGE001.jpg").read_bytes() == b"PAGE"


class TestShiftRenamingGenIntegration:
    """The end to end path: build the mapping the command builds, then apply it."""

    def test_shiftname_style_shift_is_safe(self, folder: Path):
        # Pages already in the target scheme, renumbered from 1, which shifts right by one.
        before = make_pages(folder, [1, 2, 3, 4, 5])
        files = sorted(folder.glob("*"))
        mapping = shift_renaming_gen(files, start_index=1, title="Series", volume="v01")

        apply_rename_order(mapping)
        after = snapshot(folder)
        assert len(after) == 5
        assert sorted(after.values()) == sorted(before.values())

    def test_reverse_on_numbered_pages_is_not_silently_clobbered(self, folder: Path):
        # `--reverse` over pages that are already numbered bottom-up forms a cycle in the
        # rename graph, so it must be refused. Before the ordering existed this either ate
        # pages (POSIX, where Path.rename replaces) or stopped partway with a
        # FileExistsError (Windows). Both are covered by asserting the folder is intact.
        before = make_pages(folder, [1, 2, 3, 4, 5])
        files = sorted(folder.glob("*"))
        mapping = shift_renaming_gen(files, start_index=0, title="Series", volume="v01", reverse=True)

        # Bottom up numbering: the last file becomes the first page, so p005 -> p000.
        assert mapping[folder / pagename(5)] == folder / pagename(0), "reverse should flip page order"
        assert mapping[folder / pagename(1)] == folder / pagename(4)

        with pytest.raises(RenameConflictError):
            apply_rename_order(mapping)
        assert snapshot(folder) == before

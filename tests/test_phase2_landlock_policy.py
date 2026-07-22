from __future__ import annotations

from pathlib import Path

import cvsrffi.phase2_landlock_policy as policy


def test_landlock_access_masks_distinguish_directories_and_files(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    regular = tmp_path / "regular.bin"
    regular.write_bytes(b"x")
    assert policy._read_only_access(directory) == policy.LANDLOCK_FS_READ_EXECUTE
    assert (
        policy._read_only_access(regular)
        == policy.LANDLOCK_FS_FILE_READ_EXECUTE
    )
    assert policy._read_write_access(directory) == policy.LANDLOCK_FS_READ_WRITE
    assert (
        policy._read_write_access(regular)
        == policy.LANDLOCK_FS_FILE_READ_WRITE
    )


def test_file_write_mask_does_not_grant_directory_creation_rights(
    tmp_path: Path,
) -> None:
    regular = tmp_path / "device-placeholder"
    regular.write_bytes(b"x")
    mask = policy._read_write_access(regular)
    assert mask & policy.LANDLOCK_ACCESS_FS_READ_FILE
    assert mask & policy.LANDLOCK_ACCESS_FS_WRITE_FILE
    assert not mask & policy.LANDLOCK_ACCESS_FS_MAKE_DIR
    assert not mask & policy.LANDLOCK_ACCESS_FS_MAKE_SYM

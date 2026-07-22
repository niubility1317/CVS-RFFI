from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from cvsrffi.phase2_isolated_runner import Phase2IsolatedRunnerError
from cvsrffi.phase2_landlock_isolated_runner import (
    audit_landlock_lifecycle,
    map_open_ledger,
)


def _trace(python: Path, predictor: Path) -> str:
    return "\n".join(
        [
            (
                f'100 execve("{python}", ["{python}", "/controller/launcher.py", '
                f'"--predictor-entry", "{predictor}"], 0x0) = 0'
            ),
            "100 landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION) = 4",
            "100 landlock_create_ruleset({handled_access_fs=0x7fff}, 16, 0) = 3",
            "100 landlock_add_rule(3, LANDLOCK_RULE_PATH_BENEATH, 0x1, 0) = 0",
            "100 prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) = 0",
            "100 landlock_restrict_self(3, 0) = 0",
            "100 seccomp(SECCOMP_SET_MODE_FILTER, 0, 0x1) = 0",
            f'100 execve("{python}", ["{python}", "{predictor}"], 0x0) = 0',
            f'100 openat(AT_FDCWD, "{predictor}", O_RDONLY) = 3<{predictor}>',
        ]
    )


def test_lifecycle_binds_predictor_after_landlock_and_seccomp(
) -> None:
    python = PurePosixPath("/system/python")
    predictor = PurePosixPath("/runtime/code/scripts/predictor.py")
    result = audit_landlock_lifecycle(
        _trace(python, predictor),
        expected_executable=python,
        expected_entrypoint=predictor,
    )
    assert result["status"] == "PASS"
    assert result["seccomp_filter_line"] < result["predictor_exec_line"]


def test_lifecycle_does_not_mistake_launcher_for_predictor(
) -> None:
    python = PurePosixPath("/system/python")
    predictor = PurePosixPath("/runtime/code/scripts/predictor.py")
    trace = _trace(python, predictor).replace(
        "100 seccomp(SECCOMP_SET_MODE_FILTER, 0, 0x1) = 0\n", ""
    )
    with pytest.raises(Phase2IsolatedRunnerError, match="seccomp"):
        audit_landlock_lifecycle(
            trace,
            expected_executable=python,
            expected_entrypoint=predictor,
        )


def test_host_ledger_mapping_uses_longest_exact_or_prefix_binding() -> None:
    ledger = [
        {
            "path": "/srv/run/runtime/scripts/predict.py",
            "successful_open_count": 1,
            "syscalls": ["openat"],
        },
        {
            "path": "/srv/run/package/query.npz",
            "successful_open_count": 2,
            "syscalls": ["openat"],
        },
        {
            "path": "/srv/run/package.seal.json",
            "successful_open_count": 1,
            "syscalls": ["openat"],
        },
    ]
    mapped = map_open_ledger(
        ledger,
        host_to_logical={
            "/srv/run/runtime": "/runtime/code",
            "/srv/run/package": "/sealed/package",
            "/srv/run/package.seal.json": "/sealed/package.seal.json",
        },
    )
    assert mapped == [
        {
            "path": "/runtime/code/scripts/predict.py",
            "successful_open_count": 1,
            "syscalls": ["openat"],
        },
        {
            "path": "/sealed/package.seal.json",
            "successful_open_count": 1,
            "syscalls": ["openat"],
        },
        {
            "path": "/sealed/package/query.npz",
            "successful_open_count": 2,
            "syscalls": ["openat"],
        },
    ]

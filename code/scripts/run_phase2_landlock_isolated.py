#!/usr/bin/env python
"""Run a Phase2 predictor under an irreversible Landlock filesystem allowlist."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path


SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446
PR_SET_NO_NEW_PRIVS = 38
LANDLOCK_RULE_PATH_BENEATH = 1
LANDLOCK_CREATE_RULESET_VERSION = 1
ACCESS_EXECUTE = 1 << 0
ACCESS_WRITE_FILE = 1 << 1
ACCESS_READ_FILE = 1 << 2
ACCESS_READ_DIR = 1 << 3
ACCESS_REMOVE_DIR = 1 << 4
ACCESS_REMOVE_FILE = 1 << 5
ACCESS_MAKE_CHAR = 1 << 6
ACCESS_MAKE_DIR = 1 << 7
ACCESS_MAKE_REG = 1 << 8
ACCESS_MAKE_SOCK = 1 << 9
ACCESS_MAKE_FIFO = 1 << 10
ACCESS_MAKE_BLOCK = 1 << 11
ACCESS_MAKE_SYM = 1 << 12
ACCESS_REFER = 1 << 13
ACCESS_TRUNCATE = 1 << 14
READ_FILE = ACCESS_READ_FILE | ACCESS_EXECUTE
LIST_DIR = ACCESS_READ_DIR | ACCESS_EXECUTE
READ_DIR = ACCESS_READ_FILE | ACCESS_READ_DIR | ACCESS_EXECUTE
WRITE_DIR = (
    READ_DIR | ACCESS_WRITE_FILE | ACCESS_REMOVE_DIR | ACCESS_REMOVE_FILE |
    ACCESS_MAKE_CHAR | ACCESS_MAKE_DIR | ACCESS_MAKE_REG | ACCESS_MAKE_SOCK |
    ACCESS_MAKE_FIFO | ACCESS_MAKE_BLOCK | ACCESS_MAKE_SYM | ACCESS_REFER | ACCESS_TRUNCATE
)


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathBeneathAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]


def _add_rule(libc, ruleset_fd: int, path: Path, access: int) -> None:
    flags = os.O_PATH | os.O_CLOEXEC
    fd = os.open(str(path), flags)
    try:
        attr = PathBeneathAttr(access, fd)
        rc = libc.syscall(
            SYS_LANDLOCK_ADD_RULE, ruleset_fd, LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(attr), 0,
        )
        if rc != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(path))
    finally:
        os.close(fd)


def _restrict(allowlist: dict, write_dir: Path, runtime_read_dirs: list[Path]) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    abi = int(libc.syscall(SYS_LANDLOCK_CREATE_RULESET, 0, 0, LANDLOCK_CREATE_RULESET_VERSION))
    if abi < 1:
        raise RuntimeError("Landlock unavailable")
    handled = WRITE_DIR
    attr = RulesetAttr(handled)
    ruleset_fd = int(libc.syscall(SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0))
    if ruleset_fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    try:
        for raw in allowlist["read_files"]:
            _add_rule(libc, ruleset_fd, Path(str(raw)).resolve(strict=True), READ_FILE)
        for raw in allowlist["runtime_code_list_dirs"]:
            _add_rule(libc, ruleset_fd, Path(str(raw)).resolve(strict=True), LIST_DIR)
        for path in runtime_read_dirs:
            _add_rule(libc, ruleset_fd, path.resolve(strict=True), READ_DIR)
        for raw in ("/usr", "/lib", "/lib64", "/etc", "/proc", "/sys", "/run"):
            path = Path(raw)
            if path.exists():
                _add_rule(libc, ruleset_fd, path, READ_DIR)
        _add_rule(libc, ruleset_fd, Path("/dev"), WRITE_DIR)
        _add_rule(libc, ruleset_fd, write_dir.resolve(strict=True), WRITE_DIR)
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        if libc.syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    finally:
        os.close(ruleset_fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--write-dir", type=Path, required=True)
    parser.add_argument("--runtime-read-dir", type=Path, action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("predictor command is required after --")
    allowlist = json.loads(args.allowlist.read_text(encoding="utf-8-sig"))
    if allowlist.get("schema") != "cvs_phase2_landlock_allowlist_v1":
        raise ValueError("allowlist schema drift")
    args.write_dir.mkdir(parents=True, exist_ok=True)
    sandbox_home = args.write_dir / ".sandbox_home"
    sandbox_home.mkdir(exist_ok=True)
    os.environ.update({
        "HOME": str(sandbox_home), "TMPDIR": str(args.write_dir),
        "XDG_CACHE_HOME": str(sandbox_home / ".cache"), "CUDA_CACHE_DISABLE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    _restrict(allowlist, args.write_dir, list(args.runtime_read_dir))
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())

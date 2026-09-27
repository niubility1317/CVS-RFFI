"""Apply an unprivileged Landlock v4 sandbox for a Phase2 predictor."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
from typing import Iterable


LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
PR_SET_NO_NEW_PRIVS = 38

LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
LANDLOCK_ACCESS_FS_REFER = 1 << 13
LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14

LANDLOCK_ACCESS_NET_BIND_TCP = 1 << 0
LANDLOCK_ACCESS_NET_CONNECT_TCP = 1 << 1

LANDLOCK_FS_ALL_V4 = (1 << 15) - 1
LANDLOCK_FS_READ_EXECUTE = (
    LANDLOCK_ACCESS_FS_EXECUTE
    | LANDLOCK_ACCESS_FS_READ_FILE
    | LANDLOCK_ACCESS_FS_READ_DIR
)
LANDLOCK_FS_READ_WRITE = (
    LANDLOCK_FS_READ_EXECUTE
    | LANDLOCK_ACCESS_FS_WRITE_FILE
    | LANDLOCK_ACCESS_FS_REMOVE_DIR
    | LANDLOCK_ACCESS_FS_REMOVE_FILE
    | LANDLOCK_ACCESS_FS_MAKE_DIR
    | LANDLOCK_ACCESS_FS_MAKE_REG
    | LANDLOCK_ACCESS_FS_MAKE_SOCK
    | LANDLOCK_ACCESS_FS_MAKE_FIFO
    | LANDLOCK_ACCESS_FS_MAKE_SYM
    | LANDLOCK_ACCESS_FS_REFER
    | LANDLOCK_ACCESS_FS_TRUNCATE
)
LANDLOCK_FS_FILE_READ_EXECUTE = (
    LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_READ_FILE
)
LANDLOCK_FS_FILE_READ_WRITE = (
    LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_WRITE_FILE
)
LANDLOCK_NET_ALL_V4 = (
    LANDLOCK_ACCESS_NET_BIND_TCP | LANDLOCK_ACCESS_NET_CONNECT_TCP
)


class LandlockPolicyError(RuntimeError):
    """Raised if the required unprivileged kernel sandbox cannot be applied."""


class _RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


def _syscall_numbers() -> tuple[int, int, int]:
    if os.uname().machine != "x86_64":
        raise LandlockPolicyError("Landlock syscall numbers are reviewed only for x86_64")
    return 444, 445, 446


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _raise_errno(context: str) -> None:
    value = ctypes.get_errno()
    raise LandlockPolicyError(f"{context}: [{value}] {os.strerror(value)}")


def query_landlock_abi() -> int:
    create_ruleset, _add_rule, _restrict_self = _syscall_numbers()
    result = int(
        _libc().syscall(
            create_ruleset,
            0,
            0,
            LANDLOCK_CREATE_RULESET_VERSION,
        )
    )
    if result < 0:
        _raise_errno("Landlock ABI query failed")
    return result


def _existing_path(value: str | Path) -> Path:
    source = Path(value)
    resolved = source.resolve(strict=True)
    if source.is_symlink():
        raise LandlockPolicyError(f"Landlock rule source may not be a symlink: {source}")
    return resolved


def _add_path_rule(
    ruleset_fd: int,
    path: Path,
    allowed_access: int,
) -> None:
    _create_ruleset, add_rule, _restrict_self = _syscall_numbers()
    descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        rule = _PathBeneathAttr(
            allowed_access=allowed_access,
            parent_fd=descriptor,
        )
        result = int(
            _libc().syscall(
                add_rule,
                ruleset_fd,
                LANDLOCK_RULE_PATH_BENEATH,
                ctypes.byref(rule),
                0,
            )
        )
        if result < 0:
            _raise_errno(f"Landlock path rule failed for {path}")
    finally:
        os.close(descriptor)


def _read_only_access(path: Path) -> int:
    if path.is_dir():
        return LANDLOCK_FS_READ_EXECUTE
    return LANDLOCK_FS_FILE_READ_EXECUTE


def _read_write_access(path: Path) -> int:
    if path.is_dir():
        return LANDLOCK_FS_READ_WRITE
    return LANDLOCK_FS_FILE_READ_WRITE


def apply_landlock_v4(
    *,
    read_only_paths: Iterable[str | Path],
    read_write_paths: Iterable[str | Path],
) -> dict[str, object]:
    """Deny all handled filesystem access except the exact declared paths.

    TCP connect and bind are denied by handling both Landlock v4 network rights
    without adding a network-port allow rule.
    """

    abi = query_landlock_abi()
    if abi < 4:
        raise LandlockPolicyError(f"Landlock ABI 4 is required, observed ABI {abi}")
    read_only = sorted({_existing_path(path) for path in read_only_paths})
    read_write = sorted({_existing_path(path) for path in read_write_paths})
    if not read_only:
        raise LandlockPolicyError("at least one Landlock read-only path is required")
    if not read_write:
        raise LandlockPolicyError("at least one Landlock read-write path is required")

    create_ruleset, _add_rule, restrict_self = _syscall_numbers()
    attributes = _RulesetAttr(
        handled_access_fs=LANDLOCK_FS_ALL_V4,
        handled_access_net=LANDLOCK_NET_ALL_V4,
    )
    ruleset_fd = int(
        _libc().syscall(
            create_ruleset,
            ctypes.byref(attributes),
            ctypes.sizeof(attributes),
            0,
        )
    )
    if ruleset_fd < 0:
        _raise_errno("Landlock ruleset creation failed")
    try:
        for path in read_only:
            _add_path_rule(ruleset_fd, path, _read_only_access(path))
        for path in read_write:
            _add_path_rule(ruleset_fd, path, _read_write_access(path))
        if int(_libc().prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) < 0:
            _raise_errno("PR_SET_NO_NEW_PRIVS failed")
        if int(_libc().syscall(restrict_self, ruleset_fd, 0)) < 0:
            _raise_errno("Landlock restrict_self failed")
    finally:
        os.close(ruleset_fd)
    return {
        "landlock_abi": abi,
        "filesystem_default_deny": True,
        "tcp_bind_denied": True,
        "tcp_connect_denied": True,
        "no_new_privileges": True,
        "read_only_paths": [str(path) for path in read_only],
        "read_write_paths": [str(path) for path in read_write],
    }


def apply_network_seccomp_deny() -> dict[str, object]:
    """Deny socket creation and network I/O while preserving predictor execve."""

    try:
        library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    except OSError as exc:
        raise LandlockPolicyError("libseccomp.so.2 is required") from exc
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    library.seccomp_rule_add.restype = ctypes.c_int
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_load.restype = ctypes.c_int
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_release.restype = None
    scmp_act_allow = 0x7FFF0000
    scmp_act_errno = 0x00050000 | errno.EPERM
    blocked = (
        "socket",
        "socketpair",
        "connect",
        "bind",
        "listen",
        "accept",
        "accept4",
        "sendto",
        "sendmsg",
        "sendmmsg",
        "recvfrom",
        "recvmsg",
        "recvmmsg",
        "ptrace",
        "process_vm_readv",
        "process_vm_writev",
        "pidfd_open",
        "pidfd_getfd",
        "open_by_handle_at",
        "name_to_handle_at",
        "io_uring_setup",
        "io_uring_enter",
        "io_uring_register",
    )
    context = library.seccomp_init(scmp_act_allow)
    if not context:
        raise LandlockPolicyError("seccomp_init failed")
    try:
        for name in blocked:
            number = int(library.seccomp_syscall_resolve_name(name.encode("ascii")))
            if number < 0:
                continue
            result = int(
                library.seccomp_rule_add(
                    context,
                    scmp_act_errno,
                    number,
                    0,
                )
            )
            if result != 0:
                raise LandlockPolicyError(
                    f"seccomp rule failed for {name}: {os.strerror(abs(result))}"
                )
        result = int(library.seccomp_load(context))
        if result != 0:
            raise LandlockPolicyError(
                f"seccomp_load failed: {os.strerror(abs(result))}"
            )
    finally:
        library.seccomp_release(context)
    return {
        "seccomp_default_action": "ALLOW",
        "blocked_network_syscalls": list(blocked),
        "socket_creation_denied": True,
    }


def close_inherited_fds() -> list[int]:
    """Close every inherited descriptor except stdin/stdout/stderr."""

    closed: list[int] = []
    fd_root = Path("/proc/self/fd")
    if fd_root.is_dir():
        candidates = []
        for name in os.listdir(fd_root):
            try:
                descriptor = int(name)
            except ValueError:
                continue
            if descriptor > 2:
                candidates.append(descriptor)
        for descriptor in sorted(set(candidates), reverse=True):
            try:
                os.close(descriptor)
            except OSError:
                continue
            closed.append(descriptor)
    return sorted(closed)


def is_landlock_denial(exc: OSError) -> bool:
    return int(exc.errno or 0) in {errno.EACCES, errno.EPERM}


__all__ = [
    "LandlockPolicyError",
    "apply_landlock_v4",
    "apply_network_seccomp_deny",
    "close_inherited_fds",
    "is_landlock_denial",
    "query_landlock_abi",
]

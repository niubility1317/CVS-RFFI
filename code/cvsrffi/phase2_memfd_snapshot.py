"""Linux sealed-memfd snapshots for formal Phase2 predictor inputs.

The controller copies every verified request, detached seal, package manifest,
and package member into a ``memfd`` and applies all write/resize/further-seal
locks before the predictor starts.  The predictor receives only inherited file
descriptors.  This closes the same-UID replace/restore TOCTOU gap left by
read-only paths and ordinary Landlock path rules.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Mapping

try:  # pragma: no cover - exercised by the real Linux/N607 smoke
    import fcntl as _fcntl
except ImportError:  # Windows local contract tests still import this module.
    _fcntl = None


PINNED_PACKAGE_ROOT_ENV = "CVS_PHASE2_PINNED_PACKAGE_ROOT"
PINNED_PACKAGE_FDS_ENV = "CVS_PHASE2_PINNED_PACKAGE_FDS"
PINNED_REQUEST_FD_ENV = "CVS_PHASE2_PINNED_REQUEST_FD"
PINNED_SEAL_FD_ENV = "CVS_PHASE2_PINNED_SEAL_FD"

F_ADD_SEALS = getattr(_fcntl, "F_ADD_SEALS", 1033)
F_GET_SEALS = getattr(_fcntl, "F_GET_SEALS", 1034)
F_SEAL_SEAL = getattr(_fcntl, "F_SEAL_SEAL", 0x0001)
F_SEAL_SHRINK = getattr(_fcntl, "F_SEAL_SHRINK", 0x0002)
F_SEAL_GROW = getattr(_fcntl, "F_SEAL_GROW", 0x0004)
F_SEAL_WRITE = getattr(_fcntl, "F_SEAL_WRITE", 0x0008)
REQUIRED_SEALS = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE


class Phase2MemfdSnapshotError(RuntimeError):
    """Raised when an immutable predictor snapshot cannot be proven."""


def _sha256_fd(fd: int) -> tuple[str, int]:
    duplicate = os.dup(fd)
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(duplicate, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        return digest.hexdigest(), size
    finally:
        os.close(duplicate)


def _read_regular_nofollow(path: Path) -> bytes:
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise Phase2MemfdSnapshotError(f"snapshot source is not a regular file: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise Phase2MemfdSnapshotError(f"snapshot source identity changed: {path}")
        chunks: list[bytes] = []
        remaining = int(opened.st_size)
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                raise Phase2MemfdSnapshotError(f"snapshot source was truncated: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _sealed_memfd(name: str, payload: bytes) -> int:
    if _fcntl is None or not hasattr(os, "memfd_create"):
        raise Phase2MemfdSnapshotError("os.memfd_create is unavailable")
    flags = int(getattr(os, "MFD_CLOEXEC", 0x0001)) | int(
        getattr(os, "MFD_ALLOW_SEALING", 0x0002)
    )
    fd = os.memfd_create(name, flags=flags)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise Phase2MemfdSnapshotError("short write while building sealed memfd")
            view = view[written:]
        os.lseek(fd, 0, os.SEEK_SET)
        _fcntl.fcntl(fd, F_ADD_SEALS, REQUIRED_SEALS)
        actual_seals = int(_fcntl.fcntl(fd, F_GET_SEALS))
        if actual_seals & REQUIRED_SEALS != REQUIRED_SEALS:
            raise Phase2MemfdSnapshotError("memfd immutable seal set is incomplete")
        os.set_inheritable(fd, True)
        return fd
    except Exception:
        os.close(fd)
        raise


def _logical_root(path: str | Path) -> str:
    return os.path.abspath(os.fspath(path))


def pinned_input_mode_active() -> bool:
    values = (
        os.environ.get(PINNED_PACKAGE_ROOT_ENV),
        os.environ.get(PINNED_PACKAGE_FDS_ENV),
        os.environ.get(PINNED_REQUEST_FD_ENV),
        os.environ.get(PINNED_SEAL_FD_ENV),
    )
    if any(value is not None for value in values) and not all(value is not None for value in values):
        raise Phase2MemfdSnapshotError("partial pinned-input environment is forbidden")
    return all(value is not None for value in values)


def pinned_package_root(path: str | Path) -> Path | None:
    if not pinned_input_mode_active():
        return None
    expected = str(os.environ[PINNED_PACKAGE_ROOT_ENV])
    actual = _logical_root(path)
    if actual != expected:
        raise Phase2MemfdSnapshotError("logical predictor package root drift")
    return Path(actual)


def _fd_map() -> dict[str, int]:
    if not pinned_input_mode_active():
        return {}
    try:
        raw = json.loads(os.environ[PINNED_PACKAGE_FDS_ENV])
    except json.JSONDecodeError as exc:
        raise Phase2MemfdSnapshotError("pinned package fd map is not JSON") from exc
    if not isinstance(raw, dict) or not raw:
        raise Phase2MemfdSnapshotError("pinned package fd map must be nonempty")
    result: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key or not isinstance(value, int) or value < 0:
            raise Phase2MemfdSnapshotError("pinned package fd map schema drift")
        metadata = os.fstat(value)
        if not stat.S_ISREG(metadata.st_mode):
            raise Phase2MemfdSnapshotError(f"pinned package fd is not regular: {key}")
        if _fcntl is None:
            raise Phase2MemfdSnapshotError("pinned package fds require POSIX fcntl")
        seals = int(_fcntl.fcntl(value, F_GET_SEALS))
        if seals & REQUIRED_SEALS != REQUIRED_SEALS:
            raise Phase2MemfdSnapshotError(f"pinned package fd is not immutable: {key}")
        result[key] = value
    return result


@contextmanager
def open_pinned_package_member(relative_path: str) -> Iterator[BinaryIO]:
    mapping = _fd_map()
    if relative_path not in mapping:
        raise Phase2MemfdSnapshotError(f"package member is absent from pinned fd map: {relative_path}")
    duplicate = os.dup(mapping[relative_path])
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        with os.fdopen(duplicate, "rb", closefd=False) as handle:
            yield handle
    finally:
        os.close(duplicate)


@contextmanager
def open_pinned_special(kind: str) -> Iterator[BinaryIO]:
    field = {
        "request": PINNED_REQUEST_FD_ENV,
        "seal": PINNED_SEAL_FD_ENV,
    }.get(kind)
    if field is None or not pinned_input_mode_active():
        raise Phase2MemfdSnapshotError(f"pinned special input is unavailable: {kind}")
    fd = int(os.environ[field])
    metadata = os.fstat(fd)
    if _fcntl is None:
        raise Phase2MemfdSnapshotError("pinned special fds require POSIX fcntl")
    seals = int(_fcntl.fcntl(fd, F_GET_SEALS))
    if not stat.S_ISREG(metadata.st_mode) or seals & REQUIRED_SEALS != REQUIRED_SEALS:
        raise Phase2MemfdSnapshotError(f"pinned special input is not immutable: {kind}")
    duplicate = os.dup(fd)
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        with os.fdopen(duplicate, "rb", closefd=False) as handle:
            yield handle
    finally:
        os.close(duplicate)


@dataclass
class SealedMemfdSnapshot:
    package_root: str
    member_fds: dict[str, int]
    request_fd: int
    seal_fd: int
    receipt: dict[str, object]

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return tuple(sorted({*self.member_fds.values(), self.request_fd, self.seal_fd}))

    @property
    def environment(self) -> dict[str, str]:
        return {
            PINNED_PACKAGE_ROOT_ENV: self.package_root,
            PINNED_PACKAGE_FDS_ENV: json.dumps(
                self.member_fds, sort_keys=True, separators=(",", ":")
            ),
            PINNED_REQUEST_FD_ENV: str(self.request_fd),
            PINNED_SEAL_FD_ENV: str(self.seal_fd),
        }

    def close(self) -> None:
        for fd in self.pass_fds:
            try:
                os.close(fd)
            except OSError:
                pass


def build_sealed_memfd_snapshot(
    *,
    package_root: str | Path,
    detached_seal: str | Path,
    request_json: str | Path,
    manifest: Mapping[str, object],
) -> SealedMemfdSnapshot:
    """Copy verified formal inputs into immutable inherited memfds."""

    if os.name != "posix":
        raise Phase2MemfdSnapshotError("sealed memfd snapshots require Linux/POSIX")
    root = Path(package_root).resolve(strict=True)
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise Phase2MemfdSnapshotError("verified manifest has no package members")
    descriptors = [
        {
            "relative_path": "package_manifest.json",
            "sha256": hashlib.sha256(_read_regular_nofollow(root / "package_manifest.json")).hexdigest(),
        },
        *members,
    ]
    fds: dict[str, int] = {}
    receipt_members: list[dict[str, object]] = []
    request_fd = -1
    seal_fd = -1
    try:
        for raw in descriptors:
            if not isinstance(raw, Mapping):
                raise Phase2MemfdSnapshotError("verified package descriptor schema drift")
            relative = str(raw["relative_path"])
            source = root.joinpath(*relative.split("/"))
            payload = _read_regular_nofollow(source)
            digest = hashlib.sha256(payload).hexdigest()
            expected = str(raw.get("sha256", digest))
            if digest != expected:
                raise Phase2MemfdSnapshotError(f"source changed before memfd snapshot: {relative}")
            fd = _sealed_memfd(f"cvs-phase2-{Path(relative).name}", payload)
            fds[relative] = fd
            receipt_members.append(
                {
                    "relative_path": relative,
                    "sha256": digest,
                    "size_bytes": len(payload),
                    "memfd_seals": REQUIRED_SEALS,
                }
            )
        request_payload = _read_regular_nofollow(Path(request_json))
        seal_payload = _read_regular_nofollow(Path(detached_seal))
        request_fd = _sealed_memfd("cvs-phase2-request", request_payload)
        seal_fd = _sealed_memfd("cvs-phase2-detached-seal", seal_payload)
        receipt = {
            "schema": "cvs.phase2.sealed_memfd_snapshot.v1",
            "status": "PASS",
            "immutability_mechanism": "memfd_F_SEAL_WRITE_GROW_SHRINK_SEAL",
            "same_uid_path_replace_restore_reachable": False,
            "package_root_logical": _logical_root(root),
            "package_members": receipt_members,
            "request_sha256": hashlib.sha256(request_payload).hexdigest(),
            "detached_seal_sha256": hashlib.sha256(seal_payload).hexdigest(),
        }
        return SealedMemfdSnapshot(
            package_root=_logical_root(root),
            member_fds=fds,
            request_fd=request_fd,
            seal_fd=seal_fd,
            receipt=receipt,
        )
    except Exception:
        for fd in [*fds.values(), request_fd, seal_fd]:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        raise


__all__ = [
    "PINNED_PACKAGE_FDS_ENV",
    "PINNED_PACKAGE_ROOT_ENV",
    "PINNED_REQUEST_FD_ENV",
    "PINNED_SEAL_FD_ENV",
    "Phase2MemfdSnapshotError",
    "REQUIRED_SEALS",
    "SealedMemfdSnapshot",
    "build_sealed_memfd_snapshot",
    "open_pinned_package_member",
    "open_pinned_special",
    "pinned_input_mode_active",
    "pinned_package_root",
]

"""Strict truth-free Stage2 predictor package and same-fd loader.

This module is safe to import inside the Phase2 predictor.  It intentionally
does not import dataset builders, cache loaders, training code, scorers, or
legacy runners.  The offline sealer may import these helpers, but the strict
runtime only receives a sealed package root and a detached seal digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping

import numpy as np

from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT


FORMAL_LEO_WEAK_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
PHASE2_SAMPLE_VIEW_POLICY = "leo_weak_only_no_clean_access"

PREDICTOR_PACKAGE_MANIFEST_SCHEMA = "cvs.phase2.predictor_package_manifest.v2"
PREDICTOR_PACKAGE_SEAL_SCHEMA = "cvs.phase2.predictor_package_seal.v2"
PREDICTOR_INPUT_STAGE = "phase1_offline_truth_free_predictor_package"
QUERY_SCHEMA = "cvs.phase2.unlabeled_query_iq.v2"
SUPPORT_SCHEMA = "cvs.phase2.registered_support_pool.v2"

QUERY_NPZ_MEMBERS = (
    "query_leo_weak_iq",
    "query_tokens",
    "query_overlay_tokens",
    "query_satellite_seeds",
    "query_post_channel_iq_sha256",
    "manifest_json",
)
SUPPORT_NPZ_MEMBERS = (
    "support_pool_leo_weak_iq",
    "support_pool_class_indices",
    "support_pool_rank_within_class",
    "support_pool_tokens",
    "support_pool_overlay_tokens",
    "support_pool_satellite_seeds",
    "support_pool_post_channel_iq_sha256",
    "manifest_json",
)

MANIFEST_REQUIRED_KEYS = {
    "schema",
    "artifact_stage",
    "stage",
    "receiver",
    "seed",
    "new_class_count",
    "support_pool_max_k",
    "target_channel_view",
    "target_channel_scenarios",
    "registered_class_count",
    "registered_classes",
    "candidate_lock_sha256",
    "members",
    "package_root_sha256",
    *PHASE2_FULL_CONTRACT.keys(),
}
SEAL_REQUIRED_KEYS = {
    "schema",
    "manifest_relative_path",
    "manifest_sha256",
    "manifest_size_bytes",
    "package_root_sha256",
    "artifact_member_allowlist_sha256",
}
MEMBER_DESCRIPTOR_REQUIRED_KEYS = {
    "relative_path",
    "sha256",
    "size_bytes",
    "artifact_role",
    "schema",
    "scenario",
    "npz_members",
}
OPAQUE_TOKEN_RE = re.compile(r"(?:cls|qid|sid|oid)_[0-9a-f]{32,64}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

ALLOWED_NPZ_COMPRESSION_TYPES = frozenset(
    {
        zipfile.ZIP_STORED,
        zipfile.ZIP_DEFLATED,
    }
)
MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_NPZ_COMPRESSION_RATIO = 100.0


class PredictorPackageError(ValueError):
    """Raised before untrusted IQ arrays are materialized."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iq_row_sha256(row: np.ndarray) -> str:
    value = np.ascontiguousarray(row, dtype=np.float32)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value.lower()) is not None


def validate_relative_member_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise PredictorPackageError("package member path must be nonempty POSIX-relative")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PredictorPackageError(f"unsafe package member path: {value}")
    return path.as_posix()


def _ensure_root(root: Path) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise PredictorPackageError(f"package root must be a regular directory: {root}")
    return root.resolve()


def _is_reparse_point(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_flag)


def _validate_package_root_exact_allowlist(
    root: Path,
    *,
    allowed_files: set[str],
) -> None:
    root = _ensure_root(root)
    expected_files = {validate_relative_member_path(value) for value in allowed_files}
    expected_directories: set[str] = set()
    for value in expected_files:
        parent = PurePosixPath(value).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    actual_files: set[str] = set()
    actual_directories: set[str] = set()

    def visit(directory: Path, prefix: PurePosixPath) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = (prefix / entry.name).as_posix()
                metadata = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or _is_reparse_point(metadata):
                    raise PredictorPackageError(
                        f"symlink or reparse-point package entry rejected: {relative}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    if relative not in expected_directories:
                        raise PredictorPackageError(
                            f"unexpected package directory: {relative}"
                        )
                    actual_directories.add(relative)
                    visit(Path(entry.path), prefix / entry.name)
                elif entry.is_file(follow_symlinks=False):
                    if relative not in expected_files:
                        raise PredictorPackageError(f"unexpected package file: {relative}")
                    actual_files.add(relative)
                else:
                    raise PredictorPackageError(
                        f"non-regular package root entry rejected: {relative}"
                    )

    visit(root, PurePosixPath())
    if actual_files != expected_files or actual_directories != expected_directories:
        raise PredictorPackageError(
            "package root exact allowlist mismatch: "
            f"missing_files={sorted(expected_files-actual_files)}, "
            f"missing_directories={sorted(expected_directories-actual_directories)}"
        )


@contextmanager
def open_regular_member_same_fd(root: Path, relative_path: str) -> Iterator[BinaryIO]:
    """Open a package member without following symlinks and keep one fd.

    N607/Linux additionally receives ``O_NOFOLLOW``.  On Windows every path
    component is lstat-checked and the pre-open identity is compared with the
    opened file identity.  Hashing, ZIP-member inspection, and NumPy loading all
    use the yielded descriptor.
    """

    root = _ensure_root(root)
    relative = validate_relative_member_path(relative_path)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise PredictorPackageError(f"package member escapes root: {relative}") from exc
    current = root
    for component in PurePosixPath(relative).parts:
        current = current / component
        try:
            entry = current.lstat()
        except FileNotFoundError as exc:
            raise PredictorPackageError(f"missing package member: {relative}") from exc
        if stat.S_ISLNK(entry.st_mode):
            raise PredictorPackageError(f"symlink package member rejected: {relative}")
    # ``Path.stat(follow_symlinks=...)`` is unavailable on the verified
    # Python 3.8 N607 runtime. ``os.stat`` has the same no-follow contract and
    # preserves the subsequent device/inode/size identity comparison.
    before = os.stat(candidate, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise PredictorPackageError(f"non-regular package member rejected: {relative}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise PredictorPackageError(f"opened package member is not regular: {relative}")
        if (before.st_dev, before.st_ino, before.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise PredictorPackageError(f"package member identity changed before open: {relative}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            yield handle
    finally:
        os.close(descriptor)


def _hash_handle(handle: BinaryIO) -> tuple[str, int]:
    handle.seek(0)
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    handle.seek(0)
    return digest.hexdigest(), size


def _json_from_handle(handle: BinaryIO, *, context: str) -> dict[str, Any]:
    handle.seek(0)
    raw = handle.read()
    handle.seek(0)
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictorPackageError(f"invalid JSON for {context}") from exc
    if not isinstance(payload, dict):
        raise PredictorPackageError(f"JSON root must be an object for {context}")
    return payload


def _zip_members_from_handle(handle: BinaryIO, *, context: str) -> tuple[str, ...]:
    handle.seek(0)
    try:
        with zipfile.ZipFile(handle, mode="r") as archive:
            infos = archive.infolist()
            raw_members = [info.filename for info in infos]
            if len(raw_members) != len(set(raw_members)):
                raise PredictorPackageError(f"duplicate NPZ ZIP member for {context}")
            total_uncompressed = 0
            members: list[str] = []
            for info in infos:
                raw = info.filename
                path = PurePosixPath(raw)
                if (
                    info.is_dir()
                    or path.is_absolute()
                    or any(part in {"", ".", ".."} for part in path.parts)
                ):
                    raise PredictorPackageError(
                        f"unsafe NPZ ZIP member for {context}: {raw}"
                    )
                if len(path.parts) != 1 or not raw.endswith(".npy"):
                    raise PredictorPackageError(
                        f"unexpected NPZ ZIP member for {context}: {raw}"
                    )
                if info.flag_bits & 0x1:
                    raise PredictorPackageError(
                        f"encrypted NPZ ZIP member rejected for {context}: {raw}"
                    )
                if info.compress_type not in ALLOWED_NPZ_COMPRESSION_TYPES:
                    raise PredictorPackageError(
                        f"unsupported NPZ ZIP compression for {context}: {raw}"
                    )
                if info.file_size > MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES:
                    raise PredictorPackageError(
                        f"NPZ ZIP member uncompressed size exceeds limit for {context}: {raw}"
                    )
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES:
                    raise PredictorPackageError(
                        f"NPZ ZIP total uncompressed size exceeds limit for {context}"
                    )
                if info.file_size and info.compress_size <= 0:
                    raise PredictorPackageError(
                        f"invalid NPZ ZIP compressed size for {context}: {raw}"
                    )
                ratio = info.file_size / max(info.compress_size, 1)
                if ratio > MAX_NPZ_COMPRESSION_RATIO:
                    raise PredictorPackageError(
                        f"NPZ ZIP compression ratio exceeds limit for {context}: {raw}"
                    )
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if unix_mode and stat.S_ISLNK(unix_mode):
                    raise PredictorPackageError(
                        f"symlink NPZ ZIP member rejected for {context}: {raw}"
                    )
                try:
                    with archive.open(info, mode="r") as member:
                        actual_size = 0
                        for chunk in iter(lambda: member.read(1024 * 1024), b""):
                            actual_size += len(chunk)
                            if actual_size > info.file_size:
                                raise PredictorPackageError(
                                    f"NPZ ZIP member expands beyond declared size for "
                                    f"{context}: {raw}"
                                )
                except (RuntimeError, zipfile.BadZipFile, NotImplementedError) as exc:
                    raise PredictorPackageError(
                        f"NPZ ZIP member CRC/decompression validation failed for "
                        f"{context}: {raw}"
                    ) from exc
                if actual_size != info.file_size:
                    raise PredictorPackageError(
                        f"NPZ ZIP member size validation failed for {context}: {raw}"
                    )
                members.append(raw[:-4])
    except PredictorPackageError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise PredictorPackageError(f"invalid NPZ for {context}") from exc
    finally:
        handle.seek(0)
    return tuple(members)


def make_member_descriptor(
    path: str | Path,
    *,
    relative_path: str,
    artifact_role: str,
    schema: str,
    scenario: str | None = None,
    npz_members: tuple[str, ...] = (),
) -> dict[str, Any]:
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise PredictorPackageError(f"member source must be a regular file: {value}")
    return {
        "relative_path": validate_relative_member_path(relative_path),
        "sha256": sha256_file(value),
        "size_bytes": value.stat().st_size,
        "artifact_role": str(artifact_role),
        "schema": str(schema),
        "scenario": scenario,
        "npz_members": list(npz_members),
    }


def package_root_sha256(members: list[Mapping[str, Any]]) -> str:
    ordered = sorted((dict(value) for value in members), key=lambda item: item["relative_path"])
    return sha256_bytes(canonical_json_bytes(ordered))


def _validate_registered_classes(value: Any, expected_count: int) -> None:
    if not isinstance(value, list) or len(value) != expected_count:
        raise PredictorPackageError("registered class registry size mismatch")
    handles: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"class_index", "class_handle"}:
            raise PredictorPackageError("registered class registry schema mismatch")
        if item["class_index"] != index:
            raise PredictorPackageError("registered class indices must be contiguous and ordered")
        handle = item["class_handle"]
        if not isinstance(handle, str) or OPAQUE_TOKEN_RE.fullmatch(handle) is None:
            raise PredictorPackageError("registered class handle is not opaque")
        if handle in handles:
            raise PredictorPackageError("duplicate registered class handle")
        handles.add(handle)


def _validate_member_descriptors(members: Any) -> list[dict[str, Any]]:
    if not isinstance(members, list) or not members:
        raise PredictorPackageError("package members must be a nonempty list")
    checked: list[dict[str, Any]] = []
    paths: set[str] = set()
    roles: set[str] = set()
    for raw in members:
        if not isinstance(raw, dict) or set(raw) != MEMBER_DESCRIPTOR_REQUIRED_KEYS:
            raise PredictorPackageError("package member descriptor schema mismatch")
        item = dict(raw)
        item["relative_path"] = validate_relative_member_path(item["relative_path"])
        if item["relative_path"] in paths:
            raise PredictorPackageError("duplicate package member path")
        if item["artifact_role"] in roles:
            raise PredictorPackageError("duplicate package artifact role")
        paths.add(item["relative_path"])
        roles.add(item["artifact_role"])
        if not _is_sha256(item["sha256"]):
            raise PredictorPackageError("invalid package member SHA256")
        if not isinstance(item["size_bytes"], int) or item["size_bytes"] < 0:
            raise PredictorPackageError("invalid package member size")
        if not isinstance(item["schema"], str) or not item["schema"]:
            raise PredictorPackageError("invalid package member schema")
        if item["scenario"] is not None and item["scenario"] not in FORMAL_LEO_WEAK_SCENARIOS:
            raise PredictorPackageError("invalid package member scenario")
        if not isinstance(item["npz_members"], list):
            raise PredictorPackageError("invalid NPZ member allowlist")
        checked.append(item)
    required_roles = {"checkpoint", "adapter", "head", "tta_policy"}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        required_roles.update({f"support:{scenario}", f"query:{scenario}"})
    if roles != required_roles:
        raise PredictorPackageError(
            f"package artifact role set mismatch: missing={sorted(required_roles-roles)}, "
            f"unexpected={sorted(roles-required_roles)}"
        )
    return checked


def write_predictor_package_manifest_and_seal(
    package_root: str | Path,
    *,
    manifest_metadata: Mapping[str, Any],
    members: list[Mapping[str, Any]],
    detached_seal_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    """Write a canonical manifest and a detached, non-overwriting seal."""

    root = _ensure_root(Path(package_root))
    seal_path = Path(detached_seal_path).resolve()
    try:
        seal_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise PredictorPackageError("detached package seal must be outside predictor root")
    checked_members = _validate_member_descriptors([dict(value) for value in members])
    member_paths = {item["relative_path"] for item in checked_members}
    _validate_package_root_exact_allowlist(root, allowed_files=member_paths)
    for item in checked_members:
        with open_regular_member_same_fd(root, item["relative_path"]) as handle:
            digest, size = _hash_handle(handle)
        if digest != item["sha256"] or size != item["size_bytes"]:
            raise PredictorPackageError(f"member descriptor drift: {item['relative_path']}")
    root_digest = package_root_sha256(checked_members)
    payload = {**dict(manifest_metadata), "members": checked_members, "package_root_sha256": root_digest}
    if set(payload) != MANIFEST_REQUIRED_KEYS:
        raise PredictorPackageError(
            f"package manifest schema mismatch: missing={sorted(MANIFEST_REQUIRED_KEYS-set(payload))}, "
            f"unexpected={sorted(set(payload)-MANIFEST_REQUIRED_KEYS)}"
        )
    _validate_manifest(payload)
    by_role = {item["artifact_role"]: item for item in checked_members}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        support, support_manifest = _materialize_npz(
            root, by_role[f"support:{scenario}"]
        )
        _validate_support_arrays(
            support,
            support_manifest,
            scenario=scenario,
            class_count=payload["registered_class_count"],
            max_k=payload["support_pool_max_k"],
        )
    manifest_path = root / "package_manifest.json"
    if manifest_path.exists() or seal_path.exists():
        raise FileExistsError("refusing to overwrite package manifest or detached seal")
    manifest_bytes = canonical_json_bytes(payload) + b"\n"
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    _validate_package_root_exact_allowlist(
        root,
        allowed_files=member_paths | {manifest_path.name},
    )
    seal = {
        "schema": PREDICTOR_PACKAGE_SEAL_SCHEMA,
        "manifest_relative_path": manifest_path.name,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "manifest_size_bytes": len(manifest_bytes),
        "package_root_sha256": root_digest,
        "artifact_member_allowlist_sha256": root_digest,
    }
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    with seal_path.open("xb") as handle:
        handle.write(canonical_json_bytes(seal) + b"\n")
    return manifest_path, seal_path, payload, seal


def _validate_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if set(payload) != MANIFEST_REQUIRED_KEYS:
        raise PredictorPackageError("predictor package manifest exact schema mismatch")
    expected = {
        "schema": PREDICTOR_PACKAGE_MANIFEST_SCHEMA,
        "artifact_stage": PREDICTOR_INPUT_STAGE,
        "target_channel_view": "leo_weak_only",
        **PHASE2_FULL_CONTRACT,
    }
    failed = [key for key, value in expected.items() if payload.get(key) != value]
    if failed:
        raise PredictorPackageError(f"predictor package contract failed: {failed}")
    if payload.get("stage") not in {"stage2b", "stage2c"}:
        raise PredictorPackageError("predictor package stage drift")
    if payload.get("target_channel_scenarios") != list(FORMAL_LEO_WEAK_SCENARIOS):
        raise PredictorPackageError("predictor package scenario tuple drift")
    if not isinstance(payload.get("seed"), int):
        raise PredictorPackageError("predictor package seed must be an integer")
    if not isinstance(payload.get("support_pool_max_k"), int) or payload["support_pool_max_k"] < 1:
        raise PredictorPackageError("predictor package max K must be positive")
    if not isinstance(payload.get("registered_class_count"), int) or payload["registered_class_count"] < 1:
        raise PredictorPackageError("predictor package registered class count invalid")
    if not isinstance(payload.get("new_class_count"), int) or payload["new_class_count"] < 0:
        raise PredictorPackageError("predictor package new class count invalid")
    if not _is_sha256(payload.get("candidate_lock_sha256")):
        raise PredictorPackageError("predictor package candidate lock digest invalid")
    _validate_registered_classes(payload["registered_classes"], payload["registered_class_count"])
    members = _validate_member_descriptors(payload["members"])
    if payload["package_root_sha256"] != package_root_sha256(members):
        raise PredictorPackageError("predictor package root digest mismatch")
    return members


def preflight_stage2_predictor_package(
    package_root: str | Path,
    *,
    detached_seal_path: str | Path,
    expected_seal_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify seal, manifest, paths, hashes, and NPZ allowlists before IQ load."""

    root = _ensure_root(Path(package_root))
    seal_path = Path(detached_seal_path)
    if seal_path.is_symlink() or not seal_path.is_file():
        raise PredictorPackageError("detached seal must be a regular non-symlink file")
    if not _is_sha256(expected_seal_sha256) or sha256_file(seal_path) != expected_seal_sha256:
        raise PredictorPackageError("detached seal digest mismatch")
    seal = json.loads(seal_path.read_text(encoding="utf-8-sig"))
    if not isinstance(seal, dict) or set(seal) != SEAL_REQUIRED_KEYS:
        raise PredictorPackageError("detached seal exact schema mismatch")
    if seal.get("schema") != PREDICTOR_PACKAGE_SEAL_SCHEMA:
        raise PredictorPackageError("detached seal schema drift")
    manifest_relative = validate_relative_member_path(seal["manifest_relative_path"])
    if manifest_relative != "package_manifest.json":
        raise PredictorPackageError("package manifest path drift")
    with open_regular_member_same_fd(root, manifest_relative) as handle:
        manifest_digest, manifest_size = _hash_handle(handle)
        if manifest_digest != seal["manifest_sha256"] or manifest_size != seal["manifest_size_bytes"]:
            raise PredictorPackageError("package manifest detached digest mismatch")
        manifest = _json_from_handle(handle, context="package manifest")
    members = _validate_manifest(manifest)
    if manifest["package_root_sha256"] != seal["package_root_sha256"]:
        raise PredictorPackageError("manifest/seal package root mismatch")
    if seal["artifact_member_allowlist_sha256"] != seal["package_root_sha256"]:
        raise PredictorPackageError("artifact member allowlist digest mismatch")
    _validate_package_root_exact_allowlist(
        root,
        allowed_files={item["relative_path"] for item in members} | {manifest_relative},
    )

    opened: list[dict[str, Any]] = []
    for item in members:
        with open_regular_member_same_fd(root, item["relative_path"]) as handle:
            digest, size = _hash_handle(handle)
            if digest != item["sha256"] or size != item["size_bytes"]:
                raise PredictorPackageError(f"package member digest mismatch: {item['relative_path']}")
            if item["npz_members"]:
                actual_npz = _zip_members_from_handle(handle, context=item["relative_path"])
                if actual_npz != tuple(item["npz_members"]):
                    raise PredictorPackageError(
                        f"NPZ member allowlist mismatch: {item['relative_path']}"
                    )
            opened.append(
                {
                    "relative_path": item["relative_path"],
                    "sha256": digest,
                    "size_bytes": size,
                    "member_allowlist_status": "PASS",
                }
            )
    audit = {
        "schema": "cvs.phase2.preopen_audit_receipt.v2",
        "status": "PASS",
        "package_root_sha256": seal["package_root_sha256"],
        "artifact_member_allowlist_sha256": seal["artifact_member_allowlist_sha256"],
        "manifest_sha256": manifest_digest,
        "opened_members": opened,
        "iq_payload_materialized": False,
        "path_symlink_regular_file_audit": "PASS",
        "hash_and_member_audit_same_file_descriptor": True,
    }
    return manifest, seal, audit


def _parse_embedded_manifest(raw: np.ndarray, *, context: str) -> dict[str, Any]:
    value = np.asarray(raw)
    if value.size != 1:
        raise PredictorPackageError(f"embedded manifest must be scalar: {context}")
    scalar = value.reshape(-1)[0]
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    try:
        payload = json.loads(str(scalar))
    except json.JSONDecodeError as exc:
        raise PredictorPackageError(f"embedded manifest JSON invalid: {context}") from exc
    if not isinstance(payload, dict):
        raise PredictorPackageError(f"embedded manifest must be an object: {context}")
    return payload


def _materialize_npz(
    root: Path,
    descriptor: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with open_regular_member_same_fd(root, descriptor["relative_path"]) as handle:
        digest, size = _hash_handle(handle)
        if digest != descriptor["sha256"] or size != descriptor["size_bytes"]:
            raise PredictorPackageError("artifact changed after pre-open audit")
        actual_members = _zip_members_from_handle(handle, context=descriptor["relative_path"])
        if actual_members != tuple(descriptor["npz_members"]):
            raise PredictorPackageError("artifact member set changed after pre-open audit")
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in actual_members}
    manifest = _parse_embedded_manifest(arrays.pop("manifest_json"), context=descriptor["relative_path"])
    return arrays, manifest


def _validate_tokens(values: np.ndarray, *, prefix: str, context: str) -> list[str]:
    tokens = np.asarray(values).astype(str).tolist()
    if not tokens or len(tokens) != len(set(tokens)):
        raise PredictorPackageError(f"tokens must be nonempty and unique: {context}")
    if any(not token.startswith(prefix) or OPAQUE_TOKEN_RE.fullmatch(token) is None for token in tokens):
        raise PredictorPackageError(f"non-opaque token in {context}")
    return tokens


def _validate_iq_and_hashes(iq: np.ndarray, hashes: np.ndarray, *, context: str) -> None:
    values = np.asarray(iq)
    if values.dtype != np.float32 or values.ndim != 3 or values.shape[1] != 2:
        raise PredictorPackageError(f"LEO IQ shape/dtype drift: {context}")
    declared = np.asarray(hashes).astype(str).tolist()
    if len(declared) != values.shape[0] or any(not _is_sha256(value) for value in declared):
        raise PredictorPackageError(f"post-channel IQ digest array drift: {context}")
    actual = [iq_row_sha256(values[index]) for index in range(values.shape[0])]
    if actual != declared:
        raise PredictorPackageError(f"post-channel IQ digest mismatch: {context}")


def _validate_query_arrays(
    arrays: dict[str, np.ndarray], manifest: dict[str, Any], *, scenario: str
) -> None:
    expected_manifest = {
        "schema": QUERY_SCHEMA,
        "scenario": scenario,
        "query_truth_included": False,
        "query_role_included": False,
        "query_true_batch_class_count_included": False,
        "query_class_quota_included": False,
        "query_ordering_hint_included": False,
        "token_scheme": "hmac_sha256_opaque_v1",
    }
    if manifest != expected_manifest:
        raise PredictorPackageError("unlabeled query embedded manifest drift")
    tokens = _validate_tokens(arrays["query_tokens"], prefix="qid_", context="query")
    overlay = _validate_tokens(
        arrays["query_overlay_tokens"], prefix="oid_", context="query overlay"
    )
    if len(tokens) != len(overlay):
        raise PredictorPackageError("query token/overlay count drift")
    if np.asarray(arrays["query_satellite_seeds"]).shape != (len(tokens),):
        raise PredictorPackageError("query satellite seed shape drift")
    _validate_iq_and_hashes(
        arrays["query_leo_weak_iq"],
        arrays["query_post_channel_iq_sha256"],
        context="query",
    )


def _validate_support_arrays(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    *,
    scenario: str,
    class_count: int,
    max_k: int,
) -> None:
    expected_manifest = {
        "schema": SUPPORT_SCHEMA,
        "scenario": scenario,
        "registered_support_labels_allowed": True,
        "registered_class_count": class_count,
        "support_pool_max_k": max_k,
        "token_scheme": "hmac_sha256_opaque_v1",
    }
    if manifest != expected_manifest:
        raise PredictorPackageError("registered support embedded manifest drift")
    tokens = _validate_tokens(arrays["support_pool_tokens"], prefix="sid_", context="support")
    overlay = _validate_tokens(
        arrays["support_pool_overlay_tokens"], prefix="oid_", context="support overlay"
    )
    labels = np.asarray(arrays["support_pool_class_indices"])
    ranks = np.asarray(arrays["support_pool_rank_within_class"])
    if labels.dtype != np.int64 or ranks.dtype != np.int64 or labels.shape != ranks.shape:
        raise PredictorPackageError("support class/rank dtype or shape drift")
    if labels.shape != (len(tokens),) or len(tokens) != len(overlay):
        raise PredictorPackageError("support token/class count drift")
    expected_support_count = class_count * max_k
    if len(tokens) != expected_support_count:
        raise PredictorPackageError(
            "support package does not expose exactly K samples per class"
        )
    if np.asarray(arrays["support_pool_satellite_seeds"]).shape != (len(tokens),):
        raise PredictorPackageError("support satellite seed shape drift")
    expected_pairs = [
        (class_index, rank)
        for class_index in range(class_count)
        for rank in range(max_k)
    ]
    if list(zip(labels.tolist(), ranks.tolist())) != expected_pairs:
        raise PredictorPackageError(
            "support package is not an exact ordered K-shot prefix"
        )
    _validate_iq_and_hashes(
        arrays["support_pool_leo_weak_iq"],
        arrays["support_pool_post_channel_iq_sha256"],
        context="support",
    )


def load_verified_stage2_predictor_bundle(
    package_root: str | Path,
    *,
    detached_seal_path: str | Path,
    expected_seal_sha256: str,
    scenario: str | None = None,
):
    """Preflight the full package, then materialize one or all LEO scenarios."""

    manifest, seal, audit = preflight_stage2_predictor_package(
        package_root,
        detached_seal_path=detached_seal_path,
        expected_seal_sha256=expected_seal_sha256,
    )
    scenarios = FORMAL_LEO_WEAK_SCENARIOS if scenario is None else (scenario,)
    if any(value not in FORMAL_LEO_WEAK_SCENARIOS for value in scenarios):
        raise PredictorPackageError("requested scenario is not formal LEO_weak")
    by_role = {item["artifact_role"]: item for item in manifest["members"]}
    support_by_scenario: dict[str, dict[str, np.ndarray]] = {}
    query_by_scenario: dict[str, dict[str, np.ndarray]] = {}
    observed_tokens: set[str] = set()
    reference_support_count: int | None = None
    reference_query_count: int | None = None
    root = Path(package_root)
    for value in scenarios:
        support, support_manifest = _materialize_npz(root, by_role[f"support:{value}"])
        query, query_manifest = _materialize_npz(root, by_role[f"query:{value}"])
        _validate_support_arrays(
            support,
            support_manifest,
            scenario=value,
            class_count=manifest["registered_class_count"],
            max_k=manifest["support_pool_max_k"],
        )
        _validate_query_arrays(query, query_manifest, scenario=value)
        support_tokens = np.asarray(support["support_pool_tokens"]).astype(str).tolist()
        query_tokens = np.asarray(query["query_tokens"]).astype(str).tolist()
        if set(support_tokens) & set(query_tokens):
            raise PredictorPackageError("support/query opaque token overlap")
        scenario_tokens = set(support_tokens) | set(query_tokens)
        if observed_tokens & scenario_tokens:
            raise PredictorPackageError(
                "physical sample token reuse across LEO_weak scenarios"
            )
        observed_tokens.update(scenario_tokens)
        if reference_support_count is None:
            reference_support_count = len(support_tokens)
            reference_query_count = len(query_tokens)
        elif (
            len(support_tokens) != reference_support_count
            or len(query_tokens) != reference_query_count
        ):
            raise PredictorPackageError(
                "support/query count drifts across LEO_weak scenarios"
            )
        support_by_scenario[value] = support
        query_by_scenario[value] = query
    audit = {
        **audit,
        "iq_payload_materialized": True,
        "materialized_scenarios": list(scenarios),
        "support_pool_count": int(reference_support_count or 0),
        "query_count": int(reference_query_count or 0),
        "sample_level_post_channel_iq_sha256_status": "PASS",
        "cross_scenario_physical_sample_token_disjointness": (
            "PASS"
            if tuple(scenarios) == tuple(FORMAL_LEO_WEAK_SCENARIOS)
            else "NOT_CHECKED_SINGLE_SCENARIO"
        ),
    }
    return support_by_scenario, query_by_scenario, manifest, {**audit, "seal": seal}

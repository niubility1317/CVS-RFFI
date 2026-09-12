"""Sealed, tamper-evident Stage2 prediction artifacts.

The artifact is a single ZIP container with exactly three members::

    payload.npz -> manifest.json -> seal.json

The manifest binds the exact NPZ member allowlist and payload SHA256.  The
seal binds both the payload and canonical manifest SHA256.  Publication uses
an exclusive temporary file followed by a no-replace atomic publish, so a
previous prediction can never be silently overwritten.
"""

from __future__ import annotations

import ctypes
import datetime as _datetime
import errno
import hashlib
import io
import json
import os
import re
import secrets
import stat
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT

ARTIFACT_SCHEMA = "cvs.stage2.prediction_artifact.v1"
MANIFEST_SCHEMA = "cvs.stage2.prediction_manifest.v1"
SEAL_SCHEMA = "cvs.stage2.prediction_seal.v1"

OUTER_MEMBER_ALLOWLIST = ("payload.npz", "manifest.json", "seal.json")
NPZ_FIELD_ALLOWLIST = (
    "query_tokens",
    "scenarios",
    "candidate_after",
    "candidate_before",
    "identity_after",
    "identity_before",
    "direct",
    "shared_view_counts",
)
NPZ_MEMBER_ALLOWLIST = tuple(f"{field}.npy" for field in NPZ_FIELD_ALLOWLIST)

ALLOWED_STAGES = frozenset({"Stage2-A", "Stage2-B", "Stage2-C"})
ALLOWED_SCENARIOS = frozenset(
    {"leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"}
)
ALLOWED_VIEW_COUNTS = frozenset({1, 3, 5})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_PAYLOAD_BYTES = 500 * 1024 * 1024
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class PredictionArtifactError(ValueError):
    """Raised when publication or verification fails closed."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PredictionArtifactError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _require_nonempty_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PredictionArtifactError(f"{name} must be non-empty trimmed text")
    if "\x00" in value:
        raise PredictionArtifactError(f"{name} contains a NUL byte")
    return value


def _string_vector(name: str, value: Sequence[Any] | np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise PredictionArtifactError(f"{name} must be one-dimensional")
    if array.dtype.kind not in {"U", "S"}:
        raise PredictionArtifactError(f"{name} must contain strings, not {array.dtype}")
    result = array.astype(str)
    if any(not item or "\x00" in item for item in result.tolist()):
        raise PredictionArtifactError(f"{name} contains an empty or NUL-bearing value")
    return result


def _prepare_arrays(
    *,
    query_tokens: Sequence[Any] | np.ndarray,
    scenarios: Sequence[Any] | np.ndarray,
    candidate_after: Sequence[Any] | np.ndarray,
    candidate_before: Sequence[Any] | np.ndarray,
    identity_after: Sequence[Any] | np.ndarray,
    identity_before: Sequence[Any] | np.ndarray,
    direct: Sequence[Any] | np.ndarray,
    shared_view_counts: Sequence[Any] | np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = {
        "query_tokens": _string_vector("query_tokens", query_tokens),
        "scenarios": _string_vector("scenarios", scenarios),
        "candidate_after": _string_vector("candidate_after", candidate_after),
        "candidate_before": _string_vector("candidate_before", candidate_before),
        "identity_after": _string_vector("identity_after", identity_after),
        "identity_before": _string_vector("identity_before", identity_before),
        "direct": _string_vector("direct", direct),
    }
    counts = np.asarray(shared_view_counts)
    if counts.ndim != 1 or counts.dtype.kind not in {"i", "u"}:
        raise PredictionArtifactError("shared_view_counts must be a one-dimensional integer vector")
    raw_count_values = [int(item) for item in counts.tolist()]
    invalid_counts = sorted(set(raw_count_values) - ALLOWED_VIEW_COUNTS)
    if invalid_counts:
        raise PredictionArtifactError(
            f"shared_view_counts must be drawn from 1/3/5, got {invalid_counts}"
        )
    counts = np.asarray(raw_count_values, dtype=np.uint8)
    arrays["shared_view_counts"] = counts

    lengths = {name: int(array.shape[0]) for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise PredictionArtifactError(f"prediction columns have inconsistent lengths: {lengths}")
    row_count = next(iter(lengths.values()), 0)
    if row_count <= 0:
        raise PredictionArtifactError("prediction artifact must contain at least one query")
    tokens = arrays["query_tokens"].tolist()
    invalid_scenarios = sorted(set(arrays["scenarios"].tolist()) - ALLOWED_SCENARIOS)
    if invalid_scenarios:
        raise PredictionArtifactError(f"unsupported LEO scenarios: {invalid_scenarios}")
    compound_keys = list(zip(arrays["scenarios"].tolist(), tokens))
    if len(compound_keys) != len(set(compound_keys)):
        raise PredictionArtifactError("scenario/query_token keys must be unique")
    return arrays


def _resource_receipt(shared_view_counts: np.ndarray) -> dict[str, Any]:
    counts = shared_view_counts.astype(np.int64, copy=False)
    total = int(counts.sum())
    n = int(counts.size)
    # ``method='higher'`` makes the recorded P95 an actually observed worst-tail
    # execution tier instead of an interpolated non-deployable count.
    try:
        p95 = int(np.percentile(counts, 95, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        p95 = int(np.percentile(counts, 95, interpolation="higher"))
    return {
        "query_count": n,
        "total_backbone_forward_count": total,
        "mean_backbone_forward_count": total / n,
        "p95_backbone_forward_count": p95,
        "max_backbone_forward_count": int(counts.max()),
        "view_1_trigger_count": int(np.count_nonzero(counts == 1)),
        "view_3_trigger_count": int(np.count_nonzero(counts == 3)),
        "view_5_trigger_count": int(np.count_nonzero(counts == 5)),
    }


def _adapter_resource_verification() -> dict[str, Any]:
    embedded = [
        name
        for name in (*OUTER_MEMBER_ALLOWLIST, *NPZ_FIELD_ALLOWLIST)
        if "adapter" in name.lower()
    ]
    if embedded:
        raise PredictionArtifactError(
            "adapter-bearing prediction schema requires content-based resource recomputation"
        )
    return {
        "status": "NOT_PROVABLE_FROM_PREDICTION_ARTIFACT",
        "reason_code": "ADAPTER_MATRIX_NOT_EMBEDDED",
        "adapter_matrix_embedded": False,
        "trainable_parameter_count_verified": False,
        "persistent_state_bytes_verified": False,
        "formal_adapter_resource_claim_allowed": False,
    }


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **{name: arrays[name] for name in NPZ_FIELD_ALLOWLIST})
    payload = buffer.getvalue()
    if len(payload) > _MAX_PAYLOAD_BYTES:
        raise PredictionArtifactError("prediction payload exceeds the size limit")
    _check_npz_member_names(payload)
    return payload


def _zip_names_exact(zf: zipfile.ZipFile, expected: tuple[str, ...], label: str) -> None:
    names = [item.filename for item in zf.infolist()]
    if len(names) != len(set(names)):
        raise PredictionArtifactError(f"{label} contains duplicate members")
    if set(names) != set(expected) or len(names) != len(expected):
        raise PredictionArtifactError(
            f"{label} members must exactly equal {list(expected)}, got {names}"
        )
    for item in zf.infolist():
        if item.is_dir() or item.filename.startswith(('/', '\\')) or ".." in Path(item.filename).parts:
            raise PredictionArtifactError(f"unsafe {label} member path: {item.filename!r}")


def _check_npz_member_names(payload: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as zf:
            _zip_names_exact(zf, NPZ_MEMBER_ALLOWLIST, "payload NPZ")
            if zf.testzip() is not None:
                raise PredictionArtifactError("payload NPZ has a CRC failure")
    except (zipfile.BadZipFile, OSError) as exc:
        raise PredictionArtifactError("payload is not a valid NPZ archive") from exc


def _zip_member(name: str, data: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100444 << 16
    return info, data


def _container_bytes(payload: bytes, manifest_bytes: bytes, seal_bytes: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", allowZip64=True) as zf:
        for name, data in (
            ("payload.npz", payload),
            ("manifest.json", manifest_bytes),
            ("seal.json", seal_bytes),
        ):
            info, content = _zip_member(name, data)
            zf.writestr(info, content)
    result = buffer.getvalue()
    if len(result) > _MAX_ARTIFACT_BYTES:
        raise PredictionArtifactError("prediction artifact exceeds the size limit")
    return result


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_noreplace(source: Path, target: Path) -> None:
    """Atomically publish ``source`` without ever replacing ``target``."""
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileExW
        move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        move_file.restype = ctypes.c_int
        # MOVEFILE_WRITE_THROUGH, deliberately without MOVEFILE_REPLACE_EXISTING.
        if not move_file(str(source), str(target), 0x00000008):
            error = ctypes.get_last_error()
            if error in {80, 183}:
                raise FileExistsError(errno.EEXIST, "prediction artifact already exists", str(target))
            raise OSError(error, os.strerror(error), str(target))
        return

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(target),
            1,  # RENAME_NOREPLACE
        )
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, "prediction artifact already exists", str(target))
        if error not in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
            raise OSError(error, os.strerror(error), str(target))

    # Same-directory hard-link publication is atomic and fails if target exists.
    # Read-only files can still be unlinked on POSIX because deletion is governed
    # by directory permissions rather than the file write bits.
    os.link(source, target, follow_symlinks=False)
    os.unlink(source)


def _write_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while sealing prediction artifact")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_regular_nofollow(path: Path) -> tuple[bytes, os.stat_result]:
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode):
        raise PredictionArtifactError("prediction artifact must not be a symlink")
    if not stat.S_ISREG(before.st_mode):
        raise PredictionArtifactError("prediction artifact must be a regular file")
    if before.st_size > _MAX_ARTIFACT_BYTES:
        raise PredictionArtifactError("prediction artifact exceeds the size limit")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PredictionArtifactError("prediction artifact could not be opened safely") from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise PredictionArtifactError("opened prediction artifact is not a regular file")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise PredictionArtifactError("prediction artifact changed between lstat and open")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise PredictionArtifactError("prediction artifact was truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise PredictionArtifactError("prediction artifact grew during read")
        return b"".join(chunks), opened
    finally:
        os.close(fd)


def _load_json_exact(data: bytes, label: str, expected_keys: set[str]) -> dict[str, Any]:
    if len(data) > _MAX_JSON_BYTES:
        raise PredictionArtifactError(f"{label} exceeds the size limit")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictionArtifactError(f"{label} is not canonical UTF-8 JSON") from exc
    if not isinstance(value, dict) or set(value) != expected_keys:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise PredictionArtifactError(f"{label} keys are not exact: {got}")
    if _canonical_json(value) != data:
        raise PredictionArtifactError(f"{label} is not canonical JSON")
    return value


def _validate_bindings(document: Mapping[str, Any], label: str) -> None:
    if document["stage"] not in ALLOWED_STAGES:
        raise PredictionArtifactError(f"unsupported {label} stage: {document['stage']!r}")
    _require_nonempty_text(f"{label}.row_id", document["row_id"])
    _require_nonempty_text(f"{label}.receiver", document["receiver"])
    k_shot = document["k_shot"]
    if not isinstance(k_shot, int) or isinstance(k_shot, bool):
        raise PredictionArtifactError(f"{label}.k_shot must be an integer")
    if (
        document["stage"] == "Stage2-A"
        and k_shot != 0
    ) or (
        document["stage"] != "Stage2-A"
        and k_shot <= 0
    ):
        raise PredictionArtifactError(
            f"{label}.k_shot does not match the stage"
        )
    for field in ("candidate_lock_sha256", "package_root_sha256", "package_seal_sha256"):
        _require_sha256(f"{label}.{field}", document[field])
    failed = [
        key
        for key, value in PHASE2_FULL_CONTRACT.items()
        if document.get(key) != value
    ]
    if failed:
        raise PredictionArtifactError(
            f"{label} Phase2 contract drift: {failed}"
        )


def publish_prediction_artifact(
    target: str | os.PathLike[str],
    *,
    stage: str,
    row_id: str,
    receiver: str,
    k_shot: int,
    candidate_lock_sha256: str,
    package_root_sha256: str,
    package_seal_sha256: str,
    query_tokens: Sequence[Any] | np.ndarray,
    scenarios: Sequence[Any] | np.ndarray,
    candidate_after: Sequence[Any] | np.ndarray,
    candidate_before: Sequence[Any] | np.ndarray,
    identity_after: Sequence[Any] | np.ndarray,
    identity_before: Sequence[Any] | np.ndarray,
    direct: Sequence[Any] | np.ndarray,
    shared_view_counts: Sequence[Any] | np.ndarray,
) -> dict[str, Any]:
    """Seal and atomically publish a Stage2 prediction artifact.

    The destination parent must already exist and must not be a symlink.  The
    destination itself must not exist; publication fails closed on collision.
    """
    binding = {
        "stage": stage,
        "row_id": row_id,
        "receiver": receiver,
        "k_shot": k_shot,
        "candidate_lock_sha256": candidate_lock_sha256,
        "package_root_sha256": package_root_sha256,
        "package_seal_sha256": package_seal_sha256,
        **PHASE2_FULL_CONTRACT,
    }
    _validate_bindings(binding, "publication")
    arrays = _prepare_arrays(
        query_tokens=query_tokens,
        scenarios=scenarios,
        candidate_after=candidate_after,
        candidate_before=candidate_before,
        identity_after=identity_after,
        identity_before=identity_before,
        direct=direct,
        shared_view_counts=shared_view_counts,
    )
    payload = _npz_bytes(arrays)
    payload_sha256 = _sha256(payload)
    resource = _resource_receipt(arrays["shared_view_counts"])
    resource_bytes = _canonical_json(resource)
    resource_sha256 = _sha256(resource_bytes)
    columns = {
        name: {"dtype": arrays[name].dtype.str, "shape": list(arrays[name].shape)}
        for name in NPZ_FIELD_ALLOWLIST
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        **binding,
        "payload_member": "payload.npz",
        "payload_sha256": payload_sha256,
        "payload_size_bytes": len(payload),
        "npz_member_allowlist": list(NPZ_MEMBER_ALLOWLIST),
        "row_count": int(arrays["query_tokens"].size),
        "columns": columns,
        "resource_receipt": resource,
        "resource_receipt_sha256": resource_sha256,
        "immutability": {
            "mode_octal": "0444",
            "no_overwrite": True,
            "publish_method": "o_excl_temp_then_atomic_noreplace",
            "write_bits_clear_at_first_visibility": True,
        },
    }
    manifest_bytes = _canonical_json(manifest)
    manifest_sha256 = _sha256(manifest_bytes)
    seal = {
        "schema_version": SEAL_SCHEMA,
        **binding,
        "payload_sha256": payload_sha256,
        "payload_size_bytes": len(payload),
        "npz_member_allowlist": list(NPZ_MEMBER_ALLOWLIST),
        "manifest_sha256": manifest_sha256,
        "manifest_size_bytes": len(manifest_bytes),
        "resource_receipt_sha256": resource_sha256,
        "hash_algorithm": "sha256",
    }
    seal_bytes = _canonical_json(seal)
    seal_sha256 = _sha256(seal_bytes)
    container = _container_bytes(payload, manifest_bytes, seal_bytes)
    artifact_sha256 = _sha256(container)

    destination = Path(target).absolute()
    parent = destination.parent
    try:
        parent_stat = os.lstat(parent)
    except FileNotFoundError as exc:
        raise PredictionArtifactError("destination parent does not exist") from exc
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise PredictionArtifactError("destination parent must be a non-symlink directory")
    temp = parent / f".{destination.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    published = False
    try:
        _write_exclusive(temp, container)
        os.chmod(temp, 0o444)
        temp_mode = stat.S_IMODE(os.lstat(temp).st_mode)
        if temp_mode & _WRITE_BITS:
            raise PredictionArtifactError("temporary prediction artifact could not be made read-only")
        _fsync_directory(parent)
        # Full validation happens while the artifact is still private.
        verify_prediction_artifact(
            temp,
            expected_artifact_sha256=artifact_sha256,
            expected_seal_sha256=seal_sha256,
        )
        _publish_noreplace(temp, destination)
        published = True
        _fsync_directory(parent)
    finally:
        if not published and temp.exists():
            try:
                os.chmod(temp, 0o600)
                temp.unlink()
            except OSError:
                pass

    final_stat = os.lstat(destination)
    readonly = stat.S_ISREG(final_stat.st_mode) and not (stat.S_IMODE(final_stat.st_mode) & _WRITE_BITS)
    if not readonly:
        raise PredictionArtifactError("published prediction artifact is not read-only")
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "path": str(destination),
        "artifact_sha256": artifact_sha256,
        "payload_sha256": payload_sha256,
        "manifest_sha256": manifest_sha256,
        "seal_sha256": seal_sha256,
        "resource_receipt_sha256": resource_sha256,
        "row_count": manifest["row_count"],
        "mode_octal": format(stat.S_IMODE(final_stat.st_mode), "04o"),
        "readonly": readonly,
        "immutable_state": "SEALED_READ_ONLY_ATOMIC_NOREPLACE",
        "adapter_resource_verification": _adapter_resource_verification(),
    }


def verify_prediction_artifact(
    path: str | os.PathLike[str],
    *,
    expected_artifact_sha256: str | None = None,
    expected_seal_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify every layer and return the safe prediction arrays and receipts."""
    if expected_artifact_sha256 is not None:
        _require_sha256("expected_artifact_sha256", expected_artifact_sha256)
    if expected_seal_sha256 is not None:
        _require_sha256("expected_seal_sha256", expected_seal_sha256)
    artifact_path = Path(path)
    data, opened_stat = _read_regular_nofollow(artifact_path)
    mode = stat.S_IMODE(opened_stat.st_mode)
    if mode & _WRITE_BITS:
        raise PredictionArtifactError("prediction artifact is not sealed read-only: write bits are set")
    artifact_sha256 = _sha256(data)
    if expected_artifact_sha256 is not None and artifact_sha256 != expected_artifact_sha256:
        raise PredictionArtifactError("prediction artifact SHA256 does not match the expected digest")

    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            _zip_names_exact(zf, OUTER_MEMBER_ALLOWLIST, "prediction container")
            if zf.testzip() is not None:
                raise PredictionArtifactError("prediction container has a CRC failure")
            infos = {item.filename: item for item in zf.infolist()}
            if infos["payload.npz"].file_size > _MAX_PAYLOAD_BYTES:
                raise PredictionArtifactError("prediction payload exceeds the size limit")
            if infos["manifest.json"].file_size > _MAX_JSON_BYTES or infos["seal.json"].file_size > _MAX_JSON_BYTES:
                raise PredictionArtifactError("prediction metadata exceeds the size limit")
            payload = zf.read("payload.npz")
            manifest_bytes = zf.read("manifest.json")
            seal_bytes = zf.read("seal.json")
    except (zipfile.BadZipFile, OSError) as exc:
        raise PredictionArtifactError("prediction artifact is not a valid sealed container") from exc

    _check_npz_member_names(payload)
    manifest_keys = {
        "schema_version", "created_utc", "stage", "row_id", "receiver", "k_shot",
        "candidate_lock_sha256", "package_root_sha256", "package_seal_sha256",
        "payload_member", "payload_sha256", "payload_size_bytes", "npz_member_allowlist",
        "row_count", "columns", "resource_receipt", "resource_receipt_sha256", "immutability",
        *PHASE2_FULL_CONTRACT.keys(),
    }
    seal_keys = {
        "schema_version", "stage", "row_id", "receiver", "k_shot",
        "candidate_lock_sha256", "package_root_sha256", "package_seal_sha256",
        "payload_sha256", "payload_size_bytes", "npz_member_allowlist",
        "manifest_sha256", "manifest_size_bytes", "resource_receipt_sha256", "hash_algorithm",
        *PHASE2_FULL_CONTRACT.keys(),
    }
    manifest = _load_json_exact(manifest_bytes, "manifest", manifest_keys)
    seal = _load_json_exact(seal_bytes, "seal", seal_keys)
    if manifest["schema_version"] != MANIFEST_SCHEMA or seal["schema_version"] != SEAL_SCHEMA:
        raise PredictionArtifactError("prediction manifest/seal schema is unsupported")
    _validate_bindings(manifest, "manifest")
    _validate_bindings(seal, "seal")
    for field in (
        "stage", "row_id", "receiver", "k_shot", "candidate_lock_sha256",
        "package_root_sha256", "package_seal_sha256",
        *PHASE2_FULL_CONTRACT.keys(),
    ):
        if manifest[field] != seal[field]:
            raise PredictionArtifactError(f"manifest/seal binding mismatch: {field}")

    payload_sha256 = _sha256(payload)
    manifest_sha256 = _sha256(manifest_bytes)
    seal_sha256 = _sha256(seal_bytes)
    if expected_seal_sha256 is not None and seal_sha256 != expected_seal_sha256:
        raise PredictionArtifactError("prediction seal SHA256 does not match the expected digest")
    if manifest["payload_member"] != "payload.npz":
        raise PredictionArtifactError("manifest payload member is invalid")
    if manifest["payload_sha256"] != payload_sha256 or seal["payload_sha256"] != payload_sha256:
        raise PredictionArtifactError("payload SHA256 binding failed")
    if manifest["payload_size_bytes"] != len(payload) or seal["payload_size_bytes"] != len(payload):
        raise PredictionArtifactError("payload size binding failed")
    if seal["manifest_sha256"] != manifest_sha256 or seal["manifest_size_bytes"] != len(manifest_bytes):
        raise PredictionArtifactError("manifest SHA256/size binding failed")
    if seal["hash_algorithm"] != "sha256":
        raise PredictionArtifactError("seal hash algorithm is unsupported")
    expected_members = list(NPZ_MEMBER_ALLOWLIST)
    if manifest["npz_member_allowlist"] != expected_members or seal["npz_member_allowlist"] != expected_members:
        raise PredictionArtifactError("NPZ member allowlist binding failed")

    resource = manifest["resource_receipt"]
    if not isinstance(resource, dict):
        raise PredictionArtifactError("resource receipt must be an object")
    resource_sha256 = _sha256(_canonical_json(resource))
    if manifest["resource_receipt_sha256"] != resource_sha256 or seal["resource_receipt_sha256"] != resource_sha256:
        raise PredictionArtifactError("resource receipt SHA256 binding failed")
    immutability = manifest["immutability"]
    expected_immutability = {
        "mode_octal": "0444",
        "no_overwrite": True,
        "publish_method": "o_excl_temp_then_atomic_noreplace",
        "write_bits_clear_at_first_visibility": True,
    }
    if immutability != expected_immutability:
        raise PredictionArtifactError("immutability policy is invalid")

    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if tuple(archive.files) != NPZ_FIELD_ALLOWLIST:
                raise PredictionArtifactError("loaded NPZ fields do not match the exact allowlist order")
            raw_arrays = {name: np.array(archive[name], copy=True) for name in NPZ_FIELD_ALLOWLIST}
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        if isinstance(exc, PredictionArtifactError):
            raise
        raise PredictionArtifactError("prediction NPZ arrays could not be loaded safely") from exc
    arrays = _prepare_arrays(**raw_arrays)
    columns = {
        name: {"dtype": arrays[name].dtype.str, "shape": list(arrays[name].shape)}
        for name in NPZ_FIELD_ALLOWLIST
    }
    if manifest["columns"] != columns or manifest["row_count"] != int(arrays["query_tokens"].size):
        raise PredictionArtifactError("prediction array schema/row count binding failed")
    calculated_resource = _resource_receipt(arrays["shared_view_counts"])
    if resource != calculated_resource:
        raise PredictionArtifactError("resource receipt does not match shared_view_counts")

    return {
        "schema_version": ARTIFACT_SCHEMA,
        "path": str(artifact_path.absolute()),
        "artifact_sha256": artifact_sha256,
        "payload_sha256": payload_sha256,
        "manifest_sha256": manifest_sha256,
        "seal_sha256": seal_sha256,
        "resource_receipt_sha256": resource_sha256,
        "manifest": manifest,
        "seal": seal,
        "arrays": arrays,
        "mode_octal": format(mode, "04o"),
        "readonly": True,
        "immutable_state": "SEALED_READ_ONLY_ATOMIC_NOREPLACE",
        "adapter_resource_verification": _adapter_resource_verification(),
    }


__all__ = [
    "ALLOWED_SCENARIOS",
    "NPZ_FIELD_ALLOWLIST",
    "NPZ_MEMBER_ALLOWLIST",
    "OUTER_MEMBER_ALLOWLIST",
    "PredictionArtifactError",
    "publish_prediction_artifact",
    "verify_prediction_artifact",
]

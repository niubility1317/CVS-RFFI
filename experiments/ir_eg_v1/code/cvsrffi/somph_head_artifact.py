"""Immutable raw-NPZ artifacts for validated SOMP-H head capsules.

This module proves only artifact integrity and SOMP-H capsule semantics.  It
does not grant formal Phase2 launch authority or establish protocol-complete
evidence for any experiment row.
"""

from __future__ import annotations

import hashlib
import io
import os
import secrets
import stat
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import cvsrffi.stage2_prediction_artifact as _base
from cvsrffi.somph_predictor_runtime import (
    SomphPredictorRuntimeError,
    somph_head_capsule_members,
    validate_somph_head_capsule,
)


ARTIFACT_SCHEMA = "cvs.phase2.somph_head_artifact.v1"
PROTOCOL_EVIDENCE_STATUS = "ARTIFACT_INTEGRITY_ONLY_NOT_FORMAL_PROTOCOL_EVIDENCE"

_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
_MAX_MEMBER_UNCOMPRESSED_BYTES = 512 * 1024
_MAX_TOTAL_UNCOMPRESSED_BYTES = 1024 * 1024
_MAX_COMPRESSION_RATIO = 100
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class SomphHeadArtifactError(ValueError):
    """Raised when a SOMP-H head artifact fails closed."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(name: str, value: Any) -> str:
    if not _is_sha256(value):
        raise SomphHeadArtifactError(f"{name} must be a lowercase SHA256")
    return str(value)


def _member_names() -> tuple[str, ...]:
    return tuple(f"{name}.npy" for name in somph_head_capsule_members())


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    try:
        np.lib.format.write_array(
            buffer,
            np.asarray(value),
            allow_pickle=False,
        )
    except (TypeError, ValueError) as exc:
        raise SomphHeadArtifactError(
            "SOMP-H head capsule contains an unsafe array"
        ) from exc
    return buffer.getvalue()


def _npz_bytes(capsule: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", allowZip64=False) as archive:
        for name in somph_head_capsule_members():
            info = zipfile.ZipInfo(
                f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100444 << 16
            archive.writestr(info, _npy_bytes(np.asarray(capsule[name])))
    payload = buffer.getvalue()
    if len(payload) > _MAX_ARTIFACT_BYTES:
        raise SomphHeadArtifactError("SOMP-H head artifact exceeds the size limit")
    _check_npz(payload)
    return payload


def _check_npz(payload: bytes) -> None:
    if len(payload) > _MAX_ARTIFACT_BYTES:
        raise SomphHeadArtifactError("SOMP-H head artifact exceeds the size limit")
    expected = _member_names()
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            infos = archive.infolist()
            names = tuple(info.filename for info in infos)
            if len(names) != len(set(names)):
                raise SomphHeadArtifactError(
                    "SOMP-H head NPZ contains duplicate members"
                )
            if names != expected:
                raise SomphHeadArtifactError(
                    "SOMP-H head NPZ exact member/order drift"
                )
            total_uncompressed = 0
            for info in infos:
                if (
                    info.is_dir()
                    or info.filename.startswith(("/", "\\"))
                    or ".." in Path(info.filename).parts
                ):
                    raise SomphHeadArtifactError(
                        f"unsafe SOMP-H head NPZ member path: {info.filename!r}"
                    )
                if info.flag_bits & 0x1:
                    raise SomphHeadArtifactError(
                        "SOMP-H head NPZ must not contain encrypted members"
                    )
                if info.compress_type != zipfile.ZIP_STORED:
                    raise SomphHeadArtifactError(
                        "SOMP-H head NPZ must use ZIP_STORED members"
                    )
                if (
                    info.file_size < 0
                    or info.compress_size < 0
                    or info.file_size > _MAX_MEMBER_UNCOMPRESSED_BYTES
                ):
                    raise SomphHeadArtifactError(
                        "SOMP-H head NPZ member exceeds the decompressed size limit"
                    )
                total_uncompressed += info.file_size
                if total_uncompressed > _MAX_TOTAL_UNCOMPRESSED_BYTES:
                    raise SomphHeadArtifactError(
                        "SOMP-H head NPZ exceeds the total decompressed size limit"
                    )
                if info.file_size:
                    if info.compress_size == 0:
                        raise SomphHeadArtifactError(
                            "SOMP-H head NPZ member has an invalid compressed size"
                        )
                    ratio = (
                        info.file_size + info.compress_size - 1
                    ) // info.compress_size
                    if ratio > _MAX_COMPRESSION_RATIO:
                        raise SomphHeadArtifactError(
                            "SOMP-H head NPZ member exceeds the compression-ratio limit"
                        )
            if archive.testzip() is not None:
                raise SomphHeadArtifactError(
                    "SOMP-H head NPZ has a CRC failure"
                )
    except (zipfile.BadZipFile, OSError) as exc:
        raise SomphHeadArtifactError(
            "SOMP-H head artifact is not a valid NPZ"
        ) from exc


def _validated_capsule(
    capsule: Mapping[str, np.ndarray],
    *,
    method_lock: Mapping[str, Any],
    expected_enrollment_binding_sha256: str,
) -> dict[str, Any]:
    binding_sha256 = _require_sha256(
        "expected_enrollment_binding_sha256",
        expected_enrollment_binding_sha256,
    )
    try:
        return validate_somph_head_capsule(
            capsule,
            method_lock=method_lock,
            expected_enrollment_binding_sha256=binding_sha256,
        )
    except SomphPredictorRuntimeError as exc:
        raise SomphHeadArtifactError(str(exc)) from exc


def publish_somph_head_artifact(
    target: str | os.PathLike[str],
    *,
    capsule: Mapping[str, np.ndarray],
    method_lock: Mapping[str, Any],
    expected_enrollment_binding_sha256: str,
) -> dict[str, Any]:
    """Publish a validated SOMP-H capsule as an immutable raw NPZ."""

    resource = _validated_capsule(
        capsule,
        method_lock=method_lock,
        expected_enrollment_binding_sha256=expected_enrollment_binding_sha256,
    )
    payload = _npz_bytes(capsule)
    head_capsule_sha256 = _sha256(payload)

    destination = Path(target).absolute()
    parent = destination.parent
    try:
        parent_stat = os.lstat(parent)
    except FileNotFoundError as exc:
        raise SomphHeadArtifactError("destination parent does not exist") from exc
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise SomphHeadArtifactError(
            "destination parent must be a non-symlink directory"
        )

    temp = parent / (
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    )
    published = False
    try:
        _base._write_exclusive(temp, payload)
        os.chmod(temp, 0o444)
        if stat.S_IMODE(os.lstat(temp).st_mode) & _WRITE_BITS:
            raise SomphHeadArtifactError(
                "temporary SOMP-H head artifact could not be made read-only"
            )
        _base._fsync_directory(parent)
        verify_somph_head_artifact(
            temp,
            method_lock=method_lock,
            expected_enrollment_binding_sha256=expected_enrollment_binding_sha256,
            expected_head_capsule_sha256=head_capsule_sha256,
        )
        _base._publish_noreplace(temp, destination)
        published = True
        _base._fsync_directory(parent)
    except _base.PredictionArtifactError as exc:
        raise SomphHeadArtifactError(str(exc)) from exc
    finally:
        if not published and temp.exists():
            try:
                os.chmod(temp, 0o600)
                temp.unlink()
            except OSError:
                pass

    final_stat = os.lstat(destination)
    if (
        not stat.S_ISREG(final_stat.st_mode)
        or stat.S_IMODE(final_stat.st_mode) & _WRITE_BITS
    ):
        raise SomphHeadArtifactError(
            "published SOMP-H head artifact is not sealed read-only"
        )
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "path": str(destination),
        "head_capsule_sha256": head_capsule_sha256,
        "enrollment_binding_sha256": resource["enrollment_binding_sha256"],
        "readonly": True,
        "immutable_state": "SEALED_READ_ONLY_ATOMIC_NOREPLACE",
        "protocol_evidence_status": PROTOCOL_EVIDENCE_STATUS,
        "formal_launch_authority": False,
    }


def verify_somph_head_artifact(
    path: str | os.PathLike[str],
    *,
    method_lock: Mapping[str, Any],
    expected_enrollment_binding_sha256: str,
    expected_head_capsule_sha256: str | None = None,
) -> dict[str, Any]:
    """Read once from one file descriptor and fully validate a head artifact."""

    binding_sha256 = _require_sha256(
        "expected_enrollment_binding_sha256",
        expected_enrollment_binding_sha256,
    )
    if expected_head_capsule_sha256 is not None:
        _require_sha256(
            "expected_head_capsule_sha256",
            expected_head_capsule_sha256,
        )
    artifact_path = Path(path)
    try:
        payload, opened_stat = _base._read_regular_nofollow(artifact_path)
    except _base.PredictionArtifactError as exc:
        raise SomphHeadArtifactError(str(exc)) from exc
    if len(payload) > _MAX_ARTIFACT_BYTES:
        raise SomphHeadArtifactError("SOMP-H head artifact exceeds the size limit")
    if stat.S_IMODE(opened_stat.st_mode) & _WRITE_BITS:
        raise SomphHeadArtifactError(
            "SOMP-H head artifact is not sealed read-only"
        )
    head_capsule_sha256 = _sha256(payload)
    if (
        expected_head_capsule_sha256 is not None
        and head_capsule_sha256 != expected_head_capsule_sha256
    ):
        raise SomphHeadArtifactError("SOMP-H head capsule SHA256 mismatch")

    _check_npz(payload)
    expected_fields = somph_head_capsule_members()
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if tuple(archive.files) != expected_fields:
                raise SomphHeadArtifactError(
                    "loaded SOMP-H head NPZ field order drift"
                )
            capsule = {
                name: np.array(archive[name], copy=True)
                for name in expected_fields
            }
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        if isinstance(exc, SomphHeadArtifactError):
            raise
        raise SomphHeadArtifactError(
            "SOMP-H head arrays could not be loaded safely"
        ) from exc
    resource = _validated_capsule(
        capsule,
        method_lock=method_lock,
        expected_enrollment_binding_sha256=binding_sha256,
    )
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "path": str(artifact_path.absolute()),
        "head_capsule_sha256": head_capsule_sha256,
        "enrollment_binding_sha256": resource["enrollment_binding_sha256"],
        "capsule": capsule,
        "resource": resource,
        "readonly": True,
        "immutable_state": "SEALED_READ_ONLY_ATOMIC_NOREPLACE",
        "protocol_evidence_status": PROTOCOL_EVIDENCE_STATUS,
        "formal_launch_authority": False,
    }


__all__ = [
    "ARTIFACT_SCHEMA",
    "PROTOCOL_EVIDENCE_STATUS",
    "SomphHeadArtifactError",
    "publish_somph_head_artifact",
    "verify_somph_head_artifact",
]

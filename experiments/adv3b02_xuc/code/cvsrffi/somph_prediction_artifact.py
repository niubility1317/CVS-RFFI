"""Single-stream immutable prediction artifacts for the SOMP-H Stage2 route.

The predictor-facing payload contains only opaque query tokens, LEO-weak
scenario names, opaque predicted class handles, and observed backbone forward
counts.  Query truth, old/new roles, per-class query counts, quotas, ordering
hints, and baseline streams are deliberately absent.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import io
import json
import os
import re
import secrets
import stat
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cvsrffi.stage2_prediction_artifact as _base


ARTIFACT_SCHEMA = "cvs.stage2.somph_single_prediction_artifact.v2"
MANIFEST_SCHEMA = "cvs.stage2.somph_single_prediction_manifest.v2"
SEAL_SCHEMA = "cvs.stage2.somph_single_prediction_seal.v2"

NPZ_FIELD_ALLOWLIST = (
    "query_tokens",
    "scenarios",
    "predicted_class_handles",
    "backbone_forward_counts",
)
NPZ_MEMBER_ALLOWLIST = tuple(f"{field}.npy" for field in NPZ_FIELD_ALLOWLIST)
REGISTRATION_STATES = frozenset({"before_registration", "after_registration"})

_QUERY_TOKEN_RE = re.compile(r"^qid_[0-9a-f]{64}$")
_CLASS_HANDLE_RE = re.compile(r"^cls_[0-9a-f]{64}$")
_ROW_ID_RE = re.compile(r"^row_[0-9a-f]{64}$")
_MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
_MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_NPZ_COMPRESSION_RATIO = 100
_BINDING_FIELDS = (
    "stage",
    "registration_state",
    "row_id",
    "receiver",
    "seed",
    "k_shot",
    "registered_class_count",
    "registry_snapshot_sha256",
    "method_lock_sha256",
    "row_manifest_sha256",
    "stage_input_binding_sha256",
    "package_root_sha256",
    "package_seal_sha256",
    "feature_runtime_sha256",
    "head_capsule_sha256",
    "protocol_policy_sha256",
)


class SomphPredictionArtifactError(ValueError):
    """Raised when a SOMP-H prediction artifact fails closed."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _string_vector(name: str, value: Sequence[Any] | np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise SomphPredictionArtifactError(f"{name} must be a one-dimensional string vector")
    result = array.astype(str)
    if any(not item or "\x00" in item for item in result.tolist()):
        raise SomphPredictionArtifactError(f"{name} contains an empty or NUL-bearing value")
    return result


def _prepare_arrays(
    *,
    query_tokens: Sequence[Any] | np.ndarray,
    scenarios: Sequence[Any] | np.ndarray,
    predicted_class_handles: Sequence[Any] | np.ndarray,
    backbone_forward_counts: Sequence[Any] | np.ndarray,
) -> dict[str, np.ndarray]:
    tokens = _string_vector("query_tokens", query_tokens)
    scenario_rows = _string_vector("scenarios", scenarios)
    predictions = _string_vector("predicted_class_handles", predicted_class_handles)
    if any(_QUERY_TOKEN_RE.fullmatch(value) is None for value in tokens.tolist()):
        raise SomphPredictionArtifactError("query_tokens must be opaque qid_SHA256 handles")
    if any(_CLASS_HANDLE_RE.fullmatch(value) is None for value in predictions.tolist()):
        raise SomphPredictionArtifactError(
            "predicted_class_handles must be opaque cls_SHA256 handles"
        )
    invalid_scenarios = sorted(set(scenario_rows.tolist()) - _base.ALLOWED_SCENARIOS)
    if invalid_scenarios:
        raise SomphPredictionArtifactError(f"unsupported LEO-weak scenarios: {invalid_scenarios}")
    counts = np.asarray(backbone_forward_counts)
    if counts.ndim != 1 or counts.dtype.kind not in {"i", "u"}:
        raise SomphPredictionArtifactError(
            "backbone_forward_counts must be a one-dimensional integer vector"
        )
    raw_counts = [int(value) for value in counts.tolist()]
    if any(value != 1 for value in raw_counts):
        raise SomphPredictionArtifactError(
            "SOMP-H single-view prediction must use exactly one backbone forward per query"
        )
    counts = np.asarray(raw_counts, dtype=np.uint8)
    arrays = {
        "query_tokens": tokens,
        "scenarios": scenario_rows,
        "predicted_class_handles": predictions,
        "backbone_forward_counts": counts,
    }
    lengths = {name: int(value.shape[0]) for name, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise SomphPredictionArtifactError(f"prediction columns have inconsistent lengths: {lengths}")
    if not len(tokens):
        raise SomphPredictionArtifactError("prediction artifact must contain at least one query")
    compound = list(zip(scenario_rows.tolist(), tokens.tolist()))
    if len(compound) != len(set(compound)):
        raise SomphPredictionArtifactError("scenario/query_token keys must be unique")
    return arrays


def _resource_receipt(counts: np.ndarray) -> dict[str, Any]:
    values = np.asarray(counts, dtype=np.int64)
    return {
        "query_count": int(values.size),
        "total_backbone_forward_count": int(values.sum()),
        "mean_backbone_forward_count": float(values.mean()),
        "p95_backbone_forward_count": 1,
        "max_backbone_forward_count": 1,
        "single_view_only": True,
    }


def _validate_binding(binding: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    if not isinstance(binding, dict) or set(binding) != set(_BINDING_FIELDS):
        raise SomphPredictionArtifactError(f"{context} binding exact schema drift")
    if binding["stage"] not in _base.ALLOWED_STAGES:
        raise SomphPredictionArtifactError(f"unsupported {context} stage")
    state = binding["registration_state"]
    if state not in REGISTRATION_STATES:
        raise SomphPredictionArtifactError(f"unsupported {context} registration_state")
    registered_count = binding["registered_class_count"]
    if (
        not isinstance(registered_count, int)
        or isinstance(registered_count, bool)
        or registered_count < 2
    ):
        raise SomphPredictionArtifactError(
            f"{context}.registered_class_count is invalid"
        )
    if binding["stage"] == "Stage2-B":
        if state != "before_registration":
            raise SomphPredictionArtifactError(
                "Stage2-B requires before_registration"
            )
    for field in ("row_id", "receiver"):
        _base._require_nonempty_text(f"{context}.{field}", binding[field])
    if _ROW_ID_RE.fullmatch(binding["row_id"]) is None:
        raise SomphPredictionArtifactError(
            f"{context}.row_id must be an opaque row_SHA256 handle"
        )
    if not isinstance(binding["seed"], int) or isinstance(binding["seed"], bool):
        raise SomphPredictionArtifactError(f"{context}.seed must be an integer")
    if not isinstance(binding["k_shot"], int) or isinstance(binding["k_shot"], bool) or binding["k_shot"] <= 0:
        raise SomphPredictionArtifactError(f"{context}.k_shot must be positive")
    for field in (
        "registry_snapshot_sha256",
        "method_lock_sha256",
        "row_manifest_sha256",
        "stage_input_binding_sha256",
        "package_root_sha256",
        "package_seal_sha256",
        "feature_runtime_sha256",
        "head_capsule_sha256",
        "protocol_policy_sha256",
    ):
        try:
            _base._require_sha256(f"{context}.{field}", binding[field])
        except _base.PredictionArtifactError as exc:
            raise SomphPredictionArtifactError(str(exc)) from exc
    return dict(binding)


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **{name: arrays[name] for name in NPZ_FIELD_ALLOWLIST})
    payload = buffer.getvalue()
    if len(payload) > _base._MAX_PAYLOAD_BYTES:
        raise SomphPredictionArtifactError("prediction payload exceeds the size limit")
    _check_npz(payload)
    return payload


def _check_npz(payload: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            _base._zip_names_exact(archive, NPZ_MEMBER_ALLOWLIST, "SOMP-H prediction NPZ")
            total_uncompressed = 0
            for info in archive.infolist():
                if info.flag_bits & 0x1:
                    raise SomphPredictionArtifactError(
                        "SOMP-H prediction NPZ must not contain encrypted members"
                    )
                if info.compress_type != zipfile.ZIP_STORED:
                    raise SomphPredictionArtifactError(
                        "SOMP-H prediction NPZ must use ZIP_STORED members"
                    )
                if (
                    info.file_size < 0
                    or info.compress_size < 0
                    or info.file_size > _MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES
                ):
                    raise SomphPredictionArtifactError(
                        "SOMP-H prediction NPZ member exceeds the decompressed size limit"
                    )
                total_uncompressed += info.file_size
                if total_uncompressed > _MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES:
                    raise SomphPredictionArtifactError(
                        "SOMP-H prediction NPZ exceeds the total decompressed size limit"
                    )
                if info.file_size:
                    if info.compress_size == 0:
                        raise SomphPredictionArtifactError(
                            "SOMP-H prediction NPZ member has an invalid compressed size"
                        )
                    ratio = (info.file_size + info.compress_size - 1) // info.compress_size
                    if ratio > _MAX_NPZ_COMPRESSION_RATIO:
                        raise SomphPredictionArtifactError(
                            "SOMP-H prediction NPZ member exceeds the compression-ratio limit"
                        )
            if archive.testzip() is not None:
                raise SomphPredictionArtifactError("SOMP-H prediction NPZ has a CRC failure")
    except (zipfile.BadZipFile, OSError) as exc:
        raise SomphPredictionArtifactError("SOMP-H payload is not a valid NPZ") from exc


def publish_somph_prediction_artifact(
    target: str | os.PathLike[str],
    *,
    query_tokens: Sequence[Any] | np.ndarray,
    scenarios: Sequence[Any] | np.ndarray,
    predicted_class_handles: Sequence[Any] | np.ndarray,
    backbone_forward_counts: Sequence[Any] | np.ndarray,
    **binding_values: Any,
) -> dict[str, Any]:
    """Publish one SOMP-H registration-state stream atomically and read-only."""
    binding = _validate_binding(dict(binding_values), context="publication")
    arrays = _prepare_arrays(
        query_tokens=query_tokens,
        scenarios=scenarios,
        predicted_class_handles=predicted_class_handles,
        backbone_forward_counts=backbone_forward_counts,
    )
    payload = _npz_bytes(arrays)
    payload_sha256 = _sha256(payload)
    resource = _resource_receipt(arrays["backbone_forward_counts"])
    resource_sha256 = _sha256(_canonical_json(resource))
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
    container = _base._container_bytes(payload, manifest_bytes, seal_bytes)
    artifact_sha256 = _sha256(container)

    destination = Path(target).absolute()
    parent = destination.parent
    try:
        parent_stat = os.lstat(parent)
    except FileNotFoundError as exc:
        raise SomphPredictionArtifactError("destination parent does not exist") from exc
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise SomphPredictionArtifactError("destination parent must be a non-symlink directory")
    temp = parent / f".{destination.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    published = False
    try:
        _base._write_exclusive(temp, container)
        os.chmod(temp, 0o444)
        if stat.S_IMODE(os.lstat(temp).st_mode) & _base._WRITE_BITS:
            raise SomphPredictionArtifactError("temporary artifact could not be made read-only")
        _base._fsync_directory(parent)
        verify_somph_prediction_artifact(
            temp,
            expected_artifact_sha256=artifact_sha256,
            expected_seal_sha256=seal_sha256,
        )
        _base._publish_noreplace(temp, destination)
        published = True
        _base._fsync_directory(parent)
    finally:
        if not published and temp.exists():
            try:
                os.chmod(temp, 0o600)
                temp.unlink()
            except OSError:
                pass
    final_stat = os.lstat(destination)
    if not stat.S_ISREG(final_stat.st_mode) or stat.S_IMODE(final_stat.st_mode) & _base._WRITE_BITS:
        raise SomphPredictionArtifactError("published artifact is not sealed read-only")
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "path": str(destination),
        "artifact_sha256": artifact_sha256,
        "payload_sha256": payload_sha256,
        "manifest_sha256": manifest_sha256,
        "seal_sha256": seal_sha256,
        "resource_receipt_sha256": resource_sha256,
        "row_count": manifest["row_count"],
        "readonly": True,
        "immutable_state": "SEALED_READ_ONLY_ATOMIC_NOREPLACE",
    }


def verify_somph_prediction_artifact(
    path: str | os.PathLike[str],
    *,
    expected_artifact_sha256: str | None = None,
    expected_seal_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify a SOMP-H single-stream artifact and return safe copied arrays."""
    for name, value in (
        ("expected_artifact_sha256", expected_artifact_sha256),
        ("expected_seal_sha256", expected_seal_sha256),
    ):
        if value is not None:
            try:
                _base._require_sha256(name, value)
            except _base.PredictionArtifactError as exc:
                raise SomphPredictionArtifactError(str(exc)) from exc
    artifact_path = Path(path)
    try:
        data, opened_stat = _base._read_regular_nofollow(artifact_path)
    except _base.PredictionArtifactError as exc:
        raise SomphPredictionArtifactError(str(exc)) from exc
    mode = stat.S_IMODE(opened_stat.st_mode)
    if mode & _base._WRITE_BITS:
        raise SomphPredictionArtifactError("prediction artifact is not sealed read-only")
    artifact_sha256 = _sha256(data)
    if expected_artifact_sha256 is not None and artifact_sha256 != expected_artifact_sha256:
        raise SomphPredictionArtifactError("artifact SHA256 mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as container:
            _base._zip_names_exact(container, _base.OUTER_MEMBER_ALLOWLIST, "prediction container")
            if container.testzip() is not None:
                raise SomphPredictionArtifactError("prediction container has a CRC failure")
            infos = {item.filename: item for item in container.infolist()}
            if infos["payload.npz"].file_size > _base._MAX_PAYLOAD_BYTES:
                raise SomphPredictionArtifactError("prediction payload exceeds the size limit")
            if (
                infos["manifest.json"].file_size > _base._MAX_JSON_BYTES
                or infos["seal.json"].file_size > _base._MAX_JSON_BYTES
            ):
                raise SomphPredictionArtifactError("prediction metadata exceeds the size limit")
            payload = container.read("payload.npz")
            manifest_bytes = container.read("manifest.json")
            seal_bytes = container.read("seal.json")
    except (zipfile.BadZipFile, OSError) as exc:
        raise SomphPredictionArtifactError("artifact is not a valid sealed container") from exc
    _check_npz(payload)
    manifest_keys = {
        "schema_version", "created_utc", *_BINDING_FIELDS,
        "payload_member", "payload_sha256", "payload_size_bytes",
        "npz_member_allowlist", "row_count", "columns", "resource_receipt",
        "resource_receipt_sha256", "immutability",
    }
    seal_keys = {
        "schema_version", *_BINDING_FIELDS, "payload_sha256", "payload_size_bytes",
        "npz_member_allowlist", "manifest_sha256", "manifest_size_bytes",
        "resource_receipt_sha256", "hash_algorithm",
    }
    try:
        manifest = _base._load_json_exact(manifest_bytes, "manifest", manifest_keys)
        seal = _base._load_json_exact(seal_bytes, "seal", seal_keys)
    except _base.PredictionArtifactError as exc:
        raise SomphPredictionArtifactError(str(exc)) from exc
    if manifest["schema_version"] != MANIFEST_SCHEMA or seal["schema_version"] != SEAL_SCHEMA:
        raise SomphPredictionArtifactError("manifest/seal schema is unsupported")
    _validate_binding({field: manifest[field] for field in _BINDING_FIELDS}, context="manifest")
    _validate_binding({field: seal[field] for field in _BINDING_FIELDS}, context="seal")
    for field in _BINDING_FIELDS:
        if manifest[field] != seal[field]:
            raise SomphPredictionArtifactError(f"manifest/seal binding mismatch: {field}")
    payload_sha256 = _sha256(payload)
    manifest_sha256 = _sha256(manifest_bytes)
    seal_sha256 = _sha256(seal_bytes)
    if expected_seal_sha256 is not None and seal_sha256 != expected_seal_sha256:
        raise SomphPredictionArtifactError("seal SHA256 mismatch")
    if manifest["payload_member"] != "payload.npz":
        raise SomphPredictionArtifactError("payload member drift")
    if manifest["payload_sha256"] != payload_sha256 or seal["payload_sha256"] != payload_sha256:
        raise SomphPredictionArtifactError("payload SHA256 binding failed")
    if manifest["payload_size_bytes"] != len(payload) or seal["payload_size_bytes"] != len(payload):
        raise SomphPredictionArtifactError("payload size binding failed")
    if seal["manifest_sha256"] != manifest_sha256 or seal["manifest_size_bytes"] != len(manifest_bytes):
        raise SomphPredictionArtifactError("manifest SHA256/size binding failed")
    if seal["hash_algorithm"] != "sha256":
        raise SomphPredictionArtifactError("unsupported hash algorithm")
    members = list(NPZ_MEMBER_ALLOWLIST)
    if manifest["npz_member_allowlist"] != members or seal["npz_member_allowlist"] != members:
        raise SomphPredictionArtifactError("NPZ member allowlist binding failed")
    resource = manifest["resource_receipt"]
    resource_sha256 = _sha256(_canonical_json(resource))
    if manifest["resource_receipt_sha256"] != resource_sha256 or seal["resource_receipt_sha256"] != resource_sha256:
        raise SomphPredictionArtifactError("resource receipt binding failed")
    expected_immutability = {
        "mode_octal": "0444",
        "no_overwrite": True,
        "publish_method": "o_excl_temp_then_atomic_noreplace",
        "write_bits_clear_at_first_visibility": True,
    }
    if manifest["immutability"] != expected_immutability:
        raise SomphPredictionArtifactError("immutability policy drift")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if tuple(archive.files) != NPZ_FIELD_ALLOWLIST:
                raise SomphPredictionArtifactError("loaded NPZ field order drift")
            raw = {name: np.array(archive[name], copy=True) for name in NPZ_FIELD_ALLOWLIST}
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        if isinstance(exc, SomphPredictionArtifactError):
            raise
        raise SomphPredictionArtifactError("prediction arrays could not be loaded safely") from exc
    arrays = _prepare_arrays(**raw)
    columns = {
        name: {"dtype": arrays[name].dtype.str, "shape": list(arrays[name].shape)}
        for name in NPZ_FIELD_ALLOWLIST
    }
    if manifest["columns"] != columns or manifest["row_count"] != int(arrays["query_tokens"].size):
        raise SomphPredictionArtifactError("array schema/row-count binding failed")
    if resource != _resource_receipt(arrays["backbone_forward_counts"]):
        raise SomphPredictionArtifactError("resource receipt does not match payload")
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
        "readonly": True,
        "immutable_state": "SEALED_READ_ONLY_ATOMIC_NOREPLACE",
    }


__all__ = [
    "ARTIFACT_SCHEMA",
    "MANIFEST_SCHEMA",
    "NPZ_FIELD_ALLOWLIST",
    "NPZ_MEMBER_ALLOWLIST",
    "SEAL_SCHEMA",
    "SomphPredictionArtifactError",
    "publish_somph_prediction_artifact",
    "verify_somph_prediction_artifact",
]

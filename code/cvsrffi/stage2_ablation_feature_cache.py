"""Reusable truth-free feature caches for Phase2 ablation rows.

One cache is produced once for a sealed physical input row and can then be
reused by every compatible ablation arm. The cache contains legal support
features/labels, unlabelled query features with opaque tokens, frozen Phase1
deployment prototypes, and immutable ground-spectrum state. It never contains
query labels, query roles, transmitter identities, or scorer-side files.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.full_ablation_spec import PROTOCOL_SCHEMA
from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS


FEATURE_CACHE_SCHEMA = "cvs.full_ablation.phase2.feature_cache.v2"
FEATURE_CACHE_MANIFEST_SCHEMA = (
    "cvs.full_ablation.phase2.feature_cache_manifest.v2"
)
FEATURE_DIM = 288
OLD_CLASS_COUNT = 6

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLASS_HANDLE = re.compile(r"^cls_[0-9a-f]{32,64}$")
_QUERY_TOKEN = re.compile(r"^qid_[0-9a-f]{32,64}$")
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
_STAGE_PAYLOAD_FIELDS = {
    "stage2a": ("query_features", "query_tokens"),
    "stage2b": (
        "old_support_features",
        "old_support_labels",
        "query_features",
        "query_tokens",
    ),
    "stage2c": (
        "old_support_features",
        "old_support_labels",
        "new_support_features",
        "new_support_labels",
        "query_features",
        "query_tokens",
    ),
}
_FORBIDDEN_AUDIT_KEYS = ("truth", "role_oracle", "query_label")


class Stage2AblationFeatureCacheError(ValueError):
    """Raised when a truth-free feature cache fails closed."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(value: str, *, name: str) -> str:
    result = str(value).strip().lower()
    if _SHA256.fullmatch(result) is None:
        raise Stage2AblationFeatureCacheError(f"{name} must be SHA256")
    return result


def _string_vector(
    value: Any,
    *,
    name: str,
    pattern: re.Pattern[str] | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S", "O"}:
        raise Stage2AblationFeatureCacheError(
            f"{name} must be a one-dimensional string vector"
        )
    result = array.astype(str)
    if any(
        not item
        or "\x00" in item
        or (pattern is not None and pattern.fullmatch(item) is None)
        for item in result.tolist()
    ):
        raise Stage2AblationFeatureCacheError(f"{name} value drift")
    return result


def _feature_matrix(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if (
        result.ndim != 2
        or result.shape[0] <= 0
        or result.shape[1] != FEATURE_DIM
        or not np.isfinite(result).all()
    ):
        raise Stage2AblationFeatureCacheError(
            f"{name} must be finite N x {FEATURE_DIM}"
        )
    return result


def _balanced_support(
    features: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    *,
    name: str,
) -> int:
    if len(features) != len(labels) or set(labels.tolist()) != set(classes):
        raise Stage2AblationFeatureCacheError(f"{name} registry drift")
    counts = [int(np.sum(labels == value)) for value in classes]
    if not counts or min(counts) <= 0 or len(set(counts)) != 1:
        raise Stage2AblationFeatureCacheError(
            f"{name} is not balanced K-shot support"
        )
    return counts[0]


def _audit_contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(item in lowered for item in _FORBIDDEN_AUDIT_KEYS):
                return True
            if _audit_contains_forbidden_key(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_audit_contains_forbidden_key(child) for child in value)
    return False


def _write_exclusive_readonly(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing feature cache")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)


def _payload_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    value = buffer.getvalue()
    try:
        with zipfile.ZipFile(io.BytesIO(value), "r") as archive:
            names = [item.filename for item in archive.infolist()]
            expected = [f"{name}.npy" for name in arrays]
            if (
                len(names) != len(set(names))
                or set(names) != set(expected)
                or archive.testzip() is not None
            ):
                raise Stage2AblationFeatureCacheError(
                    "feature-cache NPZ member drift"
                )
    except zipfile.BadZipFile as exc:
        raise Stage2AblationFeatureCacheError(
            "feature-cache NPZ is invalid"
        ) from exc
    return value


def publish_feature_cache(
    payload_path: str | Path,
    manifest_path: str | Path,
    *,
    receiver: str,
    method_seed: int,
    support_seed: int,
    query_seed: int,
    new_class_draw_seed: int,
    stage_scope: str,
    phase2_data_status: str,
    capsule_id: str,
    split_id: str,
    package_root_sha256: str,
    package_seal_sha256: str,
    phase1_bundle_sha256: str,
    phase1_prototype_sha256: str,
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    scenario_payloads: Mapping[str, Mapping[str, Any]],
    deployment_prototypes: Any,
    ground_basis: Any,
    ground_spectral_weights: Any,
    ground_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish one immutable, reusable feature cache and manifest."""

    scope = str(stage_scope)
    if scope not in _STAGE_PAYLOAD_FIELDS:
        raise Stage2AblationFeatureCacheError("unknown Stage2 cache scope")
    if (
        str(phase2_data_status) != "VALIDATED_ONCE"
        or not str(capsule_id).strip()
        or not str(split_id).strip()
    ):
        raise Stage2AblationFeatureCacheError(
            "validated Phase2 data handles are incomplete"
        )
    if set(scenario_payloads) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise Stage2AblationFeatureCacheError(
            "feature cache must cover exactly three formal scenarios"
        )
    old_registry = tuple(str(value) for value in old_classes)
    new_registry = tuple(str(value) for value in new_classes)
    if (
        len(old_registry) != OLD_CLASS_COUNT
        or len(set(old_registry)) != len(old_registry)
        or any(_CLASS_HANDLE.fullmatch(value) is None for value in old_registry)
        or (
            scope == "stage2c"
            and len(new_registry) not in {5, 10, 20}
        )
        or (scope != "stage2c" and len(new_registry) != 0)
        or len(set(new_registry)) != len(new_registry)
        or any(_CLASS_HANDLE.fullmatch(value) is None for value in new_registry)
        or set(old_registry) & set(new_registry)
    ):
        raise Stage2AblationFeatureCacheError("class registry drift")
    prototypes = _feature_matrix(
        deployment_prototypes, name="deployment_prototypes"
    )
    if len(prototypes) != OLD_CLASS_COUNT:
        raise Stage2AblationFeatureCacheError(
            "deployment prototype count drift"
        )
    if _audit_contains_forbidden_key(ground_audit):
        raise Stage2AblationFeatureCacheError(
            "ground audit contains forbidden truth-side state"
        )

    arrays: dict[str, np.ndarray] = {
        "deployment_prototypes": prototypes,
    }
    if scope == "stage2a":
        if ground_audit:
            raise Stage2AblationFeatureCacheError(
                "Stage2-A cache must not carry ground audit state"
            )
    else:
        basis = np.asarray(ground_basis, dtype=np.float64)
        weights = np.asarray(ground_spectral_weights, dtype=np.float64)
        if (
            basis.ndim != 2
            or basis.shape[0] != 160
            or weights.shape != (basis.shape[1],)
            or not np.isfinite(basis).all()
            or not np.isfinite(weights).all()
            or bool(np.any(weights <= 0))
        ):
            raise Stage2AblationFeatureCacheError("ground spectrum drift")
        arrays.update(
            {
                "ground_basis": basis,
                "ground_spectral_weights": weights,
            }
        )
    k_shot: int | None = None
    query_counts: dict[str, int] = {}
    scenario_token_sets: list[set[str]] = []
    payload_fields = _STAGE_PAYLOAD_FIELDS[scope]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        payload = scenario_payloads[scenario]
        if set(payload) != set(payload_fields):
            raise Stage2AblationFeatureCacheError(
                f"{scenario} payload schema drift"
            )
        query_x = _feature_matrix(
            payload["query_features"],
            name=f"{scenario}.query_features",
        )
        query_token = _string_vector(
            payload["query_tokens"],
            name=f"{scenario}.query_tokens",
            pattern=_QUERY_TOKEN,
        )
        if len(query_x) != len(query_token):
            raise Stage2AblationFeatureCacheError(
                f"{scenario} query row count drift"
            )
        prefix = scenario + "__"
        arrays[prefix + "query_features"] = query_x
        arrays[prefix + "query_tokens"] = query_token
        query_counts[scenario] = len(query_token)
        current_tokens = set(query_token.tolist())
        if len(current_tokens) != len(query_token):
            raise Stage2AblationFeatureCacheError(
                f"{scenario} contains duplicate query tokens"
            )
        if any(current_tokens & prior for prior in scenario_token_sets):
            raise Stage2AblationFeatureCacheError(
                "formal scenario query tokens must be pairwise disjoint"
            )
        scenario_token_sets.append(current_tokens)
        if scope != "stage2a":
            old_x = _feature_matrix(
                payload["old_support_features"],
                name=f"{scenario}.old_support_features",
            )
            old_y = _string_vector(
                payload["old_support_labels"],
                name=f"{scenario}.old_support_labels",
                pattern=_CLASS_HANDLE,
            )
            old_k = _balanced_support(
                old_x, old_y, old_registry, name=f"{scenario}.old_support"
            )
            arrays[prefix + "old_support_features"] = old_x
            arrays[prefix + "old_support_labels"] = old_y
            if scope == "stage2c":
                new_x = _feature_matrix(
                    payload["new_support_features"],
                    name=f"{scenario}.new_support_features",
                )
                new_y = _string_vector(
                    payload["new_support_labels"],
                    name=f"{scenario}.new_support_labels",
                    pattern=_CLASS_HANDLE,
                )
                new_k = _balanced_support(
                    new_x,
                    new_y,
                    new_registry,
                    name=f"{scenario}.new_support",
                )
                if old_k != new_k:
                    raise Stage2AblationFeatureCacheError(
                        f"{scenario} old/new K mismatch"
                    )
                arrays[prefix + "new_support_features"] = new_x
                arrays[prefix + "new_support_labels"] = new_y
            if k_shot is None:
                k_shot = old_k
            elif k_shot != old_k:
                raise Stage2AblationFeatureCacheError(
                    "K-shot drift across scenarios"
                )
    if scope == "stage2a":
        k_shot = 0
    elif k_shot not in {1, 2, 5, 10}:
        raise Stage2AblationFeatureCacheError("unsupported K-shot")

    payload_destination = Path(payload_path).absolute()
    manifest_destination = Path(manifest_path).absolute()
    if payload_destination.parent != manifest_destination.parent:
        raise Stage2AblationFeatureCacheError(
            "cache payload and manifest must share a directory"
        )
    parent = payload_destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if (
        payload_destination.exists()
        or manifest_destination.exists()
        or payload_destination.is_symlink()
        or manifest_destination.is_symlink()
    ):
        raise Stage2AblationFeatureCacheError(
            "refusing to overwrite feature-cache evidence"
        )
    payload = _payload_bytes(arrays)
    payload_sha256 = _sha256_bytes(payload)
    manifest = {
        "schema": FEATURE_CACHE_MANIFEST_SCHEMA,
        "feature_cache_schema": FEATURE_CACHE_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "stage_scope": scope,
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": str(capsule_id),
        "split_id": str(split_id),
        "receiver": str(receiver),
        "method_seed": int(method_seed),
        "support_seed": int(support_seed),
        "query_seed": int(query_seed),
        "new_class_draw_seed": int(new_class_draw_seed),
        "k_shot": int(k_shot),
        "old_classes": list(old_registry),
        "new_classes": list(new_registry),
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "package_root_sha256": _hash(
            package_root_sha256, name="package_root_sha256"
        ),
        "package_seal_sha256": _hash(
            package_seal_sha256, name="package_seal_sha256"
        ),
        "phase1_bundle_sha256": _hash(
            phase1_bundle_sha256, name="phase1_bundle_sha256"
        ),
        "phase1_prototype_sha256": _hash(
            phase1_prototype_sha256,
            name="phase1_prototype_sha256",
        ),
        "payload_file": payload_destination.name,
        "payload_sha256": payload_sha256,
        "payload_size_bytes": len(payload),
        "deployment_state_bytes": int(
            prototypes.nbytes
            + (
                arrays["ground_basis"].nbytes
                + arrays["ground_spectral_weights"].nbytes
                if scope != "stage2a"
                else 0
            )
        ),
        "array_names": sorted(arrays),
        "array_descriptors": {
            name: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
            }
            for name, value in sorted(arrays.items())
        },
        "ground_audit": dict(ground_audit) if scope != "stage2a" else {},
        "query_truth_present": False,
        "query_role_present": False,
        "clean_source_samples_present": False,
        **PHASE2_FULL_CONTRACT,
    }
    manifest_bytes = _canonical_json(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    _write_exclusive_readonly(payload_destination, payload)
    _write_exclusive_readonly(
        manifest_destination, manifest_bytes + b"\n"
    )
    return {
        "payload_path": str(payload_destination),
        "payload_sha256": payload_sha256,
        "manifest_path": str(manifest_destination),
        "manifest_sha256": manifest_sha256,
        "stage_scope": scope,
        "k_shot": k_shot,
        "query_rows_by_scenario": query_counts,
        "immutable": True,
    }


def repair_legacy_stage2b_manifest_protocol_schema(
    source_manifest_path: str | Path,
    destination_manifest_path: str | Path,
    *,
    expected_source_manifest_sha256: str,
) -> dict[str, str]:
    """Repair only the protocol field omitted by the legacy Stage2-B builder."""

    source = Path(source_manifest_path)
    destination = Path(destination_manifest_path)
    source_info = os.lstat(source)
    if stat.S_ISLNK(source_info.st_mode) or not stat.S_ISREG(source_info.st_mode):
        raise Stage2AblationFeatureCacheError(
            "legacy feature-cache manifest is not a regular file"
        )
    source_bytes = source.read_bytes()
    if source_bytes.endswith(b"\n"):
        source_bytes = source_bytes[:-1]
    source_sha256 = _sha256_bytes(source_bytes)
    if source_sha256 != _hash(
        expected_source_manifest_sha256,
        name="expected_source_manifest_sha256",
    ):
        raise Stage2AblationFeatureCacheError(
            "legacy feature-cache manifest SHA256 mismatch"
        )
    try:
        manifest = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2AblationFeatureCacheError(
            "legacy feature-cache manifest is invalid"
        ) from exc
    split_id = str(manifest.get("split_id", ""))
    split_match = re.fullmatch(
        rf"{re.escape(PROTOCOL_SCHEMA)}-rx(?P<receiver>.+)-m(?P<method>\d+)"
        rf"-s(?P<support>\d+)-q(?P<query>\d+)-d(?P<draw>\d+)"
        rf"-k(?P<k>\d+)-new(?P<new>\d+)",
        split_id,
    )
    if (
        not isinstance(manifest, dict)
        or "protocol_schema" in manifest
        or manifest.get("schema") != FEATURE_CACHE_MANIFEST_SCHEMA
        or manifest.get("feature_cache_schema") != FEATURE_CACHE_SCHEMA
        or manifest.get("stage_scope") != "stage2b"
        or manifest.get("phase2_data_status") != "VALIDATED_ONCE"
        or not str(manifest.get("capsule_id", "")).strip()
        or split_match is None
        or manifest.get("query_truth_present") is not False
        or manifest.get("query_role_present") is not False
        or manifest.get("clean_source_samples_present") is not False
        or tuple(manifest.get("scenarios", ()))
        != tuple(FORMAL_LEO_WEAK_SCENARIOS)
        or any(
            manifest.get(key) != value
            for key, value in PHASE2_FULL_CONTRACT.items()
        )
    ):
        raise Stage2AblationFeatureCacheError(
            "legacy Stage2-B manifest contract drift"
        )
    assert split_match is not None
    split_values = split_match.groupdict()
    if (
        str(manifest.get("receiver")) != split_values["receiver"]
        or int(manifest.get("method_seed", -1)) != int(split_values["method"])
        or int(manifest.get("support_seed", -1)) != int(split_values["support"])
        or int(manifest.get("query_seed", -1)) != int(split_values["query"])
        or int(manifest.get("k_shot", -1)) != int(split_values["k"])
        or re.fullmatch(
            rf"d18-reuse-validated-once-rx{re.escape(split_values['receiver'])}"
            rf"-seed\d+-m{split_values['method']}-k{split_values['k']}"
            rf"-new{split_values['new']}",
            str(manifest["capsule_id"]),
        )
        is None
    ):
        raise Stage2AblationFeatureCacheError(
            "legacy Stage2-B row identity drift"
        )
    for key in (
        "payload_sha256",
        "package_root_sha256",
        "package_seal_sha256",
        "phase1_bundle_sha256",
        "phase1_prototype_sha256",
    ):
        _hash(manifest.get(key), name=key)
    repaired = dict(manifest)
    repaired["protocol_schema"] = PROTOCOL_SCHEMA
    repaired_bytes = _canonical_json(repaired)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive_readonly(destination, repaired_bytes + b"\n")
    return {
        "source_manifest_sha256": source_sha256,
        "manifest_path": str(destination),
        "manifest_sha256": _sha256_bytes(repaired_bytes),
        "protocol_schema": PROTOCOL_SCHEMA,
    }


def load_feature_cache(
    payload_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_payload_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify and load a previously published truth-free feature cache."""

    payload_file = Path(payload_path)
    manifest_file = Path(manifest_path)
    for path, label in (
        (payload_file, "payload"),
        (manifest_file, "manifest"),
    ):
        info = os.lstat(path)
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) & _WRITE_BITS
        ):
            raise Stage2AblationFeatureCacheError(
                f"feature-cache {label} is not sealed read-only"
            )
    payload = payload_file.read_bytes()
    manifest_bytes = manifest_file.read_bytes()
    if manifest_bytes.endswith(b"\n"):
        manifest_bytes = manifest_bytes[:-1]
    if _sha256_bytes(payload) != _hash(
        expected_payload_sha256, name="expected_payload_sha256"
    ):
        raise Stage2AblationFeatureCacheError(
            "feature-cache payload SHA256 mismatch"
        )
    if _sha256_bytes(manifest_bytes) != _hash(
        expected_manifest_sha256, name="expected_manifest_sha256"
    ):
        raise Stage2AblationFeatureCacheError(
            "feature-cache manifest SHA256 mismatch"
        )
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if (
        manifest.get("schema") != FEATURE_CACHE_MANIFEST_SCHEMA
        or manifest.get("feature_cache_schema") != FEATURE_CACHE_SCHEMA
        or manifest.get("protocol_schema") != PROTOCOL_SCHEMA
        or manifest.get("payload_file") != payload_file.name
        or manifest.get("payload_sha256") != expected_payload_sha256
        or manifest.get("query_truth_present") is not False
        or manifest.get("query_role_present") is not False
        or manifest.get("clean_source_samples_present") is not False
        or manifest.get("stage_scope") not in _STAGE_PAYLOAD_FIELDS
        or manifest.get("phase2_data_status") != "VALIDATED_ONCE"
        or not str(manifest.get("capsule_id", "")).strip()
        or not str(manifest.get("split_id", "")).strip()
        or tuple(manifest.get("scenarios", ()))
        != tuple(FORMAL_LEO_WEAK_SCENARIOS)
        or any(
            manifest.get(key) != value
            for key, value in PHASE2_FULL_CONTRACT.items()
        )
    ):
        raise Stage2AblationFeatureCacheError(
            "feature-cache manifest contract drift"
        )
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != set(manifest["array_names"]):
                raise Stage2AblationFeatureCacheError(
                    "feature-cache array allowlist drift"
                )
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise Stage2AblationFeatureCacheError(
            "feature-cache payload cannot be loaded"
        ) from exc
    descriptors = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
        for name, value in sorted(arrays.items())
    }
    if descriptors != manifest["array_descriptors"]:
        raise Stage2AblationFeatureCacheError(
            "feature-cache array descriptor drift"
        )
    scope = str(manifest["stage_scope"])
    payload_fields = _STAGE_PAYLOAD_FIELDS[scope]
    expected_array_names = {"deployment_prototypes"}
    if scope != "stage2a":
        expected_array_names.update(
            {"ground_basis", "ground_spectral_weights"}
        )
    expected_array_names.update(
        f"{scenario}__{field}"
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
        for field in payload_fields
    )
    if set(arrays) != expected_array_names:
        raise Stage2AblationFeatureCacheError(
            "feature-cache stage allowlist drift"
        )
    scenario_payloads = {
        scenario: {
            field: arrays[f"{scenario}__{field}"]
            for field in payload_fields
        }
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    return {
        "manifest": manifest,
        "old_classes": tuple(manifest["old_classes"]),
        "new_classes": tuple(manifest["new_classes"]),
        "scenario_payloads": scenario_payloads,
        "deployment_prototypes_by_scenario": {
            scenario: arrays["deployment_prototypes"]
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "ground_basis": arrays.get(
            "ground_basis", np.empty((160, 0), dtype=np.float64)
        ),
        "ground_spectral_weights": arrays.get(
            "ground_spectral_weights", np.empty(0, dtype=np.float64)
        ),
        "ground_audit": dict(manifest["ground_audit"]),
    }


__all__ = [
    "FEATURE_CACHE_MANIFEST_SCHEMA",
    "FEATURE_CACHE_SCHEMA",
    "Stage2AblationFeatureCacheError",
    "load_feature_cache",
    "publish_feature_cache",
    "repair_legacy_stage2b_manifest_protocol_schema",
]

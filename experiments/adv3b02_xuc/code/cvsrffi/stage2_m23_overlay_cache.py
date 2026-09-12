"""Immutable compact overlay cache for ERBT-IDR M2.3 RFGuard.

The overlay binds a validated 288-dimensional feature cache to freshly
derived RF-lite/quality observations and the already sealed aggregate Phase1
component.  It contains support labels but no query labels, clean/source
samples, or dense floating-point ground bank.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_m23_rfguard import COMPACT_DIM, IDENTITY_DIM


M23_OVERLAY_SCHEMA = "cvs.erbt_idr.m23.overlay_cache.v1"
M23_OVERLAY_MANIFEST_SCHEMA = "cvs.erbt_idr.m23.overlay_cache_manifest.v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLASS_HANDLE = re.compile(r"^cls_[0-9a-f]{32,64}$")
_QUERY_TOKEN = re.compile(r"^qid_[0-9a-f]{32,64}$")
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
_SCENARIO_FIELDS = (
    "old_support_blocks",
    "old_support_labels",
    "old_support_quality",
    "new_support_blocks",
    "new_support_labels",
    "new_support_quality",
    "query_blocks",
    "query_tokens",
)
_GROUND_FIELDS = (
    "core_q",
    "core_scale",
    "residual_basis_q",
    "residual_basis_scale",
    "residual_coeff_q",
    "residual_coeff_scale",
    "domain_registry",
    "residual_domain_registry",
    "class_registry",
    "center_domain_handle",
)


class M23OverlayCacheError(ValueError):
    """Raised when an M2.3 overlay fails its truth-free compact schema."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(value: Any, *, name: str) -> str:
    result = str(value).strip().lower()
    if _SHA256.fullmatch(result) is None:
        raise M23OverlayCacheError(f"{name} must be SHA256")
    return result


def _strings(value: Any, *, name: str, pattern: re.Pattern[str] | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in {"U", "S", "O"}:
        raise M23OverlayCacheError(f"{name} must be a string vector")
    result = raw.astype(str)
    if any(
        not item
        or "\x00" in item
        or (pattern is not None and pattern.fullmatch(item) is None)
        for item in result.tolist()
    ):
        raise M23OverlayCacheError(f"{name} value drift")
    return result


def _blocks(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] <= 0 or result.shape[1] != COMPACT_DIM or not np.isfinite(result).all():
        raise M23OverlayCacheError(f"{name} must be finite N x {COMPACT_DIM}")
    norms = (
        np.linalg.norm(result[:, :160], axis=1),
        np.linalg.norm(result[:, 160:256], axis=1),
        np.linalg.norm(result[:, 256:], axis=1),
    )
    if any(np.any(value <= 1.0e-8) for value in norms):
        raise M23OverlayCacheError(f"{name} contains a zero compact block")
    return np.array(result, copy=True)


def _quality(value: Any, rows: int, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (rows,) or not np.isfinite(result).all() or np.any(result <= 0.0) or np.any(result > 1.0):
        raise M23OverlayCacheError(f"{name} quality must be in (0, 1]")
    return np.array(result, copy=True)


def _registry(value: Sequence[str], *, name: str, expected_count: set[int]) -> tuple[str, ...]:
    result = tuple(str(item) for item in value)
    if len(result) not in expected_count or len(set(result)) != len(result) or any(_CLASS_HANDLE.fullmatch(item) is None for item in result):
        raise M23OverlayCacheError(f"{name} class registry drift")
    return result


def _balanced(labels: np.ndarray, classes: tuple[str, ...], k_shot: int, *, name: str) -> None:
    if set(labels.tolist()) != set(classes):
        raise M23OverlayCacheError(f"{name} support registry drift")
    counts = [int(np.sum(labels == item)) for item in classes]
    if any(count != k_shot for count in counts):
        raise M23OverlayCacheError(f"{name} support is not the declared K-shot set")


def _ground(value: Mapping[str, Any], old_classes: tuple[str, ...]) -> dict[str, np.ndarray]:
    if set(value) != set(_GROUND_FIELDS):
        raise M23OverlayCacheError("ground component schema drift")
    domains = _strings(value["domain_registry"], name="domain_registry")
    residual_domains = _strings(
        value["residual_domain_registry"], name="residual_domain_registry"
    )
    classes = _strings(value["class_registry"], name="class_registry", pattern=_CLASS_HANDLE)
    center_raw = np.asarray(value["center_domain_handle"])
    if center_raw.shape != () or center_raw.dtype.kind not in {"U", "S", "O"}:
        raise M23OverlayCacheError("center-domain handle schema drift")
    center = str(center_raw.item())
    if (
        tuple(classes.tolist()) != old_classes
        or len(domains) < 2
        or len(set(domains.tolist())) != len(domains)
        or center not in set(domains.tolist())
        or set(residual_domains.tolist()) != set(domains.tolist()) - {center}
    ):
        raise M23OverlayCacheError("ground registry binding drift")
    class_count = len(classes)
    residual_count = len(residual_domains)
    core_q = np.asarray(value["core_q"])
    core_scale = np.asarray(value["core_scale"])
    basis_q = np.asarray(value["residual_basis_q"])
    basis_scale = np.asarray(value["residual_basis_scale"])
    coefficient_q = np.asarray(value["residual_coeff_q"])
    coefficient_scale = np.asarray(value["residual_coeff_scale"])
    if basis_q.ndim != 3:
        raise M23OverlayCacheError("ground residual basis schema drift")
    rank = basis_q.shape[1]
    specifications = (
        (core_q, np.dtype(np.int8), (class_count, IDENTITY_DIM), "core_q"),
        (core_scale, np.dtype(np.float16), (class_count,), "core_scale"),
        (
            basis_q,
            np.dtype(np.int8),
            (class_count, rank, IDENTITY_DIM),
            "residual_basis_q",
        ),
        (
            basis_scale,
            np.dtype(np.float16),
            (class_count, rank),
            "residual_basis_scale",
        ),
        (
            coefficient_q,
            np.dtype(np.int8),
            (residual_count, class_count, rank),
            "residual_coeff_q",
        ),
        (
            coefficient_scale,
            np.dtype(np.float16),
            (residual_count, class_count),
            "residual_coeff_scale",
        ),
    )
    for array, dtype, shape, name in specifications:
        if array.dtype != dtype or array.shape != shape or not np.isfinite(array).all():
            raise M23OverlayCacheError(f"ground {name} schema drift")
    if rank <= 0 or np.any(core_scale <= 0) or np.any(basis_scale <= 0) or np.any(coefficient_scale <= 0):
        raise M23OverlayCacheError("ground quantization scale drift")
    return {
        "core_q": np.array(core_q, copy=True),
        "core_scale": np.array(core_scale, copy=True),
        "residual_basis_q": np.array(basis_q, copy=True),
        "residual_basis_scale": np.array(basis_scale, copy=True),
        "residual_coeff_q": np.array(coefficient_q, copy=True),
        "residual_coeff_scale": np.array(coefficient_scale, copy=True),
        "domain_registry": np.array(domains, copy=True),
        "residual_domain_registry": np.array(residual_domains, copy=True),
        "class_registry": np.array(classes, copy=True),
        "center_domain_handle": np.asarray(center, dtype=np.str_),
    }


def _scenario(
    value: Mapping[str, Any],
    *,
    scenario: str,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    k_shot: int,
) -> dict[str, np.ndarray]:
    if set(value) != set(_SCENARIO_FIELDS):
        raise M23OverlayCacheError(f"{scenario} payload schema drift")
    old_blocks = _blocks(value["old_support_blocks"], name=f"{scenario}.old_support_blocks")
    new_blocks = _blocks(value["new_support_blocks"], name=f"{scenario}.new_support_blocks")
    query_blocks = _blocks(value["query_blocks"], name=f"{scenario}.query_blocks")
    old_labels = _strings(
        value["old_support_labels"], name=f"{scenario}.old_support_labels", pattern=_CLASS_HANDLE
    )
    new_labels = _strings(
        value["new_support_labels"], name=f"{scenario}.new_support_labels", pattern=_CLASS_HANDLE
    )
    tokens = _strings(value["query_tokens"], name=f"{scenario}.query_tokens", pattern=_QUERY_TOKEN)
    if len(old_labels) != len(old_blocks) or len(new_labels) != len(new_blocks) or len(tokens) != len(query_blocks):
        raise M23OverlayCacheError(f"{scenario} row-count drift")
    if len(set(tokens.tolist())) != len(tokens):
        raise M23OverlayCacheError(f"{scenario} duplicate query token")
    _balanced(old_labels, old_classes, k_shot, name=f"{scenario}.old")
    _balanced(new_labels, new_classes, k_shot, name=f"{scenario}.new")
    return {
        "old_support_blocks": old_blocks,
        "old_support_labels": old_labels,
        "old_support_quality": _quality(
            value["old_support_quality"], len(old_blocks), name=f"{scenario}.old_support"
        ),
        "new_support_blocks": new_blocks,
        "new_support_labels": new_labels,
        "new_support_quality": _quality(
            value["new_support_quality"], len(new_blocks), name=f"{scenario}.new_support"
        ),
        "query_blocks": query_blocks,
        "query_tokens": tokens,
    }


def _payload_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    result = buffer.getvalue()
    try:
        with zipfile.ZipFile(io.BytesIO(result), "r") as archive:
            names = [item.filename for item in archive.infolist()]
            expected = [f"{name}.npy" for name in arrays]
            if len(names) != len(set(names)) or set(names) != set(expected) or archive.testzip() is not None:
                raise M23OverlayCacheError("overlay NPZ member drift")
    except zipfile.BadZipFile as exc:
        raise M23OverlayCacheError("overlay payload is not a valid NPZ") from exc
    return result


def _write_exclusive_readonly(path: Path, data: bytes) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise M23OverlayCacheError("overlay destination parent must be a real directory")
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
                raise OSError("short write while publishing M2.3 overlay")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)


def publish_m23_overlay_cache(
    payload_path: str | Path,
    manifest_path: str | Path,
    *,
    receiver: str,
    k_shot: int,
    method_seed: int,
    support_seed: int,
    query_seed: int,
    new_class_draw_seed: int,
    phase2_data_status: str,
    capsule_id: str,
    split_id: str,
    base_feature_cache_payload_sha256: str,
    base_feature_cache_manifest_sha256: str,
    predictor_package_root_sha256: str,
    predictor_package_seal_sha256: str,
    phase1_bundle_sha256: str,
    phase1_component_manifest_sha256: str,
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    scenario_payloads: Mapping[str, Mapping[str, Any]],
    ground_component: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish one non-overwriting, truth-free M2.3 compact overlay."""

    if str(phase2_data_status) != "VALIDATED_ONCE" or not str(capsule_id).strip() or not str(split_id).strip():
        raise M23OverlayCacheError("validated Phase2 handles are incomplete")
    if not str(receiver).strip() or int(k_shot) <= 0:
        raise M23OverlayCacheError("receiver/K-shot binding is incomplete")
    old_registry = _registry(old_classes, name="old", expected_count={6})
    new_registry = _registry(new_classes, name="new", expected_count={5, 10, 20})
    if set(old_registry) & set(new_registry):
        raise M23OverlayCacheError("old/new class registries overlap")
    if set(scenario_payloads) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise M23OverlayCacheError("overlay must cover exactly the formal LEO-weak scenarios")
    checked_ground = _ground(ground_component, old_registry)
    checked_scenarios = {
        scenario: _scenario(
            scenario_payloads[scenario],
            scenario=scenario,
            old_classes=old_registry,
            new_classes=new_registry,
            k_shot=int(k_shot),
        )
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    token_sets = [set(value["query_tokens"].tolist()) for value in checked_scenarios.values()]
    if any(token_sets[left] & token_sets[right] for left in range(len(token_sets)) for right in range(left + 1, len(token_sets))):
        raise M23OverlayCacheError("query tokens overlap across scenarios")

    bindings = {
        "base_feature_cache_payload_sha256": _hash(
            base_feature_cache_payload_sha256, name="base_feature_cache_payload_sha256"
        ),
        "base_feature_cache_manifest_sha256": _hash(
            base_feature_cache_manifest_sha256, name="base_feature_cache_manifest_sha256"
        ),
        "predictor_package_root_sha256": _hash(
            predictor_package_root_sha256, name="predictor_package_root_sha256"
        ),
        "predictor_package_seal_sha256": _hash(
            predictor_package_seal_sha256, name="predictor_package_seal_sha256"
        ),
        "phase1_bundle_sha256": _hash(phase1_bundle_sha256, name="phase1_bundle_sha256"),
        "phase1_component_manifest_sha256": _hash(
            phase1_component_manifest_sha256, name="phase1_component_manifest_sha256"
        ),
    }
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(M23_OVERLAY_SCHEMA, dtype=np.str_),
        "old_classes": np.asarray(old_registry, dtype=np.str_),
        "new_classes": np.asarray(new_registry, dtype=np.str_),
    }
    for field in _GROUND_FIELDS:
        arrays[f"ground__{field}"] = checked_ground[field]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        for field in _SCENARIO_FIELDS:
            arrays[f"{scenario}__{field}"] = checked_scenarios[scenario][field]
    payload_bytes = _payload_bytes(arrays)
    payload_sha = _sha256(payload_bytes)
    query_counts = {
        scenario: int(len(checked_scenarios[scenario]["query_tokens"]))
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    manifest: dict[str, Any] = {
        "schema": M23_OVERLAY_MANIFEST_SCHEMA,
        "payload_schema": M23_OVERLAY_SCHEMA,
        "phase2_data_status": "VALIDATED_ONCE",
        "protocol_schema": "p2_min_v1",
        "capsule_id": str(capsule_id),
        "split_id": str(split_id),
        "receiver": str(receiver),
        "k_shot": int(k_shot),
        "method_seed": int(method_seed),
        "support_seed": int(support_seed),
        "query_seed": int(query_seed),
        "new_class_draw_seed": int(new_class_draw_seed),
        "old_class_count": len(old_registry),
        "new_class_count": len(new_registry),
        "feature_dim": COMPACT_DIM,
        "feature_block_offsets": [0, 160, 256, 266],
        "rf_quality_classifier_dimension": 0,
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "query_counts": query_counts,
        "query_truth_present": False,
        "query_role_present": False,
        "clean_source_samples_present": False,
        "dense_float_ground_bank_present": False,
        "ground_component_representation": "int8_center_lowrank_residual_direction_only",
        "payload_sha256": payload_sha,
        "payload_bytes": len(payload_bytes),
        "npz_members": list(arrays),
        **bindings,
    }
    manifest_bytes = _canonical_json(manifest)
    payload_target = Path(payload_path)
    manifest_target = Path(manifest_path)
    if payload_target == manifest_target:
        raise M23OverlayCacheError("payload and manifest destinations must differ")
    _write_exclusive_readonly(payload_target, payload_bytes)
    _write_exclusive_readonly(manifest_target, manifest_bytes)
    result = dict(manifest)
    result["manifest_sha256"] = _sha256(manifest_bytes)
    result["manifest_bytes"] = len(manifest_bytes)
    return result


def _read_readonly(path: Path, *, name: str) -> bytes:
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_mode & _WRITE_BITS:
            raise M23OverlayCacheError(f"{name} is not a sealed read-only file")
        return path.read_bytes()
    except OSError as exc:
        raise M23OverlayCacheError(f"{name} could not be read") from exc


def load_m23_overlay_cache(
    payload_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_payload_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Load and revalidate one immutable M2.3 overlay."""

    payload_bytes = _read_readonly(Path(payload_path), name="overlay payload")
    manifest_bytes = _read_readonly(Path(manifest_path), name="overlay manifest")
    if _sha256(payload_bytes) != _hash(expected_payload_sha256, name="expected_payload_sha256"):
        raise M23OverlayCacheError("overlay payload SHA256 mismatch")
    if _sha256(manifest_bytes) != _hash(expected_manifest_sha256, name="expected_manifest_sha256"):
        raise M23OverlayCacheError("overlay manifest SHA256 mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M23OverlayCacheError("overlay manifest is invalid JSON") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != M23_OVERLAY_MANIFEST_SCHEMA
        or manifest.get("payload_schema") != M23_OVERLAY_SCHEMA
        or manifest.get("payload_sha256") != _sha256(payload_bytes)
        or int(manifest.get("payload_bytes", -1)) != len(payload_bytes)
        or manifest.get("query_truth_present") is not False
        or manifest.get("clean_source_samples_present") is not False
    ):
        raise M23OverlayCacheError("overlay manifest binding drift")
    try:
        with zipfile.ZipFile(io.BytesIO(payload_bytes), "r") as archive:
            if archive.testzip() is not None:
                raise M23OverlayCacheError("overlay payload has a CRC failure")
        with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as opened:
            if list(opened.files) != manifest.get("npz_members"):
                raise M23OverlayCacheError("overlay NPZ member order drift")
            arrays = {name: np.array(opened[name], copy=True) for name in opened.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, M23OverlayCacheError):
            raise
        raise M23OverlayCacheError("overlay payload could not be loaded safely") from exc
    schema = np.asarray(arrays.pop("schema"))
    if schema.shape != () or str(schema.item()) != M23_OVERLAY_SCHEMA:
        raise M23OverlayCacheError("overlay payload schema drift")
    old_registry = _registry(arrays.pop("old_classes").astype(str).tolist(), name="old", expected_count={6})
    new_registry = _registry(arrays.pop("new_classes").astype(str).tolist(), name="new", expected_count={5, 10, 20})
    ground_raw = {field: arrays.pop(f"ground__{field}") for field in _GROUND_FIELDS}
    checked_ground = _ground(ground_raw, old_registry)
    checked_scenarios: dict[str, dict[str, np.ndarray]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        raw = {field: arrays.pop(f"{scenario}__{field}") for field in _SCENARIO_FIELDS}
        checked_scenarios[scenario] = _scenario(
            raw,
            scenario=scenario,
            old_classes=old_registry,
            new_classes=new_registry,
            k_shot=int(manifest["k_shot"]),
        )
    if arrays:
        raise M23OverlayCacheError("overlay contains non-allowlisted arrays")
    return {
        "manifest": manifest,
        "old_classes": old_registry,
        "new_classes": new_registry,
        "ground_component": checked_ground,
        "scenario_payloads": checked_scenarios,
    }


__all__ = [
    "M23_OVERLAY_MANIFEST_SCHEMA",
    "M23_OVERLAY_SCHEMA",
    "M23OverlayCacheError",
    "load_m23_overlay_cache",
    "publish_m23_overlay_cache",
]

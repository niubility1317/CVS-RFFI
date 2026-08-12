#!/usr/bin/env python
"""Fail-closed source-only postfreeze CLIC scoring primitives.

This module deliberately contains no trainer, receiver adaptation, target
capsule, or threshold-selection path.  Its only fitted state is the frozen
float64 source-L diagonal Gaussian and the three source-L, single-LEO tail
rules defined in the CLIC design card.  In particular, proxy rows are scored
only; they never enter geometry, quantile, or decision-state fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import export_phase1_clic_features as _clean
import export_phase1_clic_leo_features as _leo
from cvsrffi import phase1_clic as _clic


EXPECTED_TRAINING_RUN_LEAF = "phase1_clic12_20260812_v5"
EXPECTED_POSTFREEZE_MATRIX_ID = "phase1_clic_postfreeze_20260812_v2"
EXPECTED_PAIR_SCHEMA = "cvs.phase1.clic_postfreeze_pair.v1"
EXPECTED_GEOMETRY_SCHEMA = "cvs.phase1.clic_source_geometry.v1"
EXPECTED_POLICY_SCHEMA = "cvs.phase1.clic_source_tail_policy.v1"
EXPECTED_SOURCE_POLICY_STATE_SCHEMA = "cvs.phase1.clic_source_policy_state.v1"
EXPECTED_SCENARIOS = tuple(_clic.FORMAL_LEO_WEAK_SCENARIOS)
LOCAL_CLASS_COUNT = 4
SOURCE_RX_SLOT_COUNT = 7
VARIANCE_DDOF = 1
VARIANCE_SHRINK_CLASS = 0.9
VARIANCE_SHRINK_POOLED = 0.1
VARIANCE_FLOOR = 1.0e-6
HIGHER_Q90 = 0.90
HIGHER_Q95 = 0.95
MIN_TAIL_CELL_ROWS = 20
NONCOMPENSATING_FLOOR_DELTA_PP = -2.0
REQUIRED_FLOORS = (
    "overall_accuracy",
    "min_class_accuracy",
    "min_rx_accuracy",
    "min_day_accuracy",
)


class CLICPostfreezePairError(RuntimeError):
    """Raised when source-only CLIC postfreeze evidence cannot close safely."""


def _canonical_json_bytes(value: Any) -> bytes:
    """Encode only JSON-safe scalar/aggregate state in one deterministic form."""

    def convert(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, np.generic):
            return convert(item.item())
        if isinstance(item, Mapping):
            return {str(key): convert(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(value) for value in item]
        if isinstance(item, float):
            if not math.isfinite(item):
                raise CLICPostfreezePairError("non-finite value cannot enter sealed CLIC state")
            return item
        if isinstance(item, (str, int, bool)) or item is None:
            return item
        raise CLICPostfreezePairError(f"unsupported sealed CLIC state type: {type(item).__name__}")

    try:
        return json.dumps(
            convert(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CLICPostfreezePairError("cannot canonicalize sealed CLIC state") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CLICPostfreezePairError(f"{label} SHA256 is invalid")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CLICPostfreezePairError(f"{label} SHA256 is invalid") from exc
    if value.lower() != value:
        raise CLICPostfreezePairError(f"{label} SHA256 must be lowercase")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise CLICPostfreezePairError(f"{label} must not be boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CLICPostfreezePairError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise CLICPostfreezePairError(f"{label} is non-finite")
    return number


def _as_matrix(values: Any, *, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CLICPostfreezePairError(f"{label} cannot be converted to float64") from exc
    if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise CLICPostfreezePairError(f"{label} must be a nonempty rank-2 matrix")
    if not np.isfinite(matrix).all():
        raise CLICPostfreezePairError(f"{label} contains non-finite values")
    return matrix


def safe_totalized_l2_float64(features: Any, *, label: str = "CLIC features") -> np.ndarray:
    """Use the exact totalized L2 map: nonzero rows normalize, zero rows stay zero."""

    matrix = _as_matrix(features, label=label)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all():
        raise CLICPostfreezePairError(f"{label} has non-finite L2 norms")
    normalized = np.zeros(matrix.shape, dtype=np.float64)
    positive = norms > 0.0
    if np.any(positive):
        normalized[positive] = matrix[positive] / norms[positive, None]
    if not np.isfinite(normalized).all():
        raise CLICPostfreezePairError(f"{label} normalization is non-finite")
    return normalized


normalize_clic_float64 = safe_totalized_l2_float64


def _source_class_order(source_tx_ids: Sequence[str]) -> tuple[str, ...]:
    order = tuple(str(item) for item in source_tx_ids)
    if len(order) != LOCAL_CLASS_COUNT or len(set(order)) != LOCAL_CLASS_COUNT or any(not item for item in order):
        raise CLICPostfreezePairError("CLIC source geometry requires exactly four unique source-L TX classes")
    return order


def _geometry_payload(
    *,
    class_order: Sequence[str],
    class_counts: Mapping[str, int],
    means: np.ndarray,
    raw_variances: np.ndarray,
    pooled_variance: np.ndarray,
    variances: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema": EXPECTED_GEOMETRY_SCHEMA,
        "source_fit_role": "source_L",
        "class_order": list(class_order),
        "class_counts": {str(name): int(class_counts[str(name)]) for name in class_order},
        "feature_dim": int(means.shape[1]),
        "normalization": "float64_totalized_l2_zero_preserved",
        "variance_ddof": VARIANCE_DDOF,
        "variance_shrink_class": VARIANCE_SHRINK_CLASS,
        "variance_shrink_pooled": VARIANCE_SHRINK_POOLED,
        "variance_floor": VARIANCE_FLOOR,
        "means": means.tolist(),
        "raw_ddof1_variances": raw_variances.tolist(),
        "pooled_variance": pooled_variance.tolist(),
        "variances": variances.tolist(),
        "unknown_energy": "log4_minus_logsumexp_negative_full_diagonal_gaussian_nll",
        "fit_rows": int(sum(int(class_counts[name]) for name in class_order)),
        "threshold_fit_rows": 0,
    }


def fit_clic_source_geometry(features: Any, labels: Any, source_tx_ids: Sequence[str]) -> dict[str, Any]:
    """Fit the only learned postfreeze state from clean source-L local4 rows."""

    class_order = _source_class_order(source_tx_ids)
    matrix = safe_totalized_l2_float64(features, label="source-L geometry features")
    raw_labels = np.asarray(labels).reshape(-1)
    if raw_labels.shape[0] != matrix.shape[0]:
        raise CLICPostfreezePairError("source-L geometry labels do not align with source rows")
    normalized_labels = tuple(str(item) for item in raw_labels)
    unknown_labels = sorted(set(normalized_labels).difference(class_order))
    if unknown_labels:
        raise CLICPostfreezePairError("source-L geometry fit received validation, proxy, or non-source rows")
    counts = {name: int(sum(label == name for label in normalized_labels)) for name in class_order}
    if any(count <= VARIANCE_DDOF for count in counts.values()):
        raise CLICPostfreezePairError("each source-L class needs more than one clean fit row")
    means = np.empty((LOCAL_CLASS_COUNT, matrix.shape[1]), dtype=np.float64)
    raw_variances = np.empty_like(means)
    labels_array = np.asarray(normalized_labels, dtype=str)
    for index, name in enumerate(class_order):
        rows = matrix[labels_array == name]
        means[index] = np.mean(rows, axis=0, dtype=np.float64)
        raw_variances[index] = np.var(rows, axis=0, dtype=np.float64, ddof=VARIANCE_DDOF)
    pooled = np.mean(raw_variances, axis=0, dtype=np.float64)
    variances = np.maximum(
        VARIANCE_FLOOR,
        VARIANCE_SHRINK_CLASS * raw_variances + VARIANCE_SHRINK_POOLED * pooled[None, :],
    )
    if not (np.isfinite(means).all() and np.isfinite(raw_variances).all() and np.isfinite(pooled).all() and np.isfinite(variances).all()):
        raise CLICPostfreezePairError("source-L diagonal Gaussian state is non-finite")
    payload = _geometry_payload(
        class_order=class_order,
        class_counts=counts,
        means=means,
        raw_variances=raw_variances,
        pooled_variance=pooled,
        variances=variances,
    )
    payload["state_sha256"] = _canonical_sha256(payload)
    return payload


def _validated_geometry(geometry: Mapping[str, Any]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, str]:
    if not isinstance(geometry, Mapping):
        raise CLICPostfreezePairError("CLIC source geometry must be a mapping")
    expected_fields = {
        "schema", "source_fit_role", "class_order", "class_counts", "feature_dim", "normalization",
        "variance_ddof", "variance_shrink_class", "variance_shrink_pooled", "variance_floor", "means",
        "raw_ddof1_variances", "pooled_variance", "variances", "unknown_energy", "fit_rows",
        "threshold_fit_rows", "state_sha256",
    }
    if set(geometry) != expected_fields:
        raise CLICPostfreezePairError("CLIC geometry state fields drifted")
    if geometry.get("schema") != EXPECTED_GEOMETRY_SCHEMA or geometry.get("source_fit_role") != "source_L":
        raise CLICPostfreezePairError("CLIC geometry is not frozen source-L state")
    class_order = _source_class_order(geometry.get("class_order", ()))
    counts = geometry.get("class_counts")
    if not isinstance(counts, Mapping) or set(str(key) for key in counts) != set(class_order):
        raise CLICPostfreezePairError("CLIC geometry class counts drifted")
    if any(type(counts[name]) is not int or int(counts[name]) <= VARIANCE_DDOF for name in class_order):
        raise CLICPostfreezePairError("CLIC geometry class counts are invalid")
    feature_dim = geometry.get("feature_dim")
    if type(feature_dim) is not int or feature_dim <= 0:
        raise CLICPostfreezePairError("CLIC geometry feature dimension is invalid")
    if geometry.get("normalization") != "float64_totalized_l2_zero_preserved":
        raise CLICPostfreezePairError("CLIC geometry normalization contract drifted")
    for field, expected in (
        ("variance_ddof", VARIANCE_DDOF),
        ("variance_shrink_class", VARIANCE_SHRINK_CLASS),
        ("variance_shrink_pooled", VARIANCE_SHRINK_POOLED),
        ("variance_floor", VARIANCE_FLOOR),
        ("unknown_energy", "log4_minus_logsumexp_negative_full_diagonal_gaussian_nll"),
        ("threshold_fit_rows", 0),
    ):
        if geometry.get(field) != expected:
            raise CLICPostfreezePairError(f"CLIC geometry {field} drifted")
    try:
        means = np.asarray(geometry.get("means"), dtype=np.float64)
        variances = np.asarray(geometry.get("variances"), dtype=np.float64)
        raw_variances = np.asarray(geometry.get("raw_ddof1_variances"), dtype=np.float64)
        pooled = np.asarray(geometry.get("pooled_variance"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CLICPostfreezePairError("CLIC geometry arrays are malformed") from exc
    expected_shape = (LOCAL_CLASS_COUNT, feature_dim)
    if means.shape != expected_shape or variances.shape != expected_shape or raw_variances.shape != expected_shape or pooled.shape != (feature_dim,):
        raise CLICPostfreezePairError("CLIC geometry shape drifted")
    if not (np.isfinite(means).all() and np.isfinite(variances).all() and np.isfinite(raw_variances).all() and np.isfinite(pooled).all()):
        raise CLICPostfreezePairError("CLIC geometry contains non-finite state")
    if np.any(variances < VARIANCE_FLOOR):
        raise CLICPostfreezePairError("CLIC geometry variance floor drifted")
    if np.any(raw_variances < 0.0):
        raise CLICPostfreezePairError("CLIC geometry raw ddof1 variance is negative")
    expected_pooled = np.mean(raw_variances, axis=0, dtype=np.float64)
    expected_variances = np.maximum(
        VARIANCE_FLOOR,
        VARIANCE_SHRINK_CLASS * raw_variances + VARIANCE_SHRINK_POOLED * expected_pooled[None, :],
    )
    if not np.array_equal(pooled, expected_pooled):
        raise CLICPostfreezePairError("CLIC geometry pooled variance does not equal class-equal raw ddof1 mean")
    if not np.array_equal(variances, expected_variances):
        raise CLICPostfreezePairError("CLIC geometry shrink/floor variance does not recompute")
    expected_fit_rows = sum(int(counts[name]) for name in class_order)
    if type(geometry.get("fit_rows")) is not int or int(geometry["fit_rows"]) != expected_fit_rows:
        raise CLICPostfreezePairError("CLIC geometry fit row count does not close")
    payload = dict(geometry)
    state_sha = payload.pop("state_sha256", None)
    _require_sha256(state_sha, label="CLIC geometry state")
    if _canonical_sha256(payload) != state_sha:
        raise CLICPostfreezePairError("CLIC geometry state hash drifted")
    return class_order, means, variances, state_sha


def clic_unknown_energy(features: Any, geometry: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Score the frozen K=4 full-NLL mixture without fitting or thresholding."""

    _, means, variances, _ = _validated_geometry(geometry)
    normalized = safe_totalized_l2_float64(features, label="CLIC scored z_id")
    if normalized.shape[1] != means.shape[1]:
        raise CLICPostfreezePairError("CLIC score feature shape does not match geometry")
    difference = normalized[:, None, :] - means[None, :, :]
    d2 = np.sum(np.square(difference) / variances[None, :, :], axis=2, dtype=np.float64)
    log_terms = np.sum(np.log(2.0 * math.pi * variances), axis=1, dtype=np.float64)
    nll = 0.5 * (d2 + log_terms[None, :])
    neg_nll = -nll
    maximum = np.max(neg_nll, axis=1, keepdims=True)
    logsumexp = maximum[:, 0] + np.log(np.sum(np.exp(neg_nll - maximum), axis=1, dtype=np.float64))
    energy = math.log(float(LOCAL_CLASS_COUNT)) - logsumexp
    if not (np.isfinite(nll).all() and np.isfinite(energy).all()):
        raise CLICPostfreezePairError("CLIC Gaussian NLL or unknown energy is non-finite")
    return energy.astype(np.float64, copy=False), nll.astype(np.float64, copy=False)


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    if values.ndim != 1 or values.size <= 0 or not np.isfinite(values).all():
        raise CLICPostfreezePairError("CLIC tail quantile needs finite nonempty values")
    index = int(math.ceil(probability * values.size) - 1)
    if index < 0 or index >= values.size:
        raise CLICPostfreezePairError("CLIC higher quantile index is invalid")
    return float(np.sort(values, kind="stable")[index])


def _reject_forbidden_binding_fields(value: Any, *, label: str) -> None:
    forbidden_fragments = (
        "target", "proxy", "sample_feature", "sample_logit", "raw_iq", "clean_iq", "receiver_id", "day_id", "physical_id",
    )
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in forbidden_fragments):
                raise CLICPostfreezePairError(f"{label} contains forbidden row-level or non-source field: {key}")
            _reject_forbidden_binding_fields(item, label=label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden_binding_fields(item, label=label)


def _physical_binding_projection(binding: Mapping[str, Any]) -> tuple[str, str]:
    if not isinstance(binding, Mapping):
        raise CLICPostfreezePairError("CLIC LEO physical binding must be a mapping")
    _reject_forbidden_binding_fields(binding, label="CLIC LEO physical binding")
    received = binding.get("received_iq_sha256", binding.get("existing_received_iq_sha256"))
    physical = binding.get("physical_order_sha256")
    received_sha = _require_sha256(received, label="CLIC received-IQ")
    physical_sha = _require_sha256(physical, label="CLIC physical order")
    source_only = binding.get("source_only", binding.get("source_only_export"))
    single = binding.get("single_leo_observation", binding.get("single_leo_forward_bound"))
    if source_only is not True or single is not True:
        raise CLICPostfreezePairError("CLIC tail calibration must be source-only and one existing LEO observation")
    return received_sha, physical_sha


def _policy_rule_payload(*, scene: str, a_s: float, b_s: float) -> dict[str, Any]:
    return {
        "scene": scene,
        "unknown_direction": "higher_is_unknown",
        "thresholds": {"a_s": a_s, "b_s": b_s},
        "decision_priority": [
            "nonfinite_fail_closed",
            "zero_to_defer",
            "exact_head_tie_to_defer",
            "energy_gt_b_unknown",
            "a_lt_energy_lte_b_defer",
            "energy_lte_a_registered_unique_head",
        ],
        "higher_quantile": "ceil(p*n)-1",
        "known_tail_empirical_upper_bounds": {"energy_gt_a_s": 0.10, "energy_gt_b_s": 0.05},
        "defer_rate_not_fixed": True,
    }


def _policy_payload(
    *,
    scene: str,
    geometry_sha: str,
    class_order: Sequence[str],
    feature_dim: int,
    received_sha: str,
    physical_sha: str,
    a_s: float,
    b_s: float,
    min_total: int,
    min_positive: int,
    zero_count: int,
) -> dict[str, Any]:
    rule = _policy_rule_payload(scene=scene, a_s=a_s, b_s=b_s)
    rule_sha = _canonical_sha256(rule)
    return {
        "schema": EXPECTED_POLICY_SCHEMA,
        "scene": scene,
        "geometry_state_sha256": geometry_sha,
        "class_order_sha256": _canonical_sha256(list(class_order)),
        "class_count": LOCAL_CLASS_COUNT,
        "feature_dim": feature_dim,
        "source_rx_slot_count": SOURCE_RX_SLOT_COUNT,
        "cell_count": SOURCE_RX_SLOT_COUNT * LOCAL_CLASS_COUNT,
        "min_cell_total": min_total,
        "min_cell_positive": min_positive,
        "zero_calibration_count": zero_count,
        "tail_calibration_role": "source_L_existing_single_leo_only",
        "source_only": True,
        "single_leo_observation": True,
        "received_iq_sha256": received_sha,
        "physical_order_sha256": physical_sha,
        "higher_quantile": "ceil(p*n)-1",
        "a_s": a_s,
        "b_s": b_s,
        "unknown_direction": "higher_is_unknown",
        "known_tail_empirical_upper_bounds": {"energy_gt_a_s": 0.10, "energy_gt_b_s": 0.05},
        "defer_rate_not_fixed": True,
        "fit_rows": 0,
        "threshold_fit_rows": 0,
        "policy_rule_sha256": rule_sha,
        "rule_sha256": rule_sha,
    }


def freeze_clic_tail_policy(
    geometry: Mapping[str, Any],
    leo_z_id: Any,
    scene_rows: Any,
    rx_slot: Any,
    true_source_class: Any,
    physical_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze one scene's global max higher-q90/q95 rule from source-L LEO only."""

    class_order, means, _, geometry_sha = _validated_geometry(geometry)
    matrix = _as_matrix(leo_z_id, label="source-L LEO z_id")
    if matrix.shape[1] != means.shape[1]:
        raise CLICPostfreezePairError("source-L LEO z_id shape does not match geometry")
    scenes = np.asarray(scene_rows).reshape(-1)
    labels = np.asarray(true_source_class).reshape(-1)
    slots = np.asarray(rx_slot).reshape(-1)
    if not (scenes.size == labels.size == slots.size == matrix.shape[0]):
        raise CLICPostfreezePairError("source-L LEO tail rows do not align")
    scene_values = tuple(str(item) for item in scenes)
    unique_scenes = tuple(sorted(set(scene_values)))
    if len(unique_scenes) != 1 or unique_scenes[0] not in EXPECTED_SCENARIOS:
        raise CLICPostfreezePairError("source-L LEO tail policy needs exactly one formal scene")
    scene = unique_scenes[0]
    try:
        slot_values = np.asarray(slots, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise CLICPostfreezePairError("source-L LEO RX slots are invalid") from exc
    if not np.array_equal(slot_values, slots):
        raise CLICPostfreezePairError("source-L LEO RX slots must be integral")
    if np.any(slot_values < 0) or np.any(slot_values >= SOURCE_RX_SLOT_COUNT):
        raise CLICPostfreezePairError("source-L LEO RX slots must cover only seven frozen slots")
    label_values = np.asarray([str(item) for item in labels], dtype=str)
    if set(label_values).difference(class_order):
        raise CLICPostfreezePairError("source-L LEO tail policy received non-source class rows")
    normalized = safe_totalized_l2_float64(matrix, label="source-L LEO z_id")
    norms = np.linalg.norm(matrix, axis=1)
    positive = norms > 0.0
    energy, _ = clic_unknown_energy(matrix, geometry)
    received_sha, physical_sha = _physical_binding_projection(physical_binding)
    q90_values: list[float] = []
    q95_values: list[float] = []
    total_counts: list[int] = []
    positive_counts: list[int] = []
    for slot in range(SOURCE_RX_SLOT_COUNT):
        for class_name in class_order:
            cell = (slot_values == slot) & (label_values == class_name)
            total = int(np.count_nonzero(cell))
            positive_cell = cell & positive
            positive_count = int(np.count_nonzero(positive_cell))
            if total < MIN_TAIL_CELL_ROWS or positive_count < MIN_TAIL_CELL_ROWS:
                raise CLICPostfreezePairError("source-L LEO tail cell has fewer than 20 total or positive rows")
            total_counts.append(total)
            positive_counts.append(positive_count)
            q90_values.append(_higher_quantile(energy[positive_cell], HIGHER_Q90))
            q95_values.append(_higher_quantile(energy[positive_cell], HIGHER_Q95))
    a_s = max(q90_values)
    b_s = max(q95_values)
    if not (math.isfinite(a_s) and math.isfinite(b_s) and a_s <= b_s):
        raise CLICPostfreezePairError("source-L LEO global tail thresholds are non-finite or unordered")
    payload = _policy_payload(
        scene=scene,
        geometry_sha=geometry_sha,
        class_order=class_order,
        feature_dim=normalized.shape[1],
        received_sha=received_sha,
        physical_sha=physical_sha,
        a_s=float(a_s),
        b_s=float(b_s),
        min_total=min(total_counts),
        min_positive=min(positive_counts),
        zero_count=int(np.count_nonzero(~positive)),
    )
    payload["state_sha256"] = _canonical_sha256(payload)
    return payload


def _validated_policy(policy: Mapping[str, Any], *, geometry: Mapping[str, Any], scene: str) -> tuple[float, float, str]:
    if not isinstance(policy, Mapping) or policy.get("schema") != EXPECTED_POLICY_SCHEMA:
        raise CLICPostfreezePairError("CLIC source tail policy schema is invalid")
    expected_fields = {
        "schema", "scene", "geometry_state_sha256", "class_order_sha256", "class_count", "feature_dim",
        "source_rx_slot_count", "cell_count", "min_cell_total", "min_cell_positive", "zero_calibration_count",
        "tail_calibration_role", "source_only", "single_leo_observation", "received_iq_sha256",
        "physical_order_sha256", "higher_quantile", "a_s", "b_s", "unknown_direction",
        "known_tail_empirical_upper_bounds", "defer_rate_not_fixed", "fit_rows", "threshold_fit_rows",
        "policy_rule_sha256", "rule_sha256", "state_sha256",
    }
    if set(policy) != expected_fields:
        raise CLICPostfreezePairError("CLIC source tail policy state fields drifted")
    if scene not in EXPECTED_SCENARIOS or policy.get("scene") != scene:
        raise CLICPostfreezePairError("CLIC scene does not match source-frozen tail policy")
    class_order, means, _, geometry_sha = _validated_geometry(geometry)
    if policy.get("geometry_state_sha256") != geometry_sha:
        raise CLICPostfreezePairError("CLIC tail policy geometry SHA binding drifted")
    if policy.get("class_order_sha256") != _canonical_sha256(list(class_order)):
        raise CLICPostfreezePairError("CLIC tail policy class order SHA drifted")
    for field, expected in (
        ("class_count", LOCAL_CLASS_COUNT),
        ("feature_dim", means.shape[1]),
        ("source_rx_slot_count", SOURCE_RX_SLOT_COUNT),
        ("cell_count", SOURCE_RX_SLOT_COUNT * LOCAL_CLASS_COUNT),
        ("tail_calibration_role", "source_L_existing_single_leo_only"),
        ("source_only", True),
        ("single_leo_observation", True),
        ("higher_quantile", "ceil(p*n)-1"),
        ("unknown_direction", "higher_is_unknown"),
        ("defer_rate_not_fixed", True),
        ("fit_rows", 0),
        ("threshold_fit_rows", 0),
    ):
        if policy.get(field) != expected:
            raise CLICPostfreezePairError(f"CLIC tail policy {field} drifted")
    for field in ("min_cell_total", "min_cell_positive"):
        if type(policy.get(field)) is not int or int(policy[field]) < MIN_TAIL_CELL_ROWS:
            raise CLICPostfreezePairError(f"CLIC tail policy {field} is below the frozen minimum")
    if type(policy.get("zero_calibration_count")) is not int or int(policy["zero_calibration_count"]) < 0:
        raise CLICPostfreezePairError("CLIC tail policy zero count is invalid")
    _require_sha256(policy.get("received_iq_sha256"), label="CLIC tail policy received-IQ")
    _require_sha256(policy.get("physical_order_sha256"), label="CLIC tail policy physical order")
    a_s = _finite_float(policy.get("a_s"), label="CLIC tail a_s")
    b_s = _finite_float(policy.get("b_s"), label="CLIC tail b_s")
    if a_s > b_s:
        raise CLICPostfreezePairError("CLIC tail policy thresholds are unordered")
    bounds = policy.get("known_tail_empirical_upper_bounds")
    if bounds != {"energy_gt_a_s": 0.10, "energy_gt_b_s": 0.05}:
        raise CLICPostfreezePairError("CLIC tail policy empirical known-tail bounds drifted")
    rule = _policy_rule_payload(scene=scene, a_s=a_s, b_s=b_s)
    rule_sha = _canonical_sha256(rule)
    if policy.get("policy_rule_sha256") != rule_sha or policy.get("rule_sha256") != rule_sha:
        raise CLICPostfreezePairError("CLIC per-scene policy/rule SHA drifted")
    payload = dict(policy)
    state_sha = payload.pop("state_sha256", None)
    _require_sha256(state_sha, label="CLIC tail policy state")
    if _canonical_sha256(payload) != state_sha:
        raise CLICPostfreezePairError("CLIC tail policy state hash drifted")
    return a_s, b_s, rule_sha


def decide_clic_open_set(
    e_unknown: Any,
    tx_logits: Any,
    *,
    a_s: Any,
    b_s: Any,
    zero_flag: bool,
) -> dict[str, Any]:
    """Apply the frozen decision priority without truth, role, fitting, or feedback."""

    energy = _finite_float(e_unknown, label="CLIC unknown energy")
    lower = _finite_float(a_s, label="CLIC a_s")
    upper = _finite_float(b_s, label="CLIC b_s")
    if lower > upper:
        raise CLICPostfreezePairError("CLIC decision thresholds are unordered")
    try:
        logits = np.asarray(tx_logits, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CLICPostfreezePairError("CLIC exact-head logits are invalid") from exc
    if logits.ndim != 1 or logits.size != LOCAL_CLASS_COUNT or not np.isfinite(logits).all():
        raise CLICPostfreezePairError("CLIC exact-head logits are non-finite or shape-invalid")
    if not isinstance(zero_flag, (bool, np.bool_)):
        raise CLICPostfreezePairError("CLIC zero_flag must be a boolean")
    if bool(zero_flag):
        return {"decision": "defer", "predicted_index": None, "zero_flag": True, "head_unique": False}
    maximum = np.max(logits)
    winners = np.flatnonzero(logits == maximum)
    if winners.size != 1:
        return {"decision": "defer", "predicted_index": None, "zero_flag": False, "head_unique": False}
    predicted_index = int(winners[0])
    if energy > upper:
        decision = "unknown"
    elif energy > lower:
        decision = "defer"
    else:
        decision = "registered"
    return {
        "decision": decision,
        "predicted_index": predicted_index if decision == "registered" else None,
        "zero_flag": False,
        "head_unique": True,
    }


def score_clic_open_set(
    geometry: Mapping[str, Any],
    policy: Mapping[str, Any],
    z_id: Any,
    tx_logits: Any,
    scene: str,
    *,
    expected_proxy_count: int | None = None,
) -> dict[str, Any]:
    """Score rows using frozen state only; this function has zero fit/threshold rows."""

    class_order, means, _, geometry_sha = _validated_geometry(geometry)
    a_s, b_s, rule_sha = _validated_policy(policy, geometry=geometry, scene=str(scene))
    matrix = _as_matrix(z_id, label="CLIC runtime z_id")
    if matrix.shape[1] != means.shape[1]:
        raise CLICPostfreezePairError("CLIC runtime z_id shape does not match source geometry")
    try:
        logits = np.asarray(tx_logits, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CLICPostfreezePairError("CLIC runtime exact-head logits are invalid") from exc
    if logits.ndim != 2 or logits.shape != (matrix.shape[0], LOCAL_CLASS_COUNT) or not np.isfinite(logits).all():
        raise CLICPostfreezePairError("CLIC runtime exact-head logits are non-finite or shape-invalid")
    if expected_proxy_count is not None:
        if type(expected_proxy_count) is not int or expected_proxy_count <= 0 or matrix.shape[0] != expected_proxy_count:
            raise CLICPostfreezePairError("fixed400 proxy row count does not match the frozen score-only contract")
    normalized = safe_totalized_l2_float64(matrix, label="CLIC runtime z_id")
    zero_flags = np.linalg.norm(matrix, axis=1) == 0.0
    energy, nll = clic_unknown_energy(matrix, geometry)
    decision_rows = [
        decide_clic_open_set(energy[index], logits[index], a_s=a_s, b_s=b_s, zero_flag=bool(zero_flags[index]))
        for index in range(matrix.shape[0])
    ]
    decisions = np.asarray([row["decision"] for row in decision_rows], dtype=str)
    predicted_indices = np.asarray(
        [-1 if row["predicted_index"] is None else int(row["predicted_index"]) for row in decision_rows], dtype=np.int64
    )
    predicted_classes = np.asarray(
        ["" if index < 0 else class_order[int(index)] for index in predicted_indices], dtype=str
    )
    return {
        "e_unknown": energy,
        "nll": nll,
        "decision": decisions,
        "predicted_index": predicted_indices,
        "predicted_class": predicted_classes,
        "zero_flag": zero_flags.astype(bool, copy=False),
        "scene": str(scene),
        "geometry_state_sha256": geometry_sha,
        "policy_rule_sha256": rule_sha,
        "fit_rows": 0,
        "threshold_fit_rows": 0,
        "normalization": normalized,
    }


def build_clic_source_policy_state(
    *,
    fold_index: int,
    arm: str,
    operator_mode: str,
    geometry: Mapping[str, Any],
    policies: Mapping[str, Any],
    checkpoint_sha256: str,
    terminal_receipt_sha256: str,
) -> dict[str, Any]:
    """Seal one C/G aggregate-only predictor policy state inside its pair row."""

    normalized_arm = str(arm).upper()
    if type(fold_index) is not int or fold_index not in range(1, 7) or normalized_arm not in {"C", "G"}:
        raise CLICPostfreezePairError("CLIC source policy state fold/arm is invalid")
    expected_operator = "raw_phase_control" if normalized_arm == "C" else "complex_local_invariant_curvature"
    if operator_mode != expected_operator:
        raise CLICPostfreezePairError("CLIC source policy state operator does not bind C/G arm")
    _, _, _, geometry_sha = _validated_geometry(geometry)
    if not isinstance(policies, Mapping) or set(str(key) for key in policies) != set(EXPECTED_SCENARIOS):
        raise CLICPostfreezePairError("CLIC source policy state lacks exact three-scene policies")
    normalized_policies: dict[str, Any] = {}
    for scene in EXPECTED_SCENARIOS:
        policy = policies[scene]
        _validated_policy(policy, geometry=geometry, scene=scene)
        normalized_policies[scene] = dict(policy)
    _require_sha256(checkpoint_sha256, label="CLIC source policy checkpoint")
    _require_sha256(terminal_receipt_sha256, label="CLIC source policy terminal receipt")
    payload: dict[str, Any] = {
        "schema": EXPECTED_SOURCE_POLICY_STATE_SCHEMA,
        "fold_index": fold_index,
        "arm": normalized_arm,
        "operator_mode": expected_operator,
        "geometry": dict(geometry),
        "policies": normalized_policies,
        "geometry_state_sha256": geometry_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "terminal_receipt_sha256": terminal_receipt_sha256,
    }
    payload["state_sha256"] = _canonical_sha256(payload)
    return payload


def _validated_clic_source_policy_state(
    state: Mapping[str, Any],
    *,
    fold_index: int | None = None,
    arm: str | None = None,
    checkpoint_sha256: str | None = None,
    terminal_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the exact aggregate-only source policy state, never a model state."""

    if not isinstance(state, Mapping):
        raise CLICPostfreezePairError("CLIC source policy state must be a mapping")
    expected_fields = {
        "schema", "fold_index", "arm", "operator_mode", "geometry", "policies", "geometry_state_sha256",
        "checkpoint_sha256", "terminal_receipt_sha256", "state_sha256",
    }
    if set(state) != expected_fields or state.get("schema") != EXPECTED_SOURCE_POLICY_STATE_SCHEMA:
        raise CLICPostfreezePairError("CLIC source policy state fields/schema drifted")
    observed_fold = state.get("fold_index")
    observed_arm = str(state.get("arm", "")).upper()
    if type(observed_fold) is not int or observed_fold not in range(1, 7) or observed_arm not in {"C", "G"}:
        raise CLICPostfreezePairError("CLIC source policy state fold/arm is invalid")
    if fold_index is not None and observed_fold != fold_index:
        raise CLICPostfreezePairError("CLIC source policy state fold binding drifted")
    if arm is not None and observed_arm != str(arm).upper():
        raise CLICPostfreezePairError("CLIC source policy state arm binding drifted")
    expected_operator = "raw_phase_control" if observed_arm == "C" else "complex_local_invariant_curvature"
    if state.get("operator_mode") != expected_operator:
        raise CLICPostfreezePairError("CLIC source policy state operator binding drifted")
    geometry = state.get("geometry")
    _, _, _, geometry_sha = _validated_geometry(geometry)
    if state.get("geometry_state_sha256") != geometry_sha:
        raise CLICPostfreezePairError("CLIC source policy state geometry SHA drifted")
    policies = state.get("policies")
    if not isinstance(policies, Mapping) or set(str(key) for key in policies) != set(EXPECTED_SCENARIOS):
        raise CLICPostfreezePairError("CLIC source policy state three-scene policy coverage drifted")
    for scene in EXPECTED_SCENARIOS:
        _validated_policy(policies[scene], geometry=geometry, scene=scene)
    for field, expected in (("checkpoint_sha256", checkpoint_sha256), ("terminal_receipt_sha256", terminal_receipt_sha256)):
        observed = _require_sha256(state.get(field), label=f"CLIC source policy {field}")
        if expected is not None and observed != expected:
            raise CLICPostfreezePairError(f"CLIC source policy state {field} binding drifted")
    payload = dict(state)
    supplied_sha = payload.pop("state_sha256")
    _require_sha256(supplied_sha, label="CLIC source policy state")
    if _canonical_sha256(payload) != supplied_sha:
        raise CLICPostfreezePairError("CLIC source policy state hash drifted")
    return dict(state)


def _auroc_unknown(known_energy: Any, proxy_energy: Any) -> float:
    """Exact tie-aware AUROC: larger frozen energy means more unknown."""

    known = np.asarray(known_energy, dtype=np.float64).reshape(-1)
    proxy = np.asarray(proxy_energy, dtype=np.float64).reshape(-1)
    if known.size <= 0 or proxy.size <= 0:
        raise CLICPostfreezePairError("CLIC proxy AUROC needs nonempty source-V and proxy rows")
    if not np.isfinite(known).all() or not np.isfinite(proxy).all():
        raise CLICPostfreezePairError("CLIC proxy AUROC energy is non-finite")
    greater = proxy[:, None] > known[None, :]
    tied = proxy[:, None] == known[None, :]
    value = float(np.mean(greater, dtype=np.float64) + 0.5 * np.mean(tied, dtype=np.float64))
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise CLICPostfreezePairError("CLIC proxy AUROC is invalid")
    return value


def compute_clic_proxy_diagnostic(
    source_l_features: Any,
    source_l_tx_ids: Any,
    source_validation_features: Any,
    proxy_features: Any,
    proxy_tx_ids: Any,
    source_tx_ids: Sequence[str],
) -> dict[str, Any]:
    """Fit source-L geometry once; score only V and the sealed fixed400 proxy.

    This is intentionally a continuous diagnostic.  It does not receive LEO
    rows, logits, decisions, a tail policy, target rows, or any threshold
    selection input.  ``u_gap`` is the proxy-minus-source-V mean frozen energy.
    """

    source_order = _source_class_order(source_tx_ids)
    geometry = fit_clic_source_geometry(source_l_features, source_l_tx_ids, source_order)
    source_v = _as_matrix(source_validation_features, label="CLIC source-V proxy diagnostic z_id")
    proxy = _as_matrix(proxy_features, label="CLIC fixed400 proxy diagnostic z_id")
    if proxy.shape[0] != 400:
        raise CLICPostfreezePairError("CLIC proxy diagnostic requires exactly fixed400 proxy rows")
    proxy_labels = np.asarray(proxy_tx_ids).reshape(-1)
    if proxy_labels.size != proxy.shape[0]:
        raise CLICPostfreezePairError("CLIC proxy TX IDs do not align with fixed400 proxy rows")
    proxy_order = tuple(str(value) for value in proxy_labels)
    if len(set(proxy_order)) != 1 or not proxy_order[0] or proxy_order[0] in set(source_order):
        raise CLICPostfreezePairError("CLIC fixed400 proxy must contain exactly one TX disjoint from source local4")
    if source_v.shape[1] != int(geometry["feature_dim"]) or proxy.shape[1] != int(geometry["feature_dim"]):
        raise CLICPostfreezePairError("CLIC proxy diagnostic feature shape does not match source-L geometry")
    source_v_energy, _ = clic_unknown_energy(source_v, geometry)
    proxy_energy, _ = clic_unknown_energy(proxy, geometry)
    source_v_mean = float(np.mean(source_v_energy, dtype=np.float64))
    proxy_mean = float(np.mean(proxy_energy, dtype=np.float64))
    u_gap = proxy_mean - source_v_mean
    auroc = _auroc_unknown(source_v_energy, proxy_energy)
    if not all(math.isfinite(value) for value in (source_v_mean, proxy_mean, u_gap, auroc)):
        raise CLICPostfreezePairError("CLIC continuous proxy diagnostic is non-finite")
    return {
        "schema": "cvs.phase1.clic_proxy_diagnostic.v1",
        "geometry": geometry,
        "geometry_state_sha256": str(geometry["state_sha256"]),
        "fit": {
            "role": "source_L_only",
            "fit_rows": int(geometry["fit_rows"]),
            "threshold_fit_rows": 0,
            "class_counts": dict(geometry["class_counts"]),
            "feature_dim": int(geometry["feature_dim"]),
            "normalization": "float64_totalized_l2_zero_preserved",
            "variance_ddof": VARIANCE_DDOF,
            "variance_shrink_class": VARIANCE_SHRINK_CLASS,
            "variance_shrink_pooled": VARIANCE_SHRINK_POOLED,
            "variance_floor": VARIANCE_FLOOR,
        },
        "source_validation_known": {
            "role": "source_validation_known",
            "count": int(source_v_energy.size),
            "mean_e_unknown": source_v_mean,
            "min_e_unknown": float(np.min(source_v_energy)),
            "max_e_unknown": float(np.max(source_v_energy)),
            "fit_rows": 0,
            "threshold_fit_rows": 0,
        },
        "proxy_unknown": {
            "role": "proxy_unknown",
            "tx_id": proxy_order[0],
            "count": int(proxy_energy.size),
            "mean_e_unknown": proxy_mean,
            "min_e_unknown": float(np.min(proxy_energy)),
            "max_e_unknown": float(np.max(proxy_energy)),
            "fit_rows": 0,
            "threshold_fit_rows": 0,
        },
        "AUROC_unknown": auroc,
        "u_gap": u_gap,
        "proxy_minus_known_heldout_mean_e_unknown": u_gap,
        "score_rule": "log4_minus_logsumexp_negative_full_diagonal_gaussian_nll",
        "threshold_used": False,
        "tail_policy_used": False,
        "source_validation_fit_rows": 0,
        "proxy_fit_rows": 0,
        "source_validation_threshold_rows": 0,
        "proxy_threshold_rows": 0,
    }


def validate_clic_common_training_binding(c_receipt: Mapping[str, Any], g_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Require same-fold C/G source ordering and common three-scene receipt digests."""

    if not isinstance(c_receipt, Mapping) or not isinstance(g_receipt, Mapping):
        raise CLICPostfreezePairError("C/G common training binding needs two receipt mappings")
    sha_fields = (
        "physical_order_sha256",
        "class_order_sha256",
        "source_split_sha256",
        "common_batch_sequence_sha256",
    )
    required = ("arm", "fold_index", "training_run_root", "scene_order", "physical_row_count", *sha_fields)
    for label, receipt, arm in (("C", c_receipt, "C"), ("G", g_receipt, "G")):
        missing = [field for field in required if field not in receipt]
        if missing:
            raise CLICPostfreezePairError(f"{label} common training binding is missing {missing[0]}")
        if receipt.get("arm") != arm:
            raise CLICPostfreezePairError(f"{label} common training receipt arm drifted")
        if type(receipt.get("fold_index")) is not int or int(receipt["fold_index"]) not in range(1, 7):
            raise CLICPostfreezePairError(f"{label} common training fold is invalid")
        if not isinstance(receipt.get("training_run_root"), str) or not str(receipt["training_run_root"]):
            raise CLICPostfreezePairError(f"{label} common training run root is invalid")
        if tuple(str(item) for item in receipt.get("scene_order", ())) != EXPECTED_SCENARIOS:
            raise CLICPostfreezePairError(f"{label} common training scene order drifted")
        if type(receipt.get("physical_row_count")) is not int or int(receipt["physical_row_count"]) <= 0:
            raise CLICPostfreezePairError(f"{label} common training physical row count is invalid")
        for field in sha_fields:
            _require_sha256(receipt.get(field), label=f"{label} common training {field}")
    common = {field: c_receipt[field] for field in required if field != "arm"}
    for field, expected in common.items():
        if g_receipt.get(field) != expected or type(g_receipt.get(field)) is not type(expected):
            raise CLICPostfreezePairError(f"C/G common physical/order binding differs: {field}")
    return {
        "passed": True,
        "same_fold": True,
        "training_run_root": str(common["training_run_root"]),
        "common_binding_sha256": _canonical_sha256(common),
        "fields": common,
    }


def clic_noncompensating_gates(
    *,
    clean_delta_pp: Mapping[str, Any],
    leo_delta_pp: Mapping[str, Mapping[str, Any]],
    proxy_guard: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the source-only four-floor and fixed-proxy gates without compensation."""

    if not isinstance(clean_delta_pp, Mapping) or not isinstance(leo_delta_pp, Mapping) or not isinstance(proxy_guard, Mapping):
        raise CLICPostfreezePairError("CLIC non-compensating gates need mapping inputs")
    clean: dict[str, float] = {}
    for field in REQUIRED_FLOORS:
        if field not in clean_delta_pp:
            raise CLICPostfreezePairError(f"CLIC clean non-compensating floor is absent: {field}")
        clean[field] = _finite_float(clean_delta_pp[field], label=f"clean delta {field}")
    leo: dict[str, dict[str, float]] = {}
    if set(str(key) for key in leo_delta_pp) != set(EXPECTED_SCENARIOS):
        raise CLICPostfreezePairError("CLIC LEO non-compensating scene coverage drifted")
    for scene in EXPECTED_SCENARIOS:
        values = leo_delta_pp[scene]
        if not isinstance(values, Mapping):
            raise CLICPostfreezePairError(f"CLIC LEO non-compensating metrics are invalid: {scene}")
        leo[scene] = {}
        for field in REQUIRED_FLOORS:
            if field not in values:
                raise CLICPostfreezePairError(f"CLIC LEO non-compensating floor is absent: {scene}.{field}")
            leo[scene][field] = _finite_float(values[field], label=f"LEO delta {scene}.{field}")
    expected_proxy = ("strict_AUROC_improvement", "strict_proxy_known_gap_improvement")
    if any(proxy_guard.get(field) is not True for field in expected_proxy):
        proxy_passed = False
    else:
        proxy_passed = True
    clean_passed = all(value >= NONCOMPENSATING_FLOOR_DELTA_PP for value in clean.values())
    leo_passed = all(
        value >= NONCOMPENSATING_FLOOR_DELTA_PP
        for metrics in leo.values() for value in metrics.values()
    )
    return {
        "non_compensating": bool(clean_passed and leo_passed and proxy_passed),
        "required_floors": list(REQUIRED_FLOORS),
        "floor_delta_limit_pp": NONCOMPENSATING_FLOOR_DELTA_PP,
        "clean_passed": clean_passed,
        "leo_passed": leo_passed,
        "proxy_passed": proxy_passed,
        "clean_delta_pp": clean,
        "leo_delta_pp": leo,
        "proxy_guard": {field: bool(proxy_guard.get(field) is True) for field in expected_proxy},
    }


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise CLICPostfreezePairError(f"{label} raw artifact is missing: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CLICPostfreezePairError(f"{label} raw artifact JSON is invalid") from exc
    if not isinstance(value, dict):
        raise CLICPostfreezePairError(f"{label} raw artifact JSON must be an object")
    return value


def _require_regular_existing(value: Any, *, label: str) -> Path:
    target = Path(str(value)).resolve()
    if not target.is_file():
        raise CLICPostfreezePairError(f"{label} raw artifact is missing or tampered: {target}")
    return target


def _text_rows(values: Any, *, label: str, row_count: int) -> np.ndarray:
    """Read one non-object text column without silently stringifying scalars."""

    array = np.asarray(values)
    if array.dtype.hasobject or array.dtype.kind not in {"U", "S"}:
        raise CLICPostfreezePairError(f"{label} must be a non-object text array")
    flattened = array.reshape(-1)
    if flattened.size != row_count:
        raise CLICPostfreezePairError(f"{label} does not align with feature rows")
    result = np.asarray([str(item) for item in flattened], dtype=str)
    if np.any(result == ""):
        raise CLICPostfreezePairError(f"{label} contains an empty row value")
    return result


def _finite_feature_matrix(values: Any, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.hasobject or array.dtype.kind != "f":
        raise CLICPostfreezePairError(f"{label} must be a floating non-object array")
    try:
        matrix = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CLICPostfreezePairError(f"{label} cannot be represented as float64") from exc
    if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise CLICPostfreezePairError(f"{label} must be a nonempty rank-2 matrix")
    if not np.isfinite(matrix).all():
        raise CLICPostfreezePairError(f"{label} contains non-finite values")
    return matrix


def _manifest_from_npz_member(value: Any, *, label: str) -> dict[str, Any]:
    raw = np.asarray(value)
    if raw.dtype.hasobject or raw.size != 1:
        raise CLICPostfreezePairError(f"{label} manifest_json must be one safe scalar")
    scalar = raw.reshape(-1)[0]
    if isinstance(scalar, bytes):
        try:
            text = scalar.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CLICPostfreezePairError(f"{label} manifest_json is not UTF-8") from exc
    elif isinstance(scalar, str):
        text = scalar
    else:
        raise CLICPostfreezePairError(f"{label} manifest_json must be text")
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CLICPostfreezePairError(f"{label} manifest_json is invalid") from exc
    if not isinstance(manifest, dict):
        raise CLICPostfreezePairError(f"{label} manifest_json must encode an object")
    return manifest


def _load_feature_npz(
    path: str | Path,
    expected_schema: str,
    expected_arm: str,
) -> dict[str, Any]:
    """Strictly materialize a source-only feature NPZ for postfreeze reopening.

    This reader is deliberately shared by clean and existing-LEO artifacts so
    no evaluator branch can accept object arrays, raw IQ, non-finite features,
    mismatched rows, a non-source manifest, or an unsealed role vocabulary.
    It does not fit geometry or thresholds.
    """

    arm = str(expected_arm).upper()
    if arm not in {"C", "G"}:
        raise CLICPostfreezePairError("feature NPZ expected arm must be C or G")
    label = f"{arm} feature"
    common = {
        "z_id", "features", "tx_logits", "raw_labels", "domain_labels", "tx_ids", "rx_ids",
        "day_ids", "eq_ids", "sig_ids", "dataset_role", "channel_views", "sat_scenarios", "manifest_json",
    }
    if expected_schema == _clean.EXPECTED_LV_EXPORT_SCHEMA:
        required = common
        expected_roles = ("labeled_fit", "source_validation_known", "proxy_unknown")
    elif expected_schema == "cvs.phase1.clic_leo_export.v1":
        required = common | {"physical_sample_id", "source_rx_slot"}
        expected_roles = ("source_L_leo_calibration",)
    else:
        raise CLICPostfreezePairError("feature NPZ schema is not a frozen CLIC clean or LEO schema")
    source = _require_regular_existing(path, label=label)
    try:
        with np.load(source, allow_pickle=False) as archive:
            names = set(archive.files)
            if names != required:
                raise CLICPostfreezePairError(f"{label} NPZ exact member allowlist drifted")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except CLICPostfreezePairError:
        raise
    except (OSError, ValueError) as exc:
        raise CLICPostfreezePairError(f"{label} feature NPZ is unreadable or unsafe") from exc
    if any(array.dtype.hasobject for array in arrays.values()):
        raise CLICPostfreezePairError(f"{label} feature NPZ contains an object array")
    z_id = _finite_feature_matrix(arrays["z_id"], label=f"{label} z_id")
    features = _finite_feature_matrix(arrays["features"], label=f"{label} features")
    if features.shape != z_id.shape or not np.array_equal(features, z_id):
        raise CLICPostfreezePairError(f"{label} features/z_id exact binding drifted")
    logits = _finite_feature_matrix(arrays["tx_logits"], label=f"{label} tx_logits")
    if logits.shape != (z_id.shape[0], LOCAL_CLASS_COUNT):
        raise CLICPostfreezePairError(f"{label} exact-head logits do not align with local4 z_id rows")
    for name, array in arrays.items():
        if name == "manifest_json":
            continue
        if array.ndim <= 0 or array.shape[0] != z_id.shape[0]:
            raise CLICPostfreezePairError(f"{label} NPZ member row alignment drifted: {name}")
    tx_ids = _text_rows(arrays["tx_ids"], label=f"{label} TX IDs", row_count=z_id.shape[0])
    roles = _text_rows(arrays["dataset_role"], label=f"{label} dataset roles", row_count=z_id.shape[0])
    allowed_roles = tuple(str(item) for item in expected_roles)
    if not allowed_roles or set(roles).difference(allowed_roles):
        raise CLICPostfreezePairError(f"{label} feature roles drift from the sealed source-only contract")
    if set(roles) != set(allowed_roles):
        raise CLICPostfreezePairError(f"{label} feature NPZ does not cover every required sealed role")
    manifest = _manifest_from_npz_member(arrays["manifest_json"], label=label)
    if manifest.get("schema") != expected_schema or manifest.get("source_only") is not True:
        raise CLICPostfreezePairError(f"{label} feature manifest schema/source-only contract drifted")
    if manifest.get("method") != _clean.EXPECTED_METHOD:
        raise CLICPostfreezePairError(f"{label} feature manifest method drifted")
    candidate = str(manifest.get("candidate_id", ""))
    candidate_match = _clean.EXPECTED_CANDIDATE_PATTERN.fullmatch(candidate)
    if candidate_match is None or candidate_match.group(2) != arm:
        raise CLICPostfreezePairError(f"{label} feature manifest candidate/arm binding drifted")
    if expected_schema == _clean.EXPECTED_LV_EXPORT_SCHEMA:
        if manifest.get("run_id") != _clean.EXPECTED_TRAINING_RUN_ID or manifest.get("training_run_contract") != _clean.EXPECTED_TRAINING_RUN_ID:
            raise CLICPostfreezePairError(f"{label} clean manifest training-run binding drifted")
        if manifest.get("clic_enabled") is not (arm == "G"):
            raise CLICPostfreezePairError(f"{label} clean manifest C/G enabled binding drifted")
        if manifest.get("forwarded_roles") != list(allowed_roles):
            raise CLICPostfreezePairError(f"{label} clean manifest role order drifted")
        for field, expected_count in (
            ("labeled_row_count", int(np.sum(roles == "labeled_fit"))),
            ("source_validation_row_count", int(np.sum(roles == "source_validation_known"))),
            ("proxy_row_count", int(np.sum(roles == "proxy_unknown"))),
        ):
            if type(manifest.get(field)) is not int or int(manifest[field]) != expected_count:
                raise CLICPostfreezePairError(f"{label} clean manifest {field} does not bind rows")
        if manifest.get("proxy_row_count") != 400:
            raise CLICPostfreezePairError(f"{label} clean manifest fixed400/U-zero contract drifted")
        _require_sha256(manifest.get("source_checkpoint_sha256"), label=f"{label} source_checkpoint_sha256")
        _require_sha256(manifest.get("terminal_receipt_sha256"), label=f"{label} terminal_receipt_sha256")
        if manifest.get("unlabeled_loader_constructed") is not False or manifest.get("unlabeled_forward_rows") != 0:
            raise CLICPostfreezePairError(f"{label} clean manifest U-zero contract drifted")
    else:
        fold = int(candidate_match.group(1))
        if manifest.get("arm") != arm or manifest.get("fold_index") != fold:
            raise CLICPostfreezePairError(f"{label} LEO manifest candidate/fold/arm binding drifted")
        if manifest.get("single_leo_observation_required") is not True or manifest.get("single_leo_forward_count") != int(z_id.shape[0]):
            raise CLICPostfreezePairError(f"{label} LEO manifest single-forward row count drifted")
        if tuple(str(item) for item in manifest.get("satellite_scenarios", ())) != EXPECTED_SCENARIOS:
            raise CLICPostfreezePairError(f"{label} LEO manifest formal scene order drifted")
        if manifest.get("fit_rows") != 0 or manifest.get("threshold_fit_rows") != 0:
            raise CLICPostfreezePairError(f"{label} LEO manifest fit/threshold counters drifted")
        for field in ("checkpoint_sha256", "terminal_receipt_sha256", "received_iq_sha256", "physical_order_sha256"):
            _require_sha256(manifest.get(field), label=f"{label} {field}")
        _text_rows(arrays["physical_sample_id"], label=f"{label} physical IDs", row_count=z_id.shape[0])
        scenes = _text_rows(arrays["sat_scenarios"], label=f"{label} LEO scenes", row_count=z_id.shape[0])
        if set(scenes) != set(EXPECTED_SCENARIOS):
            raise CLICPostfreezePairError(f"{label} LEO rows do not cover the formal three scenes")
        slots = np.asarray(arrays["source_rx_slot"])
        if slots.dtype.hasobject or slots.dtype.kind not in {"i", "u"} or slots.reshape(-1).size != z_id.shape[0]:
            raise CLICPostfreezePairError(f"{label} LEO source RX slots are invalid")
        if np.any(slots < 0) or np.any(slots >= SOURCE_RX_SLOT_COUNT):
            raise CLICPostfreezePairError(f"{label} LEO source RX slots drift from 0..6")
    return {
        "path": source,
        "sha256": _sha256_file(source),
        "manifest": manifest,
        "manifest_sha256": _canonical_sha256(manifest),
        "z_id": z_id,
        "tx_logits": logits,
        "tx_ids": tx_ids,
        "roles": roles,
        "row_count": int(z_id.shape[0]),
        "arrays": arrays,
    }


def _load_binding_json(path: str | Path, expected_arm: str) -> dict[str, Any]:
    """Read only a strict source-L / single-LEO aggregate binding."""

    arm = str(expected_arm).upper()
    if arm not in {"C", "G"}:
        raise CLICPostfreezePairError("LEO binding expected arm must be C or G")
    label = f"{arm} LEO binding"
    source = _require_regular_existing(path, label=label)
    binding = _load_json(source, label=label)
    if binding.get("schema") != _leo.EXPECTED_BINDING_SCHEMA:
        raise CLICPostfreezePairError(f"{label} LEO binding schema drifted")
    if binding.get("method") != _clean.EXPECTED_METHOD:
        raise CLICPostfreezePairError(f"{label} LEO binding method drifted")
    candidate = str(binding.get("candidate_id", ""))
    candidate_match = _clean.EXPECTED_CANDIDATE_PATTERN.fullmatch(candidate)
    if candidate_match is None or candidate_match.group(2) != arm or binding.get("arm") != arm:
        raise CLICPostfreezePairError(f"{label} LEO binding candidate/arm drifted")
    if binding.get("fold_index") != int(candidate_match.group(1)):
        raise CLICPostfreezePairError(f"{label} LEO binding fold drifted")
    if binding.get("source_only") is not True or binding.get("single_leo_observation") is not True:
        raise CLICPostfreezePairError(f"{label} LEO binding is not source-only single-LEO")
    if binding.get("single_leo_forward_bound") is not True or binding.get("common_physical_order_bound") is not True:
        raise CLICPostfreezePairError(f"{label} LEO binding forward/order contract drifted")
    if tuple(str(item) for item in binding.get("satellite_scenarios", ())) != EXPECTED_SCENARIOS:
        raise CLICPostfreezePairError(f"{label} LEO binding formal scene order drifted")
    source_tx_ids = binding.get("source_tx_ids")
    if not isinstance(source_tx_ids, list) or len(source_tx_ids) != LOCAL_CLASS_COUNT:
        raise CLICPostfreezePairError(f"{label} LEO binding source TX ordering is invalid")
    _source_class_order(source_tx_ids)
    for field in (
        "checkpoint_sha256", "terminal_receipt_sha256", "received_iq_sha256",
        "physical_order_sha256", "leo_npz_sha256", "leo_manifest_sha256",
    ):
        _require_sha256(binding.get(field), label=f"{label} {field}")
    if binding.get("policy_fit_rows") != 0 or binding.get("threshold_fit_rows") != 0:
        raise CLICPostfreezePairError(f"{label} LEO binding fit/threshold counters drifted")
    npz_path = _require_regular_existing(binding.get("leo_npz_path"), label=f"{label} NPZ")
    if Path(str(binding.get("output_npz_path", ""))).resolve() != npz_path:
        raise CLICPostfreezePairError(f"{label} LEO output NPZ path binding drifted")
    feature = _load_feature_npz(npz_path, "cvs.phase1.clic_leo_export.v1", arm)
    if binding.get("leo_npz_sha256") != feature["sha256"] or binding.get("leo_manifest_sha256") != feature["manifest_sha256"]:
        raise CLICPostfreezePairError(f"{label} LEO binding NPZ/manifest hash drifted")
    manifest = feature["manifest"]
    for field in ("checkpoint_sha256", "terminal_receipt_sha256", "received_iq_sha256", "physical_order_sha256"):
        if binding.get(field) != manifest.get(field):
            raise CLICPostfreezePairError(f"{label} LEO binding/NPZ {field} drifted")
    if binding.get("physical_row_count") != feature["row_count"]:
        raise CLICPostfreezePairError(f"{label} LEO binding physical row count drifted")
    return {
        "path": source,
        "sha256": _sha256_file(source),
        "binding": binding,
        "feature": feature,
    }


def _open_checkpoint_arm(
    *,
    checkpoint_path: str | Path,
    terminal_path: str | Path,
    expected_arm: str,
    fold_index: int,
    training_run_root: str,
    source_tx_ids: Sequence[str],
) -> dict[str, Any]:
    """Strictly reopen one current final checkpoint plus external terminal."""

    arm = str(expected_arm).upper()
    if arm not in {"C", "G"} or fold_index not in range(1, 7):
        raise CLICPostfreezePairError("PAIR checkpoint arm/fold is invalid")
    checkpoint_file = _require_regular_existing(checkpoint_path, label=f"PAIR {arm} checkpoint")
    terminal_file = _require_regular_existing(terminal_path, label=f"PAIR {arm} terminal receipt")
    if checkpoint_file.parent.name != f"F{fold_index}{arm}_CLIC12" or checkpoint_file.parent.parent.name != str(training_run_root):
        raise CLICPostfreezePairError(f"PAIR {arm} checkpoint candidate/training-run path drifted")
    try:
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CLICPostfreezePairError(f"PAIR {arm} checkpoint is unreadable") from exc
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("args"), Mapping):
        raise CLICPostfreezePairError(f"PAIR {arm} checkpoint payload is malformed")
    checkpoint_args = checkpoint["args"]
    try:
        known = _clean._parse_csv(
            checkpoint_args.get("phase1_source_known_validation_tx_ids", ""), label=f"PAIR {arm} held validation TX IDs"
        )
        proxy = _clean._parse_csv(
            checkpoint_args.get("phase1_source_proxy_unknown_tx_ids", ""), label=f"PAIR {arm} proxy TX IDs"
        )
        args, receipt, observed_arm = _clean.validate_clic_training_checkpoint(
            checkpoint,
            checkpoint_path=checkpoint_file,
            terminal_receipt_path=terminal_file,
            source_tx_ids=source_tx_ids,
            known_validation_tx_ids=known,
            proxy_unknown_tx_ids=proxy,
        )
    except _clean.CLICSplitExportError as exc:
        raise CLICPostfreezePairError(f"PAIR {arm} checkpoint/terminal strict reopen failed: {exc}") from exc
    if observed_arm != arm or str(args.get("candidate_id")) != f"F{fold_index}{arm}_CLIC12":
        raise CLICPostfreezePairError(f"PAIR {arm} checkpoint arm/fold receipt drifted")
    if len(known) != 1 or len(proxy) != 1:
        raise CLICPostfreezePairError(f"PAIR {arm} held/proxy TX cardinality drifted")
    return {
        "arm": arm,
        "checkpoint_path": checkpoint_file,
        "terminal_path": terminal_file,
        "checkpoint": checkpoint,
        "checkpoint_args": args,
        "receipt": receipt,
        "checkpoint_sha256": _sha256_file(checkpoint_file),
        "terminal_receipt_sha256": _sha256_file(terminal_file),
        "source_tx_ids": tuple(str(item) for item in source_tx_ids),
        "known_validation_tx_ids": known,
        "proxy_unknown_tx_ids": proxy,
    }


def _clean_masks_for_arm(clean: Mapping[str, Any], *, opened: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Verify C/G clean role identities before geometry or proxy scoring."""

    manifest = clean.get("manifest")
    if not isinstance(manifest, Mapping):
        raise CLICPostfreezePairError("PAIR clean manifest is absent")
    if manifest.get("source_checkpoint_sha256") != opened["checkpoint_sha256"]:
        raise CLICPostfreezePairError("PAIR clean manifest/checkpoint SHA drifted")
    if "terminal_receipt_sha256" in manifest and manifest.get("terminal_receipt_sha256") != opened["terminal_receipt_sha256"]:
        raise CLICPostfreezePairError("PAIR clean manifest/terminal SHA drifted")
    if tuple(str(item) for item in manifest.get("source_tx_ids", ())) != tuple(opened["source_tx_ids"]):
        raise CLICPostfreezePairError("PAIR clean manifest source TX ordering drifted")
    if tuple(str(item) for item in manifest.get("known_validation_tx_ids", ())) != tuple(opened["known_validation_tx_ids"]):
        raise CLICPostfreezePairError("PAIR clean manifest held-validation TX ordering drifted")
    if tuple(str(item) for item in manifest.get("proxy_unknown_tx_ids", ())) != tuple(opened["proxy_unknown_tx_ids"]):
        raise CLICPostfreezePairError("PAIR clean manifest proxy TX ordering drifted")
    roles = np.asarray(clean["roles"], dtype=str).reshape(-1)
    tx_ids = np.asarray(clean["tx_ids"], dtype=str).reshape(-1)
    labeled = roles == "labeled_fit"
    validation = roles == "source_validation_known"
    proxy = roles == "proxy_unknown"
    if not (np.any(labeled) and np.any(validation) and int(np.sum(proxy)) == 400):
        raise CLICPostfreezePairError("PAIR clean source-L/source-V/fixed400 role counts do not close")
    if set(tx_ids[labeled]) != set(opened["source_tx_ids"]):
        raise CLICPostfreezePairError("PAIR clean source-L labels are not exactly local4")
    if any(int(np.sum(tx_ids[labeled] == tx)) <= VARIANCE_DDOF for tx in opened["source_tx_ids"]):
        raise CLICPostfreezePairError("PAIR clean source-L local4 class count is insufficient")
    # The feature-export role ``source_validation_known`` is the held-out V
    # slice of the current local4 source partition.  The one disjoint
    # checkpoint validation TX remains a manifest/terminal audit identity and
    # is intentionally never materialized as postfreeze feature rows.
    if set(tx_ids[validation]) != set(opened["source_tx_ids"]):
        raise CLICPostfreezePairError("PAIR clean source-V local4 labels drifted")
    if set(tx_ids[proxy]) != set(opened["proxy_unknown_tx_ids"]):
        raise CLICPostfreezePairError("PAIR clean fixed400 proxy TX labels drifted")
    return labeled, validation, proxy


def _load_clean_for_proxy_diagnostic(path: str | Path) -> dict[str, Any]:
    """Open a formal clean artifact and derive its C/G arm from candidate ID."""

    source = _require_regular_existing(path, label="CLIC proxy clean feature")
    try:
        with np.load(source, allow_pickle=False) as archive:
            if "manifest_json" not in archive.files:
                raise CLICPostfreezePairError("CLIC proxy clean feature has no manifest")
            manifest = _manifest_from_npz_member(archive["manifest_json"], label="CLIC proxy clean feature")
    except CLICPostfreezePairError:
        raise
    except (OSError, ValueError) as exc:
        raise CLICPostfreezePairError("CLIC proxy clean feature is unreadable or unsafe") from exc
    candidate = str(manifest.get("candidate_id", ""))
    match = _clean.EXPECTED_CANDIDATE_PATTERN.fullmatch(candidate)
    if match is None or match.group(2) not in {"C", "G"}:
        raise CLICPostfreezePairError("CLIC proxy clean feature candidate/arm binding drifted")
    return _load_feature_npz(source, _clean.EXPECTED_LV_EXPORT_SCHEMA, match.group(2))


def _proxy_inputs_from_clean_artifact(
    clean: Mapping[str, Any],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Extract the only legal source-L/V/fixed400 inputs for proxy scoring."""

    manifest = clean.get("manifest")
    if not isinstance(manifest, Mapping):
        raise CLICPostfreezePairError("CLIC proxy clean manifest is absent")
    source_tx_ids = _source_class_order(tuple(str(value) for value in manifest.get("source_tx_ids", ())))
    known_validation_tx_ids = tuple(str(value) for value in manifest.get("known_validation_tx_ids", ()))
    proxy_unknown_tx_ids = tuple(str(value) for value in manifest.get("proxy_unknown_tx_ids", ()))
    if len(known_validation_tx_ids) != 1 or len(proxy_unknown_tx_ids) != 1:
        raise CLICPostfreezePairError("CLIC proxy clean held/proxy TX cardinality drifted")
    if set(source_tx_ids).intersection(known_validation_tx_ids) or set(source_tx_ids).intersection(proxy_unknown_tx_ids):
        raise CLICPostfreezePairError("CLIC proxy clean source/held/proxy TX partition is not disjoint")
    roles = np.asarray(clean.get("roles"), dtype=str).reshape(-1)
    tx_ids = np.asarray(clean.get("tx_ids"), dtype=str).reshape(-1)
    row_count = int(clean.get("row_count", -1))
    if roles.size != row_count or tx_ids.size != row_count:
        raise CLICPostfreezePairError("CLIC proxy clean role/TX row alignment drifted")
    labeled = roles == "labeled_fit"
    validation = roles == "source_validation_known"
    proxy = roles == "proxy_unknown"
    if not (np.any(labeled) and np.any(validation) and int(np.sum(proxy)) == 400):
        raise CLICPostfreezePairError("CLIC proxy clean source-L/source-V/fixed400 rows do not close")
    if set(tx_ids[labeled]) != set(source_tx_ids):
        raise CLICPostfreezePairError("CLIC proxy clean source-L TX labels drifted")
    if any(int(np.sum(tx_ids[labeled] == tx_id)) <= VARIANCE_DDOF for tx_id in source_tx_ids):
        raise CLICPostfreezePairError("CLIC proxy clean source-L class count is insufficient")
    if set(tx_ids[validation]) != set(source_tx_ids):
        raise CLICPostfreezePairError("CLIC proxy clean source-V local4 TX labels drifted")
    if set(tx_ids[proxy]) != set(proxy_unknown_tx_ids):
        raise CLICPostfreezePairError("CLIC proxy clean fixed400 proxy TX labels drifted")
    return source_tx_ids, labeled, validation, proxy


def _atomic_write_proxy_diagnostic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise CLICPostfreezePairError(f"refusing to overwrite immutable CLIC proxy diagnostic: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise CLICPostfreezePairError(f"refusing to overwrite temporary CLIC proxy diagnostic: {temporary}")
    temporary.write_bytes(_canonical_json_bytes(dict(payload)) + b"\n")
    temporary.replace(path)


def export_clic_common_training_receipt(
    checkpoint_path: str | Path,
    terminal_receipt_path: str | Path,
    output_json_path: str | Path,
    *,
    expected_arm: str,
    fold_index: int,
    training_run_root: str,
) -> dict[str, Any]:
    """Project one strict training terminal into the PAIR common binding.

    No caller-supplied row, scene, or digest can enter this artifact.  Every
    value is re-opened from the frozen final checkpoint and its validated
    terminal envelope, then the input bytes are checked again before the
    immutable aggregate-only JSON is sealed.
    """

    arm = str(expected_arm).upper()
    if arm not in {"C", "G"}:
        raise CLICPostfreezePairError("CLIC common receipt arm must be C or G")
    if type(fold_index) is not int or fold_index not in range(1, 7):
        raise CLICPostfreezePairError("CLIC common receipt fold must be F1..F6")
    if str(training_run_root) != EXPECTED_TRAINING_RUN_LEAF:
        raise CLICPostfreezePairError("CLIC common receipt training run drifted")

    checkpoint_file = _require_regular_existing(
        checkpoint_path, label=f"CLIC common {arm} checkpoint"
    )
    terminal_file = _require_regular_existing(
        terminal_receipt_path, label=f"CLIC common {arm} terminal receipt"
    )
    expected_candidate = f"F{fold_index}{arm}_CLIC12"
    if (
        checkpoint_file.name != "final_ssdg.pth"
        or checkpoint_file.parent.name != expected_candidate
        or checkpoint_file.parent.parent.name != EXPECTED_TRAINING_RUN_LEAF
        or terminal_file.parent != checkpoint_file.parent
    ):
        raise CLICPostfreezePairError("CLIC common checkpoint/terminal path binding drifted")

    input_sha_before = {
        "checkpoint": _sha256_file(checkpoint_file),
        "terminal": _sha256_file(terminal_file),
    }
    try:
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CLICPostfreezePairError("CLIC common checkpoint is unreadable") from exc
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("args"), Mapping):
        raise CLICPostfreezePairError("CLIC common checkpoint payload is malformed")
    raw_args = checkpoint["args"]
    try:
        source_tx_ids = _clean._parse_csv(
            raw_args.get("phase1_source_train_tx_ids", ""),
            label="CLIC common source TX IDs",
        )
        known_validation_tx_ids = _clean._parse_csv(
            raw_args.get("phase1_source_known_validation_tx_ids", ""),
            label="CLIC common held TX IDs",
        )
        proxy_unknown_tx_ids = _clean._parse_csv(
            raw_args.get("phase1_source_proxy_unknown_tx_ids", ""),
            label="CLIC common proxy TX IDs",
        )
        args, receipt, observed_arm = _clean.validate_clic_training_checkpoint(
            checkpoint,
            checkpoint_path=checkpoint_file,
            terminal_receipt_path=terminal_file,
            source_tx_ids=source_tx_ids,
            known_validation_tx_ids=known_validation_tx_ids,
            proxy_unknown_tx_ids=proxy_unknown_tx_ids,
        )
    except _clean.CLICSplitExportError as exc:
        raise CLICPostfreezePairError(
            f"CLIC common checkpoint/terminal strict reopen failed: {exc}"
        ) from exc
    if (
        observed_arm != arm
        or args.get("candidate_id") != expected_candidate
        or args.get("run_id") != EXPECTED_TRAINING_RUN_LEAF
    ):
        raise CLICPostfreezePairError("CLIC common arm/fold/run binding drifted")

    physical_count = receipt.get("physical_order_count")
    if type(physical_count) is not int or int(physical_count) <= 0:
        raise CLICPostfreezePairError("CLIC common physical row count is invalid")
    sha_fields = (
        "physical_order_sha256",
        "class_order_sha256",
        "source_split_sha256",
        "common_batch_sequence_sha256",
    )
    for field in sha_fields:
        _require_sha256(receipt.get(field), label=f"CLIC common {field}")
    payload = {
        "arm": arm,
        "fold_index": fold_index,
        "training_run_root": EXPECTED_TRAINING_RUN_LEAF,
        "scene_order": list(EXPECTED_SCENARIOS),
        "physical_row_count": int(physical_count),
        **{field: str(receipt[field]) for field in sha_fields},
        "source_only": True,
    }
    mirror = {**payload, "arm": "G" if arm == "C" else "C"}
    validate_clic_common_training_binding(
        payload if arm == "C" else mirror,
        mirror if arm == "C" else payload,
    )
    input_sha_after = {
        "checkpoint": _sha256_file(checkpoint_file),
        "terminal": _sha256_file(terminal_file),
    }
    if input_sha_after != input_sha_before:
        raise CLICPostfreezePairError("CLIC common input bytes changed during export")
    output = Path(output_json_path).resolve()
    _atomic_write_proxy_diagnostic_json(output, payload)
    return {
        "output_json": str(output),
        "checkpoint_sha256": input_sha_before["checkpoint"],
        "terminal_receipt_sha256": input_sha_before["terminal"],
        "common_binding_sha256": _canonical_sha256(payload),
    }


def export_clic_proxy_diagnostic(
    *,
    clean_npz_path: str | Path,
    output_json_path: str | Path,
) -> dict[str, Any]:
    """Seal a fixed400 source-only continuous proxy diagnostic from clean rows.

    The writer has no model, LEO, target, policy, threshold, or selection
    input.  It records only aggregate geometry and continuous V/proxy scores
    together with the exact clean feature artifact SHA.
    """

    clean = _load_clean_for_proxy_diagnostic(clean_npz_path)
    source_tx_ids, labeled, validation, proxy = _proxy_inputs_from_clean_artifact(clean)
    diagnostic = compute_clic_proxy_diagnostic(
        clean["z_id"][labeled], clean["tx_ids"][labeled], clean["z_id"][validation],
        clean["z_id"][proxy], clean["tx_ids"][proxy], source_tx_ids,
    )
    payload = dict(diagnostic)
    payload["clean_npz_sha256"] = str(clean["sha256"])
    output = Path(output_json_path).resolve()
    _atomic_write_proxy_diagnostic_json(output, payload)
    return {
        "output_json": str(output),
        "clean_npz_sha256": payload["clean_npz_sha256"],
        "geometry_state_sha256": payload["geometry_state_sha256"],
    }


def _leo_for_arm(
    *,
    leo_npz_path: str | Path,
    leo_binding_path: str | Path,
    opened: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cross-bind an existing-LEO NPZ to its source-only physical binding."""

    loaded = _load_binding_json(leo_binding_path, str(opened["arm"]))
    binding = loaded["binding"]
    feature = loaded["feature"]
    if Path(leo_npz_path).resolve() != Path(feature["path"]).resolve():
        raise CLICPostfreezePairError("PAIR LEO NPZ path does not equal LEO binding NPZ path")
    for field in ("checkpoint_sha256", "terminal_receipt_sha256"):
        expected = opened[field]
        if binding.get(field) != expected:
            raise CLICPostfreezePairError(f"PAIR LEO binding {field} drifted")
    if tuple(str(item) for item in binding.get("source_tx_ids", ())) != tuple(opened["source_tx_ids"]):
        raise CLICPostfreezePairError("PAIR LEO binding source TX ordering drifted")
    arrays = feature.get("arrays")
    if not isinstance(arrays, Mapping):
        raise CLICPostfreezePairError("PAIR LEO feature arrays are absent")
    tx_ids = np.asarray(feature["tx_ids"], dtype=str).reshape(-1)
    scenes = _text_rows(arrays["sat_scenarios"], label="PAIR LEO scenes", row_count=feature["row_count"])
    rx_ids = _text_rows(arrays["rx_ids"], label="PAIR LEO RX IDs", row_count=feature["row_count"])
    day_ids = _text_rows(arrays["day_ids"], label="PAIR LEO day IDs", row_count=feature["row_count"])
    physical_ids = _text_rows(arrays["physical_sample_id"], label="PAIR LEO physical IDs", row_count=feature["row_count"])
    slots = np.asarray(arrays["source_rx_slot"])
    if slots.dtype.kind not in {"i", "u"} or slots.reshape(-1).size != feature["row_count"]:
        raise CLICPostfreezePairError("PAIR LEO source RX slots are invalid")
    slots = np.asarray(slots, dtype=np.int64).reshape(-1)
    if np.any(slots < 0) or np.any(slots >= SOURCE_RX_SLOT_COUNT):
        raise CLICPostfreezePairError("PAIR LEO source RX slots drifted")
    if set(tx_ids).difference(opened["source_tx_ids"]) or set(scenes) != set(EXPECTED_SCENARIOS):
        raise CLICPostfreezePairError("PAIR LEO source/scene coverage drifted")
    if not all(str(value) for value in physical_ids):
        raise CLICPostfreezePairError("PAIR LEO physical_sample_id is empty")
    # A physical ID is globally unique.  The scene names only the one frozen
    # observation condition; it must never turn the same physical sample into
    # a distinct calibration row.
    if len(set(physical_ids)) != int(feature["row_count"]):
        raise CLICPostfreezePairError("PAIR LEO physical_sample_id must be globally unique across scenes")
    physical_keys = [
        "|".join((tx_ids[index], rx_ids[index], day_ids[index], physical_ids[index]))
        for index in range(int(feature["row_count"]))
    ]
    if len(physical_keys) != len(set(physical_keys)):
        raise CLICPostfreezePairError("PAIR LEO physical row order contains duplicates")
    if binding.get("physical_keys") != physical_keys or _canonical_sha256(physical_keys) != binding.get("physical_order_sha256"):
        raise CLICPostfreezePairError("PAIR LEO binding physical order does not equal current NPZ rows")
    return feature, {
        "received_iq_sha256": binding["received_iq_sha256"],
        "physical_order_sha256": binding["physical_order_sha256"],
        "source_only": True,
        "single_leo_observation": True,
        "binding_sha256": loaded["sha256"],
        "leo_npz_sha256": feature["sha256"],
        "leo_manifest_sha256": feature["manifest_sha256"],
        "slots": slots,
        "scenes": scenes,
    }


def _load_common_receipt(
    path: str | Path,
    *,
    expected_arm: str,
    fold_index: int,
    training_run_root: str,
    terminal_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _load_json(path, label=f"PAIR {expected_arm} common receipt")
    required_fields = {
        "arm",
        "fold_index",
        "training_run_root",
        "scene_order",
        "physical_row_count",
        "physical_order_sha256",
        "class_order_sha256",
        "source_split_sha256",
        "common_batch_sequence_sha256",
        "source_only",
    }
    if set(receipt) != required_fields:
        raise CLICPostfreezePairError(
            f"PAIR {expected_arm} common receipt exact field/schema drifted"
        )
    if receipt.get("source_only") is not True:
        raise CLICPostfreezePairError(f"PAIR {expected_arm} common receipt source-only drifted")
    if receipt.get("arm") != expected_arm or receipt.get("fold_index") != fold_index or receipt.get("training_run_root") != training_run_root:
        raise CLICPostfreezePairError(f"PAIR {expected_arm} common receipt arm/fold/run drifted")
    for field in ("physical_order_sha256", "class_order_sha256", "source_split_sha256", "common_batch_sequence_sha256"):
        _require_sha256(receipt.get(field), label=f"PAIR {expected_arm} common receipt {field}")
        if terminal_receipt.get(field) != receipt.get(field):
            raise CLICPostfreezePairError(f"PAIR {expected_arm} common receipt/terminal {field} drifted")
    if tuple(str(item) for item in receipt.get("scene_order", ())) != EXPECTED_SCENARIOS:
        raise CLICPostfreezePairError(f"PAIR {expected_arm} common receipt scene order drifted")
    if type(receipt.get("physical_row_count")) is not int or int(receipt["physical_row_count"]) <= 0:
        raise CLICPostfreezePairError(f"PAIR {expected_arm} common receipt physical row count is invalid")
    if receipt["physical_row_count"] != terminal_receipt.get("physical_order_count"):
        raise CLICPostfreezePairError(
            f"PAIR {expected_arm} common receipt/terminal physical row count drifted"
        )
    return receipt


def _load_proxy_diagnostic_declaration(
    path: str | Path,
    *,
    arm: str,
    expected_diagnostic: Mapping[str, Any],
    expected_clean_npz_sha256: str,
) -> dict[str, Any]:
    """Accept only a declaration whose supplied fields equal clean recomputation.

    The formal writer emits the complete payload including the clean artifact
    SHA.  A compact declaration may be read as an untrusted transport record,
    but it is never used as a source of state: every field it supplies must
    equal the canonical source-only recomputation below.  This lets F6 reopen
    legacy sealed paths without letting any hand-written value steer a result.
    """

    source = _require_regular_existing(path, label=f"PAIR {arm} proxy diagnostic")
    declared = _load_json(source, label=f"PAIR {arm} proxy diagnostic")
    expected = dict(expected_diagnostic)
    expected["clean_npz_sha256"] = _require_sha256(
        expected_clean_npz_sha256, label=f"PAIR {arm} clean feature"
    )
    # Sealed PAIR/F6 accepts only the formal writer's complete canonical
    # schema.  A compact subset cannot act as a legacy declaration because it
    # would omit byte/provenance or geometry bindings while still looking
    # numerically plausible.
    if set(declared) != set(expected):
        raise CLICPostfreezePairError(f"PAIR {arm} proxy diagnostic exact field/schema drifted")
    for field in expected:
        if _canonical_json_bytes(declared[field]) != _canonical_json_bytes(expected[field]):
            raise CLICPostfreezePairError(f"PAIR {arm} proxy diagnostic recomputation drifted: {field}")
    return {
        "path": str(source),
        "sha256": _sha256_file(source),
        "payload": dict(declared),
    }


def _derive_arm_postfreeze_state(
    *,
    arm: str,
    checkpoint_path: str | Path,
    terminal_path: str | Path,
    clean_npz_path: str | Path,
    leo_npz_path: str | Path,
    leo_binding_path: str | Path,
    common_receipt_path: str | Path,
    proxy_diagnostic_path: str | Path,
    fold_index: int,
    training_run_root: str,
    source_tx_ids: Sequence[str],
) -> dict[str, Any]:
    opened = _open_checkpoint_arm(
        checkpoint_path=checkpoint_path, terminal_path=terminal_path, expected_arm=arm,
        fold_index=fold_index, training_run_root=training_run_root, source_tx_ids=source_tx_ids,
    )
    clean = _load_feature_npz(clean_npz_path, _clean.EXPECTED_LV_EXPORT_SCHEMA, arm)
    labeled, validation, proxy = _clean_masks_for_arm(clean, opened=opened)
    diagnostic = compute_clic_proxy_diagnostic(
        clean["z_id"][labeled], clean["tx_ids"][labeled], clean["z_id"][validation],
        clean["z_id"][proxy], clean["tx_ids"][proxy], opened["source_tx_ids"],
    )
    leo, physical_binding = _leo_for_arm(
        leo_npz_path=leo_npz_path, leo_binding_path=leo_binding_path, opened=opened
    )
    declared_proxy = _load_proxy_diagnostic_declaration(
        proxy_diagnostic_path,
        arm=arm,
        expected_diagnostic=diagnostic,
        expected_clean_npz_sha256=str(clean["sha256"]),
    )
    policies: dict[str, Any] = {}
    for scene in EXPECTED_SCENARIOS:
        scene_mask = physical_binding["scenes"] == scene
        policies[scene] = freeze_clic_tail_policy(
            diagnostic["geometry"], leo["z_id"][scene_mask], physical_binding["scenes"][scene_mask],
            physical_binding["slots"][scene_mask], leo["tx_ids"][scene_mask], physical_binding,
        )
    common_receipt = _load_common_receipt(
        common_receipt_path, expected_arm=arm, fold_index=fold_index,
        training_run_root=training_run_root, terminal_receipt=opened["receipt"],
    )
    source_policy_state = build_clic_source_policy_state(
        fold_index=fold_index,
        arm=arm,
        operator_mode=str(opened["checkpoint_args"].get("phase1_clic_operator_mode", "")),
        geometry=diagnostic["geometry"],
        policies=policies,
        checkpoint_sha256=opened["checkpoint_sha256"],
        terminal_receipt_sha256=opened["terminal_receipt_sha256"],
    )
    return {
        "opened": opened,
        "clean": clean,
        "leo": leo,
        "physical_binding": physical_binding,
        "common_receipt": common_receipt,
        "proxy_declaration": declared_proxy,
        "proxy_diagnostic": diagnostic,
        "geometry": diagnostic["geometry"],
        "policies": policies,
        "source_policy_state": source_policy_state,
    }


def _atomic_write_pair_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise CLICPostfreezePairError(f"refusing to overwrite immutable PAIR JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise CLICPostfreezePairError(f"refusing to overwrite temporary PAIR JSON: {temporary}")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Reopen one C/G postfreeze pair, fit source-L state, and write no row data."""

    fold_index = int(args.fold_index)
    training_run_root = str(args.training_run_root)
    source_tx_ids = _source_class_order(_clean._parse_csv(args.source_tx_ids, label="PAIR source TX IDs"))
    if fold_index not in range(1, 7) or training_run_root != EXPECTED_TRAINING_RUN_LEAF:
        raise CLICPostfreezePairError("PAIR fold/training-run contract drifted")
    postfreeze_matrix_id = getattr(args, "postfreeze_matrix_id", None)
    if postfreeze_matrix_id is None:
        postfreeze_matrix_id = EXPECTED_POSTFREEZE_MATRIX_ID
    if str(postfreeze_matrix_id) != EXPECTED_POSTFREEZE_MATRIX_ID:
        raise CLICPostfreezePairError("PAIR postfreeze matrix contract drifted")
    expected_scenarios = getattr(args, "expected_scenarios", None)
    if expected_scenarios is None:
        expected_scenarios = ",".join(EXPECTED_SCENARIOS)
    if tuple(str(item) for item in str(expected_scenarios).split(",")) != EXPECTED_SCENARIOS:
        raise CLICPostfreezePairError("PAIR expected formal scene order drifted")
    states = {
        arm: _derive_arm_postfreeze_state(
            arm=arm,
            checkpoint_path=getattr(args, f"{arm.lower()}_checkpoint"),
            terminal_path=getattr(args, f"{arm.lower()}_terminal_receipt_json"),
            clean_npz_path=getattr(args, f"{arm.lower()}_clean_npz"),
            leo_npz_path=getattr(args, f"{arm.lower()}_leo_npz"),
            leo_binding_path=getattr(args, f"{arm.lower()}_leo_binding_json"),
            common_receipt_path=getattr(args, f"{arm.lower()}_common_receipt_json"),
            proxy_diagnostic_path=getattr(args, f"{arm.lower()}_proxy_diagnostic_json"),
            fold_index=fold_index,
            training_run_root=training_run_root,
            source_tx_ids=source_tx_ids,
        )
        for arm in ("C", "G")
    }
    common_binding = validate_clic_common_training_binding(states["C"]["common_receipt"], states["G"]["common_receipt"])
    for field in ("received_iq_sha256", "physical_order_sha256"):
        if states["C"]["physical_binding"][field] != states["G"]["physical_binding"][field]:
            raise CLICPostfreezePairError(f"PAIR C/G single-LEO {field} binding drifted")
    payload = {
        "schema": EXPECTED_PAIR_SCHEMA,
        "postfreeze_matrix_id": EXPECTED_POSTFREEZE_MATRIX_ID,
        "training_run_root": training_run_root,
        "fold_index": fold_index,
        "same_fold": True,
        "source_only": True,
        "target_artifacts_present": False,
        "source_tx_ids": list(source_tx_ids),
        "common_binding": common_binding,
        "geometry": {arm: states[arm]["geometry"] for arm in ("C", "G")},
        "policies": {arm: states[arm]["policies"] for arm in ("C", "G")},
        "clic_source_policy_state": {arm: states[arm]["source_policy_state"] for arm in ("C", "G")},
        "proxy_diagnostic": {arm: states[arm]["proxy_diagnostic"] for arm in ("C", "G")},
        "single_leo_common_binding": {
            "received_iq_sha256": states["C"]["physical_binding"]["received_iq_sha256"],
            "physical_order_sha256": states["C"]["physical_binding"]["physical_order_sha256"],
            "source_only": True,
            "single_leo_observation": True,
        },
        "raw_artifacts": {
            arm: {
                "checkpoint": str(states[arm]["opened"]["checkpoint_path"]),
                "terminal": str(states[arm]["opened"]["terminal_path"]),
                "clean": str(states[arm]["clean"]["path"]),
                "leo": str(states[arm]["leo"]["path"]),
                "leo_binding": str(states[arm]["physical_binding"]["binding_sha256"]),
                "common_receipt": str(states[arm]["common_receipt"] and Path(getattr(args, f"{arm.lower()}_common_receipt_json")).resolve()),
                "proxy_diagnostic": str(states[arm]["proxy_declaration"]["path"]),
                "proxy_diagnostic_sha256": str(states[arm]["proxy_declaration"]["sha256"]),
            }
            for arm in ("C", "G")
        },
    }
    # Replace the internal binding digest with its actual immutable path only
    # after all scientific state is derived; the digest itself remains in the
    # per-arm policy and is never a substitute for reopening its bytes.
    for arm in ("C", "G"):
        payload["raw_artifacts"][arm]["leo_binding"] = str(Path(getattr(args, f"{arm.lower()}_leo_binding_json")).resolve())
    _atomic_write_pair_json(Path(args.output_pair_json).resolve(), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-common-training-receipt",
        action="store_true",
        help="write one immutable source-only common binding from checkpoint+terminal",
    )
    parser.add_argument("--checkpoint", help="final checkpoint for common-receipt export only")
    parser.add_argument(
        "--terminal-receipt-json",
        help="versioned terminal envelope for common-receipt export only",
    )
    parser.add_argument(
        "--output-common-receipt-json",
        help="immutable common-receipt output for common-receipt export only",
    )
    parser.add_argument("--expected-arm", choices=("C", "G"), help="bound arm for common-receipt export")
    parser.add_argument(
        "--export-proxy-diagnostic",
        action="store_true",
        help="write one immutable fixed400 source-only proxy diagnostic from --clean-npz",
    )
    parser.add_argument("--clean-npz", help="formal clean NPZ for --export-proxy-diagnostic only")
    parser.add_argument(
        "--output-proxy-diagnostic-json",
        help="immutable proxy JSON output for --export-proxy-diagnostic only",
    )
    for arm in ("c", "g"):
        parser.add_argument(f"--{arm}-checkpoint")
        parser.add_argument(f"--{arm}-terminal-receipt-json")
        parser.add_argument(f"--{arm}-clean-npz")
        parser.add_argument(f"--{arm}-leo-npz")
        parser.add_argument(f"--{arm}-leo-binding-json")
        parser.add_argument(f"--{arm}-common-receipt-json")
        parser.add_argument(f"--{arm}-proxy-diagnostic-json")
    parser.add_argument("--fold-index", type=int)
    parser.add_argument("--training-run-root")
    parser.add_argument("--source-tx-ids")
    parser.add_argument("--postfreeze-matrix-id")
    parser.add_argument("--expected-scenarios")
    parser.add_argument("--output-pair-json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pair_fields = (
        "c_checkpoint", "g_checkpoint", "c_terminal_receipt_json", "g_terminal_receipt_json",
        "c_clean_npz", "g_clean_npz", "c_leo_npz", "g_leo_npz",
        "c_leo_binding_json", "g_leo_binding_json", "c_common_receipt_json", "g_common_receipt_json",
        "c_proxy_diagnostic_json", "g_proxy_diagnostic_json", "fold_index", "training_run_root",
        "source_tx_ids", "output_pair_json",
    )
    common_fields = (
        "checkpoint", "terminal_receipt_json", "output_common_receipt_json", "expected_arm",
    )
    pair_metadata_fields = ("postfreeze_matrix_id", "expected_scenarios")
    if args.export_common_training_receipt:
        if args.export_proxy_diagnostic:
            parser.error("common-receipt and proxy export modes are mutually exclusive")
        missing_common = [field.replace("_", "-") for field in common_fields if getattr(args, field) is None]
        if missing_common or args.fold_index is None or args.training_run_root is None:
            parser.error(
                "--export-common-training-receipt requires --checkpoint, --terminal-receipt-json, "
                "--output-common-receipt-json, --expected-arm, --fold-index and --training-run-root"
            )
        if any(getattr(args, field) is not None for field in pair_fields if field not in {"fold_index", "training_run_root"}):
            parser.error("common-receipt export is mutually exclusive with PAIR C/G inputs")
        if args.clean_npz is not None or args.output_proxy_diagnostic_json is not None:
            parser.error("common-receipt export is mutually exclusive with proxy inputs")
        if any(getattr(args, field) is not None for field in pair_metadata_fields):
            parser.error("common-receipt export is mutually exclusive with PAIR metadata")
        result = export_clic_common_training_receipt(
            args.checkpoint,
            args.terminal_receipt_json,
            args.output_common_receipt_json,
            expected_arm=args.expected_arm,
            fold_index=args.fold_index,
            training_run_root=args.training_run_root,
        )
    elif args.export_proxy_diagnostic:
        if any(getattr(args, field) is not None for field in common_fields):
            parser.error("proxy export is mutually exclusive with common-receipt inputs")
        if not args.clean_npz or not args.output_proxy_diagnostic_json:
            parser.error("--export-proxy-diagnostic requires --clean-npz and --output-proxy-diagnostic-json")
        if any(getattr(args, field) is not None for field in pair_fields):
            parser.error("--export-proxy-diagnostic is mutually exclusive with PAIR C/G inputs")
        if any(getattr(args, field) is not None for field in pair_metadata_fields):
            parser.error("--export-proxy-diagnostic is mutually exclusive with PAIR metadata")
        result = export_clic_proxy_diagnostic(
            clean_npz_path=args.clean_npz,
            output_json_path=args.output_proxy_diagnostic_json,
        )
    else:
        if any(getattr(args, field) is not None for field in common_fields):
            parser.error("common-receipt inputs require --export-common-training-receipt")
        if args.clean_npz is not None or args.output_proxy_diagnostic_json is not None:
            parser.error("--clean-npz and --output-proxy-diagnostic-json require --export-proxy-diagnostic")
        missing = [field.replace("_", "-") for field in pair_fields if getattr(args, field) is None]
        if missing:
            parser.error("PAIR evaluation is missing required arguments: " + ", ".join(missing))
        result = evaluate(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def reopen_f6_raw_artifacts(
    *,
    prior_pair_metrics: Sequence[str | Path],
    current_fold: int,
    expected_matrix_id: str,
    expected_training_run: str,
    current_checkpoint: str | Path,
    raw_artifacts_by_fold: Mapping[int, Any],
) -> dict[str, Any]:
    """F6 reopens every F1--F5 C/G raw chain instead of trusting summaries.

    Each previous pair must be backed by both strict checkpoint/terminal
    reopenings, current clean and existing-LEO bytes, binding-content closure,
    a source-only continuous proxy recomputation, C/G common binding, and a
    real G bundle state rebuild.  No target or performance artifact is read.
    """

    if current_fold != 6 or expected_matrix_id != EXPECTED_POSTFREEZE_MATRIX_ID or expected_training_run != EXPECTED_TRAINING_RUN_LEAF:
        raise CLICPostfreezePairError("F6 current fold, matrix, or training-run contract drifted")
    current = _require_regular_existing(current_checkpoint, label="F6 current checkpoint")
    if current.parent.name != "F6G_CLIC12" or current.parent.parent.name != expected_training_run:
        raise CLICPostfreezePairError("F6 current G checkpoint candidate/training-run path drifted")
    # Current F6 itself is reopened with the co-located versioned envelope so
    # the raw-reopen gate never relies on a checkpoint filename alone.
    try:
        current_payload = torch.load(current, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CLICPostfreezePairError("F6 current checkpoint is unreadable") from exc
    if not isinstance(current_payload, Mapping) or not isinstance(current_payload.get("args"), Mapping):
        raise CLICPostfreezePairError("F6 current checkpoint payload is malformed")
    try:
        current_source = _clean._parse_csv(current_payload["args"].get("phase1_source_train_tx_ids", ""), label="F6 current source TX IDs")
    except _clean.CLICSplitExportError as exc:
        raise CLICPostfreezePairError("F6 current source TX IDs are invalid") from exc
    _open_checkpoint_arm(
        checkpoint_path=current,
        terminal_path=current.parent / "phase1_clic_terminal_receipt.json",
        expected_arm="G",
        fold_index=6,
        training_run_root=expected_training_run,
        source_tx_ids=current_source,
    )
    if len(prior_pair_metrics) != 5:
        raise CLICPostfreezePairError("F6 must reopen exactly F1--F5 prior raw pair artifacts")

    base_fields = {
        "checkpoint", "terminal", "clean", "leo", "leo_binding", "common_receipt", "proxy_diagnostic",
    }
    records: dict[int, dict[str, Any]] = {}
    reopened: dict[int, dict[str, Any]] = {}
    artifact_hashes: dict[str, dict[str, dict[str, str]]] = {}
    for item in prior_pair_metrics:
        record = _load_json(item, label="F6 prior pair")
        if record.get("schema") != EXPECTED_PAIR_SCHEMA:
            raise CLICPostfreezePairError("F6 prior pair schema drifted")
        fold = record.get("fold_index")
        if type(fold) is not int or fold not in range(1, 6) or fold in records:
            raise CLICPostfreezePairError("F6 prior pair fold identity is invalid")
        if record.get("postfreeze_matrix_id") != expected_matrix_id or record.get("training_run_root") != expected_training_run:
            raise CLICPostfreezePairError("F6 prior pair matrix/training root drifted")
        raw = record.get("raw_artifacts")
        if not isinstance(raw, Mapping) or set(str(key) for key in raw) != {"C", "G"}:
            raise CLICPostfreezePairError("F6 prior pair C/G raw artifact map is incomplete")
        external = raw_artifacts_by_fold.get(fold)
        if not isinstance(external, Mapping) or set(str(key).upper() for key in external) != {"C", "G"}:
            raise CLICPostfreezePairError(f"F6 F{fold} C/G external raw artifact binding is incomplete")
        states: dict[str, Any] = {}
        artifact_hashes[str(fold)] = {}
        for arm in ("C", "G"):
            persisted = raw.get(arm)
            supplied = external.get(arm, external.get(arm.lower()))
            if not isinstance(persisted, Mapping) or not isinstance(supplied, Mapping):
                raise CLICPostfreezePairError(f"F6 F{fold} {arm} raw artifact record is invalid")
            if not base_fields.issubset(set(str(key) for key in persisted)) or not base_fields.issubset(set(str(key) for key in supplied)):
                raise CLICPostfreezePairError(f"F6 F{fold} {arm} raw artifact set is incomplete")
            for field in base_fields:
                recorded_path = _require_regular_existing(persisted.get(field), label=f"F6 F{fold} {arm} recorded {field}")
                supplied_path = _require_regular_existing(supplied.get(field), label=f"F6 F{fold} {arm} supplied {field}")
                if recorded_path != supplied_path:
                    raise CLICPostfreezePairError(f"F6 F{fold} {arm} raw artifact path binding drifted: {field}")
            states[arm] = _derive_arm_postfreeze_state(
                arm=arm,
                checkpoint_path=supplied["checkpoint"],
                terminal_path=supplied["terminal"],
                clean_npz_path=supplied["clean"],
                leo_npz_path=supplied["leo"],
                leo_binding_path=supplied["leo_binding"],
                common_receipt_path=supplied["common_receipt"],
                proxy_diagnostic_path=supplied["proxy_diagnostic"],
                fold_index=fold,
                training_run_root=expected_training_run,
                source_tx_ids=_source_class_order(
                    _clean._parse_csv(
                        torch.load(supplied["checkpoint"], map_location="cpu", weights_only=False)["args"].get("phase1_source_train_tx_ids", ""),
                        label=f"F6 F{fold} {arm} source TX IDs",
                    )
                ),
            )
            artifact_hashes[str(fold)][arm] = {
                field: _sha256_file(_require_regular_existing(supplied[field], label=f"F6 F{fold} {arm} {field}"))
                for field in base_fields
            }
            persisted_proxy_sha = _require_sha256(
                persisted.get("proxy_diagnostic_sha256"),
                label=f"F6 F{fold} {arm} persisted proxy diagnostic",
            )
            supplied_proxy_sha = _require_sha256(
                supplied.get("proxy_diagnostic_sha256"),
                label=f"F6 F{fold} {arm} supplied proxy diagnostic",
            )
            recomputed_proxy_sha = states[arm]["proxy_declaration"]["sha256"]
            if persisted_proxy_sha != recomputed_proxy_sha or supplied_proxy_sha != recomputed_proxy_sha:
                raise CLICPostfreezePairError(f"F6 F{fold} {arm} proxy diagnostic artifact SHA drifted")
        common = validate_clic_common_training_binding(states["C"]["common_receipt"], states["G"]["common_receipt"])
        for field in ("received_iq_sha256", "physical_order_sha256"):
            if states["C"]["physical_binding"][field] != states["G"]["physical_binding"][field]:
                raise CLICPostfreezePairError(f"F6 F{fold} C/G LEO {field} binding drifted")
        persisted_common = record.get("common_binding")
        if not isinstance(persisted_common, Mapping) or persisted_common.get("common_binding_sha256") != common["common_binding_sha256"]:
            raise CLICPostfreezePairError(f"F6 F{fold} persisted C/G common binding does not reopen")
        for field, derived in (
            ("geometry", {arm: states[arm]["geometry"] for arm in ("C", "G")} ),
            ("policies", {arm: states[arm]["policies"] for arm in ("C", "G")} ),
            ("proxy_diagnostic", {arm: states[arm]["proxy_diagnostic"] for arm in ("C", "G")} ),
        ):
            if record.get(field) != derived:
                raise CLICPostfreezePairError(f"F6 F{fold} persisted {field} does not equal raw recomputation")
        persisted_policy_state = record.get("clic_source_policy_state")
        if not isinstance(persisted_policy_state, Mapping) or set(str(key) for key in persisted_policy_state) != {"C", "G"}:
            raise CLICPostfreezePairError(f"F6 F{fold} C/G source policy state is incomplete")
        for arm in ("C", "G"):
            validated_policy_state = _validated_clic_source_policy_state(
                persisted_policy_state[arm],
                fold_index=fold,
                arm=arm,
                checkpoint_sha256=states[arm]["opened"]["checkpoint_sha256"],
                terminal_receipt_sha256=states[arm]["opened"]["terminal_receipt_sha256"],
            )
            if validated_policy_state != states[arm]["source_policy_state"]:
                raise CLICPostfreezePairError(f"F6 F{fold} {arm} source policy state does not equal raw recomputation")
        bundle_path = _require_regular_existing(
            external["G"].get("bundle"), label=f"F6 F{fold} G deployment bundle"
        )
        try:
            import export_phase1_clic_deployment_bundle as _bundle
            verified_bundle = _bundle.verify_clic_bundle(bundle_path)
        except Exception as exc:
            raise CLICPostfreezePairError(f"F6 F{fold} G deployment bundle strict verification failed") from exc
        if (
            verified_bundle.get("state_origin") != "checkpoint_model_exact"
            or verified_bundle.get("real_checkpoint_state_rebuild_verified") is not True
            or verified_bundle.get("real_checkpoint_reload_verified") is not False
            or verified_bundle.get("checkpoint_sha256") != states["G"]["opened"]["checkpoint_sha256"]
            or verified_bundle.get("terminal_receipt_sha256") != states["G"]["opened"]["terminal_receipt_sha256"]
        ):
            raise CLICPostfreezePairError(f"F6 F{fold} G deployment bundle real checkpoint marker/binding drifted")
        rule = verified_bundle.get("source_frozen_unknown_rule")
        if not isinstance(rule, Mapping) or rule.get("geometry_state_sha256") != states["G"]["geometry"]["state_sha256"]:
            raise CLICPostfreezePairError(f"F6 F{fold} G deployment bundle geometry binding drifted")
        if rule.get("per_scene_policies") != states["G"]["policies"]:
            raise CLICPostfreezePairError(f"F6 F{fold} G deployment bundle source policy recomputation drifted")
        if verified_bundle.get("clic_source_policy_state") != states["G"]["source_policy_state"]:
            raise CLICPostfreezePairError(f"F6 F{fold} G deployment bundle pair-sealed source policy state drifted")
        artifact_hashes[str(fold)]["G"]["bundle"] = _sha256_file(bundle_path)
        records[fold] = record
        reopened[fold] = {"common": common, "states": states}
    if set(records) != set(range(1, 6)):
        raise CLICPostfreezePairError("F6 prior pair set does not close F1--F5")
    return {
        "passed": True,
        "current_fold": current_fold,
        "current_checkpoint_sha256": _sha256_file(current),
        "prior_raw_artifact_sha256": artifact_hashes,
        "c_g_checkpoint_sha256": {
            str(fold): {
                arm: reopened[fold]["states"][arm]["opened"]["checkpoint_sha256"] for arm in ("C", "G")
            }
            for fold in range(1, 6)
        },
        "raw_reopen_only": True,
        "target_artifacts_reopened": False,
    }


__all__ = [
    "CLICPostfreezePairError",
    "EXPECTED_GEOMETRY_SCHEMA",
    "EXPECTED_PAIR_SCHEMA",
    "EXPECTED_POLICY_SCHEMA",
    "EXPECTED_SOURCE_POLICY_STATE_SCHEMA",
    "EXPECTED_POSTFREEZE_MATRIX_ID",
    "EXPECTED_SCENARIOS",
    "EXPECTED_TRAINING_RUN_LEAF",
    "build_clic_source_policy_state",
    "build_parser",
    "clic_noncompensating_gates",
    "clic_unknown_energy",
    "compute_clic_proxy_diagnostic",
    "decide_clic_open_set",
    "evaluate",
    "export_clic_common_training_receipt",
    "export_clic_proxy_diagnostic",
    "fit_clic_source_geometry",
    "freeze_clic_tail_policy",
    "main",
    "normalize_clic_float64",
    "reopen_f6_raw_artifacts",
    "safe_totalized_l2_float64",
    "score_clic_open_set",
    "validate_clic_common_training_binding",
]


if __name__ == "__main__":
    raise SystemExit(main())

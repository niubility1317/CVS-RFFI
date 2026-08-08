#!/usr/bin/env python
"""Final-only source-paired C/G diagnostics for frozen P1-CCPC-LEO checkpoints.

This evaluator has no fit, calibration, threshold, or model-selection path.
It only verifies frozen source-clean/source-LEO bindings and reports classifier
and z_id geometry diagnostics for an already-completed C/G fold pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPECTED_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
EXPECTED_SOURCE_DAYS = ("2021_03_01", "2021_03_08")
EXPECTED_SOURCE_RXS = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1")
METADATA_FIELDS = (
    "dataset_role",
    "tx_ids",
    "rx_ids",
    "day_ids",
    "eq_ids",
    "sig_ids",
    "channel_views",
    "sat_scenarios",
)

# File-scoped traceability is intentionally retained here because the frozen
# handoff authorizes exactly this evaluator, its launcher and its focused test.
FROZEN_POSTFREEZE_CONTRACT = {
    "POSTFREEZE-01": "source-only final diagnostics; no fit/calibration/selection",
    "POSTFREEZE-02": "C/G clean/LEO physical and ordered metadata binding",
    "POSTFREEZE-03": "three disjoint LEO scenarios with TX/RX coverage",
    "POSTFREEZE-04": "four classifier floors plus source-clean z_id diagnostics",
    "POSTFREEZE-05": "strict checkpoint reconstruction and same-arm export binding",
}


class PostfreezePairError(RuntimeError):
    """Raised when a frozen postfreeze pair does not close its evidence binding."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_items(value: str | Sequence[str], *, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        out = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        out = tuple(str(item).strip() for item in value if str(item).strip())
    if not out or len(set(out)) != len(out):
        raise PostfreezePairError(f"{field} must contain a non-empty, duplicate-free ordered list")
    return out


def _as_str_array(value: Any, n: int, *, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == ():
        array = np.repeat(array.reshape(1), int(n))
    array = array.reshape(-1)
    if int(array.size) != int(n):
        raise PostfreezePairError(f"{field} row count mismatch: expected={n} observed={array.size}")
    return np.asarray([str(item).strip() for item in array.tolist()], dtype=object)


def _manifest_from_npz(data: Any, path: Path) -> dict[str, Any]:
    if "manifest_json" not in data.files:
        raise PostfreezePairError(f"{path} is missing manifest_json")
    try:
        raw = np.asarray(data["manifest_json"])
        item = raw.item() if raw.shape == () else raw.reshape(-1)[0]
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        manifest = json.loads(str(item))
    except Exception as exc:
        raise PostfreezePairError(f"{path} has invalid manifest_json: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PostfreezePairError(f"{path} manifest_json must encode an object")
    return manifest


def _load_npz(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise PostfreezePairError(f"missing NPZ: {source}")
    required = ("features", "tx_logits", *METADATA_FIELDS)
    with np.load(source, allow_pickle=False) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise PostfreezePairError(f"{source} is missing required arrays: {','.join(missing)}")
        features = np.asarray(data["features"], dtype=np.float32)
        logits = np.asarray(data["tx_logits"], dtype=np.float32)
        if features.ndim != 2 or features.shape[0] <= 0:
            raise PostfreezePairError(f"{source} features must be non-empty rank-2")
        if logits.ndim != 2 or logits.shape[0] != features.shape[0]:
            raise PostfreezePairError(f"{source} tx_logits/features row mismatch")
        if not np.isfinite(features).all() or not np.isfinite(logits).all():
            raise PostfreezePairError(f"{source} contains non-finite features or logits")
        n = int(features.shape[0])
        payload: dict[str, Any] = {
            "path": str(source),
            "features": features,
            "tx_logits": logits,
            "manifest": _manifest_from_npz(data, source),
        }
        for field in METADATA_FIELDS:
            payload[field] = _as_str_array(data[field], n, field=f"{source}:{field}")
    return payload


def _physical_keys(payload: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            "\x1f".join(
                str(payload[field][index])
                for field in ("tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids")
            )
            for index in range(int(payload["features"].shape[0]))
        ],
        dtype=object,
    )


def _require_unique_physical_keys(payload: Mapping[str, Any], *, label: str) -> np.ndarray:
    keys = _physical_keys(payload)
    if len(set(keys.tolist())) != int(keys.size):
        raise PostfreezePairError(f"{label} contains duplicate physical keys")
    return keys


def _checkpoint_sha256_from_manifest(payload: Mapping[str, Any], *, label: str) -> str:
    manifest = payload["manifest"]
    checkpoint_sha256 = str(manifest.get("source_checkpoint_sha256", "")).strip().lower()
    if len(checkpoint_sha256) != 64 or any(char not in "0123456789abcdef" for char in checkpoint_sha256):
        raise PostfreezePairError(f"{label} lacks a valid source checkpoint SHA256")
    if manifest.get("checkpoint_load_strict") is not True:
        raise PostfreezePairError(f"{label} checkpoint export was not strict-loaded")
    audit = manifest.get("checkpoint_load_audit")
    if not isinstance(audit, Mapping) or audit.get("checkpoint_load_strict") is not True:
        raise PostfreezePairError(f"{label} lacks a strict checkpoint-load audit")
    for field in ("missing_keys", "unexpected_keys", "skipped_mismatch"):
        try:
            manifest_count = int(manifest.get(field, -1))
            audit_count = int(audit.get(field, -1))
        except (TypeError, ValueError) as exc:
            raise PostfreezePairError(f"{label} strict checkpoint-load audit has invalid {field}") from exc
        if manifest_count != 0 or audit_count != 0:
            raise PostfreezePairError(f"{label} strict checkpoint-load audit has nonzero {field}")
    return checkpoint_sha256


def _validate_logit_contract(payload: Mapping[str, Any], source_tx_ids: Sequence[str], *, label: str) -> None:
    manifest = payload["manifest"]
    class_order = tuple(str(item).strip() for item in manifest.get("class_id_to_tx", []))
    expected = tuple(source_tx_ids)
    if class_order != expected:
        raise PostfreezePairError(
            f"{label} class label/order mismatch: expected={list(expected)} observed={list(class_order)}"
        )
    logit_order = list(manifest.get("logit_class_order", []))
    if logit_order != list(range(len(expected))):
        raise PostfreezePairError(f"{label} logit class order is not the frozen contiguous source order")
    if int(payload["tx_logits"].shape[1]) != len(expected):
        raise PostfreezePairError(
            f"{label} tx_logits width mismatch: expected={len(expected)} observed={payload['tx_logits'].shape[1]}"
        )
    feature_name = str(manifest.get("feature_name", manifest.get("feature_key", "")))
    if feature_name != "z_id":
        raise PostfreezePairError(f"{label} must export z_id, got {feature_name!r}")
    _checkpoint_sha256_from_manifest(payload, label=label)


def _role_count(payload: Mapping[str, Any], role: str) -> int:
    return int(np.sum(payload["dataset_role"] == str(role)))


def _source_mask(payload: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(payload["dataset_role"] == "source", dtype=bool)


def _validate_clean_payload(
    payload: Mapping[str, Any],
    source_tx_ids: Sequence[str],
    expected_source_count: int,
    expected_target_old_count: int,
    expected_proxy_count: int,
    expected_days: Sequence[str],
    expected_rxs: Sequence[str],
    *,
    label: str,
) -> np.ndarray:
    _validate_logit_contract(payload, source_tx_ids, label=label)
    manifest = payload["manifest"]
    if manifest.get("source_only_export") is not False:
        raise PostfreezePairError(f"{label} clean manifest must not be source-only")
    if str(manifest.get("satellite_tta_policy", "")) != "none":
        raise PostfreezePairError(f"{label} clean manifest must use satellite_tta_policy=none")
    roles = set(payload["dataset_role"].tolist())
    expected_roles = {"source", "target_old", "proxy_unknown"}
    if roles != expected_roles:
        raise PostfreezePairError(f"{label} clean roles mismatch: expected={sorted(expected_roles)} observed={sorted(roles)}")
    required_counts = {
        "source": int(expected_source_count),
        "target_old": int(expected_target_old_count),
        "proxy_unknown": int(expected_proxy_count),
    }
    for role, expected in required_counts.items():
        observed = _role_count(payload, role)
        if observed != expected:
            raise PostfreezePairError(f"{label} clean role count mismatch for {role}: expected={expected} observed={observed}")
    if set(payload["channel_views"].tolist()) != {"clean"}:
        raise PostfreezePairError(f"{label} clean payload has a non-clean channel view")
    if any(str(item) for item in payload["sat_scenarios"].tolist()):
        raise PostfreezePairError(f"{label} clean payload must not assign satellite scenarios")
    source_mask = _source_mask(payload)
    if set(payload["tx_ids"][source_mask].tolist()) != set(source_tx_ids):
        raise PostfreezePairError(f"{label} clean source TX set mismatch")
    if set(payload["day_ids"][source_mask].tolist()) != set(expected_days):
        raise PostfreezePairError(f"{label} clean source day set mismatch")
    if set(payload["rx_ids"][source_mask].tolist()) != set(expected_rxs):
        raise PostfreezePairError(f"{label} clean source RX set mismatch")
    keys = _require_unique_physical_keys(payload, label=label)
    return keys[source_mask]


def _validate_leo_payload(
    payload: Mapping[str, Any],
    source_tx_ids: Sequence[str],
    expected_source_count: int,
    expected_scenarios: Sequence[str],
    expected_days: Sequence[str],
    expected_rxs: Sequence[str],
    source_sat_seed: int,
    *,
    label: str,
) -> np.ndarray:
    _validate_logit_contract(payload, source_tx_ids, label=label)
    if set(payload["dataset_role"].tolist()) != {"source"}:
        raise PostfreezePairError(f"{label} LEO payload must contain source rows only")
    if int(payload["features"].shape[0]) != int(expected_source_count):
        raise PostfreezePairError(
            f"{label} LEO source row count mismatch: expected={expected_source_count} observed={payload['features'].shape[0]}"
        )
    if any(item == "clean" or not item for item in payload["channel_views"].tolist()):
        raise PostfreezePairError(f"{label} LEO payload has invalid clean/empty channel views")
    scenarios = payload["sat_scenarios"]
    if set(scenarios.tolist()) != set(expected_scenarios):
        raise PostfreezePairError(f"{label} LEO scenario set mismatch")
    if set(payload["tx_ids"].tolist()) != set(source_tx_ids):
        raise PostfreezePairError(f"{label} LEO source TX set mismatch")
    if set(payload["day_ids"].tolist()) != set(expected_days):
        raise PostfreezePairError(f"{label} LEO source day set mismatch")
    if set(payload["rx_ids"].tolist()) != set(expected_rxs):
        raise PostfreezePairError(f"{label} LEO source RX set mismatch")
    manifest = payload["manifest"]
    if manifest.get("source_only_export") is not True:
        raise PostfreezePairError(f"{label} LEO manifest is not source-only")
    if str(manifest.get("star_ground_channel_impl", "")) != "simplified_leo_residual":
        raise PostfreezePairError(f"{label} LEO manifest does not use simplified_leo_residual")
    if str(manifest.get("satellite_tta_policy", "")) != "none":
        raise PostfreezePairError(f"{label} LEO manifest must use satellite_tta_policy=none")
    profile = manifest.get("channel_profile", {}).get("source", {})
    if not isinstance(profile, Mapping) or str(profile.get("view", "")) != "satellite":
        raise PostfreezePairError(f"{label} LEO source channel profile is not satellite")
    if tuple(str(item) for item in profile.get("scenarios", [])) != tuple(expected_scenarios):
        raise PostfreezePairError(f"{label} LEO source channel scenarios are not frozen")
    if int(profile.get("sat_seed", -1)) != int(source_sat_seed):
        raise PostfreezePairError(f"{label} LEO source satellite seed mismatch")
    keys = _require_unique_physical_keys(payload, label=label)
    physical_by_scenario: dict[str, set[str]] = {}
    for scenario in expected_scenarios:
        mask = np.asarray(scenarios == scenario, dtype=bool)
        if set(payload["tx_ids"][mask].tolist()) != set(source_tx_ids):
            raise PostfreezePairError(f"{label} scenario {scenario} lacks full source TX coverage")
        if set(payload["rx_ids"][mask].tolist()) != set(expected_rxs):
            raise PostfreezePairError(f"{label} scenario {scenario} lacks full source RX coverage")
        physical_by_scenario[scenario] = set(keys[mask].tolist())
    for left_index, left in enumerate(expected_scenarios):
        for right in expected_scenarios[left_index + 1 :]:
            if physical_by_scenario[left] & physical_by_scenario[right]:
                raise PostfreezePairError(f"{label} scenarios {left}/{right} reuse physical keys")
    return keys


def _assert_pair_metadata(c_payload: Mapping[str, Any], g_payload: Mapping[str, Any], *, label: str) -> None:
    c_n = int(c_payload["features"].shape[0])
    g_n = int(g_payload["features"].shape[0])
    if c_n != g_n:
        raise PostfreezePairError(f"C/G {label} metadata row count mismatch")
    if int(c_payload["features"].shape[1]) != int(g_payload["features"].shape[1]):
        raise PostfreezePairError(f"C/G {label} z_id dimension mismatch")
    for field in METADATA_FIELDS:
        if not np.array_equal(c_payload[field], g_payload[field]):
            raise PostfreezePairError(f"C/G {label} metadata/scenario mismatch in {field}")


def _rate(values: np.ndarray) -> float:
    if int(values.size) <= 0:
        raise PostfreezePairError("cannot compute a rate on zero rows")
    return float(np.mean(values.astype(np.float64)))


def _min_group_accuracy(groups: np.ndarray, correct: np.ndarray) -> float:
    rates = [_rate(correct[groups == group]) for group in sorted(set(groups.tolist()))]
    if not rates:
        raise PostfreezePairError("cannot compute a group floor on zero groups")
    return float(min(rates))


def _predicted_tx(payload: Mapping[str, Any], source_tx_ids: Sequence[str]) -> np.ndarray:
    indices = np.asarray(payload["tx_logits"].argmax(axis=1), dtype=np.int64)
    return np.asarray([source_tx_ids[int(index)] for index in indices.tolist()], dtype=object)


def _classification_summary(payload: Mapping[str, Any], mask: np.ndarray, source_tx_ids: Sequence[str]) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    predicted = _predicted_tx(payload, source_tx_ids)
    correct = np.asarray(predicted == payload["tx_ids"], dtype=bool)
    tx = payload["tx_ids"][mask]
    rx = payload["rx_ids"][mask]
    day = payload["day_ids"][mask]
    selected = correct[mask]
    return {
        "count": int(mask.sum()),
        "overall_accuracy": _rate(selected),
        "min_class_accuracy": _min_group_accuracy(tx, selected),
        "min_rx_accuracy": _min_group_accuracy(rx, selected),
        "min_day_accuracy": _min_group_accuracy(day, selected),
    }


def _delta_pp(c_metrics: Mapping[str, Any], g_metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        metric: 100.0 * (float(g_metrics[metric]) - float(c_metrics[metric]))
        for metric in ("overall_accuracy", "min_class_accuracy", "min_rx_accuracy", "min_day_accuracy")
    }


def _normalize_rows(features: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
        raise PostfreezePairError(f"{label} contains a zero/non-finite z_id row")
    return values / norms


def _geometry_summary(
    clean: Mapping[str, Any],
    leo: Mapping[str, Any],
    leo_mask: np.ndarray,
    source_tx_ids: Sequence[str],
) -> dict[str, float]:
    clean_mask = _source_mask(clean)
    clean_keys = _physical_keys(clean)[clean_mask]
    clean_z = _normalize_rows(clean["features"][clean_mask], label="clean source")
    key_to_clean = {key: clean_z[index] for index, key in enumerate(clean_keys.tolist())}
    if len(key_to_clean) != int(clean_keys.size):
        raise PostfreezePairError("clean source geometry bank has duplicate physical keys")
    clean_tx = clean["tx_ids"][clean_mask]
    centroids: dict[str, np.ndarray] = {}
    for tx in source_tx_ids:
        rows = clean_z[clean_tx == tx]
        if int(rows.shape[0]) <= 0:
            raise PostfreezePairError(f"clean source geometry bank lacks class {tx}")
        centroid = rows.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            raise PostfreezePairError(f"clean source geometry centroid is invalid for class {tx}")
        centroids[tx] = centroid / norm
    leo_mask = np.asarray(leo_mask, dtype=bool)
    leo_z = _normalize_rows(leo["features"][leo_mask], label="LEO source")
    leo_keys = _physical_keys(leo)[leo_mask]
    leo_tx = leo["tx_ids"][leo_mask]
    paired_distance: list[float] = []
    margins: list[float] = []
    for z, key, tx in zip(leo_z, leo_keys.tolist(), leo_tx.tolist()):
        if key not in key_to_clean:
            raise PostfreezePairError("LEO geometry row is not bound to a clean source physical key")
        paired_distance.append(float(1.0 - np.clip(np.dot(z, key_to_clean[key]), -1.0, 1.0)))
        own_distance = 1.0 - float(np.clip(np.dot(z, centroids[str(tx)]), -1.0, 1.0))
        other_distance = min(
            1.0 - float(np.clip(np.dot(z, centroids[other]), -1.0, 1.0))
            for other in source_tx_ids
            if other != str(tx)
        )
        margins.append(float(other_distance - own_distance))
    if not paired_distance or not margins:
        raise PostfreezePairError("geometry summary has zero selected LEO rows")
    return {
        "paired_clean_leo_cosine_distance_mean": float(np.mean(paired_distance)),
        "nearest_other_class_centroid_margin_mean": float(np.mean(margins)),
    }


def _geometry_delta(c_metrics: Mapping[str, Any], g_metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        metric: float(g_metrics[metric]) - float(c_metrics[metric])
        for metric in (
            "paired_clean_leo_cosine_distance_mean",
            "nearest_other_class_centroid_margin_mean",
        )
    }


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise PostfreezePairError(f"refusing to overwrite final-only pair output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        raise PostfreezePairError(f"refusing to overwrite temporary pair output: {temporary}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    source_tx_ids = _parse_items(args.source_tx_ids, field="source_tx_ids")
    expected_scenarios = _parse_items(args.expected_scenarios, field="expected_scenarios")
    if expected_scenarios != EXPECTED_SCENARIOS:
        raise PostfreezePairError("expected_scenarios are frozen to the three leo_*_weak scenarios")
    expected_days = _parse_items(args.expected_source_days, field="expected_source_days")
    expected_rxs = _parse_items(args.expected_source_rxs, field="expected_source_rxs")
    if expected_days != EXPECTED_SOURCE_DAYS:
        raise PostfreezePairError("expected_source_days do not match the frozen WRC LEO v2 slice")
    if expected_rxs != EXPECTED_SOURCE_RXS:
        raise PostfreezePairError("expected_source_rxs do not match the frozen WRC LEO v2 slice")
    expected_source_count = int(args.expected_source_count)
    expected_target_old_count = int(args.expected_target_old_count)
    expected_proxy_count = int(args.expected_proxy_count)
    if min(expected_source_count, expected_target_old_count, expected_proxy_count) <= 0:
        raise PostfreezePairError("expected role counts must be positive")

    c_clean = _load_npz(args.c_clean_npz)
    g_clean = _load_npz(args.g_clean_npz)
    c_leo = _load_npz(args.c_leo_npz)
    g_leo = _load_npz(args.g_leo_npz)
    _assert_pair_metadata(c_clean, g_clean, label="clean")
    _assert_pair_metadata(c_leo, g_leo, label="LEO")

    c_clean_keys = _validate_clean_payload(
        c_clean,
        source_tx_ids,
        expected_source_count,
        expected_target_old_count,
        expected_proxy_count,
        expected_days,
        expected_rxs,
        label="C clean",
    )
    g_clean_keys = _validate_clean_payload(
        g_clean,
        source_tx_ids,
        expected_source_count,
        expected_target_old_count,
        expected_proxy_count,
        expected_days,
        expected_rxs,
        label="G clean",
    )
    c_leo_keys = _validate_leo_payload(
        c_leo,
        source_tx_ids,
        expected_source_count,
        expected_scenarios,
        expected_days,
        expected_rxs,
        int(args.source_sat_seed),
        label="C LEO",
    )
    g_leo_keys = _validate_leo_payload(
        g_leo,
        source_tx_ids,
        expected_source_count,
        expected_scenarios,
        expected_days,
        expected_rxs,
        int(args.source_sat_seed),
        label="G LEO",
    )
    if set(c_clean_keys.tolist()) != set(c_leo_keys.tolist()):
        raise PostfreezePairError("C clean/LEO source physical key sets differ")
    if set(g_clean_keys.tolist()) != set(g_leo_keys.tolist()):
        raise PostfreezePairError("G clean/LEO source physical key sets differ")
    if set(c_clean_keys.tolist()) != set(g_clean_keys.tolist()):
        raise PostfreezePairError("C/G clean source physical key sets differ")
    if set(c_leo_keys.tolist()) != set(g_leo_keys.tolist()):
        raise PostfreezePairError("C/G LEO source physical key sets differ")
    c_checkpoint_sha256 = _checkpoint_sha256_from_manifest(c_clean, label="C clean")
    g_checkpoint_sha256 = _checkpoint_sha256_from_manifest(g_clean, label="G clean")
    if c_checkpoint_sha256 != _checkpoint_sha256_from_manifest(c_leo, label="C LEO"):
        raise PostfreezePairError("C clean/LEO source checkpoint SHA256 differs")
    if g_checkpoint_sha256 != _checkpoint_sha256_from_manifest(g_leo, label="G LEO"):
        raise PostfreezePairError("G clean/LEO source checkpoint SHA256 differs")

    c_clean_summary = _classification_summary(c_clean, _source_mask(c_clean), source_tx_ids)
    g_clean_summary = _classification_summary(g_clean, _source_mask(g_clean), source_tx_ids)
    scenario_metrics: dict[str, Any] = {}
    for scenario in expected_scenarios:
        c_mask = np.asarray(c_leo["sat_scenarios"] == scenario, dtype=bool)
        g_mask = np.asarray(g_leo["sat_scenarios"] == scenario, dtype=bool)
        c_summary = _classification_summary(c_leo, c_mask, source_tx_ids)
        g_summary = _classification_summary(g_leo, g_mask, source_tx_ids)
        c_geometry = _geometry_summary(c_clean, c_leo, c_mask, source_tx_ids)
        g_geometry = _geometry_summary(g_clean, g_leo, g_mask, source_tx_ids)
        scenario_metrics[scenario] = {
            "C": c_summary,
            "G": g_summary,
            "G_minus_C_pp": _delta_pp(c_summary, g_summary),
            "geometry_diagnostic_source_clean_only": {
                "C": c_geometry,
                "G": g_geometry,
                "G_minus_C": _geometry_delta(c_geometry, g_geometry),
            },
        }
    metrics = {
        "schema": "cvs.phase1.ccpc_leo_postfreeze_pair.v1",
        "candidate_pair": str(args.candidate_pair),
        "evidence_boundary": "PHASE1_SOURCE_ONLY_FINAL_ONLY_DIAGNOSTIC",
        "frozen_contract": dict(FROZEN_POSTFREEZE_CONTRACT),
        "policy": {
            "fit_performed": False,
            "calibration_performed": False,
            "threshold_used": False,
            "model_selection_performed": False,
            "proxy_rows_used_for_pair_metrics": 0,
            "target_old_rows_used_for_pair_metrics": 0,
            "proxy_or_target_rows_used_for_geometry_bank": 0,
            "geometry_bank": "source_clean_z_id_only",
        },
        "source_tx_ids": list(source_tx_ids),
        "expected_source_days": list(expected_days),
        "expected_source_rxs": list(expected_rxs),
        "expected_role_counts": {
            "source": expected_source_count,
            "target_old": expected_target_old_count,
            "proxy_unknown": expected_proxy_count,
        },
        "expected_scenarios": list(expected_scenarios),
        "source_sat_seed": int(args.source_sat_seed),
        "bindings": {
            "c_clean_npz_sha256": _sha256_file(args.c_clean_npz),
            "g_clean_npz_sha256": _sha256_file(args.g_clean_npz),
            "c_leo_npz_sha256": _sha256_file(args.c_leo_npz),
            "g_leo_npz_sha256": _sha256_file(args.g_leo_npz),
            "c_source_checkpoint_sha256": c_checkpoint_sha256,
            "g_source_checkpoint_sha256": g_checkpoint_sha256,
        },
        "clean_source": {
            "C": c_clean_summary,
            "G": g_clean_summary,
            "G_minus_C_pp": _delta_pp(c_clean_summary, g_clean_summary),
        },
        "leo_scenarios": scenario_metrics,
    }
    _atomic_write_json(args.output_metrics_json, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-clean-npz", required=True)
    parser.add_argument("--g-clean-npz", required=True)
    parser.add_argument("--c-leo-npz", required=True)
    parser.add_argument("--g-leo-npz", required=True)
    parser.add_argument("--source-tx-ids", required=True)
    parser.add_argument("--candidate-pair", required=True)
    parser.add_argument("--expected-scenarios", default=",".join(EXPECTED_SCENARIOS))
    parser.add_argument("--expected-source-days", default=",".join(EXPECTED_SOURCE_DAYS))
    parser.add_argument("--expected-source-rxs", default=",".join(EXPECTED_SOURCE_RXS))
    parser.add_argument("--source-sat-seed", type=int, default=7281718)
    parser.add_argument("--expected-source-count", type=int, default=1600)
    parser.add_argument("--expected-target-old-count", type=int, default=400)
    parser.add_argument("--expected-proxy-count", type=int, default=400)
    parser.add_argument("--output-metrics-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    metrics = evaluate(build_parser().parse_args(argv))
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Final-only source-paired C/G closure for frozen P1-CB-SFCE checkpoints.

The evaluator only consumes immutable clean/LEO feature exports and the fixed
source-calibrated proxy diagnostic JSONs already produced by the postfreeze
launcher.  It never loads checkpoint weights, trains, fits, calibrates,
thresholds, sweeps, or selects a model.
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
# ``channel_views`` is the post-channel runtime view.  The exporter emits
# ``single`` for one fixed satellite observation, while its manifest records
# the source profile itself as ``satellite``.
EXPECTED_LEO_RUNTIME_VIEW = "single"
FROZEN_FOLD_SOURCE_TX = {
    1: ("20-15", "20-19", "6-15", "8-20"),
    2: ("14-10", "20-19", "6-15", "8-20"),
    3: ("14-10", "14-7", "6-15", "8-20"),
    4: ("14-10", "14-7", "20-15", "8-20"),
    5: ("14-10", "14-7", "20-15", "20-19"),
    6: ("14-7", "20-15", "20-19", "6-15"),
}
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
CLASSIFICATION_METRICS = (
    "overall_accuracy",
    "min_class_accuracy",
    "min_rx_accuracy",
    "min_day_accuracy",
)
FLOOR_DELTA_LIMIT_PP = -2.0


FROZEN_POSTFREEZE_CONTRACT = {
    "CBSFCE-POSTFREEZE-01": "final-only source diagnostics; no fit/calibration/selection",
    "CBSFCE-POSTFREEZE-02": "C/G ordered clean/LEO metadata, physical keys, and strict checkpoint SHA binding",
    "CBSFCE-POSTFREEZE-03": "single runtime LEO view plus satellite manifest profile and three disjoint scenarios",
    "CBSFCE-POSTFREEZE-04": "four source classifier floors and fixed proxy AUROC/FAR guardrail only",
    "CBSFCE-POSTFREEZE-05": "six-fold non-compensating clean, LEO, fold-equal, and 18-cell-equal gates",
}


class CBSFCEPostfreezePairError(RuntimeError):
    """Raised when a frozen CB-SFCE postfreeze evidence binding does not close."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if not root.is_dir():
        raise CBSFCEPostfreezePairError(f"postfreeze output root must already exist: {root}")
    return root


def _require_under_root(path: str | Path, root: Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CBSFCEPostfreezePairError(f"{label} is outside the frozen postfreeze output root") from exc
    return resolved


def _parse_items(value: str | Sequence[str], *, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        out = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        out = tuple(str(item).strip() for item in value if str(item).strip())
    if not out or len(set(out)) != len(out):
        raise CBSFCEPostfreezePairError(f"{field} must contain a non-empty, duplicate-free ordered list")
    return out


def _as_str_array(value: Any, n: int, *, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == ():
        array = np.repeat(array.reshape(1), int(n))
    array = array.reshape(-1)
    if int(array.size) != int(n):
        raise CBSFCEPostfreezePairError(f"{field} row count mismatch: expected={n} observed={array.size}")
    return np.asarray([str(item).strip() for item in array.tolist()], dtype=object)


def _manifest_from_npz(data: Any, path: Path) -> dict[str, Any]:
    if "manifest_json" not in data.files:
        raise CBSFCEPostfreezePairError(f"{path} is missing manifest_json")
    try:
        raw = np.asarray(data["manifest_json"])
        item = raw.item() if raw.shape == () else raw.reshape(-1)[0]
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        manifest = json.loads(str(item))
    except Exception as exc:
        raise CBSFCEPostfreezePairError(f"{path} has invalid manifest_json: {exc}") from exc
    if not isinstance(manifest, dict):
        raise CBSFCEPostfreezePairError(f"{path} manifest_json must encode an object")
    return manifest


def _load_npz(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise CBSFCEPostfreezePairError(f"missing NPZ: {source}")
    required = ("features", "tx_logits", *METADATA_FIELDS)
    with np.load(source, allow_pickle=False) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise CBSFCEPostfreezePairError(f"{source} is missing required arrays: {','.join(missing)}")
        features = np.asarray(data["features"], dtype=np.float32)
        logits = np.asarray(data["tx_logits"], dtype=np.float32)
        if features.ndim != 2 or features.shape[0] <= 0:
            raise CBSFCEPostfreezePairError(f"{source} features must be non-empty rank-2")
        if logits.ndim != 2 or logits.shape[0] != features.shape[0]:
            raise CBSFCEPostfreezePairError(f"{source} tx_logits/features row mismatch")
        if not np.isfinite(features).all() or not np.isfinite(logits).all():
            raise CBSFCEPostfreezePairError(f"{source} contains non-finite features or logits")
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
        raise CBSFCEPostfreezePairError(f"{label} contains duplicate physical keys")
    return keys


def _checkpoint_sha256_from_manifest(payload: Mapping[str, Any], *, label: str) -> str:
    manifest = payload["manifest"]
    checkpoint_sha256 = str(manifest.get("source_checkpoint_sha256", "")).strip().lower()
    if len(checkpoint_sha256) != 64 or any(char not in "0123456789abcdef" for char in checkpoint_sha256):
        raise CBSFCEPostfreezePairError(f"{label} lacks a valid source checkpoint SHA256")
    if manifest.get("checkpoint_load_strict") is not True:
        raise CBSFCEPostfreezePairError(f"{label} checkpoint export was not strict-loaded")
    audit = manifest.get("checkpoint_load_audit")
    if not isinstance(audit, Mapping) or audit.get("checkpoint_load_strict") is not True:
        raise CBSFCEPostfreezePairError(f"{label} lacks a strict checkpoint-load audit")
    for field in ("missing_keys", "unexpected_keys", "skipped_mismatch"):
        try:
            manifest_count = int(manifest.get(field, -1))
            audit_count = int(audit.get(field, -1))
        except (TypeError, ValueError) as exc:
            raise CBSFCEPostfreezePairError(f"{label} strict checkpoint-load audit has invalid {field}") from exc
        if manifest_count != 0 or audit_count != 0:
            raise CBSFCEPostfreezePairError(f"{label} strict checkpoint-load audit has nonzero {field}")
    return checkpoint_sha256


def _bind_final_checkpoint(path: str | Path, expected_sha256: str, *, label: str) -> str:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CBSFCEPostfreezePairError(f"missing {label} final checkpoint: {checkpoint}")
    observed = _sha256_file(checkpoint)
    if observed != str(expected_sha256).lower():
        raise CBSFCEPostfreezePairError(f"{label} final checkpoint SHA256 does not bind its NPZ exports")
    return observed


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(manifest), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_logit_contract(payload: Mapping[str, Any], source_tx_ids: Sequence[str], *, label: str) -> None:
    manifest = payload["manifest"]
    expected = tuple(source_tx_ids)
    class_order = tuple(str(item).strip() for item in manifest.get("class_id_to_tx", []))
    if class_order != expected:
        raise CBSFCEPostfreezePairError(
            f"{label} class label/order mismatch: expected={list(expected)} observed={list(class_order)}"
        )
    logit_order = list(manifest.get("logit_class_order", []))
    if logit_order != list(range(len(expected))):
        raise CBSFCEPostfreezePairError(f"{label} logit class order is not the frozen contiguous source order")
    if int(payload["tx_logits"].shape[1]) != len(expected):
        raise CBSFCEPostfreezePairError(
            f"{label} tx_logits width mismatch: expected={len(expected)} observed={payload['tx_logits'].shape[1]}"
        )
    feature_name = str(manifest.get("feature_name", manifest.get("feature_key", "")))
    if feature_name != "z_id":
        raise CBSFCEPostfreezePairError(f"{label} must export z_id, got {feature_name!r}")
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
        raise CBSFCEPostfreezePairError(f"{label} clean manifest must not be source-only")
    if str(manifest.get("satellite_tta_policy", "")) != "none":
        raise CBSFCEPostfreezePairError(f"{label} clean manifest must use satellite_tta_policy=none")
    roles = set(payload["dataset_role"].tolist())
    expected_roles = {"source", "target_old", "proxy_unknown"}
    if roles != expected_roles:
        raise CBSFCEPostfreezePairError(f"{label} clean roles mismatch: expected={sorted(expected_roles)} observed={sorted(roles)}")
    required_counts = {
        "source": int(expected_source_count),
        "target_old": int(expected_target_old_count),
        "proxy_unknown": int(expected_proxy_count),
    }
    for role, expected in required_counts.items():
        observed = _role_count(payload, role)
        if observed != expected:
            raise CBSFCEPostfreezePairError(
                f"{label} clean role count mismatch for {role}: expected={expected} observed={observed}"
            )
    if set(payload["channel_views"].tolist()) != {"clean"}:
        raise CBSFCEPostfreezePairError(f"{label} clean payload has a non-clean channel view")
    if any(str(item) for item in payload["sat_scenarios"].tolist()):
        raise CBSFCEPostfreezePairError(f"{label} clean payload must not assign satellite scenarios")
    source = _source_mask(payload)
    if set(payload["tx_ids"][source].tolist()) != set(source_tx_ids):
        raise CBSFCEPostfreezePairError(f"{label} clean source TX set mismatch")
    if set(payload["day_ids"][source].tolist()) != set(expected_days):
        raise CBSFCEPostfreezePairError(f"{label} clean source day set mismatch")
    if set(payload["rx_ids"][source].tolist()) != set(expected_rxs):
        raise CBSFCEPostfreezePairError(f"{label} clean source RX set mismatch")
    return _require_unique_physical_keys(payload, label=label)[source]


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
        raise CBSFCEPostfreezePairError(f"{label} LEO payload must contain source rows only")
    if int(payload["features"].shape[0]) != int(expected_source_count):
        raise CBSFCEPostfreezePairError(
            f"{label} LEO source row count mismatch: expected={expected_source_count} observed={payload['features'].shape[0]}"
        )
    if set(payload["channel_views"].tolist()) != {EXPECTED_LEO_RUNTIME_VIEW}:
        raise CBSFCEPostfreezePairError(
            f"{label} LEO payload must use exactly channel_view={EXPECTED_LEO_RUNTIME_VIEW}"
        )
    scenarios = payload["sat_scenarios"]
    if set(scenarios.tolist()) != set(expected_scenarios):
        raise CBSFCEPostfreezePairError(f"{label} LEO scenario set mismatch")
    if set(payload["tx_ids"].tolist()) != set(source_tx_ids):
        raise CBSFCEPostfreezePairError(f"{label} LEO source TX set mismatch")
    if set(payload["day_ids"].tolist()) != set(expected_days):
        raise CBSFCEPostfreezePairError(f"{label} LEO source day set mismatch")
    if set(payload["rx_ids"].tolist()) != set(expected_rxs):
        raise CBSFCEPostfreezePairError(f"{label} LEO source RX set mismatch")
    manifest = payload["manifest"]
    if manifest.get("source_only_export") is not True:
        raise CBSFCEPostfreezePairError(f"{label} LEO manifest is not source-only")
    if str(manifest.get("star_ground_channel_impl", "")) != "simplified_leo_residual":
        raise CBSFCEPostfreezePairError(f"{label} LEO manifest does not use simplified_leo_residual")
    if str(manifest.get("satellite_tta_policy", "")) != "none":
        raise CBSFCEPostfreezePairError(f"{label} LEO manifest must use satellite_tta_policy=none")
    profile_root = manifest.get("channel_profile")
    profile = profile_root.get("source") if isinstance(profile_root, Mapping) else None
    if not isinstance(profile, Mapping) or str(profile.get("view", "")) != "satellite":
        raise CBSFCEPostfreezePairError(f"{label} LEO source channel profile is not satellite")
    if tuple(str(item) for item in profile.get("scenarios", [])) != tuple(expected_scenarios):
        raise CBSFCEPostfreezePairError(f"{label} LEO source channel scenarios are not frozen")
    if int(profile.get("sat_seed", -1)) != int(source_sat_seed):
        raise CBSFCEPostfreezePairError(f"{label} LEO source satellite seed mismatch")
    keys = _require_unique_physical_keys(payload, label=label)
    physical_by_scenario: dict[str, set[str]] = {}
    for scenario in expected_scenarios:
        mask = np.asarray(scenarios == scenario, dtype=bool)
        if set(payload["tx_ids"][mask].tolist()) != set(source_tx_ids):
            raise CBSFCEPostfreezePairError(f"{label} scenario {scenario} lacks full source TX coverage")
        if set(payload["rx_ids"][mask].tolist()) != set(expected_rxs):
            raise CBSFCEPostfreezePairError(f"{label} scenario {scenario} lacks full source RX coverage")
        physical_by_scenario[scenario] = set(keys[mask].tolist())
    for index, left in enumerate(expected_scenarios):
        for right in expected_scenarios[index + 1 :]:
            if physical_by_scenario[left] & physical_by_scenario[right]:
                raise CBSFCEPostfreezePairError(f"{label} scenarios {left}/{right} reuse physical keys")
    return keys


def _assert_pair_metadata(c_payload: Mapping[str, Any], g_payload: Mapping[str, Any], *, label: str) -> None:
    if int(c_payload["features"].shape[0]) != int(g_payload["features"].shape[0]):
        raise CBSFCEPostfreezePairError(f"C/G {label} metadata row count mismatch")
    if int(c_payload["features"].shape[1]) != int(g_payload["features"].shape[1]):
        raise CBSFCEPostfreezePairError(f"C/G {label} z_id dimension mismatch")
    for field in METADATA_FIELDS:
        if not np.array_equal(c_payload[field], g_payload[field]):
            raise CBSFCEPostfreezePairError(f"C/G {label} metadata/scenario mismatch in {field}")


def _rate(values: np.ndarray) -> float:
    if int(values.size) <= 0:
        raise CBSFCEPostfreezePairError("cannot compute a rate on zero rows")
    return float(np.mean(values.astype(np.float64)))


def _min_group_accuracy(groups: np.ndarray, correct: np.ndarray) -> float:
    rates = [_rate(correct[groups == group]) for group in sorted(set(groups.tolist()))]
    if not rates:
        raise CBSFCEPostfreezePairError("cannot compute a group floor on zero groups")
    return float(min(rates))


def _predicted_tx(payload: Mapping[str, Any], source_tx_ids: Sequence[str]) -> np.ndarray:
    indices = np.asarray(payload["tx_logits"].argmax(axis=1), dtype=np.int64)
    return np.asarray([source_tx_ids[int(index)] for index in indices.tolist()], dtype=object)


def _classification_summary(payload: Mapping[str, Any], mask: np.ndarray, source_tx_ids: Sequence[str]) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    predicted = _predicted_tx(payload, source_tx_ids)
    correct = np.asarray(predicted == payload["tx_ids"], dtype=bool)
    return {
        "count": int(selected.sum()),
        "overall_accuracy": _rate(correct[selected]),
        "min_class_accuracy": _min_group_accuracy(payload["tx_ids"][selected], correct[selected]),
        "min_rx_accuracy": _min_group_accuracy(payload["rx_ids"][selected], correct[selected]),
        "min_day_accuracy": _min_group_accuracy(payload["day_ids"][selected], correct[selected]),
    }


def _delta_pp(c_metrics: Mapping[str, Any], g_metrics: Mapping[str, Any]) -> dict[str, float]:
    return {metric: 100.0 * (float(g_metrics[metric]) - float(c_metrics[metric])) for metric in CLASSIFICATION_METRICS}


def _load_proxy_metrics(
    path: str | Path,
    clean_payload: Mapping[str, Any],
    source_tx_ids: Sequence[str],
    expected_source_count: int,
    expected_proxy_count: int,
    *,
    label: str,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise CBSFCEPostfreezePairError(f"missing proxy diagnostic JSON: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CBSFCEPostfreezePairError(f"{label} proxy diagnostic JSON is invalid: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise CBSFCEPostfreezePairError(f"{label} proxy diagnostic JSON must encode an object")
    if tuple(str(item) for item in raw.get("source_tx_ids", [])) != tuple(source_tx_ids):
        raise CBSFCEPostfreezePairError(f"{label} proxy source TX order does not bind its clean NPZ")
    if list(raw.get("known_query_roles", [])) != ["source"]:
        raise CBSFCEPostfreezePairError(f"{label} proxy known role is not frozen to source")
    if list(raw.get("unknown_query_roles", [])) != ["proxy_unknown"]:
        raise CBSFCEPostfreezePairError(f"{label} proxy unknown role is not frozen to proxy_unknown")
    if str(raw.get("threshold_scope", "")) != "source_calibrated_only_no_target_support_no_unknown_query_tuning":
        raise CBSFCEPostfreezePairError(f"{label} proxy threshold scope is not the frozen source-only diagnostic")
    if int(raw.get("known_query_count", -1)) != int(expected_source_count):
        raise CBSFCEPostfreezePairError(f"{label} proxy source row count mismatch")
    if int(raw.get("unknown_query_count", -1)) != int(expected_proxy_count):
        raise CBSFCEPostfreezePairError(f"{label} proxy unknown row count mismatch")
    proxy_manifest = raw.get("manifest")
    if not isinstance(proxy_manifest, Mapping) or _manifest_sha256(proxy_manifest) != _manifest_sha256(clean_payload["manifest"]):
        raise CBSFCEPostfreezePairError(f"{label} proxy manifest does not bind its clean NPZ")
    feature_npz = Path(str(raw.get("feature_npz", "")))
    try:
        same_feature_file = feature_npz.resolve() == Path(str(clean_payload["path"])).resolve()
    except OSError:
        same_feature_file = False
    if not same_feature_file:
        raise CBSFCEPostfreezePairError(f"{label} proxy feature NPZ path does not bind its clean NPZ")
    values: dict[str, float] = {}
    for field in ("AUROC_unknown", "unknown_FAR"):
        try:
            value = float(raw[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise CBSFCEPostfreezePairError(f"{label} proxy diagnostic lacks finite {field}") from exc
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise CBSFCEPostfreezePairError(f"{label} proxy diagnostic has invalid {field}")
        values[field] = value
    return {"path": str(source), "sha256": _sha256_file(source), **values}


def _proxy_guardrail(c_proxy: Mapping[str, Any], g_proxy: Mapping[str, Any]) -> dict[str, Any]:
    auroc_delta = float(g_proxy["AUROC_unknown"]) - float(c_proxy["AUROC_unknown"])
    far_delta_pp = 100.0 * (float(g_proxy["unknown_FAR"]) - float(c_proxy["unknown_FAR"]))
    auroc_ok = auroc_delta >= 0.0
    far_ok = far_delta_pp <= 0.0
    return {
        "C": {"AUROC_unknown": float(c_proxy["AUROC_unknown"]), "unknown_FAR": float(c_proxy["unknown_FAR"])},
        "G": {"AUROC_unknown": float(g_proxy["AUROC_unknown"]), "unknown_FAR": float(g_proxy["unknown_FAR"])},
        "G_minus_C": {"AUROC_unknown": auroc_delta, "unknown_FAR_pp": far_delta_pp},
        "AUROC_unknown_non_decrease": auroc_ok,
        "unknown_FAR_non_increase": far_ok,
        "passed": bool(auroc_ok and far_ok),
        "diagnostic_only_non_compensating": True,
    }


def _floor_gate(deltas: Mapping[str, Any]) -> dict[str, Any]:
    metric_passes = {metric: float(deltas[metric]) >= FLOOR_DELTA_LIMIT_PP for metric in CLASSIFICATION_METRICS}
    return {"metric_passes": metric_passes, "passed": bool(all(metric_passes.values()))}


def _fold_gates(
    clean_delta: Mapping[str, Any],
    leo_scenarios: Mapping[str, Mapping[str, Any]],
    proxy_guardrail: Mapping[str, Any],
    expected_scenarios: Sequence[str],
) -> dict[str, Any]:
    clean = _floor_gate(clean_delta)
    scenario_floor = {
        scenario: _floor_gate(leo_scenarios[scenario]["G_minus_C_pp"])
        for scenario in expected_scenarios
    }
    scenario_overall_deltas = [float(leo_scenarios[scenario]["G_minus_C_pp"]["overall_accuracy"]) for scenario in expected_scenarios]
    fold_equal_overall = float(np.mean(np.asarray(scenario_overall_deltas, dtype=np.float64)))
    leo_floor_passed = bool(all(gate["passed"] for gate in scenario_floor.values()))
    fold_overall_passed = fold_equal_overall >= 0.0
    passed = bool(clean["passed"] and leo_floor_passed and fold_overall_passed and proxy_guardrail["passed"])
    return {
        "technical_binding": {"passed": True},
        "clean_four_floors_ge_minus2pp": clean,
        "leo_scenario_four_floors_ge_minus2pp": {"by_scenario": scenario_floor, "passed": leo_floor_passed},
        "fold_three_scenario_equal_weight_overall_delta_pp": {
            "value": fold_equal_overall,
            "passed": fold_overall_passed,
        },
        "proxy_guardrail": dict(proxy_guardrail),
        "fold_verdict": "PENDING_GLOBAL_18_GRID" if passed else "REJECT_CB_SFCE_PERMANENT",
    }


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise CBSFCEPostfreezePairError(f"refusing to overwrite final-only pair output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        raise CBSFCEPostfreezePairError(f"refusing to overwrite temporary pair output: {temporary}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def _load_prior_pair(
    path: str | Path,
    *,
    expected_scenarios: Sequence[str],
    source_sat_seed: int,
    matrix_id: str,
    output_root: Path,
) -> dict[str, Any]:
    source = _require_under_root(path, output_root, label="prior pair metrics JSON")
    if not source.is_file():
        raise CBSFCEPostfreezePairError(f"missing prior pair metrics JSON: {source}")
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CBSFCEPostfreezePairError(f"prior pair metrics JSON is invalid: {source}: {exc}") from exc
    if not isinstance(record, dict):
        raise CBSFCEPostfreezePairError(f"prior pair metrics JSON must encode an object: {source}")
    if record.get("schema") != "cvs.phase1.cb_sfce_postfreeze_pair.v1":
        raise CBSFCEPostfreezePairError(f"prior pair schema mismatch: {source}")
    if tuple(str(item) for item in record.get("expected_scenarios", [])) != tuple(expected_scenarios):
        raise CBSFCEPostfreezePairError(f"prior pair scenario contract mismatch: {source}")
    if int(record.get("source_sat_seed", -1)) != int(source_sat_seed):
        raise CBSFCEPostfreezePairError(f"prior pair satellite seed mismatch: {source}")
    if str(record.get("postfreeze_matrix_id", "")) != str(matrix_id):
        raise CBSFCEPostfreezePairError(f"prior pair matrix_id mismatch: {source}")
    if str(record.get("postfreeze_output_root", "")) != str(output_root):
        raise CBSFCEPostfreezePairError(f"prior pair output root mismatch: {source}")
    if record.get("matrix_aggregate") is not None:
        raise CBSFCEPostfreezePairError(f"prior pair must be a per-fold record, not an aggregate: {source}")
    record["_input_path"] = str(source)
    record["_input_sha256"] = _sha256_file(source)
    return record


def _as_fold_index(record: Mapping[str, Any], *, label: str) -> int:
    try:
        fold_index = int(record["fold_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CBSFCEPostfreezePairError(f"{label} lacks a valid fold_index") from exc
    if fold_index not in range(1, 7):
        raise CBSFCEPostfreezePairError(f"{label} fold_index must be in [1,6]")
    return fold_index


def _validate_pair_record_contract(record: Mapping[str, Any], *, output_root: Path, matrix_id: str, label: str) -> int:
    fold_index = _as_fold_index(record, label=label)
    expected_pair = f"F{fold_index}_C_vs_G"
    if str(record.get("candidate_pair", "")) != expected_pair:
        raise CBSFCEPostfreezePairError(f"{label} candidate_pair does not match frozen fold {fold_index}")
    if tuple(str(item) for item in record.get("source_tx_ids", [])) != FROZEN_FOLD_SOURCE_TX[fold_index]:
        raise CBSFCEPostfreezePairError(f"{label} source TX order does not match frozen fold {fold_index}")
    if str(record.get("postfreeze_matrix_id", "")) != str(matrix_id):
        raise CBSFCEPostfreezePairError(f"{label} matrix_id mismatch")
    if str(record.get("postfreeze_output_root", "")) != str(output_root):
        raise CBSFCEPostfreezePairError(f"{label} output root mismatch")
    policy = record.get("policy")
    if not isinstance(policy, Mapping):
        raise CBSFCEPostfreezePairError(f"{label} lacks policy receipt")
    for field in (
        "fit_performed",
        "calibration_performed",
        "threshold_used_for_pair_metrics",
        "model_selection_performed",
        "checkpoint_weights_loaded",
    ):
        if policy.get(field) is not False:
            raise CBSFCEPostfreezePairError(f"{label} policy {field} is not strictly false")
    for field in ("proxy_rows_used_for_pair_metrics", "target_old_rows_used_for_pair_metrics"):
        if type(policy.get(field)) is not int or int(policy[field]) != 0:
            raise CBSFCEPostfreezePairError(f"{label} policy {field} is not strictly zero")
    gates = record.get("postfreeze_gates")
    if not isinstance(gates, Mapping) or not isinstance(gates.get("technical_binding"), Mapping):
        raise CBSFCEPostfreezePairError(f"{label} lacks technical binding receipt")
    if gates["technical_binding"].get("passed") is not True:
        raise CBSFCEPostfreezePairError(f"{label} technical binding is not strictly true")
    proxy = record.get("proxy_guardrail")
    if not isinstance(proxy, Mapping):
        raise CBSFCEPostfreezePairError(f"{label} lacks proxy guardrail receipt")
    try:
        c_auroc = float(proxy["C"]["AUROC_unknown"])
        g_auroc = float(proxy["G"]["AUROC_unknown"])
        c_far = float(proxy["C"]["unknown_FAR"])
        g_far = float(proxy["G"]["unknown_FAR"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CBSFCEPostfreezePairError(f"{label} proxy guardrail is malformed") from exc
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in (c_auroc, g_auroc, c_far, g_far)):
        raise CBSFCEPostfreezePairError(f"{label} proxy guardrail has non-finite or out-of-range value")
    expected_auroc = g_auroc >= c_auroc
    expected_far = g_far <= c_far
    if proxy.get("AUROC_unknown_non_decrease") is not expected_auroc:
        raise CBSFCEPostfreezePairError(f"{label} proxy AUROC guardrail is not strictly bound")
    if proxy.get("unknown_FAR_non_increase") is not expected_far:
        raise CBSFCEPostfreezePairError(f"{label} proxy FAR guardrail is not strictly bound")
    if proxy.get("passed") is not bool(expected_auroc and expected_far):
        raise CBSFCEPostfreezePairError(f"{label} proxy passed receipt is not strictly bound")
    return fold_index


def _record_deltas(record: Mapping[str, Any], scenarios: Sequence[str], *, label: str) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    try:
        clean = record["clean_source"]["G_minus_C_pp"]
        leo = record["leo_scenarios"]
    except (KeyError, TypeError) as exc:
        raise CBSFCEPostfreezePairError(f"{label} lacks classifier deltas") from exc
    clean_out: dict[str, float] = {}
    leo_out: dict[str, dict[str, float]] = {}
    for metric in CLASSIFICATION_METRICS:
        try:
            value = float(clean[metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise CBSFCEPostfreezePairError(f"{label} clean delta lacks {metric}") from exc
        if not math.isfinite(value):
            raise CBSFCEPostfreezePairError(f"{label} clean delta is non-finite for {metric}")
        clean_out[metric] = value
    for scenario in scenarios:
        try:
            delta = leo[scenario]["G_minus_C_pp"]
        except (KeyError, TypeError) as exc:
            raise CBSFCEPostfreezePairError(f"{label} lacks LEO scenario {scenario}") from exc
        leo_out[scenario] = {}
        for metric in CLASSIFICATION_METRICS:
            try:
                value = float(delta[metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise CBSFCEPostfreezePairError(f"{label} LEO delta lacks {scenario}/{metric}") from exc
            if not math.isfinite(value):
                raise CBSFCEPostfreezePairError(f"{label} LEO delta is non-finite for {scenario}/{metric}")
            leo_out[scenario][metric] = value
    return clean_out, leo_out


def _matrix_aggregate(
    current: Mapping[str, Any],
    prior_paths: Sequence[str],
    *,
    expected_scenarios: Sequence[str],
    output_root: Path,
    matrix_id: str,
) -> dict[str, Any]:
    fold_index = _validate_pair_record_contract(
        current, output_root=output_root, matrix_id=matrix_id, label="current pair"
    )
    if fold_index != 6:
        raise CBSFCEPostfreezePairError("matrix aggregate is frozen to the sixth and final pair")
    if len(prior_paths) != 5:
        raise CBSFCEPostfreezePairError("sixth pair requires exactly five prior per-fold metrics JSONs")
    records = [
        _load_prior_pair(
            path,
            expected_scenarios=expected_scenarios,
            source_sat_seed=int(current["source_sat_seed"]),
            matrix_id=matrix_id,
            output_root=output_root,
        )
        for path in prior_paths
    ]
    records.append(dict(current))
    fold_indices = [
        _validate_pair_record_contract(record, output_root=output_root, matrix_id=matrix_id, label="pair record")
        for record in records
    ]
    if set(fold_indices) != set(range(1, 7)) or len(set(fold_indices)) != len(fold_indices):
        raise CBSFCEPostfreezePairError("matrix aggregate must contain exactly folds 1..6 once")
    records.sort(key=lambda record: _as_fold_index(record, label="pair record"))

    clean_passes: list[bool] = []
    leo_passes: list[bool] = []
    fold_equal_overall: dict[str, float] = {}
    technical_passes: list[bool] = []
    proxy_passes: list[bool] = []
    deltas_by_metric: dict[str, list[float]] = {metric: [] for metric in CLASSIFICATION_METRICS}
    for record in records:
        clean_delta, leo_delta = _record_deltas(record, expected_scenarios, label=f"fold{record['fold_index']}")
        clean_passes.append(all(value >= FLOOR_DELTA_LIMIT_PP for value in clean_delta.values()))
        all_leo = [value for scenario in expected_scenarios for value in leo_delta[scenario].values()]
        leo_passes.append(all(value >= FLOOR_DELTA_LIMIT_PP for value in all_leo))
        fold_overall = float(np.mean([leo_delta[scenario]["overall_accuracy"] for scenario in expected_scenarios]))
        fold_equal_overall[f"F{record['fold_index']}"] = fold_overall
        for scenario in expected_scenarios:
            for metric in CLASSIFICATION_METRICS:
                deltas_by_metric[metric].append(leo_delta[scenario][metric])
        technical_passes.append(record["postfreeze_gates"]["technical_binding"]["passed"] is True)
        proxy = record["proxy_guardrail"]
        proxy_auroc_pass = float(proxy["G"]["AUROC_unknown"]) >= float(proxy["C"]["AUROC_unknown"])
        proxy_far_pass = float(proxy["G"]["unknown_FAR"]) <= float(proxy["C"]["unknown_FAR"])
        proxy_passes.append(bool(proxy_auroc_pass and proxy_far_pass))

    global_18 = {metric: float(np.mean(np.asarray(values, dtype=np.float64))) for metric, values in deltas_by_metric.items()}
    technical_passed = bool(all(technical_passes))
    clean_passed = bool(all(clean_passes))
    leo_passed = bool(all(leo_passes))
    fold_overall_passed = bool(all(value >= 0.0 for value in fold_equal_overall.values()))
    global_overall_passed = global_18["overall_accuracy"] >= 0.0
    proxy_passed = bool(all(proxy_passes))
    passed = bool(technical_passed and clean_passed and leo_passed and fold_overall_passed and global_overall_passed and proxy_passed)
    prior_bindings = [
        {"fold_index": int(record["fold_index"]), "metrics_json": record["_input_path"], "sha256": record["_input_sha256"]}
        for record in records
        if "_input_path" in record
    ]
    return {
        "fold_indices": [int(record["fold_index"]) for record in records],
        "prior_pair_metrics_bindings": prior_bindings,
        "global_18_cell_equal_weight_G_minus_C_pp": global_18,
        "gates": {
            "technical_binding": {"passed": technical_passed},
            "clean_6of6_four_floors_ge_minus2pp": {"passed": clean_passed, "by_fold": clean_passes},
            "leo_18of18_four_floors_ge_minus2pp": {"passed": leo_passed, "by_fold": leo_passes},
            "fold_three_scenario_equal_weight_overall_delta_pp": {"values": fold_equal_overall, "passed": fold_overall_passed},
            "global_18_cell_equal_weight_overall_delta_pp": {
                "value": global_18["overall_accuracy"],
                "passed": global_overall_passed,
            },
            "proxy_AUROC_non_decrease_and_FAR_non_increase": {"passed": proxy_passed},
        },
        "verdict": "PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW" if passed else "REJECT_CB_SFCE_PERMANENT",
        "phase3_unknown_capability_claim": "NOT_EVALUATED",
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    source_tx_ids = _parse_items(args.source_tx_ids, field="source_tx_ids")
    if len(source_tx_ids) != 4:
        raise CBSFCEPostfreezePairError("P1-CB-SFCE postfreeze is frozen to local4 source TX classes")
    fold_index = int(args.fold_index)
    if fold_index not in range(1, 7):
        raise CBSFCEPostfreezePairError("fold_index must be in [1,6]")
    if source_tx_ids != FROZEN_FOLD_SOURCE_TX[fold_index]:
        raise CBSFCEPostfreezePairError(f"source_tx_ids do not match frozen fold {fold_index}")
    if str(args.candidate_pair) != f"F{fold_index}_C_vs_G":
        raise CBSFCEPostfreezePairError(f"candidate_pair does not match frozen fold {fold_index}")
    matrix_id = str(args.postfreeze_matrix_id).strip()
    if not matrix_id:
        raise CBSFCEPostfreezePairError("postfreeze_matrix_id must be non-empty")
    output_root = _canonical_root(args.postfreeze_output_root)
    for path, label in (
        (args.c_clean_npz, "C clean NPZ"),
        (args.g_clean_npz, "G clean NPZ"),
        (args.c_leo_npz, "C LEO NPZ"),
        (args.g_leo_npz, "G LEO NPZ"),
        (args.c_proxy_metrics_json, "C proxy metrics JSON"),
        (args.g_proxy_metrics_json, "G proxy metrics JSON"),
        (args.output_metrics_json, "pair output JSON"),
    ):
        _require_under_root(path, output_root, label=label)
    expected_scenarios = _parse_items(args.expected_scenarios, field="expected_scenarios")
    if expected_scenarios != EXPECTED_SCENARIOS:
        raise CBSFCEPostfreezePairError("expected_scenarios are frozen to the three leo_*_weak scenarios")
    expected_days = _parse_items(args.expected_source_days, field="expected_source_days")
    expected_rxs = _parse_items(args.expected_source_rxs, field="expected_source_rxs")
    if expected_days != EXPECTED_SOURCE_DAYS:
        raise CBSFCEPostfreezePairError("expected_source_days do not match the frozen WRC LEO v2 slice")
    if expected_rxs != EXPECTED_SOURCE_RXS:
        raise CBSFCEPostfreezePairError("expected_source_rxs do not match the frozen WRC LEO v2 slice")
    expected_source_count = int(args.expected_source_count)
    expected_target_old_count = int(args.expected_target_old_count)
    expected_proxy_count = int(args.expected_proxy_count)
    if min(expected_source_count, expected_target_old_count, expected_proxy_count) <= 0:
        raise CBSFCEPostfreezePairError("expected role counts must be positive")
    aggregate_prior_paths = _parse_items(args.aggregate_prior_pair_metrics_json, field="aggregate_prior_pair_metrics_json") if args.aggregate_prior_pair_metrics_json else ()
    if fold_index < 6 and aggregate_prior_paths:
        raise CBSFCEPostfreezePairError("only the sixth pair may aggregate prior pair metrics")
    if fold_index == 6 and len(aggregate_prior_paths) != 5:
        raise CBSFCEPostfreezePairError("sixth pair requires five prior pair metrics JSONs for the 18-cell gate")

    c_clean = _load_npz(args.c_clean_npz)
    g_clean = _load_npz(args.g_clean_npz)
    c_leo = _load_npz(args.c_leo_npz)
    g_leo = _load_npz(args.g_leo_npz)
    _assert_pair_metadata(c_clean, g_clean, label="clean")
    _assert_pair_metadata(c_leo, g_leo, label="LEO")
    if int(c_clean["features"].shape[1]) != int(c_leo["features"].shape[1]):
        raise CBSFCEPostfreezePairError("C clean/LEO z_id dimension mismatch")
    if int(g_clean["features"].shape[1]) != int(g_leo["features"].shape[1]):
        raise CBSFCEPostfreezePairError("G clean/LEO z_id dimension mismatch")

    c_clean_keys = _validate_clean_payload(
        c_clean, source_tx_ids, expected_source_count, expected_target_old_count, expected_proxy_count,
        expected_days, expected_rxs, label="C clean"
    )
    g_clean_keys = _validate_clean_payload(
        g_clean, source_tx_ids, expected_source_count, expected_target_old_count, expected_proxy_count,
        expected_days, expected_rxs, label="G clean"
    )
    c_leo_keys = _validate_leo_payload(
        c_leo, source_tx_ids, expected_source_count, expected_scenarios, expected_days, expected_rxs,
        int(args.source_sat_seed), label="C LEO"
    )
    g_leo_keys = _validate_leo_payload(
        g_leo, source_tx_ids, expected_source_count, expected_scenarios, expected_days, expected_rxs,
        int(args.source_sat_seed), label="G LEO"
    )
    if set(c_clean_keys.tolist()) != set(c_leo_keys.tolist()):
        raise CBSFCEPostfreezePairError("C clean/LEO source physical key sets differ")
    if set(g_clean_keys.tolist()) != set(g_leo_keys.tolist()):
        raise CBSFCEPostfreezePairError("G clean/LEO source physical key sets differ")
    if set(c_clean_keys.tolist()) != set(g_clean_keys.tolist()):
        raise CBSFCEPostfreezePairError("C/G clean source physical key sets differ")
    if set(c_leo_keys.tolist()) != set(g_leo_keys.tolist()):
        raise CBSFCEPostfreezePairError("C/G LEO source physical key sets differ")
    c_checkpoint_sha256 = _checkpoint_sha256_from_manifest(c_clean, label="C clean")
    g_checkpoint_sha256 = _checkpoint_sha256_from_manifest(g_clean, label="G clean")
    if c_checkpoint_sha256 != _checkpoint_sha256_from_manifest(c_leo, label="C LEO"):
        raise CBSFCEPostfreezePairError("C clean/LEO source checkpoint SHA256 differs")
    if g_checkpoint_sha256 != _checkpoint_sha256_from_manifest(g_leo, label="G LEO"):
        raise CBSFCEPostfreezePairError("G clean/LEO source checkpoint SHA256 differs")
    c_final_checkpoint_sha256 = _bind_final_checkpoint(
        args.c_final_checkpoint, c_checkpoint_sha256, label="C"
    )
    g_final_checkpoint_sha256 = _bind_final_checkpoint(
        args.g_final_checkpoint, g_checkpoint_sha256, label="G"
    )

    c_proxy = _load_proxy_metrics(
        args.c_proxy_metrics_json, c_clean, source_tx_ids, expected_source_count, expected_proxy_count, label="C"
    )
    g_proxy = _load_proxy_metrics(
        args.g_proxy_metrics_json, g_clean, source_tx_ids, expected_source_count, expected_proxy_count, label="G"
    )
    proxy_guardrail = _proxy_guardrail(c_proxy, g_proxy)
    c_clean_summary = _classification_summary(c_clean, _source_mask(c_clean), source_tx_ids)
    g_clean_summary = _classification_summary(g_clean, _source_mask(g_clean), source_tx_ids)
    scenario_metrics: dict[str, Any] = {}
    for scenario in expected_scenarios:
        c_mask = np.asarray(c_leo["sat_scenarios"] == scenario, dtype=bool)
        g_mask = np.asarray(g_leo["sat_scenarios"] == scenario, dtype=bool)
        c_summary = _classification_summary(c_leo, c_mask, source_tx_ids)
        g_summary = _classification_summary(g_leo, g_mask, source_tx_ids)
        scenario_metrics[scenario] = {
            "C": c_summary,
            "G": g_summary,
            "G_minus_C_pp": _delta_pp(c_summary, g_summary),
        }
    clean_delta = _delta_pp(c_clean_summary, g_clean_summary)
    postfreeze_gates = _fold_gates(clean_delta, scenario_metrics, proxy_guardrail, expected_scenarios)
    metrics: dict[str, Any] = {
        "schema": "cvs.phase1.cb_sfce_postfreeze_pair.v1",
        "candidate_pair": str(args.candidate_pair),
        "fold_index": fold_index,
        "postfreeze_matrix_id": matrix_id,
        "postfreeze_output_root": str(output_root),
        "evidence_boundary": "PHASE1_SOURCE_ONLY_FINAL_ONLY_DIAGNOSTIC",
        "frozen_contract": dict(FROZEN_POSTFREEZE_CONTRACT),
        "policy": {
            "fit_performed": False,
            "calibration_performed": False,
            "threshold_used_for_pair_metrics": False,
            "model_selection_performed": False,
            "checkpoint_weights_loaded": False,
            "proxy_rows_used_for_pair_metrics": 0,
            "target_old_rows_used_for_pair_metrics": 0,
            "proxy_guardrail_only": True,
            "proxy_guardrail_non_compensating": True,
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
            "c_final_checkpoint_sha256": c_final_checkpoint_sha256,
            "g_final_checkpoint_sha256": g_final_checkpoint_sha256,
            "c_proxy_metrics_json_sha256": c_proxy["sha256"],
            "g_proxy_metrics_json_sha256": g_proxy["sha256"],
            "checkpoint_weight_reading": "DISALLOWED",
        },
        "clean_source": {"C": c_clean_summary, "G": g_clean_summary, "G_minus_C_pp": clean_delta},
        "leo_scenarios": scenario_metrics,
        "proxy_guardrail": proxy_guardrail,
        "postfreeze_gates": postfreeze_gates,
        "matrix_aggregate": None,
    }
    if fold_index == 6:
        metrics["matrix_aggregate"] = _matrix_aggregate(
            metrics,
            aggregate_prior_paths,
            expected_scenarios=expected_scenarios,
            output_root=output_root,
            matrix_id=matrix_id,
        )
    _atomic_write_json(args.output_metrics_json, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-clean-npz", required=True)
    parser.add_argument("--g-clean-npz", required=True)
    parser.add_argument("--c-leo-npz", required=True)
    parser.add_argument("--g-leo-npz", required=True)
    parser.add_argument("--c-final-checkpoint", required=True)
    parser.add_argument("--g-final-checkpoint", required=True)
    parser.add_argument("--c-proxy-metrics-json", required=True)
    parser.add_argument("--g-proxy-metrics-json", required=True)
    parser.add_argument("--source-tx-ids", required=True)
    parser.add_argument("--candidate-pair", required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--postfreeze-matrix-id", required=True)
    parser.add_argument("--postfreeze-output-root", required=True)
    parser.add_argument("--expected-scenarios", default=",".join(EXPECTED_SCENARIOS))
    parser.add_argument("--expected-source-days", default=",".join(EXPECTED_SOURCE_DAYS))
    parser.add_argument("--expected-source-rxs", default=",".join(EXPECTED_SOURCE_RXS))
    parser.add_argument("--source-sat-seed", type=int, default=7281718)
    parser.add_argument("--expected-source-count", type=int, default=1600)
    parser.add_argument("--expected-target-old-count", type=int, default=400)
    parser.add_argument("--expected-proxy-count", type=int, default=400)
    parser.add_argument("--aggregate-prior-pair-metrics-json", default="")
    parser.add_argument("--output-metrics-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    metrics = evaluate(build_parser().parse_args(argv))
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

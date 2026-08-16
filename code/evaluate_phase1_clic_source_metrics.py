"""Score held source-V clean/LEO evidence without target or selection access.

The evaluator consumes only immutable source-side artifacts: training-v5 final
checkpoint envelopes, clean-v4 feature evidence, Task1 V-only received-IQ
receipt, V-only feature exports and the sealed PAIR-v3 source policy/proxy
state.  It never rebuilds source-L geometry, opens proxy/L feature rows for
scoring, reads a target artifact, or updates a model/threshold/selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

import build_phase1_clic_source_v_leo_iq as _cache
import evaluate_phase1_clic_postfreeze_pair as _pair
import export_phase1_clic_features as _clean
import export_phase1_clic_source_v_leo_features as _source_v


SOURCE_METRICS_PAIR_SCHEMA = "cvs.phase1.clic_source_metrics_pair.v1"
SOURCE_METRICS_AGGREGATE_SCHEMA = "cvs.phase1.clic_source_metrics_aggregate.v1"
EXPECTED_SCENARIOS = tuple(_pair.EXPECTED_SCENARIOS)
EXPECTED_RUN_ID = _cache.EXPECTED_CACHE_RUN_ID
FLOOR_FIELDS = (
    "overall_accuracy",
    "min_class_accuracy",
    "min_rx_accuracy",
    "min_day_accuracy",
)
NONCOMPENSATING_DELTA_PP = -2.0


class CLICSourceMetricsError(RuntimeError):
    """Raised only for malformed, unsafe or incomplete source evidence."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CLICSourceMetricsError("source metric state cannot be canonicalized") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CLICSourceMetricsError(f"{label} SHA256 is absent or invalid")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CLICSourceMetricsError(f"{label} SHA256 is not hexadecimal") from exc
    return value


def validate_clean_v_cache_identity(
    *,
    cache_receipt: Mapping[str, Any],
    clean_manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind both clean-v4 held-V exports to Task1's immutable validation identities."""

    if not isinstance(cache_receipt, Mapping) or set(clean_manifests) != {"C", "G"}:
        raise CLICSourceMetricsError("clean-v4/source-V cache identity binding is incomplete")
    for field in ("source_validation_indices_sha256", "source_validation_physical_order_sha256"):
        expected = _require_sha256(cache_receipt.get(field), label=f"source-V cache {field}")
        for arm in ("C", "G"):
            manifest = clean_manifests[arm]
            if not isinstance(manifest, Mapping):
                raise CLICSourceMetricsError(f"{arm} clean-v4 manifest identity is malformed")
            observed = _require_sha256(manifest.get(field), label=f"{arm} clean-v4 {field}")
            if observed != expected:
                raise CLICSourceMetricsError(f"{arm} clean-v4/source-V cache {field} binding drifted")


def validate_source_v_feature_cache_metadata(
    *, feature_axes: Mapping[str, Any], cache_axes: Mapping[str, Any]
) -> None:
    """Require a reopened V feature export to preserve every cache metric axis."""

    required = {"tx_ids", "rx_ids", "day_ids", "physical_ids", "scenes"}
    if not isinstance(feature_axes, Mapping) or not isinstance(cache_axes, Mapping) or set(feature_axes) != required or set(cache_axes) != required:
        raise CLICSourceMetricsError("source-V feature/cache metadata fields are incomplete")
    cache_physical = _text_rows(
        cache_axes["physical_ids"], label="source-V cache physical IDs", row_count=np.asarray(cache_axes["physical_ids"]).reshape(-1).size
    )
    row_count = int(cache_physical.size)
    if row_count <= 0 or len(set(cache_physical.tolist())) != row_count:
        raise CLICSourceMetricsError("source-V cache physical metadata is invalid")
    labels = {
        "tx_ids": "TX",
        "rx_ids": "RX",
        "day_ids": "day",
        "physical_ids": "physical",
        "scenes": "scene",
    }
    for field, label in labels.items():
        expected = _text_rows(cache_axes[field], label=f"source-V cache {label}", row_count=row_count)
        observed = _text_rows(feature_axes[field], label=f"source-V feature {label}", row_count=row_count)
        if not np.array_equal(observed, expected):
            raise CLICSourceMetricsError(f"source-V feature/cache {label} metadata binding drifted")


def _source_order(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        result = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        result = tuple(str(item) for item in value)
    if len(result) != 4 or len(set(result)) != 4:
        raise CLICSourceMetricsError("source metrics require exactly four source TX IDs")
    return result


def _text_rows(values: Any, *, label: str, row_count: int, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.hasobject or array.dtype.kind not in {"U", "S"}:
        raise CLICSourceMetricsError(f"{label} must be a non-object text array")
    result = np.asarray(array.reshape(-1), dtype=str)
    if result.size != row_count or (not allow_empty and np.any(result == "")):
        raise CLICSourceMetricsError(f"{label} row alignment is invalid")
    return result


def _finite_matrix(values: Any, *, label: str, row_count: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.hasobject or array.dtype.kind != "f":
        raise CLICSourceMetricsError(f"{label} must be a floating non-object array")
    result = np.asarray(array, dtype=np.float64)
    if result.ndim != 2 or (row_count is not None and result.shape[0] != row_count) or not np.isfinite(result).all():
        raise CLICSourceMetricsError(f"{label} is non-finite or shape-invalid")
    return result


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise CLICSourceMetricsError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CLICSourceMetricsError(f"{label} is non-finite")
    return result


def _checked_accuracy(correct: int, denominator: int, *, label: str) -> dict[str, Any]:
    if type(correct) is not int or type(denominator) is not int or denominator <= 0 or correct < 0 or correct > denominator:
        raise CLICSourceMetricsError(f"{label} raw numerator/denominator is invalid")
    return {"correct": correct, "denominator": denominator, "accuracy": correct / denominator}


def _physical_keys(
    tx_ids: np.ndarray, rx_ids: np.ndarray, day_ids: np.ndarray, eq_ids: np.ndarray, sig_ids: np.ndarray
) -> np.ndarray:
    return np.asarray(
        ["\x1f".join(values) for values in zip(tx_ids, rx_ids, day_ids, eq_ids, sig_ids, strict=True)],
        dtype=str,
    )


def _validate_known_role(role: str, scene: str | None) -> None:
    if role == "source_validation_known_clean":
        if scene is not None:
            raise CLICSourceMetricsError("clean known metrics must not carry a LEO scene")
    elif role == _source_v.SOURCE_V_ROLE:
        if scene not in EXPECTED_SCENARIOS:
            raise CLICSourceMetricsError("source-V LEO metrics must carry one formal scene")
    else:
        raise CLICSourceMetricsError("source metric role is not the held known clean/V role")


def score_known_source_rows(
    *,
    truth_tx_ids: Any,
    predicted_tx_ids: Any,
    decisions: Any,
    rx_ids: Any,
    day_ids: Any,
    physical_ids: Any,
    role: str,
    scene: str | None,
    source_tx_ids: Sequence[str],
) -> dict[str, Any]:
    """Score known source rows with raw cells; unknown/defer are explicit errors."""

    _validate_known_role(str(role), scene)
    source_order = _source_order(source_tx_ids)
    truth = np.asarray(truth_tx_ids, dtype=str).reshape(-1)
    count = int(truth.size)
    if count <= 0:
        raise CLICSourceMetricsError("known source metrics require a positive denominator")
    predicted = np.asarray(predicted_tx_ids, dtype=str).reshape(-1)
    decision_rows = np.asarray(decisions, dtype=str).reshape(-1)
    rx = np.asarray(rx_ids, dtype=str).reshape(-1)
    day = np.asarray(day_ids, dtype=str).reshape(-1)
    physical = np.asarray(physical_ids, dtype=str).reshape(-1)
    if any(values.size != count for values in (predicted, decision_rows, rx, day, physical)):
        raise CLICSourceMetricsError("known source metric rows do not align")
    if (
        np.any(truth == "")
        or np.any(rx == "")
        or np.any(day == "")
        or np.any(physical == "")
        or len(set(physical.tolist())) != count
    ):
        raise CLICSourceMetricsError("known source metric physical/axis IDs are invalid or reused")
    if set(truth.tolist()).difference(source_order) or set(truth.tolist()) != set(source_order):
        raise CLICSourceMetricsError("known source metric truth classes do not close local4")
    allowed_decisions = {"registered", "unknown", "defer"}
    if set(decision_rows.tolist()).difference(allowed_decisions):
        raise CLICSourceMetricsError("known source metric has an invalid decision")
    unknown_errors = int(np.sum(decision_rows == "unknown"))
    defer_errors = int(np.sum(decision_rows == "defer"))
    correct_mask = (decision_rows == "registered") & (predicted == truth)

    def grouped(axis: np.ndarray, *, expected: Sequence[str] | None, label: str) -> dict[str, dict[str, Any]]:
        labels = tuple(expected) if expected is not None else tuple(sorted(set(axis.tolist())))
        if not labels or set(labels) != set(axis.tolist()):
            raise CLICSourceMetricsError(f"known source metric {label} coverage is incomplete")
        result: dict[str, dict[str, Any]] = {}
        for value in labels:
            mask = axis == value
            result[str(value)] = _checked_accuracy(int(np.sum(correct_mask[mask])), int(np.sum(mask)), label=f"{label}={value}")
        return result

    by_class = grouped(truth, expected=source_order, label="class")
    by_rx = grouped(rx, expected=None, label="receiver")
    by_day = grouped(day, expected=None, label="day")
    overall = _checked_accuracy(int(np.sum(correct_mask)), count, label="overall")
    class_acc = [float(cell["accuracy"]) for cell in by_class.values()]
    rx_acc = [float(cell["accuracy"]) for cell in by_rx.values()]
    day_acc = [float(cell["accuracy"]) for cell in by_day.values()]
    if not all(math.isfinite(value) for value in (*class_acc, *rx_acc, *day_acc, float(overall["accuracy"]))):
        raise CLICSourceMetricsError("known source metric accuracy is non-finite")
    return {
        "role": str(role),
        "scene": scene,
        "row_count": count,
        "overall": overall,
        "macro_accuracy": float(sum(class_acc) / len(class_acc)),
        "by_class": by_class,
        "by_rx": by_rx,
        "by_day": by_day,
        "floors": {
            "overall_accuracy": float(overall["accuracy"]),
            "min_class_accuracy": min(class_acc),
            "min_rx_accuracy": min(rx_acc),
            "min_day_accuracy": min(day_acc),
        },
        "known_unknown_errors": unknown_errors,
        "known_defer_errors": defer_errors,
        "decision_counts": {decision: int(np.sum(decision_rows == decision)) for decision in ("registered", "unknown", "defer")},
        "correctness_rule": "registered_and_unique_local4_prediction_equals_known_truth;unknown_or_defer_is_error",
        "fit_rows": 0,
        "threshold_fit_rows": 0,
        "selection_access": False,
    }


def _validate_metric_block(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CLICSourceMetricsError(f"{label} metric block is absent")
    for key in ("overall", "macro_accuracy", "by_class", "by_rx", "by_day", "floors"):
        if key not in value:
            raise CLICSourceMetricsError(f"{label} metric block lacks {key}")
    overall = value["overall"]
    if not isinstance(overall, Mapping):
        raise CLICSourceMetricsError(f"{label} overall raw cell is invalid")
    _checked_accuracy(overall.get("correct"), overall.get("denominator"), label=f"{label} overall")
    if not math.isclose(float(overall["accuracy"]), int(overall["correct"]) / int(overall["denominator"]), rel_tol=0.0, abs_tol=1e-12):
        raise CLICSourceMetricsError(f"{label} overall accuracy does not bind raw cells")
    for axis in ("by_class", "by_rx", "by_day"):
        cells = value[axis]
        if not isinstance(cells, Mapping) or not cells:
            raise CLICSourceMetricsError(f"{label} {axis} raw cells are absent")
        for cell_label, cell in cells.items():
            if not isinstance(cell, Mapping):
                raise CLICSourceMetricsError(f"{label} {axis}.{cell_label} is invalid")
            _checked_accuracy(cell.get("correct"), cell.get("denominator"), label=f"{label} {axis}.{cell_label}")
            if not math.isclose(float(cell["accuracy"]), int(cell["correct"]) / int(cell["denominator"]), rel_tol=0.0, abs_tol=1e-12):
                raise CLICSourceMetricsError(f"{label} {axis}.{cell_label} accuracy does not bind raw cells")
        if (
            sum(int(cell["correct"]) for cell in cells.values()) != int(overall["correct"])
            or sum(int(cell["denominator"]) for cell in cells.values()) != int(overall["denominator"])
        ):
            raise CLICSourceMetricsError(f"{label} {axis} raw axis totals do not bind overall")
    macro = _finite_number(value["macro_accuracy"], label=f"{label} macro")
    expected_macro = float(np.mean([float(item["accuracy"]) for item in value["by_class"].values()], dtype=np.float64))
    if not math.isclose(macro, expected_macro, rel_tol=0.0, abs_tol=1e-12):
        raise CLICSourceMetricsError(f"{label} macro accuracy does not bind class cells")
    floors = value["floors"]
    if not isinstance(floors, Mapping):
        raise CLICSourceMetricsError(f"{label} floors are invalid")
    expected_floors = {
        "overall_accuracy": float(value["overall"]["accuracy"]),
        "min_class_accuracy": min(float(item["accuracy"]) for item in value["by_class"].values()),
        "min_rx_accuracy": min(float(item["accuracy"]) for item in value["by_rx"].values()),
        "min_day_accuracy": min(float(item["accuracy"]) for item in value["by_day"].values()),
    }
    for field, expected in expected_floors.items():
        actual = _finite_number(floors.get(field), label=f"{label} {field}")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise CLICSourceMetricsError(f"{label} {field} does not bind raw cells")
    return dict(value)


def _pair_core(receipt: Mapping[str, Any]) -> tuple[int, Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(receipt, Mapping):
        raise CLICSourceMetricsError("source metric pair receipt must be a mapping")
    if receipt.get("schema") != SOURCE_METRICS_PAIR_SCHEMA:
        raise CLICSourceMetricsError("source metric pair receipt schema drifted")
    fold = receipt.get("fold_index")
    if type(fold) is not int or fold not in range(1, 7):
        raise CLICSourceMetricsError("source metric pair fold is invalid")
    if (
        receipt.get("source_only") is not True
        or receipt.get("post_target_completion_audit_non_selection") is not True
        or receipt.get("completion_audit") != "POST_TARGET_COMPLETION_AUDIT_NON_SELECTION"
    ):
        raise CLICSourceMetricsError("source metric pair is not a post-target source-only audit")
    _source_order(receipt.get("source_tx_ids", ()))
    if tuple(str(item) for item in receipt.get("formal_scenarios", ())) != EXPECTED_SCENARIOS:
        raise CLICSourceMetricsError("source metric pair formal scene order drifted")
    if type(receipt.get("source_validation_row_count")) is not int or receipt["source_validation_row_count"] <= 0:
        raise CLICSourceMetricsError("source metric pair V row count is invalid")
    required_zero_access = {
        "source_l_rows_read": 0,
        "proxy_rows_read": 0,
        "target_access": False,
        "fit_rows": 0,
        "threshold_fit_rows": 0,
        "selection_access": False,
        "retry_access": False,
    }
    for field, expected in required_zero_access.items():
        if receipt.get(field) != expected or type(receipt.get(field)) is not type(expected):
            raise CLICSourceMetricsError(f"source metric pair {field} access boundary drifted")
    arms = receipt.get("arms")
    proxy = receipt.get("proxy")
    if not isinstance(arms, Mapping) or set(arms) != {"C", "G"} or not isinstance(proxy, Mapping) or set(proxy) != {"C", "G"}:
        raise CLICSourceMetricsError("source metric pair C/G arm or proxy coverage drifted")
    for arm in ("C", "G"):
        state = arms[arm]
        if not isinstance(state, Mapping) or set(state).difference({"clean", "scenes"}) or "clean" not in state or "scenes" not in state:
            raise CLICSourceMetricsError(f"source metric pair {arm} arm structure drifted")
        _validate_metric_block(state["clean"], label=f"{arm} clean")
        scenes = state["scenes"]
        if not isinstance(scenes, Mapping) or set(scenes) != set(EXPECTED_SCENARIOS):
            raise CLICSourceMetricsError(f"source metric pair {arm} scene coverage drifted")
        for scene in EXPECTED_SCENARIOS:
            _validate_metric_block(scenes[scene], label=f"{arm} {scene}")
        diagnostic = proxy[arm]
        if not isinstance(diagnostic, Mapping):
            raise CLICSourceMetricsError(f"source metric pair {arm} proxy diagnostic is invalid")
        for field in ("AUROC_unknown", "u_gap"):
            _finite_number(diagnostic.get(field), label=f"{arm} proxy {field}")
        for field in ("fit_rows", "threshold_fit_rows"):
            if diagnostic.get(field) != 0:
                raise CLICSourceMetricsError(f"{arm} proxy {field} must remain zero")
    return fold, arms, proxy


def _delta_floors(c_metrics: Mapping[str, Any], g_metrics: Mapping[str, Any], *, label: str) -> dict[str, float]:
    c = _validate_metric_block(c_metrics, label=f"C {label}")
    g = _validate_metric_block(g_metrics, label=f"G {label}")
    return {
        field: 100.0 * (float(g["floors"][field]) - float(c["floors"][field]))
        for field in FLOOR_FIELDS
    }


def evaluate_pair_noncompensating_gates(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate all required C/G source gates; poor performance returns false, not an exception."""

    fold, arms, proxy = _pair_core(receipt)
    clean_delta = _delta_floors(arms["C"]["clean"], arms["G"]["clean"], label="clean")
    scene_delta = {
        scene: _delta_floors(arms["C"]["scenes"][scene], arms["G"]["scenes"][scene], label=scene)
        for scene in EXPECTED_SCENARIOS
    }
    all_floor_values = list(clean_delta.values()) + [value for scene in EXPECTED_SCENARIOS for value in scene_delta[scene].values()]
    floor_passed = all(value >= NONCOMPENSATING_DELTA_PP for value in all_floor_values)
    scene_equal = float(np.mean([scene_delta[scene]["overall_accuracy"] for scene in EXPECTED_SCENARIOS], dtype=np.float64))
    scene_equal_passed = scene_equal >= NONCOMPENSATING_DELTA_PP
    proxy_delta_auroc = float(proxy["G"]["AUROC_unknown"]) - float(proxy["C"]["AUROC_unknown"])
    proxy_delta_gap = float(proxy["G"]["u_gap"]) - float(proxy["C"]["u_gap"])
    proxy_passed = proxy_delta_auroc > 0.0 and proxy_delta_gap > 0.0
    return {
        "fold_index": fold,
        "floor_delta_limit_pp": NONCOMPENSATING_DELTA_PP,
        "clean_delta_pp": clean_delta,
        "scene_delta_pp": scene_delta,
        "fold_scene_equal_overall_delta_pp": scene_equal,
        "fold_scene_equal_overall_passed": scene_equal_passed,
        "proxy_delta_AUROC": proxy_delta_auroc,
        "proxy_delta_u_gap": proxy_delta_gap,
        "proxy_strict_AUROC_improvement": proxy_delta_auroc > 0.0,
        "proxy_strict_u_gap_improvement": proxy_delta_gap > 0.0,
        "floor_passed": floor_passed,
        "proxy_passed": proxy_passed,
        "passed": bool(floor_passed and scene_equal_passed and proxy_passed),
        "non_selection": "POST_TARGET_COMPLETION_AUDIT_NON_SELECTION",
    }


def aggregate_source_metric_receipts(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require complete F1..F6/18-cell coverage and apply the global equal-scene gate."""

    if not isinstance(receipts, Sequence) or len(receipts) != 6:
        raise CLICSourceMetricsError("source metric aggregation requires exactly six fold receipts")
    records: dict[int, Mapping[str, Any]] = {}
    gates: dict[int, dict[str, Any]] = {}
    all_scene_deltas: list[float] = []
    for receipt in receipts:
        fold, arms, _proxy = _pair_core(receipt)
        if fold in records:
            raise CLICSourceMetricsError("source metric aggregation folds repeat")
        records[fold] = receipt
        gate = evaluate_pair_noncompensating_gates(receipt)
        gates[fold] = gate
        all_scene_deltas.extend(gate["scene_delta_pp"][scene]["overall_accuracy"] for scene in EXPECTED_SCENARIOS)
        # Access exactly once to retain the structural coverage audit before return.
        if set(arms["C"]["scenes"]) != set(EXPECTED_SCENARIOS):
            raise CLICSourceMetricsError("source metric aggregation scene coverage drifted")
    if set(records) != set(range(1, 7)) or len(all_scene_deltas) != 18:
        raise CLICSourceMetricsError("source metric aggregation does not close F1..F6 x three scenes")
    global_equal = float(np.mean(all_scene_deltas, dtype=np.float64))
    global_passed = global_equal >= NONCOMPENSATING_DELTA_PP
    return {
        "schema": SOURCE_METRICS_AGGREGATE_SCHEMA,
        "source_only": True,
        "post_target_completion_audit_non_selection": True,
        "completion_audit": "POST_TARGET_COMPLETION_AUDIT_NON_SELECTION",
        "folds": {f"F{fold}": gates[fold] for fold in range(1, 7)},
        "global_18_scene_equal_overall_delta_pp": global_equal,
        "global_18_scene_equal_overall_passed": global_passed,
        "floor_delta_limit_pp": NONCOMPENSATING_DELTA_PP,
        "passed": bool(global_passed and all(gate["passed"] for gate in gates.values())),
        "non_selection": "POST_TARGET_COMPLETION_AUDIT_NON_SELECTION",
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CLICSourceMetricsError(f"{label} is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CLICSourceMetricsError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise CLICSourceMetricsError(f"{label} must be an object")
    return value


def _unique_argmax_predictions(logits: np.ndarray, *, source_tx_ids: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    if logits.ndim != 2 or logits.shape[1] != len(source_tx_ids) or not np.isfinite(logits).all():
        raise CLICSourceMetricsError("clean local4 logits are non-finite or shape-invalid")
    maximum = np.max(logits, axis=1, keepdims=True)
    winner_count = np.sum(logits == maximum, axis=1)
    winner = np.argmax(logits, axis=1)
    decisions = np.where(winner_count == 1, "registered", "defer").astype(str)
    predicted = np.asarray([source_tx_ids[int(index)] if decisions[row] == "registered" else "" for row, index in enumerate(winner)], dtype=str)
    return predicted, decisions


def _load_clean_v_evidence(
    *,
    path: Path,
    expected_arm: str,
    fold_index: int,
    source_tx_ids: tuple[str, ...],
    checkpoint_sha256: str,
    terminal_sha256: str,
) -> dict[str, Any]:
    """Use only held-V clean rows for scoring, while metadata checks L/V/proxy disjointness."""

    if not path.is_file():
        raise CLICSourceMetricsError("clean-v4 feature NPZ is missing")
    sha_before = _sha256_file(path)
    required = {
        "z_id", "features", "tx_logits", "raw_labels", "domain_labels", "tx_ids", "rx_ids", "day_ids",
        "eq_ids", "sig_ids", "dataset_role", "channel_views", "sat_scenarios", "manifest_json",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise CLICSourceMetricsError("clean-v4 feature member allowlist drifted")
            manifest_item = np.array(archive["manifest_json"], copy=True)
            roles = np.array(archive["dataset_role"], copy=True)
            tx = np.array(archive["tx_ids"], copy=True)
            rx = np.array(archive["rx_ids"], copy=True)
            day = np.array(archive["day_ids"], copy=True)
            eq = np.array(archive["eq_ids"], copy=True)
            sig = np.array(archive["sig_ids"], copy=True)
            views = np.array(archive["channel_views"], copy=True)
            scenes = np.array(archive["sat_scenarios"], copy=True)
            logits = np.array(archive["tx_logits"], copy=True)
    except CLICSourceMetricsError:
        raise
    except (OSError, ValueError) as exc:
        raise CLICSourceMetricsError("clean-v4 feature NPZ is unreadable") from exc
    if _sha256_file(path) != sha_before:
        raise CLICSourceMetricsError("clean-v4 feature NPZ changed while opening")
    try:
        manifest_text = str(np.asarray(manifest_item).reshape(()).item())
        manifest = json.loads(manifest_text)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CLICSourceMetricsError("clean-v4 manifest is invalid") from exc
    if not isinstance(manifest, Mapping):
        raise CLICSourceMetricsError("clean-v4 manifest must be an object")
    candidate = f"F{fold_index}{expected_arm}_CLIC12"
    required_manifest = {
        "schema": _clean.EXPECTED_LV_EXPORT_SCHEMA,
        "method": _clean.EXPECTED_METHOD,
        "candidate_id": candidate,
        "run_id": _clean.EXPECTED_TRAINING_RUN_ID,
        "training_run_contract": _clean.EXPECTED_TRAINING_RUN_ID,
        "source_only": True,
        "clic_enabled": expected_arm == "G",
        "unlabeled_loader_constructed": False,
        "unlabeled_forward_rows": 0,
        "labeled_validation_physical_disjoint": True,
        "labeled_validation_proxy_physical_disjoint": True,
    }
    for field, expected in required_manifest.items():
        if manifest.get(field) != expected or type(manifest.get(field)) is not type(expected):
            raise CLICSourceMetricsError(f"clean-v4 manifest {field} drifted")
    if tuple(str(item) for item in manifest.get("source_tx_ids", ())) != source_tx_ids:
        raise CLICSourceMetricsError("clean-v4 source class order drifted")
    if manifest.get("source_checkpoint_sha256") != checkpoint_sha256 or manifest.get("terminal_receipt_sha256") != terminal_sha256:
        raise CLICSourceMetricsError("clean-v4 checkpoint/terminal SHA binding drifted")
    for field in ("source_validation_indices_sha256", "source_validation_physical_order_sha256"):
        _require_sha256(manifest.get(field), label=f"clean-v4 {field}")
    row_count = int(np.asarray(tx).reshape(-1).size)
    roles_text = _text_rows(roles, label="clean-v4 roles", row_count=row_count)
    tx_text = _text_rows(tx, label="clean-v4 TX", row_count=row_count)
    rx_text = _text_rows(rx, label="clean-v4 RX", row_count=row_count)
    day_text = _text_rows(day, label="clean-v4 day", row_count=row_count)
    eq_text = _text_rows(eq, label="clean-v4 EQ", row_count=row_count)
    sig_text = _text_rows(sig, label="clean-v4 sig", row_count=row_count)
    views_text = _text_rows(views, label="clean-v4 views", row_count=row_count)
    _text_rows(scenes, label="clean-v4 scenes", row_count=row_count, allow_empty=True)
    if set(roles_text.tolist()) != {"labeled_fit", "source_validation_known", "proxy_unknown"}:
        raise CLICSourceMetricsError("clean-v4 role coverage drifted")
    physical = _physical_keys(tx_text, rx_text, day_text, eq_text, sig_text)
    role_masks = {role: roles_text == role for role in ("labeled_fit", "source_validation_known", "proxy_unknown")}
    role_sets = {role: set(physical[mask].tolist()) for role, mask in role_masks.items()}
    if any(not values for values in role_sets.values()) or role_sets["labeled_fit"] & role_sets["source_validation_known"] or role_sets["labeled_fit"] & role_sets["proxy_unknown"] or role_sets["source_validation_known"] & role_sets["proxy_unknown"]:
        raise CLICSourceMetricsError("clean-v4 source-L/V/proxy physical role overlap drifted")
    v_mask = role_masks["source_validation_known"]
    if not np.all(views_text[v_mask] == "clean"):
        raise CLICSourceMetricsError("clean-v4 held-V channel view drifted")
    v_keys = physical[v_mask]
    if len(set(v_keys.tolist())) != int(np.sum(v_mask)):
        raise CLICSourceMetricsError("clean-v4 held-V physical IDs repeat")
    if _canonical_sha256(v_keys.tolist()) != manifest.get("source_validation_physical_order_sha256"):
        raise CLICSourceMetricsError("clean-v4 held-V physical order SHA drifted")
    if int(manifest.get("source_validation_row_count", -1)) != int(np.sum(v_mask)):
        raise CLICSourceMetricsError("clean-v4 held-V row count drifted")
    logits_matrix = _finite_matrix(logits, label="clean-v4 tx logits", row_count=row_count)
    if logits_matrix.shape[1] != 4:
        raise CLICSourceMetricsError("clean-v4 logits do not bind local4")
    if set(tx_text[v_mask].tolist()) != set(source_tx_ids):
        raise CLICSourceMetricsError("clean-v4 held-V truth classes drifted")
    return {
        "path": path,
        "sha256": sha_before,
        "manifest": dict(manifest),
        "manifest_sha256": _canonical_sha256(manifest),
        "v_truth": tx_text[v_mask],
        "v_rx": rx_text[v_mask],
        "v_day": day_text[v_mask],
        "v_physical_keys": v_keys,
        "v_logits": logits_matrix[v_mask],
        "v_count": int(np.sum(v_mask)),
        "source_l_rows_read": 0,
        "proxy_rows_read": 0,
    }


def _load_v_feature_export(
    *,
    feature_path: Path,
    binding_path: Path,
    expected_arm: str,
    fold_index: int,
    source_tx_ids: tuple[str, ...],
    checkpoint_sha256: str,
    terminal_sha256: str,
    clean_sha256: str,
    clean_validation_indices_sha256: str,
    clean_validation_order_sha256: str,
    cache_sha256: str,
    cache_receipt_sha256: str,
    pair_sha256: str,
    policy_state_sha256: str,
    expected_physical: np.ndarray,
    expected_scenes: np.ndarray,
    expected_tx_ids: np.ndarray,
    expected_rx_ids: np.ndarray,
    expected_day_ids: np.ndarray,
) -> dict[str, Any]:
    if not feature_path.is_file() or not binding_path.is_file():
        raise CLICSourceMetricsError("source-V feature NPZ or binding is missing")
    feature_sha = _sha256_file(feature_path)
    binding_sha = _sha256_file(binding_path)
    binding = _load_json(binding_path, label="source-V feature binding")
    candidate = f"F{fold_index}{expected_arm}_CLIC12"
    binding_expected = {
        "schema": _source_v.SOURCE_V_FEATURE_BINDING_SCHEMA,
        "method": "P1_CLIC",
        "candidate_id": candidate,
        "fold_index": fold_index,
        "arm": expected_arm,
        "source_only": True,
        "post_target_completion_audit_non_selection": True,
        "completion_audit": "POST_TARGET_COMPLETION_AUDIT_NON_SELECTION",
        "checkpoint_sha256": checkpoint_sha256,
        "terminal_receipt_sha256": terminal_sha256,
        "clean_v4_sha256": clean_sha256,
        "source_validation_indices_sha256": clean_validation_indices_sha256,
        "source_validation_physical_order_sha256": clean_validation_order_sha256,
        "source_v_cache_sha256": cache_sha256,
        "source_v_cache_receipt_sha256": cache_receipt_sha256,
        "pair_v3_sha256": pair_sha256,
        "pair_policy_state_sha256": policy_state_sha256,
        "source_tx_ids": list(source_tx_ids),
        "source_v_feature_npz_path": str(feature_path),
        "source_v_feature_npz_sha256": feature_sha,
        "source_l_forward_rows": 0,
        "proxy_forward_rows": 0,
        "target_access": False,
        "fit_rows": 0,
        "threshold_fit_rows": 0,
        "selection_access": False,
    }
    for field, expected in binding_expected.items():
        if binding.get(field) != expected or type(binding.get(field)) is not type(expected):
            raise CLICSourceMetricsError(f"source-V feature binding {field} drifted")
    expected_members = {
        "features", "z_id", "tx_logits", "raw_labels", "domain_labels", "tx_ids", "rx_ids", "day_ids",
        "eq_ids", "sig_ids", "dataset_role", "channel_views", "sat_scenarios", "physical_sample_id", "manifest_json",
    }
    try:
        with np.load(feature_path, allow_pickle=False) as archive:
            if set(archive.files) != expected_members:
                raise CLICSourceMetricsError("source-V feature NPZ member allowlist drifted")
            arrays = {name: np.array(archive[name], copy=True) for name in expected_members}
    except CLICSourceMetricsError:
        raise
    except (OSError, ValueError) as exc:
        raise CLICSourceMetricsError("source-V feature NPZ is unreadable") from exc
    if _sha256_file(feature_path) != feature_sha or _sha256_file(binding_path) != binding_sha:
        raise CLICSourceMetricsError("source-V feature/binding changed while opening")
    try:
        manifest = json.loads(str(np.asarray(arrays["manifest_json"]).reshape(()).item()))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CLICSourceMetricsError("source-V feature manifest is invalid") from exc
    if not isinstance(manifest, Mapping):
        raise CLICSourceMetricsError("source-V feature manifest must be an object")
    manifest_expected = {
        "schema": _source_v.SOURCE_V_FEATURE_SCHEMA,
        "method": "P1_CLIC",
        "candidate_id": candidate,
        "fold_index": fold_index,
        "arm": expected_arm,
        "source_only": True,
        "post_target_completion_audit_non_selection": True,
        "completion_audit": "POST_TARGET_COMPLETION_AUDIT_NON_SELECTION",
        "checkpoint_sha256": checkpoint_sha256,
        "terminal_receipt_sha256": terminal_sha256,
        "clean_v4_sha256": clean_sha256,
        "source_validation_indices_sha256": clean_validation_indices_sha256,
        "source_validation_physical_order_sha256": clean_validation_order_sha256,
        "source_v_cache_sha256": cache_sha256,
        "source_v_cache_receipt_sha256": cache_receipt_sha256,
        "pair_v3_sha256": pair_sha256,
        "pair_policy_state_sha256": policy_state_sha256,
        "source_tx_ids": list(source_tx_ids),
        "source_validation_role": _source_v.SOURCE_V_ROLE,
        "single_leo_observation": True,
        "single_leo_forward_bound": True,
        "source_l_rows_read": 0,
        "proxy_rows_read": 0,
        "target_access": False,
        "fit_rows": 0,
        "threshold_fit_rows": 0,
        "selection_access": False,
    }
    for field, expected in manifest_expected.items():
        if manifest.get(field) != expected or type(manifest.get(field)) is not type(expected):
            raise CLICSourceMetricsError(f"source-V feature manifest {field} drifted")
    row_count = int(np.asarray(arrays["tx_ids"]).reshape(-1).size)
    if row_count <= 0 or manifest.get("single_leo_forward_count") != row_count or binding.get("single_leo_forward_count") != row_count:
        raise CLICSourceMetricsError("source-V feature one-forward row count drifted")
    features = _finite_matrix(arrays["features"], label="source-V features", row_count=row_count)
    z_id = _finite_matrix(arrays["z_id"], label="source-V z_id", row_count=row_count)
    logits = _finite_matrix(arrays["tx_logits"], label="source-V logits", row_count=row_count)
    if features.shape != z_id.shape or not np.array_equal(features, z_id) or logits.shape != (row_count, 4):
        raise CLICSourceMetricsError("source-V feature/z_id/local4 logits binding drifted")
    tx = _text_rows(arrays["tx_ids"], label="source-V feature TX", row_count=row_count)
    rx = _text_rows(arrays["rx_ids"], label="source-V feature RX", row_count=row_count)
    day = _text_rows(arrays["day_ids"], label="source-V feature day", row_count=row_count)
    sig = _text_rows(arrays["sig_ids"], label="source-V feature sig", row_count=row_count)
    physical = _text_rows(arrays["physical_sample_id"], label="source-V feature physical", row_count=row_count)
    roles = _text_rows(arrays["dataset_role"], label="source-V feature role", row_count=row_count)
    views = _text_rows(arrays["channel_views"], label="source-V feature view", row_count=row_count)
    scenes = _text_rows(arrays["sat_scenarios"], label="source-V feature scene", row_count=row_count)
    if set(tx.tolist()).difference(source_tx_ids) or set(roles.tolist()) != {_source_v.SOURCE_V_ROLE} or set(views.tolist()) != {"received_existing"}:
        raise CLICSourceMetricsError("source-V feature source-only role/view class contract drifted")
    validate_source_v_feature_cache_metadata(
        feature_axes={
            "tx_ids": tx,
            "rx_ids": rx,
            "day_ids": day,
            "physical_ids": physical,
            "scenes": scenes,
        },
        cache_axes={
            "tx_ids": expected_tx_ids,
            "rx_ids": expected_rx_ids,
            "day_ids": expected_day_ids,
            "physical_ids": expected_physical,
            "scenes": expected_scenes,
        },
    )
    if not np.array_equal(sig, physical):
        raise CLICSourceMetricsError("source-V feature signal/physical identity drifted")
    if manifest.get("physical_order_sha256") != _canonical_sha256(physical.tolist()) or binding.get("physical_order_sha256") != manifest.get("physical_order_sha256"):
        raise CLICSourceMetricsError("source-V feature physical-order SHA drifted")
    if binding.get("source_v_feature_manifest_sha256") != _canonical_sha256(manifest):
        raise CLICSourceMetricsError("source-V binding manifest SHA drifted")
    return {
        "path": feature_path,
        "sha256": feature_sha,
        "binding_path": binding_path,
        "binding_sha256": binding_sha,
        "manifest": dict(manifest),
        "z_id": z_id,
        "tx_logits": logits,
        "tx_ids": tx,
        "rx_ids": rx,
        "day_ids": day,
        "physical_ids": physical,
        "scenes": scenes,
        "row_count": row_count,
    }


def _open_checkpoint_arm(
    *,
    checkpoint_path: Path,
    terminal_path: Path,
    expected_arm: str,
    fold_index: int,
    training_root: Path,
    source_tx_ids: tuple[str, ...],
) -> dict[str, Any]:
    if checkpoint_path != training_root / f"F{fold_index}{expected_arm}_CLIC12" / "final_ssdg.pth" or terminal_path.parent != checkpoint_path.parent:
        raise CLICSourceMetricsError("source metrics checkpoint/terminal path binding drifted")
    if not checkpoint_path.is_file() or not terminal_path.is_file():
        raise CLICSourceMetricsError("source metrics checkpoint/terminal is missing")
    checkpoint_sha = _sha256_file(checkpoint_path)
    terminal_sha = _sha256_file(terminal_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CLICSourceMetricsError("source metrics final checkpoint is unreadable") from exc
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("args"), Mapping):
        raise CLICSourceMetricsError("source metrics checkpoint payload is malformed")
    checkpoint_args = checkpoint["args"]
    try:
        known = _clean._parse_csv(checkpoint_args.get("phase1_source_known_validation_tx_ids", ""), label="source metrics known TX")
        proxy = _clean._parse_csv(checkpoint_args.get("phase1_source_proxy_unknown_tx_ids", ""), label="source metrics proxy TX")
        _args, receipt, observed_arm = _clean.validate_clic_training_checkpoint(
            checkpoint,
            checkpoint_path=checkpoint_path,
            terminal_receipt_path=terminal_path,
            source_tx_ids=source_tx_ids,
            known_validation_tx_ids=known,
            proxy_unknown_tx_ids=proxy,
        )
    except _clean.CLICSplitExportError as exc:
        raise CLICSourceMetricsError(f"source metrics checkpoint/terminal strict reopen failed: {exc}") from exc
    if observed_arm != expected_arm:
        raise CLICSourceMetricsError("source metrics checkpoint arm drifted")
    return {"checkpoint_sha256": checkpoint_sha, "terminal_sha256": terminal_sha, "terminal": receipt}


def _assert_input_hashes(paths: Mapping[str, Path], expected: Mapping[str, str]) -> None:
    for name, path in paths.items():
        if _sha256_file(path) != expected[name]:
            raise CLICSourceMetricsError(f"source metrics input changed during evaluation: {name}")


def _publish_immutable_json(path: Path, payload: Mapping[str, Any], *, label: str) -> Any:
    """Use Task1's pre-sealed no-replace publication and verify it immediately."""

    try:
        publication = _cache._atomic_write_json(path, payload)
        digest = publication.sha256
        if not isinstance(digest, str):
            raise CLICSourceMetricsError(f"{label} publication lacks a pre-publish SHA seal")
        _cache._assert_publication_current(publication, expected_sha256=digest, label=label)
        return publication
    except CLICSourceMetricsError:
        raise
    except Exception as exc:
        raise CLICSourceMetricsError(f"{label} immutable publication failed: {exc}") from exc


def score_source_metrics_pair(args: argparse.Namespace) -> dict[str, Any]:
    """Write one immutable F* C/G source-only clean+V metrics receipt."""

    fold = int(args.fold_index)
    if fold not in range(1, 7):
        raise CLICSourceMetricsError("source metrics fold must be F1..F6")
    source_tx_ids = _source_order(args.source_tx_ids)
    training_root = Path(args.training_run_root).resolve()
    clean_root = Path(args.clean_run_root).resolve()
    cache_root = Path(args.cache_run_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_json = Path(args.output_metrics_json).resolve()
    if (
        training_root.name != _clean.EXPECTED_TRAINING_RUN_ID
        or clean_root.name != _cache.EXPECTED_CLEAN_RUN_ID
        or cache_root.name != EXPECTED_RUN_ID
        or output_root.name != EXPECTED_RUN_ID
        or not (training_root.parent == clean_root.parent == cache_root.parent == output_root.parent)
    ):
        raise CLICSourceMetricsError("source metrics training/clean/cache/output root binding drifted")
    expected_output = output_root / f"F{fold}_PAIR" / "source_metrics_pair.json"
    if output_json != expected_output:
        raise CLICSourceMetricsError("source metrics pair output path must be canonical and immutable")
    if output_json.exists():
        raise CLICSourceMetricsError("refusing to overwrite source metrics pair receipt")
    paths: dict[str, Path] = {
        "c_checkpoint": Path(args.c_ckpt).resolve(),
        "c_terminal": Path(args.c_terminal_receipt_json).resolve(),
        "g_checkpoint": Path(args.g_ckpt).resolve(),
        "g_terminal": Path(args.g_terminal_receipt_json).resolve(),
        "c_clean": Path(args.c_clean_npz).resolve(),
        "g_clean": Path(args.g_clean_npz).resolve(),
        "cache": Path(args.source_v_received_iq_npz).resolve(),
        "cache_receipt": Path(args.source_v_received_iq_receipt_json).resolve(),
        "pair": Path(args.pair_json).resolve(),
        "c_feature": Path(args.c_source_v_feature_npz).resolve(),
        "g_feature": Path(args.g_source_v_feature_npz).resolve(),
        "c_binding": Path(args.c_source_v_binding_json).resolve(),
        "g_binding": Path(args.g_source_v_binding_json).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise CLICSourceMetricsError("source metrics input artifact is missing")
    expected_cache_dir = cache_root / f"F{fold}_SHARED"
    if paths["cache"] != expected_cache_dir / "source_validation_known_leo_weak.npz" or paths["cache_receipt"] != expected_cache_dir / "source_validation_known_leo_weak.receipt.json":
        raise CLICSourceMetricsError("source metrics shared source-V cache path drifted")
    input_sha = {name: _sha256_file(path) for name, path in paths.items()}
    opened = {
        arm: _open_checkpoint_arm(
            checkpoint_path=paths[f"{arm.lower()}_checkpoint"],
            terminal_path=paths[f"{arm.lower()}_terminal"],
            expected_arm=arm,
            fold_index=fold,
            training_root=training_root,
            source_tx_ids=source_tx_ids,
        )
        for arm in ("C", "G")
    }
    cache_snapshot = _source_v.read_source_v_cache_snapshot(
        cache_path=paths["cache"],
        cache_receipt_path=paths["cache_receipt"],
        fold_index=fold,
        source_tx_ids=source_tx_ids,
    )
    if cache_snapshot["cache_sha256"] != input_sha["cache"] or cache_snapshot["cache_receipt_sha256"] != input_sha["cache_receipt"]:
        raise CLICSourceMetricsError("source metrics cache snapshot SHA drifted")
    for arm in ("C", "G"):
        if cache_snapshot["receipt"].get("checkpoint_sha256_by_arm", {}).get(arm) != opened[arm]["checkpoint_sha256"]:
            raise CLICSourceMetricsError("source metrics cache/checkpoint binding drifted")
        if cache_snapshot["receipt"].get("terminal_receipt_sha256_by_arm", {}).get(arm) != opened[arm]["terminal_sha256"]:
            raise CLICSourceMetricsError("source metrics cache/terminal binding drifted")
    pair_payload = _load_json(paths["pair"], label="PAIR-v3 source policy receipt")
    if (
        pair_payload.get("schema") != _pair.EXPECTED_PAIR_SCHEMA
        or pair_payload.get("fold_index") != fold
        or pair_payload.get("source_only") is not True
        or pair_payload.get("target_artifacts_present") is not False
        or tuple(str(item) for item in pair_payload.get("source_tx_ids", ())) != source_tx_ids
    ):
        raise CLICSourceMetricsError("PAIR-v3 source policy receipt binding drifted")
    try:
        common_binding = _source_v.validate_pair_single_leo_common_binding(pair_payload.get("single_leo_common_binding"))
    except _source_v.CLICSourceVFeatureExportError as exc:
        raise CLICSourceMetricsError(f"PAIR-v3 single-LEO source-L binding drifted: {exc}") from exc
    states = pair_payload.get("clic_source_policy_state")
    proxy = pair_payload.get("proxy_diagnostic")
    if not isinstance(states, Mapping) or set(states) != {"C", "G"} or not isinstance(proxy, Mapping) or set(proxy) != {"C", "G"}:
        raise CLICSourceMetricsError("PAIR-v3 policy/proxy coverage drifted")
    policy_state: dict[str, dict[str, Any]] = {}
    proxy_readonly: dict[str, dict[str, Any]] = {}
    for arm in ("C", "G"):
        try:
            policy_state[arm] = _pair._validated_clic_source_policy_state(
                states[arm],
                fold_index=fold,
                arm=arm,
                checkpoint_sha256=opened[arm]["checkpoint_sha256"],
                terminal_receipt_sha256=opened[arm]["terminal_sha256"],
            )
        except _pair.CLICPostfreezePairError as exc:
            raise CLICSourceMetricsError(f"PAIR-v3 source-L policy state drifted: {exc}") from exc
        try:
            _source_v.validate_pair_source_l_policy_binding(common_binding, policy_state[arm]["policies"])
        except _source_v.CLICSourceVFeatureExportError as exc:
            raise CLICSourceMetricsError(f"PAIR-v3 source-L policy/binding drifted: {exc}") from exc
        diagnostic = proxy[arm]
        if not isinstance(diagnostic, Mapping) or diagnostic.get("schema") != "cvs.phase1.clic_proxy_diagnostic.v1":
            raise CLICSourceMetricsError("PAIR-v3 proxy diagnostic drifted")
        for field in ("AUROC_unknown", "u_gap"):
            _finite_number(diagnostic.get(field), label=f"PAIR {arm} {field}")
        try:
            _source_v._validate_pair_proxy_diagnostic(diagnostic)
        except _source_v.CLICSourceVFeatureExportError as exc:
            raise CLICSourceMetricsError(f"PAIR-v3 proxy diagnostic contract drifted: {exc}") from exc
        proxy_readonly[arm] = {
            "AUROC_unknown": float(diagnostic["AUROC_unknown"]),
            "u_gap": float(diagnostic["u_gap"]),
            "fit_rows": 0,
            "threshold_fit_rows": 0,
        }
    clean = {
        arm: _load_clean_v_evidence(
            path=paths[f"{arm.lower()}_clean"],
            expected_arm=arm,
            fold_index=fold,
            source_tx_ids=source_tx_ids,
            checkpoint_sha256=opened[arm]["checkpoint_sha256"],
            terminal_sha256=opened[arm]["terminal_sha256"],
        )
        for arm in ("C", "G")
    }
    if not np.array_equal(clean["C"]["v_physical_keys"], clean["G"]["v_physical_keys"]):
        raise CLICSourceMetricsError("C/G clean-v4 held-V physical key binding drifted")
    if clean["C"]["v_count"] != clean["G"]["v_count"]:
        raise CLICSourceMetricsError("C/G clean-v4 held-V row-count binding drifted")
    validate_clean_v_cache_identity(
        cache_receipt=cache_snapshot["receipt"],
        clean_manifests={"C": clean["C"]["manifest"], "G": clean["G"]["manifest"]},
    )
    features = {
        arm: _load_v_feature_export(
            feature_path=paths[f"{arm.lower()}_feature"],
            binding_path=paths[f"{arm.lower()}_binding"],
            expected_arm=arm,
            fold_index=fold,
            source_tx_ids=source_tx_ids,
            checkpoint_sha256=opened[arm]["checkpoint_sha256"],
            terminal_sha256=opened[arm]["terminal_sha256"],
            clean_sha256=input_sha[f"{arm.lower()}_clean"],
            clean_validation_indices_sha256=clean[arm]["manifest"]["source_validation_indices_sha256"],
            clean_validation_order_sha256=clean[arm]["manifest"]["source_validation_physical_order_sha256"],
            cache_sha256=input_sha["cache"],
            cache_receipt_sha256=input_sha["cache_receipt"],
            pair_sha256=input_sha["pair"],
            policy_state_sha256=policy_state[arm]["state_sha256"],
            expected_physical=cache_snapshot["physical_ids"],
            expected_scenes=cache_snapshot["sat_scenarios"],
            expected_tx_ids=cache_snapshot["tx_ids"],
            expected_rx_ids=cache_snapshot["rx_ids"],
            expected_day_ids=cache_snapshot["day_ids"],
        )
        for arm in ("C", "G")
    }
    if features["C"]["binding_sha256"] == "" or features["C"]["manifest"].get("source_v_cache_sha256") != features["G"]["manifest"].get("source_v_cache_sha256"):
        raise CLICSourceMetricsError("C/G source-V cache SHA binding drifted")
    if not np.array_equal(features["C"]["physical_ids"], features["G"]["physical_ids"]) or not np.array_equal(features["C"]["scenes"], features["G"]["scenes"]):
        raise CLICSourceMetricsError("C/G source-V physical/scene binding drifted")
    metrics_arms: dict[str, dict[str, Any]] = {}
    used_physical_by_arm_scene: dict[str, dict[str, set[str]]] = {"C": {}, "G": {}}
    for arm in ("C", "G"):
        clean_predicted, clean_decisions = _unique_argmax_predictions(clean[arm]["v_logits"], source_tx_ids=source_tx_ids)
        clean_metrics = score_known_source_rows(
            truth_tx_ids=clean[arm]["v_truth"],
            predicted_tx_ids=clean_predicted,
            decisions=clean_decisions,
            rx_ids=clean[arm]["v_rx"],
            day_ids=clean[arm]["v_day"],
            physical_ids=clean[arm]["v_physical_keys"],
            role="source_validation_known_clean",
            scene=None,
            source_tx_ids=source_tx_ids,
        )
        scene_metrics: dict[str, Any] = {}
        for scene in EXPECTED_SCENARIOS:
            mask = features[arm]["scenes"] == scene
            if int(np.sum(mask)) <= 0:
                raise CLICSourceMetricsError("source-V formal scene has zero denominator")
            current_physical = features[arm]["physical_ids"][mask]
            current_set = set(current_physical.tolist())
            existing_physical = set().union(*used_physical_by_arm_scene[arm].values()) if used_physical_by_arm_scene[arm] else set()
            if current_set & existing_physical:
                raise CLICSourceMetricsError("source-V physical row reused across formal scenes")
            used_physical_by_arm_scene[arm][scene] = current_set
            try:
                scored = _pair.score_clic_open_set(
                    policy_state[arm]["geometry"],
                    policy_state[arm]["policies"][scene],
                    features[arm]["z_id"][mask],
                    features[arm]["tx_logits"][mask],
                    scene,
                )
            except _pair.CLICPostfreezePairError as exc:
                raise CLICSourceMetricsError(f"source-V score-only policy application failed: {exc}") from exc
            if scored.get("fit_rows") != 0 or scored.get("threshold_fit_rows") != 0:
                raise CLICSourceMetricsError("source-V score-only policy fit/threshold contract drifted")
            scene_metrics[scene] = score_known_source_rows(
                truth_tx_ids=features[arm]["tx_ids"][mask],
                predicted_tx_ids=scored["predicted_class"],
                decisions=scored["decision"],
                rx_ids=features[arm]["rx_ids"][mask],
                day_ids=features[arm]["day_ids"][mask],
                physical_ids=current_physical,
                role=_source_v.SOURCE_V_ROLE,
                scene=scene,
                source_tx_ids=source_tx_ids,
            )
        metrics_arms[arm] = {"clean": clean_metrics, "scenes": scene_metrics}
    if any(set(used_physical_by_arm_scene[arm]) != set(EXPECTED_SCENARIOS) for arm in ("C", "G")):
        raise CLICSourceMetricsError("source-V formal scene usage is incomplete")
    _assert_input_hashes(paths, input_sha)
    receipt = {
        "schema": SOURCE_METRICS_PAIR_SCHEMA,
        "method": "P1_CLIC",
        "fold_index": fold,
        "source_only": True,
        "post_target_completion_audit_non_selection": True,
        "completion_audit": "POST_TARGET_COMPLETION_AUDIT_NON_SELECTION",
        "non_selection": "POST_TARGET_COMPLETION_AUDIT_NON_SELECTION",
        "training_run_id": _clean.EXPECTED_TRAINING_RUN_ID,
        "clean_evidence_run_id": _cache.EXPECTED_CLEAN_RUN_ID,
        "source_tx_ids": list(source_tx_ids),
        "formal_scenarios": list(EXPECTED_SCENARIOS),
        "input_sha256": input_sha,
        "shared_source_v_cache_sha256": input_sha["cache"],
        "shared_source_v_cache_receipt_sha256": input_sha["cache_receipt"],
        "source_validation_row_count": int(cache_snapshot["row_count"]),
        "arms": metrics_arms,
        "proxy": proxy_readonly,
        "source_l_rows_read": 0,
        "proxy_rows_read": 0,
        "target_access": False,
        "fit_rows": 0,
        "threshold_fit_rows": 0,
        "selection_access": False,
        "retry_access": False,
    }
    receipt["gates"] = evaluate_pair_noncompensating_gates(receipt)
    publication: Any | None = None
    try:
        publication = _publish_immutable_json(output_json, receipt, label="source metrics pair receipt")
        _assert_input_hashes(paths, input_sha)
        _cache._assert_publication_current(
            publication, expected_sha256=publication.sha256, label="source metrics pair receipt before return"
        )
    except Exception:
        if publication is not None:
            _cache._unlink_if_owned(publication)
        raise
    return receipt


def _aggregate_from_paths(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    output_json = Path(args.output_metrics_json).resolve()
    if output_root.name != EXPECTED_RUN_ID or output_json != output_root / "source_metrics_aggregate.json":
        raise CLICSourceMetricsError("source metrics aggregate output path must be canonical")
    if output_json.exists():
        raise CLICSourceMetricsError("refusing to overwrite source metrics aggregate receipt")
    paths = [Path(item).resolve() for item in args.input_pair_metrics_json]
    if len(paths) != 6:
        raise CLICSourceMetricsError("source metrics aggregate requires exactly six pair receipt paths")
    if len(set(paths)) != len(paths):
        raise CLICSourceMetricsError("source metrics aggregate input receipt paths must be distinct")
    sha_before = {str(path): _sha256_file(path) for path in paths}
    receipts = [_load_json(path, label="source metrics pair receipt") for path in paths]
    aggregate = aggregate_source_metric_receipts(receipts)
    aggregate["input_pair_receipt_sha256"] = dict(sha_before)
    aggregate["non_selection"] = "POST_TARGET_COMPLETION_AUDIT_NON_SELECTION"
    if any(_sha256_file(path) != sha_before[str(path)] for path in paths):
        raise CLICSourceMetricsError("source metrics pair receipt changed during aggregation")
    publication: Any | None = None
    try:
        publication = _publish_immutable_json(output_json, aggregate, label="source metrics aggregate receipt")
        if any(_sha256_file(path) != sha_before[str(path)] for path in paths):
            raise CLICSourceMetricsError("source metrics pair receipt changed after aggregate publish")
        _cache._assert_publication_current(
            publication, expected_sha256=publication.sha256, label="source metrics aggregate receipt before return"
        )
    except Exception:
        if publication is not None:
            _cache._unlink_if_owned(publication)
        raise
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-folds", action="store_true")
    parser.add_argument("--fold-index", type=int)
    parser.add_argument("--training-run-root")
    parser.add_argument("--clean-run-root")
    parser.add_argument("--cache-run-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-metrics-json", required=True)
    parser.add_argument("--source-tx-ids")
    for arm in ("c", "g"):
        parser.add_argument(f"--{arm}-ckpt")
        parser.add_argument(f"--{arm}-terminal-receipt-json")
        parser.add_argument(f"--{arm}-clean-npz")
        parser.add_argument(f"--{arm}-source-v-feature-npz")
        parser.add_argument(f"--{arm}-source-v-binding-json")
    parser.add_argument("--source-v-received-iq-npz")
    parser.add_argument("--source-v-received-iq-receipt-json")
    parser.add_argument("--pair-json")
    parser.add_argument("--input-pair-metrics-json", nargs="*")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.aggregate_folds:
        forbidden = (
            "fold_index", "training_run_root", "clean_run_root", "cache_run_root", "source_tx_ids",
            "c_ckpt", "g_ckpt", "c_terminal_receipt_json", "g_terminal_receipt_json", "c_clean_npz", "g_clean_npz",
            "c_source_v_feature_npz", "g_source_v_feature_npz", "c_source_v_binding_json", "g_source_v_binding_json",
            "source_v_received_iq_npz", "source_v_received_iq_receipt_json", "pair_json",
        )
        if any(getattr(args, field) is not None for field in forbidden) or not args.input_pair_metrics_json:
            parser.error("--aggregate-folds accepts only --output-root, --output-metrics-json and six --input-pair-metrics-json paths")
        result = _aggregate_from_paths(args)
    else:
        required = (
            "fold_index", "training_run_root", "clean_run_root", "cache_run_root", "source_tx_ids",
            "c_ckpt", "g_ckpt", "c_terminal_receipt_json", "g_terminal_receipt_json", "c_clean_npz", "g_clean_npz",
            "c_source_v_feature_npz", "g_source_v_feature_npz", "c_source_v_binding_json", "g_source_v_binding_json",
            "source_v_received_iq_npz", "source_v_received_iq_receipt_json", "pair_json",
        )
        missing = [field.replace("_", "-") for field in required if getattr(args, field) is None]
        if missing or args.input_pair_metrics_json:
            parser.error("source metrics pair mode requires: " + ", ".join(missing) if missing else "pair mode forbids --input-pair-metrics-json")
        result = score_source_metrics_pair(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLICSourceMetricsError",
    "EXPECTED_SCENARIOS",
    "SOURCE_METRICS_AGGREGATE_SCHEMA",
    "SOURCE_METRICS_PAIR_SCHEMA",
    "aggregate_source_metric_receipts",
    "build_parser",
    "evaluate_pair_noncompensating_gates",
    "main",
    "score_known_source_rows",
    "score_source_metrics_pair",
    "validate_clean_v_cache_identity",
    "validate_source_v_feature_cache_metadata",
]

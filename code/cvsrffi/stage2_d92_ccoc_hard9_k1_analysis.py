"""Truth-last analyzer and sole frozen verdict for D92 CCOC Hard9+K1.

The analyzer consumes only sealed matrix/job/score artifacts.  It never opens
query truth before immutable prediction and score artifacts have been bound,
and it never feeds a result back into a runner or a method decision.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cvsrffi import stage2_d92_ccoc_hard9_k1 as matrix


EIGHT_PARETO_METRICS = (
    "h_old_new",
    "old_balanced_accuracy",
    "c_old_acc",
    "old_floor",
    "seen_new_acc",
    "average_forgetting",
    "new_to_old_rate",
    "old_to_new_rate",
)
PARETO_METRICS = EIGHT_PARETO_METRICS
VERDICTS = (
    "REJECT_ROUTE",
    "REVISE_ONCE",
    "ADVANCE_TO_TARGET125_CANDIDATE",
)
SCENES = tuple(matrix.SCENES)
ARM_ID = matrix.ARM_ID
CANDIDATE_ID = matrix.CANDIDATE_ID
QUERY_ZERO_FIELDS = tuple(matrix.QUERY_ZERO_FIELDS)
_TOLERANCE = 1.0e-12
PREDICTION_CLOSURE_SHA_FIELDS = (
    "before_prediction_sha256",
    "after_prediction_sha256",
    "before_commit_sha256",
    "after_commit_sha256",
    "before_fit_audit_sha256",
    "after_fit_audit_sha256",
    "before_resource_audit_sha256",
    "after_resource_audit_sha256",
    "before_execution_receipt_sha256",
    "after_execution_receipt_sha256",
)
_PREDICTION_CLOSURE_FILES = {
    "before_prediction_sha256": ("before", "prediction_artifact.npz"),
    "after_prediction_sha256": ("after", "prediction_artifact.npz"),
    "before_commit_sha256": ("before", "COMMIT.json"),
    "after_commit_sha256": ("after", "COMMIT.json"),
    "before_fit_audit_sha256": ("before", "fit_audit.json"),
    "after_fit_audit_sha256": ("after", "fit_audit.json"),
    "before_resource_audit_sha256": ("before", "resource_audit.json"),
    "after_resource_audit_sha256": ("after", "resource_audit.json"),
    "before_execution_receipt_sha256": ("before", "execution_receipt.json"),
    "after_execution_receipt_sha256": ("after", "execution_receipt.json"),
}
_PREDICTION_ARTIFACT_MEMBERS = (
    "query_tokens",
    "scenarios",
    "predicted_class_handles",
)

# These are the already-used D92 direction/magnitude gates.  They are copied
# as analysis constants only; CCOC does not tune or redefine them.
STRICT_PARETO_THRESHOLDS = {
    "h_old_new": 0.010,
    "old_balanced_accuracy": 0.015,
    "c_old_acc": 0.010,
    "old_floor": 0.040,
    "seen_new_acc": 0.005,
    "average_forgetting": -0.015,
    "new_to_old_rate": -0.005,
    "old_to_new_rate": -0.005,
}
DIRECTION_GATE = {
    metric: {
        "direction": ">" if metric in EIGHT_PARETO_METRICS[:5] else "<",
        "magnitude": threshold,
    }
    for metric, threshold in STRICT_PARETO_THRESHOLDS.items()
}


class D92CCOCHard9K1AnalysisError(ValueError):
    """Raised when a frozen CCOC evidence join is incomplete or detached."""


def _fail(message: str) -> D92CCOCHard9K1AnalysisError:
    return D92CCOCHard9K1AnalysisError(message)


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise _fail(f"missing regular artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_value(value: Any, label: str) -> str:
    result = str(value or "").lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise _fail(f"invalid SHA256 {label}")
    return result


def _prediction_closure_hashes(job_root: Path) -> dict[str, str]:
    """Recompute the ten Task1 immutable prediction-closure hashes."""

    return {
        field: _sha(_state_root(job_root, state) / name)
        for field, (state, name) in _PREDICTION_CLOSURE_FILES.items()
    }


def _validate_closure_surface(
    surface: Mapping[str, Any],
    closure_hashes: Mapping[str, str],
    *,
    label: str,
) -> None:
    for field in PREDICTION_CLOSURE_SHA_FIELDS:
        actual = _sha256_value(surface.get(field), f"{label}/{field}")
        if actual != closure_hashes[field]:
            raise _fail(f"Task1 prediction closure SHA drift: {label}/{field}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise _fail(f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise _fail(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise _fail(f"JSON object required: {path}")
    return value


def _read_csv(path: Path, *, expected_sha256: str | None = None) -> list[dict[str, str]]:
    actual = _sha(path)
    if expected_sha256 and actual.lower() != str(expected_sha256).lower():
        raise _fail(f"frozen CSV SHA drift: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error) as error:
        raise _fail(f"invalid CSV artifact: {path}") from error


def _finite(
    value: Any,
    label: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise _fail(f"non-numeric {label}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise _fail(f"non-numeric {label}") from error
    if not math.isfinite(result):
        raise _fail(f"non-finite {label}")
    if lower is not None and result < lower:
        raise _fail(f"out-of-range {label}")
    if upper is not None and result > upper:
        raise _fail(f"out-of-range {label}")
    return result


def _integer(value: Any, label: str, *, lower: int = 0) -> int:
    number = _finite(value, label, lower=float(lower))
    if number != float(int(number)):
        raise _fail(f"non-integer {label}")
    return int(number)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise _fail("empty mean")
    return float(statistics.fmean(values))


def _p90(values: Sequence[float]) -> float:
    if not values:
        raise _fail("empty p90")
    return float(sorted(values)[max(0, math.ceil(0.90 * len(values)) - 1)])


def compute_confusion_rates(score: Mapping[str, Any]) -> dict[str, float]:
    """Compute count-weighted old/new confusion from sealed score surfaces."""

    before = score.get("before")
    after = score.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise _fail("score state surface missing")
    before_scene = before.get("by_scenario")
    after_scene = after.get("by_scenario")
    if not isinstance(before_scene, Mapping) or not isinstance(after_scene, Mapping):
        return {
            "new_to_old_rate": _finite(after.get("new_to_old_rate"), "new-to-old", lower=0.0, upper=1.0),
            "old_to_new_rate": _finite(after.get("old_to_new_rate"), "old-to-new", lower=0.0, upper=1.0),
        }
    if set(before_scene) != set(after_scene) or set(after_scene) != set(SCENES):
        raise _fail("bidirectional scenario closure drift")
    old_total = new_total = old_to_new = new_to_old = 0.0
    for scene in SCENES:
        before_row = before_scene.get(scene)
        after_row = after_scene.get(scene)
        if not isinstance(before_row, Mapping) or not isinstance(after_row, Mapping):
            raise _fail(f"scenario row missing: {scene}")
        old_count = _finite(before_row.get("query_count"), f"before query count {scene}", lower=0.0)
        total_count = _finite(after_row.get("query_count"), f"after query count {scene}", lower=0.0)
        new_count = total_count - old_count
        if new_count < -_TOLERANCE:
            raise _fail(f"scenario query count regressed: {scene}")
        new_count = max(0.0, new_count)
        new_rate = _finite(after_row.get("new_to_old_rate"), f"new-to-old {scene}", lower=0.0, upper=1.0)
        old_rate = _finite(after_row.get("old_to_new_rate"), f"old-to-new {scene}", lower=0.0, upper=1.0)
        old_total += old_count
        new_total += new_count
        old_to_new += old_rate * old_count
        new_to_old += new_rate * new_count
    if old_total <= 0 or new_total <= 0:
        raise _fail("old/new query count closure is empty")
    result = {
        "new_to_old_rate": new_to_old / new_total,
        "old_to_new_rate": old_to_new / old_total,
    }
    for name, calculated in result.items():
        aggregate = after.get(name)
        if aggregate is not None and abs(_finite(aggregate, name, lower=0.0, upper=1.0) - calculated) > 1.0e-9:
            raise _fail(f"aggregate {name} disagrees with scenario weighting")
    return result


def compute_old_balanced_accuracy(by_tx: Mapping[str, Any]) -> float:
    values = [
        _finite(row.get("accuracy"), f"old accuracy {tx}", lower=0.0, upper=1.0)
        for tx, row in by_tx.items()
        if isinstance(row, Mapping) and row.get("role") == "target_old"
    ]
    if len(values) != 6:
        raise _fail(f"expected six old classes, got {len(values)}")
    return _mean(values)


def compute_score_metrics(score: Mapping[str, Any]) -> dict[str, Any]:
    before = score.get("before")
    after = score.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise _fail("score state surface missing")
    by_tx = after.get("by_tx")
    if not isinstance(by_tx, Mapping):
        raise _fail("score by_tx surface missing")
    old_accuracy = {
        str(tx): _finite(row.get("accuracy"), f"old accuracy {tx}", lower=0.0, upper=1.0)
        for tx, row in by_tx.items()
        if isinstance(row, Mapping) and row.get("role") == "target_old"
    }
    if len(old_accuracy) != 6:
        raise _fail("old class closure drift")
    confusion = compute_confusion_rates(score)
    after_old = _finite(after.get("old_acc"), "c_old_acc", lower=0.0, upper=1.0)
    before_old = _finite(before.get("old_acc"), "DA1_REG0 old accuracy", lower=0.0, upper=1.0)
    forgetting = (
        _finite(score.get("old_forgetting_pp"), "average_forgetting") / 100.0
        if score.get("old_forgetting_pp") is not None
        else before_old - after_old
    )
    query_value = after.get("query_macs", score.get("query_macs"))
    state_value = after.get("after_state_bytes", score.get("state_bytes"))
    query_macs = _integer(query_value, "query MACs") if query_value is not None else -1
    state_bytes = _integer(state_value, "state bytes") if state_value is not None else -1
    return {
        "h_old_new": _finite(after.get("h_old_new"), "H_old_new", lower=0.0, upper=1.0),
        "old_balanced_accuracy": compute_old_balanced_accuracy(by_tx),
        "c_old_acc": after_old,
        "old_floor": min(old_accuracy.values()),
        "seen_new_acc": _finite(after.get("seen_new_acc"), "seen-new accuracy", lower=0.0, upper=1.0),
        "average_forgetting": forgetting,
        "new_to_old_rate": confusion["new_to_old_rate"],
        "old_to_new_rate": confusion["old_to_new_rate"],
        "new_to_old_error": confusion["new_to_old_rate"],
        "old_to_new_error": confusion["old_to_new_rate"],
        "old_class_accuracy": old_accuracy,
        "old_class_count": len(old_accuracy),
        "query_macs": query_macs,
        "state_bytes": state_bytes,
        "da1_reg0_old_acc": before_old,
        "da1_reg0_old_floor": _finite(
            score.get("per_old_class_floor_before"),
            "DA1_REG0 old floor",
            lower=0.0,
            upper=1.0,
        ),
    }


def strict_pareto_deltas(candidate: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, float]:
    aliases = {"new_to_old_rate": "new_to_old_error", "old_to_new_rate": "old_to_new_error"}
    result: dict[str, float] = {}
    for metric in EIGHT_PARETO_METRICS:
        c_key = metric if metric in candidate else aliases.get(metric, metric)
        b_key = metric if metric in baseline else aliases.get(metric, metric)
        result[metric] = float(candidate[c_key]) - float(baseline[b_key])
    return result


def _strict_row_ok(deltas: Mapping[str, float]) -> bool:
    return all(
        float(deltas[metric]) > _TOLERANCE
        if metric in EIGHT_PARETO_METRICS[:5]
        else float(deltas[metric]) < -_TOLERANCE
        for metric in EIGHT_PARETO_METRICS
    )


def _magnitude_ok(deltas: Mapping[str, float]) -> bool:
    return all(
        float(deltas[metric]) >= STRICT_PARETO_THRESHOLDS[metric] - _TOLERANCE
        if metric in EIGHT_PARETO_METRICS[:5]
        else float(deltas[metric]) <= STRICT_PARETO_THRESHOLDS[metric] + _TOLERANCE
        for metric in EIGHT_PARETO_METRICS
    )


def decide_verdict(gate_state: Mapping[str, bool]) -> str:
    """Apply the sole frozen precedence: hard rejection, one revision, advance."""

    required = (
        "complete_artifact_closure",
        "performance_outer_closure",
        "all_strict_pareto",
        "all_magnitude",
        "stability",
        "resource_integrity",
        "resource_hard",
        "resource_target",
    )
    if any(name not in gate_state for name in required):
        return "REJECT_ROUTE"
    if not all(bool(gate_state[name]) for name in ("complete_artifact_closure", "performance_outer_closure", "all_strict_pareto", "resource_integrity", "resource_hard")):
        return "REJECT_ROUTE"
    if not bool(gate_state["all_magnitude"]) or not bool(gate_state["stability"]) or not bool(gate_state["resource_target"]):
        return "REVISE_ONCE"
    return "ADVANCE_TO_TARGET125_CANDIDATE"


def validate_truth_binding(
    score: Mapping[str, Any],
    receipt: Mapping[str, Any],
    job: Mapping[str, Any],
    truth_path: str | Path,
) -> str:
    """Bind manifest, receipt, score and the actual immutable truth sidecar."""

    actual_path = Path(truth_path).resolve()
    manifest_path = Path(str(job.get("truth_sidecar", ""))).resolve()
    expected = str(job.get("truth_sidecar_sha256", "")).lower()
    outer_key = str(job.get("outer_key", ""))
    expected_suffix = ("jobs", outer_key, "offline", "scorer", "truth_sidecar.json")
    def suffix(path: Path) -> tuple[str, ...]:
        return tuple(part for part in str(path).replace("\\", "/").split("/") if part)[-5:]
    same_logical_path = bool(outer_key) and suffix(manifest_path) == expected_suffix and suffix(actual_path) == expected_suffix
    if not (manifest_path == actual_path or same_logical_path) or not actual_path.is_file() or actual_path.is_symlink() or len(expected) != 64:
        raise _fail("truth sidecar path/hash closure drift")
    actual = _sha(actual_path)
    if actual != expected or str(receipt.get("truth_sidecar_sha256", "")).lower() != actual or str(score.get("truth_sidecar_sha256", "")).lower() != actual:
        raise _fail("truth sidecar hash binding drift")
    return actual


def validate_per_old_class_join(
    rows: Sequence[Mapping[str, Any]],
    raw_score: Mapping[str, Any],
    *,
    outer_key: str,
) -> dict[str, dict[str, float]]:
    """Join each frozen per-old row to the same outer's E0 raw by-TX values."""

    by_tx = raw_score.get("after", {}).get("by_tx") if isinstance(raw_score.get("after"), Mapping) else None
    if not isinstance(by_tx, Mapping):
        raise _fail(f"per-old raw by_tx missing for {outer_key}")
    raw_old = {
        str(tx): _finite(value.get("accuracy"), f"raw old accuracy {tx}", lower=0.0, upper=1.0)
        for tx, value in by_tx.items()
        if isinstance(value, Mapping) and value.get("role") == "target_old"
    }
    if len(raw_old) != 6:
        raise _fail(f"per-old raw old-class closure missing for {outer_key}")
    if not isinstance(rows, Sequence) or len(rows) != 6:
        raise _fail(f"per-old row closure drift for {outer_key}")
    joined: dict[str, dict[str, float]] = {}
    for row in rows:
        if str(row.get("outer_key")) != str(outer_key):
            raise _fail(f"per-old outer join drift for {outer_key}")
        tx = str(row.get("tx"))
        if tx in joined or tx not in raw_old:
            raise _fail(f"per-old TX join drift for {outer_key}/{tx}")
        historical_candidate = _finite(row.get("candidate_accuracy"), f"historical candidate {tx}", lower=0.0, upper=1.0)
        # The historical per-old row is an E0 receipt, so it must agree with
        # the same outer's sealed raw score. It is not a candidate backfill.
        if abs(historical_candidate - raw_old[tx]) > 1.0e-9:
            raise _fail(f"per-old E0 accuracy mismatch for {outer_key}/{tx}")
        baseline_accuracy = _finite(row.get("baseline_accuracy", historical_candidate), f"historical baseline {tx}", lower=0.0, upper=1.0)
        delta = _finite(row.get("delta_accuracy", historical_candidate - baseline_accuracy), f"historical delta {tx}")
        joined[tx] = {
            "candidate_accuracy": historical_candidate,
            "e0_accuracy": raw_old[tx],
            "historical_baseline_accuracy": baseline_accuracy,
            "historical_delta_accuracy": delta,
        }
    if set(joined) != set(raw_old):
        raise _fail(f"per-old TX set closure drift for {outer_key}")
    return joined


def validate_paired_e0_row(
    row: Mapping[str, Any],
    raw_score: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    outer_key: str,
    metrics: Mapping[str, Any] | None = None,
) -> None:
    """Bind the selected historical paired E0 row to its sealed raw score.

    The paired CSV is an independent frozen receipt, not a positional hint.
    Its selected row must describe this exact outer and agree with the raw E0
    score values it claims to summarize.
    """

    if str(row.get("outer_key", "")) != outer_key:
        raise _fail(f"paired E0 outer-key drift: {outer_key}")
    identity = {
        "receiver": str(job["receiver"]),
        "seed": _integer(job["seed"], "job seed", lower=0),
        "k_shot": _integer(job["k_shot"], "job K", lower=1),
        "new_class_count": _integer(job["new_class_count"], "job new-class count", lower=1),
        "slice": f"K{job['k_shot']}_new{job['new_class_count']}",
    }
    for field, expected in identity.items():
        actual = row.get(field)
        if field in {"seed", "k_shot", "new_class_count"}:
            if _integer(actual, f"paired E0 {field}", lower=0) != expected:
                raise _fail(f"paired E0 identity drift: {outer_key}/{field}")
        elif str(actual) != str(expected):
            raise _fail(f"paired E0 identity drift: {outer_key}/{field}")

    metrics = dict(metrics) if metrics is not None else compute_score_metrics(raw_score)
    evidence = {
        "candidate_h_old_new": metrics["h_old_new"],
        "candidate_old_acc": metrics["c_old_acc"],
        "candidate_old_floor": metrics["old_floor"],
        "candidate_seen_new_acc": metrics["seen_new_acc"],
        "candidate_forgetting": metrics["average_forgetting"],
        "candidate_da1_reg0_old_acc": metrics["da1_reg0_old_acc"],
        "candidate_da1_reg0_old_floor": metrics["da1_reg0_old_floor"],
    }
    for field, expected in evidence.items():
        actual = _finite(
            row.get(field),
            f"paired E0 {field}",
            lower=None if field == "candidate_forgetting" else 0.0,
        )
        if abs(actual - float(expected)) > 1.0e-9:
            raise _fail(f"paired E0/raw-score drift: {outer_key}/{field}")

    resources = _e0_scene_resources(job, outer_key)
    query_macs = {
        _integer(value.get("query_macs"), f"E0 query MACs {scene}")
        for scene, value in resources.items()
    }
    state_bytes = {
        _integer(value.get("state_bytes"), f"E0 state bytes {scene}", lower=1)
        for scene, value in resources.items()
    }
    if len(query_macs) != 1 or len(state_bytes) != 1:
        raise _fail(f"paired E0 query/state scene drift: {outer_key}")
    if _integer(row.get("query_macs"), "paired E0 query MACs") != next(iter(query_macs)):
        raise _fail(f"paired E0 query-MAC drift: {outer_key}")
    if _integer(row.get("state_bytes"), "paired E0 state bytes", lower=1) != next(iter(state_bytes)):
        raise _fail(f"paired E0 state-byte drift: {outer_key}")


def _logical_truth_path(job: Mapping[str, Any], truth_sidecar_root: str | Path | None) -> Path:
    if truth_sidecar_root is None:
        return Path(str(job["truth_sidecar"])).resolve(strict=True)
    root = Path(truth_sidecar_root).resolve(strict=True)
    outer = str(job["outer_key"])
    return root / "jobs" / outer / "offline" / "scorer" / "truth_sidecar.json"


def _state_root(job_root: Path, state: str) -> Path:
    candidates = (job_root / "diag" / state, job_root / state)
    for candidate in candidates:
        if candidate.is_dir() and not candidate.is_symlink():
            return candidate
    raise _fail(f"{state} prediction state missing: {job_root}")


def _validate_commit_state(state_root: Path) -> dict[str, Any]:
    commit_path = state_root / "COMMIT.json"
    commit = _read_json(commit_path)
    members = commit.get("members")
    if not isinstance(members, list):
        raise _fail(f"commit members missing: {state_root}")
    by_path: dict[str, Mapping[str, Any]] = {}
    for item in members:
        if not isinstance(item, Mapping) or not isinstance(item.get("relative_path"), str) or item["relative_path"] in by_path:
            raise _fail(f"commit member drift: {state_root}")
        by_path[str(item["relative_path"])] = item
    required = {"execution_receipt.json", "fit_audit.json", "prediction_artifact.npz", "resource_audit.json"}
    if not required.issubset(by_path):
        raise _fail(f"commit member closure drift: {state_root}")
    for relative, item in by_path.items():
        path = (state_root / relative).resolve()
        if state_root.resolve() not in path.parents or not path.is_file() or path.is_symlink():
            raise _fail(f"commit member missing: {relative}")
        actual = _sha(path)
        if actual != str(item.get("sha256", "")).lower():
            raise _fail(f"commit member SHA drift: {relative}")
        size = item.get("size_bytes")
        if size is not None and _integer(size, f"commit size {relative}") != path.stat().st_size:
            raise _fail(f"commit member size drift: {relative}")
    prediction_sha = _sha(state_root / "prediction_artifact.npz")
    if str(commit.get("prediction_artifact_sha256", "")).lower() != prediction_sha:
        raise _fail(f"commit prediction SHA drift: {state_root}")
    return {"path": str(commit_path), "sha256": _sha(commit_path), "prediction_sha256": prediction_sha}


def _validate_fit_rows(
    job: Mapping[str, Any],
    job_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    after_root = _state_root(job_root, "after")
    fit_path = after_root / "fit_audit.json"
    payload: Any = None
    if fit_path.is_file() and not fit_path.is_symlink():
        try:
            payload = json.loads(fit_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            raise _fail("invalid fit audit JSON") from error
    if not isinstance(payload, list) or len(payload) != len(SCENES) or any(not isinstance(row, Mapping) for row in payload):
        raise _fail("fit audit scene closure drift")
    by_scene: dict[str, Mapping[str, Any]] = {}
    for row in payload:
        scene = str(row.get("scenario", ""))
        if scene in by_scene or scene not in SCENES:
            raise _fail("fit audit duplicate/missing outer_key+scene")
        by_scene[scene] = row
        if row.get("arm_id") != ARM_ID or row.get("candidate_id") != CANDIDATE_ID:
            raise _fail("fit audit arm/candidate identity drift")
        for field in QUERY_ZERO_FIELDS:
            for alias in (field, f"d92_e0d_{field}", f"d92_e0d_ccoc_{field}"):
                if row.get(alias) is not False:
                    raise _fail("fit audit query access is not zero")
        k_shot = _integer(job.get("k_shot"), "job K", lower=1)
        prefix = "d92_e0d_ccoc_"
        inventory = row.get("after_actual_component_inventory")
        if not isinstance(inventory, Mapping):
            raise _fail("fit audit component inventory missing")
        total = _integer(row.get("after_total_component_fit_count"), "fit total")
        actual = _integer(inventory.get("actual_component_fit_count"), "fit actual")
        mode = str(row.get("after_registered_d_mode_effective", ""))
        if k_shot > 2:
            expected = (matrix.FIT_GATE["k_gt_2_total"], matrix.FIT_GATE["k_gt_2_actual"], "ccoc_full")
            if (total, actual, mode) != expected or row.get(prefix + "active") is not True or row.get(prefix + "fallback_active") is not False or row.get(prefix + "fallback_reason") is not None or _integer(row.get(prefix + "candidate_attempt_fit_count"), "candidate fit") != 1 or _integer(row.get(prefix + "fallback_reference_fit_count"), "fallback reference fit") != 0 or row.get(prefix + "candidate_statistic_receipt_available") is not True or row.get(prefix + "paired_e0_codec_state_equal") is not None or row.get(prefix + "g0_eligible") is not True or row.get(prefix + "g0_block_reason") is not None or _integer(row.get(prefix + "query_rows_used"), "query rows") != 0:
                raise _fail("fit audit CCOC active lifecycle drift")
        else:
            expected = (matrix.FIT_GATE["k1_total"], matrix.FIT_GATE["k1_actual"], "d92_full_alias")
            if (total, actual, mode) != expected or row.get(prefix + "active") is not False or row.get(prefix + "fallback_active") is not False or row.get(prefix + "fallback_reason") != matrix.FIT_GATE["k1_alias"] or _integer(row.get(prefix + "candidate_attempt_fit_count"), "candidate fit") != 0 or _integer(row.get(prefix + "fallback_reference_fit_count"), "fallback reference fit") != 0 or row.get(prefix + "candidate_statistic_receipt_available") is not False or row.get(prefix + "paired_e0_codec_state_equal") is not None or row.get(prefix + "g0_eligible") is not False or row.get(prefix + "g0_block_reason") != matrix.FIT_GATE["k1_alias"] or _integer(row.get(prefix + "query_rows_used"), "query rows") != 0:
                raise _fail("fit audit K1 alias lifecycle drift")
        if row.get("after_state_postprocess_mode") is not None:
            raise _fail("fit audit postprocess drift")
        class_count = _integer(row.get("registered_class_count"), "registered class count", lower=1)
        query_macs = _integer(row.get("query_macs"), "candidate query MACs")
        state_bytes = _integer(row.get("after_state_bytes"), "candidate state bytes", lower=1)
        if query_macs != class_count * 288:
            raise _fail("fit audit query MAC receipt drift")
        resource = row.get("after_registration_resource")
        if not isinstance(resource, Mapping):
            raise _fail("fit audit resource receipt missing")
        _finite(resource.get("registration_wall_time_ns"), "candidate registration wall", lower=0.0)
        _finite(resource.get("registration_incremental_peak_working_set_bytes"), "candidate registration peak", lower=0.0)
        # Fit-audit state/query values are retained for the scene keyed join.
        row = dict(row)
        row["_query_macs"] = query_macs
        row["_state_bytes"] = state_bytes
        row["_class_count"] = class_count
        by_scene[scene] = row
    if set(by_scene) != set(SCENES):
        raise _fail("fit audit missing scene")
    # The before state also has to be an immutable prediction/fit closure.
    _validate_commit_state(_state_root(job_root, "before"))
    _validate_commit_state(_state_root(job_root, "after"))
    after_rows = [dict(by_scene[scene]) for scene in SCENES]
    return after_rows, dict(by_scene)


def _e0_scene_resources(job: Mapping[str, Any], outer_key: str) -> dict[str, Mapping[str, Any]]:
    record = job.get("e0_resource")
    if not isinstance(record, Mapping):
        raise _fail(f"E0 resource record missing: {outer_key}")
    scenes = record.get("scenes")
    if not isinstance(scenes, Mapping) or set(scenes) != set(SCENES):
        raise _fail(f"E0 resource scene identity drift: {outer_key}")
    result: dict[str, Mapping[str, Any]] = {}
    for scene in SCENES:
        item = scenes.get(scene)
        if not isinstance(item, Mapping):
            raise _fail(f"E0 resource scene missing: {outer_key}/{scene}")
        result[scene] = item
        for field in ("registration_wall_time_ns", "registration_incremental_peak_working_set_bytes", "query_macs", "state_bytes"):
            _finite(item.get(field), f"E0 {field} {outer_key}/{scene}", lower=0.0)
    return result


def _resource_join(job: Mapping[str, Any], job_root: Path, fit_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    e0_by_scene = _e0_scene_resources(job, str(job["outer_key"]))
    candidate_by_scene = {str(row["scenario"]): row for row in fit_rows}
    if set(candidate_by_scene) != set(SCENES) or len(candidate_by_scene) != len(fit_rows):
        raise _fail("candidate resource scene duplicate/missing")
    result: list[dict[str, Any]] = []
    for scene in SCENES:
        candidate = candidate_by_scene[scene]
        baseline = e0_by_scene[scene]
        resource = candidate["after_registration_resource"]
        wall = _finite(resource.get("registration_wall_time_ns"), "candidate wall", lower=0.0)
        peak = _finite(resource.get("registration_incremental_peak_working_set_bytes"), "candidate peak", lower=0.0)
        base_wall = _finite(baseline.get("registration_wall_time_ns"), "E0 wall", lower=0.0)
        if base_wall <= 0:
            raise _fail("E0 registration wall is zero")
        ratio = wall / base_wall
        query_exact = int(candidate["_query_macs"]) == _integer(baseline.get("query_macs"), "E0 query MACs")
        state_exact = int(candidate["_state_bytes"]) == _integer(baseline.get("state_bytes"), "E0 state bytes", lower=1)
        result.append({
            "outer_key": str(job["outer_key"]),
            "arm_id": str(job.get("arm_id", ARM_ID)),
            "scenario": scene,
            "k_shot": int(job["k_shot"]),
            "candidate_wall_ns": wall,
            "e0_wall_ns": base_wall,
            "wall_ratio": ratio,
            "candidate_peak_bytes": peak,
            "e0_peak_bytes": _finite(baseline.get("registration_incremental_peak_working_set_bytes"), "E0 peak", lower=0.0),
            "candidate_query_macs": int(candidate["_query_macs"]),
            "e0_query_macs": int(baseline["query_macs"]),
            "candidate_state_bytes": int(candidate["_state_bytes"]),
            "e0_state_bytes": int(baseline["state_bytes"]),
            "query_state_exact": bool(query_exact and state_exact),
            "wall_hard_pass": wall <= 150_000_000,
            "ratio_hard_pass": ratio <= 1.50,
            "peak_hard_pass": peak <= float(matrix.RESOURCE_GATE["candidate_peak_hard_max_bytes"]),
            "wall_target_pass": wall <= 120_000_000,
            "ratio_target_pass": ratio <= 1.25,
            "peak_target_pass": peak <= float(matrix.RESOURCE_GATE["candidate_peak_target_max_bytes"]),
        })
    return result


def evaluate_resource_gate(candidate_rows: Sequence[Mapping[str, Any]], *, query_state_exact: bool = True) -> dict[str, Any]:
    """Evaluate keyed per-scene CCOC resource receipts.

    Candidate peak is absolute.  E0 peak is reported for traceability only and
    never offsets the 1 MiB hard ceiling.
    """

    if not candidate_rows:
        raise _fail("resource row closure drift")
    hard_scene = all(bool(row.get("wall_hard_pass")) and bool(row.get("ratio_hard_pass")) and bool(row.get("peak_hard_pass")) and bool(row.get("query_state_exact")) for row in candidate_rows)
    walls = [_finite(row.get("candidate_wall_ns"), "candidate wall", lower=0.0) for row in candidate_rows]
    ratios = [_finite(row.get("wall_ratio"), "wall ratio", lower=0.0) for row in candidate_rows]
    peaks = [_finite(row.get("candidate_peak_bytes"), "candidate peak", lower=0.0) for row in candidate_rows]
    wall_p90 = _p90(walls)
    ratio_p90 = _p90(ratios)
    peak_max = max(peaks)
    target = (
        wall_p90 <= float(matrix.RESOURCE_GATE["registration_wall_p90_target_max_ns"])
        and ratio_p90 <= float(matrix.RESOURCE_GATE["registration_wall_ratio_target_max"])
        and peak_max <= float(matrix.RESOURCE_GATE["candidate_peak_target_max_bytes"])
    )
    return {
        "query_state_exact": bool(query_state_exact) and all(bool(row.get("query_state_exact")) for row in candidate_rows),
        "hard_limits_passed": bool(hard_scene and query_state_exact),
        "hard_passed": bool(hard_scene and query_state_exact),
        "target_limits_passed": bool(target),
        "target_passed": bool(target and hard_scene and query_state_exact),
        "wall_p90_ns": wall_p90,
        "wall_ratio_p90": ratio_p90,
        "wall_ratio_median": float(statistics.median(ratios)),
        "candidate_peak_max_bytes": peak_max,
        "candidate_peak_target_pass": peak_max <= float(matrix.RESOURCE_GATE["candidate_peak_target_max_bytes"]),
        "scene_count": len(candidate_rows),
        "failed_scenes": [
            f"{row.get('outer_key')}/{row.get('scenario')}"
            for row in candidate_rows
            if not (row.get("wall_hard_pass") and row.get("ratio_hard_pass") and row.get("peak_hard_pass") and row.get("query_state_exact"))
        ],
    }


def _scenario_closure(score: Mapping[str, Any], label: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    before = score.get("before")
    after = score.get("after")
    before_scene = before.get("by_scenario") if isinstance(before, Mapping) else None
    after_scene = after.get("by_scenario") if isinstance(after, Mapping) else None
    if not isinstance(before_scene, Mapping) or not isinstance(after_scene, Mapping) or set(before_scene) != set(SCENES) or set(after_scene) != set(SCENES):
        raise _fail(f"{label} 3-scene closure drift")
    for scene in SCENES:
        if not isinstance(before_scene[scene], Mapping) or not isinstance(after_scene[scene], Mapping):
            raise _fail(f"{label} scene row invalid: {scene}")
    return before_scene, after_scene


def _canonical_query_identity(scene: str, tokens: Sequence[str]) -> str:
    """Hash a scene's explicit query-ID set, independent of artifact order."""

    if not tokens or len(tokens) != len(set(tokens)):
        raise _fail(f"query token closure drift: {scene}")
    payload = {
        "scenario": str(scene),
        "query_tokens": sorted(str(token) for token in tokens),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_prediction_artifact(path: str | Path, *, label: str) -> dict[str, list[str]]:
    """Read the exact immutable diag-cosine NPZ surface without a sidecar."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise _fail(f"{label} prediction artifact missing or symlinked")
    try:
        with np.load(source, allow_pickle=False) as archive:
            if tuple(archive.files) != _PREDICTION_ARTIFACT_MEMBERS:
                raise _fail(f"{label} prediction artifact exact schema drift")
            values = {name: archive[name].astype(str).tolist() for name in archive.files}
    except D92CCOCHard9K1AnalysisError:
        raise
    except Exception as error:  # malformed ZIP/NPZ evidence is an audit failure
        raise _fail(f"{label} prediction artifact unreadable") from error
    lengths = {len(value) for value in values.values()}
    if lengths == {0} or len(lengths) != 1:
        raise _fail(f"{label} prediction artifact rows are empty or misaligned")
    keys = list(zip(values["scenarios"], values["query_tokens"]))
    if len(keys) != len(set(keys)):
        raise _fail(f"{label} prediction scenario/token key is duplicated")
    if set(values["scenarios"]) != set(SCENES):
        raise _fail(f"{label} prediction scenario registry drift")
    return values


def _read_truth_surface(path: str | Path) -> dict[str, dict[str, str]]:
    """Parse the same manifest-bound truth sidecar used by E0 and CCOC."""

    payload = _read_json(Path(path))
    rows = payload.get("rows")
    if payload.get("schema") != "cvs.phase2.query_truth_sidecar.v2" or not isinstance(rows, list):
        raise _fail("truth sidecar schema drift")
    result: dict[str, dict[str, str]] = {}
    required = {"query_token", "true_class_handle", "transmitter_label", "evaluation_role"}
    for row in rows:
        if not isinstance(row, Mapping) or not required.issubset(row):
            raise _fail("truth sidecar row schema drift")
        token = str(row["query_token"])
        role = str(row["evaluation_role"])
        if not token or token in result or role not in {"target_old", "target_new"}:
            raise _fail("truth token/role drift")
        result[token] = {
            "true_class_handle": str(row["true_class_handle"]),
            "transmitter_label": str(row["transmitter_label"]),
            "evaluation_role": role,
        }
    if not result:
        raise _fail("truth sidecar is empty")
    return result


def _harmonic(old_accuracy: float, new_accuracy: float) -> float:
    total = old_accuracy + new_accuracy
    return 0.0 if total <= 0.0 else 2.0 * old_accuracy * new_accuracy / total


def _surface_metrics(
    prediction: Mapping[str, Sequence[str]],
    truth: Mapping[str, Mapping[str, str]],
    *,
    label: str,
) -> dict[str, Any]:
    """Recompute the full score surface directly from prediction plus truth."""

    handle_roles: dict[str, str] = {}
    for truth_row in truth.values():
        handle = str(truth_row["true_class_handle"])
        role = str(truth_row["evaluation_role"])
        previous = handle_roles.setdefault(handle, role)
        if previous != role:
            raise _fail("registered class handle has mixed roles")
    rows: list[dict[str, Any]] = []
    for scene, token, predicted in zip(
        prediction["scenarios"],
        prediction["query_tokens"],
        prediction["predicted_class_handles"],
    ):
        if token not in truth:
            raise _fail(f"{label} prediction token is absent from truth sidecar")
        target = truth[token]
        rows.append(
            {
                "scene": str(scene),
                "token": str(token),
                "tx": str(target["transmitter_label"]),
                "role": str(target["evaluation_role"]),
                "true": str(target["true_class_handle"]),
                "predicted": str(predicted),
                "predicted_role": handle_roles.get(str(predicted)),
                "correct": int(str(predicted) == str(target["true_class_handle"])),
            }
        )

    def summarize(selected: Sequence[Mapping[str, Any]], *, scene: str | None) -> dict[str, Any]:
        old = [row for row in selected if row["role"] == "target_old"]
        new = [row for row in selected if row["role"] == "target_new"]
        if not old:
            raise _fail(f"{label} old-class surface is empty")
        old_by_tx: dict[str, list[int]] = defaultdict(list)
        by_tx: dict[str, list[int]] = defaultdict(list)
        roles_by_tx: dict[str, str] = {}
        for row in selected:
            tx = str(row["tx"])
            by_tx[tx].append(int(row["correct"]))
            previous_role = roles_by_tx.setdefault(tx, str(row["role"]))
            if previous_role != row["role"]:
                raise _fail(f"{label} transmitter role drift: {tx}")
            if row["role"] == "target_old":
                old_by_tx[tx].append(int(row["correct"]))
        if len(old_by_tx) != 6:
            raise _fail(f"{label} expected six old classes, got {len(old_by_tx)}")
        old_class_accuracy = {tx: _mean(values) for tx, values in sorted(old_by_tx.items())}
        old_acc = _mean(int(row["correct"]) for row in old)
        new_acc = _mean(int(row["correct"]) for row in new) if new else None
        old_to_new = _mean(int(row["predicted_role"] == "target_new") for row in old)
        new_to_old = _mean(int(row["predicted_role"] == "target_old") for row in new) if new else None
        token_values = [str(row["token"]) for row in selected]
        return {
            "query_count": len(selected),
            "old_query_count": len(old),
            "new_query_count": len(new),
            "query_identity_sha256": _canonical_query_identity(scene or "all", token_values),
            "query_tokens": tuple(sorted(token_values)),
            "old_acc": old_acc,
            "seen_new_acc": new_acc,
            "h_old_new": _harmonic(old_acc, new_acc) if new_acc is not None else None,
            "old_to_new_rate": old_to_new,
            "new_to_old_rate": new_to_old,
            "old_class_accuracy": old_class_accuracy,
            "old_balanced_accuracy": _mean(old_class_accuracy.values()),
            "c_old_acc": old_acc,
            "old_floor": min(old_class_accuracy.values()),
            "by_tx": {
                tx: {
                    "role": roles_by_tx[tx],
                    "count": len(values),
                    "accuracy": _mean(values),
                }
                for tx, values in sorted(by_tx.items())
            },
        }

    scenes: dict[str, dict[str, Any]] = {}
    for scene in SCENES:
        scenes[scene] = summarize([row for row in rows if row["scene"] == scene], scene=scene)
    return {"scenes": scenes, "aggregate": summarize(rows, scene=None)}


def _raw_e0_prediction_paths(raw_score_path: Path, raw_score: Mapping[str, Any], *, outer_key: str) -> dict[str, Path]:
    """Derive only the frozen E0 sibling artifacts from a raw-score path."""

    if raw_score_path.name != "diag_cosine_score.json" or raw_score_path.parent.name != "scorer":
        raise _fail(f"E0 raw-score path shape drift: {outer_key}")
    root = raw_score_path.parent.parent
    result: dict[str, Path] = {}
    for state in ("before", "after"):
        path = root / "diag" / state / "prediction_artifact.npz"
        expected = _sha256_value(raw_score.get(f"{state}_prediction_sha256"), f"E0 {state} prediction {outer_key}")
        if path.is_symlink() or not path.is_file():
            raise _fail(f"E0 {state} prediction artifact missing or symlinked: {outer_key}")
        if _sha(path) != expected:
            raise _fail(f"E0 {state} prediction SHA drift: {outer_key}")
        result[state] = path
    return result


def _compare_score_metric(actual: Any, expected: Any, *, label: str, required: bool = False) -> None:
    if actual is None:
        if required and expected is not None:
            raise _fail(f"score aggregate missing: {label}")
        return
    if expected is None:
        raise _fail(f"score aggregate unexpected: {label}")
    if abs(_finite(actual, label) - float(expected)) > 1.0e-9:
        raise _fail(f"score/artifact aggregate drift: {label}")


def _validate_score_against_surface(score: Mapping[str, Any], surface: Mapping[str, Any], *, label: str) -> None:
    """Cross-check all raw-score values that actually exist against real rows."""

    for state in ("before", "after"):
        score_state = score.get(state)
        expected_state = surface[state]
        if not isinstance(score_state, Mapping) or not isinstance(expected_state, Mapping):
            raise _fail(f"{label} score state surface missing")
        score_scenes = score_state.get("by_scenario")
        expected_scenes = expected_state["scenes"]
        if not isinstance(score_scenes, Mapping) or set(score_scenes) != set(SCENES):
            raise _fail(f"{label} score 3-scene closure drift")
        for scene in SCENES:
            actual_scene = score_scenes[scene]
            if not isinstance(actual_scene, Mapping):
                raise _fail(f"{label} score scene row invalid: {scene}")
            expected_scene = expected_scenes[scene]
            for field in (
                "query_count",
                "old_acc",
                "seen_new_acc",
                "h_old_new",
                "old_to_new_rate",
                "new_to_old_rate",
            ):
                if field in {"query_count", "old_acc", "seen_new_acc", "h_old_new"} and field not in actual_scene:
                    raise _fail(f"score aggregate missing: {label}/{state}/{scene}/{field}")
                _compare_score_metric(
                    actual_scene.get(field),
                    expected_scene[field],
                    label=f"{label}/{state}/{scene}/{field}",
                    required=field in {"query_count", "old_acc", "seen_new_acc", "h_old_new"},
                )
        actual_by_tx = score_state.get("by_tx")
        expected_by_tx = expected_state["aggregate"]["by_tx"]
        if not isinstance(actual_by_tx, Mapping) or set(actual_by_tx) != set(expected_by_tx):
            raise _fail(f"{label} score by-TX closure drift: {state}")
        for tx, expected in expected_by_tx.items():
            actual = actual_by_tx[tx]
            if not isinstance(actual, Mapping) or actual.get("role") != expected["role"]:
                raise _fail(f"{label} score by-TX role drift: {state}/{tx}")
            if _integer(actual.get("count"), f"{label}/{state}/{tx}/count") != expected["count"]:
                raise _fail(f"{label} score by-TX count drift: {state}/{tx}")
            _compare_score_metric(actual.get("accuracy"), expected["accuracy"], label=f"{label}/{state}/{tx}/accuracy", required=True)
        for field in ("query_count", "old_acc", "seen_new_acc", "h_old_new", "old_to_new_rate", "new_to_old_rate"):
            if field in {"query_count", "old_acc", "seen_new_acc", "h_old_new"} and field not in score_state:
                raise _fail(f"score aggregate missing: {label}/{state}/{field}")
            _compare_score_metric(
                score_state.get(field),
                expected_state["aggregate"][field],
                label=f"{label}/{state}/{field}",
                required=field in {"query_count", "old_acc", "seen_new_acc", "h_old_new"},
            )
    before = surface["before"]["aggregate"]
    after = surface["after"]["aggregate"]
    _compare_score_metric(
        score.get("old_forgetting_pp"),
        100.0 * (float(before["old_acc"]) - float(after["old_acc"])),
        label=f"{label}/old_forgetting_pp",
        required=True,
    )
    _compare_score_metric(score.get("per_old_class_floor_before"), before["old_floor"], label=f"{label}/per_old_class_floor_before", required=True)
    _compare_score_metric(score.get("per_old_class_floor_after"), after["old_floor"], label=f"{label}/per_old_class_floor_after", required=True)


def _surface_pair(
    before_path: Path,
    after_path: Path,
    truth: Mapping[str, Mapping[str, str]],
    *,
    label: str,
) -> dict[str, Any]:
    before = _surface_metrics(_read_prediction_artifact(before_path, label=f"{label}/before"), truth, label=f"{label}/before")
    after = _surface_metrics(_read_prediction_artifact(after_path, label=f"{label}/after"), truth, label=f"{label}/after")
    for scene in SCENES:
        if before["scenes"][scene]["seen_new_acc"] is not None or after["scenes"][scene]["seen_new_acc"] is None:
            raise _fail(f"{label} before/after registration role coverage drift: {scene}")
        if set(before["scenes"][scene]["old_class_accuracy"]) != set(after["scenes"][scene]["old_class_accuracy"]):
            raise _fail(f"{label} matched old-class registry drift: {scene}")
    return {"before": before, "after": after}


def _validate_candidate_e0_surface(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    outer_key: str,
    candidate_before_path: Path,
    candidate_after_path: Path,
    raw_score_path: Path,
    truth_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind candidate and E0 to one real truth/query surface before comparison."""

    job_truth = _sha256_value(job.get("truth_sidecar_sha256"), f"job truth {outer_key}")
    candidate_truth = _sha256_value(candidate.get("truth_sidecar_sha256"), f"candidate truth {outer_key}")
    baseline_truth = _sha256_value(baseline.get("truth_sidecar_sha256"), f"E0 raw-score truth {outer_key}")
    if candidate_truth != job_truth or baseline_truth != job_truth:
        raise _fail(f"candidate/E0 truth surface drift: {outer_key}")
    raw_paths = _raw_e0_prediction_paths(raw_score_path, baseline, outer_key=outer_key)
    truth = _read_truth_surface(truth_path)
    candidate_surface = _surface_pair(candidate_before_path, candidate_after_path, truth, label=f"candidate {outer_key}")
    baseline_surface = _surface_pair(raw_paths["before"], raw_paths["after"], truth, label=f"E0 {outer_key}")
    _validate_score_against_surface(candidate, candidate_surface, label=f"candidate {outer_key}")
    _validate_score_against_surface(baseline, baseline_surface, label=f"E0 {outer_key}")
    for state in ("before", "after"):
        for scene in SCENES:
            candidate_scene = candidate_surface[state]["scenes"][scene]
            baseline_scene = baseline_surface[state]["scenes"][scene]
            if (
                candidate_scene["query_count"] != baseline_scene["query_count"]
                or candidate_scene["query_tokens"] != baseline_scene["query_tokens"]
                or candidate_scene["query_identity_sha256"] != baseline_scene["query_identity_sha256"]
            ):
                raise _fail(f"candidate/E0 query surface drift: {outer_key}/{state}/{scene}")
    return candidate_surface, baseline_surface


def _metrics_from_surface_pair(surface: Mapping[str, Any]) -> dict[str, Any]:
    before = surface["before"]["aggregate"]
    after = surface["after"]["aggregate"]
    return {
        "h_old_new": after["h_old_new"],
        "old_balanced_accuracy": after["old_balanced_accuracy"],
        "c_old_acc": after["c_old_acc"],
        "old_floor": after["old_floor"],
        "seen_new_acc": after["seen_new_acc"],
        "average_forgetting": float(before["old_acc"]) - float(after["old_acc"]),
        "new_to_old_rate": after["new_to_old_rate"],
        "old_to_new_rate": after["old_to_new_rate"],
        "new_to_old_error": after["new_to_old_rate"],
        "old_to_new_error": after["old_to_new_rate"],
        "old_class_accuracy": dict(after["old_class_accuracy"]),
        "old_class_count": len(after["old_class_accuracy"]),
        "query_macs": -1,
        "state_bytes": -1,
        "da1_reg0_old_acc": before["old_acc"],
        "da1_reg0_old_floor": before["old_floor"],
    }


def _scenario_rows(outer_key: str, job: Mapping[str, Any], candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate_before = candidate["before"]["scenes"]
    candidate_after = candidate["after"]["scenes"]
    baseline_before = baseline["before"]["scenes"]
    baseline_after = baseline["after"]["scenes"]
    rows: list[dict[str, Any]] = []
    for scene in SCENES:
        c = candidate_after[scene]
        b = baseline_after[scene]
        c_before = candidate_before[scene]
        b_before = baseline_before[scene]
        c_old = _finite(c_before["query_count"], f"candidate old count {scene}", lower=0.0)
        b_old = _finite(b_before["query_count"], f"E0 old count {scene}", lower=0.0)
        c_total = _finite(c["query_count"], f"candidate total count {scene}", lower=0.0)
        b_total = _finite(b["query_count"], f"E0 total count {scene}", lower=0.0)
        values = {
            "h_old_new": (_finite(c["h_old_new"], f"candidate H {scene}", lower=0.0, upper=1.0), _finite(b["h_old_new"], f"E0 H {scene}", lower=0.0, upper=1.0)),
            "old_balanced_accuracy": (_finite(c["old_balanced_accuracy"], f"candidate old balanced accuracy {scene}", lower=0.0, upper=1.0), _finite(b["old_balanced_accuracy"], f"E0 old balanced accuracy {scene}", lower=0.0, upper=1.0)),
            "c_old_acc": (_finite(c["c_old_acc"], f"candidate C-old accuracy {scene}", lower=0.0, upper=1.0), _finite(b["c_old_acc"], f"E0 C-old accuracy {scene}", lower=0.0, upper=1.0)),
            "old_floor": (_finite(c["old_floor"], f"candidate old floor {scene}", lower=0.0, upper=1.0), _finite(b["old_floor"], f"E0 old floor {scene}", lower=0.0, upper=1.0)),
            "seen_new_acc": (_finite(c["seen_new_acc"], f"candidate new acc {scene}", lower=0.0, upper=1.0), _finite(b["seen_new_acc"], f"E0 new acc {scene}", lower=0.0, upper=1.0)),
            "average_forgetting": (_finite(c_before["old_acc"], f"candidate before old accuracy {scene}", lower=0.0, upper=1.0) - _finite(c["old_acc"], f"candidate after old accuracy {scene}", lower=0.0, upper=1.0), _finite(b_before["old_acc"], f"E0 before old accuracy {scene}", lower=0.0, upper=1.0) - _finite(b["old_acc"], f"E0 after old accuracy {scene}", lower=0.0, upper=1.0)),
            "new_to_old_rate": (_finite(c["new_to_old_rate"], f"candidate new-to-old {scene}", lower=0.0, upper=1.0), _finite(b["new_to_old_rate"], f"E0 new-to-old {scene}", lower=0.0, upper=1.0)),
            "old_to_new_rate": (_finite(c["old_to_new_rate"], f"candidate old-to-new {scene}", lower=0.0, upper=1.0), _finite(b["old_to_new_rate"], f"E0 old-to-new {scene}", lower=0.0, upper=1.0)),
        }
        row: dict[str, Any] = {
            "outer_key": outer_key,
            "arm_id": str(job.get("arm_id", ARM_ID)),
            "outer_role": str(job.get("outer_role", "")),
            "receiver": str(job["receiver"]),
            "seed": int(job["seed"]),
            "k_shot": int(job["k_shot"]),
            "new_class_count": int(job["new_class_count"]),
            "slice": f"K{job['k_shot']}_new{job['new_class_count']}",
            "scenario": scene,
            "old_query_count": c_old,
            "new_query_count": c_total - c_old,
            "e0_old_query_count": b_old,
            "e0_new_query_count": b_total - b_old,
        }
        for metric, (candidate_value, baseline_value) in values.items():
            row[f"candidate_{metric}"] = candidate_value
            row[f"e0_{metric}"] = baseline_value
            row[f"delta_{metric}"] = candidate_value - baseline_value
        rows.append(row)
    return rows


def _group_stability(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for key, members in sorted(grouped.items()):
        deltas = {
            metric: _mean(
                float(row[f"candidate_{metric}"]) - float(row[f"e0_{metric}"])
                for row in members
            )
            for metric in EIGHT_PARETO_METRICS
        }
        result[key] = {
            "row_count": len(members),
            **{f"mean_delta_{metric}": delta for metric, delta in deltas.items()},
            "passed": _strict_row_ok(deltas),
        }
    return result


def _job_key(job: Mapping[str, Any]) -> str:
    return str(job.get("outer_key", ""))


def _validate_manifest_shape(manifest: Mapping[str, Any], lock_sha: str) -> None:
    try:
        matrix.validate_hard9_k1_manifest(manifest, expected_method_lock_sha256=lock_sha, require_package_hashes=True)
    except Exception as error:
        raise _fail("matrix manifest closure drift") from error


def _validate_job_identity(job: Mapping[str, Any], manifest_sha: str, lock_sha: str) -> None:
    outer = _job_key(job)
    if not outer or str(job.get("arm_id")) != ARM_ID or str(job.get("candidate")) != CANDIDATE_ID or str(job.get("method_lock_sha256", lock_sha)).lower() != lock_sha.lower():
        raise _fail("job arm/candidate/method identity drift")
    if str(job.get("job_id", "")) != f"{outer}__arm_{ARM_ID.lower()}":
        raise _fail("job id/outer identity drift")
    role = str(job.get("outer_role", ""))
    k_shot = _integer(job.get("k_shot"), "job K", lower=1)
    if (role == "liveness") != (k_shot == 1):
        raise _fail("K1 entered performance aggregation")
    if role not in {"performance", "liveness"}:
        raise _fail("job role drift")


def _validate_receipt(
    job: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    manifest_sha: str,
    lock_sha: str,
    score_path: Path,
    before_path: Path,
    after_path: Path,
    closure_hashes: Mapping[str, str],
) -> None:
    expected = {
        "schema": matrix.JOB_RECEIPT_SCHEMA,
        "status": "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE",
        "job_id": job["job_id"],
        "outer_key": job["outer_key"],
        "outer_role": job["outer_role"],
        "k_shot": job["k_shot"],
        "arm_id": ARM_ID,
        "candidate": CANDIDATE_ID,
        "matrix_manifest_sha256": manifest_sha,
        "method_lock_sha256": lock_sha,
        "selection_sha256": matrix.CANONICAL_SELECTION_SHA256,
        "truth_sidecar_exposed_to_predictor": False,
        "query_truth_joined_only_after_immutable_predictions": True,
        "query_truth_fed_back_to_predictor": False,
        "prediction_and_scorer_processes_isolated": True,
        "fresh_run_retry_authorized": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise _fail(f"job receipt binding drift: {key}")
    _validate_closure_surface(receipt, closure_hashes, label="job receipt")
    prediction_closure = receipt.get("prediction_closure")
    if not isinstance(prediction_closure, Mapping):
        raise _fail("job receipt prediction_closure missing")
    _validate_closure_surface(prediction_closure, closure_hashes, label="job receipt prediction_closure")
    if str(receipt.get("score_sha256", "")).lower() != _sha(score_path):
        raise _fail("job receipt prediction/score SHA drift")
    if _sha(before_path) != closure_hashes["before_prediction_sha256"] or _sha(after_path) != closure_hashes["after_prediction_sha256"]:
        raise _fail("recomputed prediction closure SHA drift")
    truth_sha = _sha256_value(job.get("truth_sidecar_sha256"), "job truth sidecar")
    if (
        str(receipt.get("truth_sidecar_sha256", "")).lower() != truth_sha
        or str(receipt.get("truth_sidecar_sha256_before_score", "")).lower() != truth_sha
        or str(receipt.get("truth_sidecar_sha256_after_score", "")).lower() != truth_sha
    ):
        raise _fail("job receipt truth-sidecar SHA drift")


def _validate_score_binding(
    score: Mapping[str, Any],
    receipt: Mapping[str, Any],
    before_path: Path,
    after_path: Path,
    *,
    job: Mapping[str, Any],
    job_root: Path,
    manifest_sha: str,
    lock_sha: str,
    closure_hashes: Mapping[str, str],
) -> None:
    if (
        score.get("schema") != "cvs.phase2.diag_cosine_dev_pair_score.v1"
        or score.get("candidate") != CANDIDATE_ID
        or score.get("query_truth_fed_back_to_predictor") is not False
        or score.get("query_truth_joined_only_after_immutable_predictions") is not True
    ):
        raise _fail("score candidate/query binding drift")
    before_sha = closure_hashes["before_prediction_sha256"]
    after_sha = closure_hashes["after_prediction_sha256"]
    if str(score.get("before_prediction_sha256", "")).lower() != before_sha or str(score.get("after_prediction_sha256", "")).lower() != after_sha:
        raise _fail("score prediction binding drift")
    binding_value = receipt.get("score_binding")
    expected_binding_path = (job_root / "score_binding.json").resolve()
    if not isinstance(binding_value, str) or not binding_value:
        raise _fail("score binding evidence missing")
    binding_path = Path(binding_value).resolve()
    if binding_path != expected_binding_path or not binding_path.is_file() or binding_path.is_symlink():
        raise _fail("score binding path drift")
    binding_sha = _sha(binding_path)
    if str(receipt.get("score_binding_sha256", "")).lower() != binding_sha:
        raise _fail("score binding SHA drift")
    truth_sha = str(job["truth_sidecar_sha256"]).lower()
    expected_binding = {
        "schema": "cvs.phase2.d92_ccoc_hard9_k1.score_binding.v1",
        "job_id": job["job_id"],
        "outer_key": job["outer_key"],
        "outer_role": job["outer_role"],
        "arm_id": ARM_ID,
        "candidate": CANDIDATE_ID,
        "matrix_manifest_sha256": manifest_sha,
        "method_lock_sha256": lock_sha,
        "truth_sidecar": str(job["truth_sidecar"]),
        "truth_sidecar_sha256": truth_sha,
        **closure_hashes,
        "performance_result_allowed": False,
    }
    binding = _read_json(binding_path)
    for key, expected in expected_binding.items():
        actual = binding.get(key)
        if key.endswith("sha256"):
            actual = str(actual or "").lower()
        if actual != expected:
            raise _fail(f"score binding drift: {key}")
    _validate_closure_surface(binding, closure_hashes, label="score binding")
    evidence = receipt.get("score_evidence")
    if not isinstance(evidence, Mapping):
        raise _fail("score binding evidence missing")
    expected_evidence = {
        "job_id": job["job_id"],
        "outer_key": job["outer_key"],
        "arm_id": ARM_ID,
        "candidate": CANDIDATE_ID,
        "matrix_manifest_sha256": manifest_sha,
        "method_lock_sha256": lock_sha,
        "score_artifact_sha256": str(receipt.get("score_sha256", "")).lower(),
        "truth_sidecar_sha256": truth_sha,
        **closure_hashes,
    }
    for key, expected in expected_evidence.items():
        actual = evidence.get(key)
        if key.endswith("sha256"):
            actual = str(actual or "").lower()
        if actual != expected:
            raise _fail(f"score binding evidence drift: {key}")
    _validate_closure_surface(evidence, closure_hashes, label="score evidence")


def _job_artifacts(
    job: Mapping[str, Any],
    root: Path,
    *,
    manifest_sha: str,
    lock_sha: str,
    truth_sidecar_root: str | Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    outer = str(job["outer_key"])
    job_root = root / "jobs" / outer / ARM_ID
    receipt = _read_json(job_root / "job_receipt.json")
    before_root = _state_root(job_root, "before")
    after_root = _state_root(job_root, "after")
    before_path = before_root / "prediction_artifact.npz"
    after_path = after_root / "prediction_artifact.npz"
    score_path = job_root / "scorer" / "diag_cosine_score.json"
    if not before_path.is_file() or not after_path.is_file() or before_path.is_symlink() or after_path.is_symlink():
        raise _fail(f"prediction artifact closure drift: {outer}")
    closure_hashes = _prediction_closure_hashes(job_root)
    _validate_receipt(job, receipt, manifest_sha=manifest_sha, lock_sha=lock_sha, score_path=score_path, before_path=before_path, after_path=after_path, closure_hashes=closure_hashes)
    score = _read_json(score_path)
    _validate_score_binding(
        score,
        receipt,
        before_path,
        after_path,
        job=job,
        job_root=job_root,
        manifest_sha=manifest_sha,
        lock_sha=lock_sha,
        closure_hashes=closure_hashes,
    )
    truth_path = _logical_truth_path(job, truth_sidecar_root)
    validate_truth_binding(score, receipt, job, truth_path)
    fit_rows, _ = _validate_fit_rows(job, job_root)
    resources = _resource_join(job, job_root, fit_rows)
    return score, resources, {
        "outer_key": outer,
        "receipt": receipt,
        "fit_rows": fit_rows,
        "truth_path": truth_path,
        "before_path": before_path,
        "after_path": after_path,
    }


def _baseline_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        outer = str(row.get("outer_key", ""))
        if not outer or outer in result:
            raise _fail("baseline duplicate outer_key")
        result[outer] = row
    return result


def _raw_score_lookup(manifest: Mapping[str, Any], lock: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    specs = lock.get("historical_baseline", {}).get("e0_raw_scores") if isinstance(lock.get("historical_baseline"), Mapping) else None
    if not isinstance(specs, Mapping):
        specs = {}
    result: dict[str, Path] = {}
    for job in jobs:
        outer = str(job["outer_key"])
        spec = specs.get(outer)
        if not isinstance(spec, Mapping):
            # The matrix module carries the same frozen map for callers that
            # load a lock fixture without its historical section.
            path = Path(matrix.RAW_SCORE_ROOT) / outer / "E0_FULL_ONLY" / "scorer" / "diag_cosine_score.json"
            expected_sha = matrix.RAW_SCORE_SHA.get(outer)
        else:
            path = Path(str(spec.get("path", "")))
            expected_sha = spec.get("sha256")
        result[outer] = path
        if not path.is_file() or path.is_symlink():
            raise _fail(f"E0 raw score missing: {outer}")
        if expected_sha and _sha(path) != str(expected_sha).lower():
            raise _fail(f"E0 raw score SHA drift: {outer}")
    return result


def _baseline_path_arg(path: str | Path | None, default: str) -> Path:
    return Path(path if path is not None else default).resolve(strict=True)


def _analyze_d92_ccoc_hard9_k1(
    matrix_manifest_path: str | Path,
    *,
    run_root: str | Path | None,
    method_lock_path: str | Path,
    baseline_paired_rows_path: str | Path | None = None,
    per_old_class_rows_path: str | Path | None = None,
    truth_sidecar_root: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze one frozen Hard9+K1 run and return the sole verdict package."""

    manifest_path = Path(matrix_manifest_path).resolve(strict=True)
    lock_path = Path(method_lock_path).resolve(strict=True)
    manifest_sha = _sha(manifest_path)
    lock_sha = _sha(lock_path)
    manifest = _read_json(manifest_path)
    lock = _read_json(lock_path)
    try:
        matrix.validate_method_lock(lock)
    except Exception as error:
        raise _fail("method lock identity drift") from error
    _validate_manifest_shape(manifest, lock_sha)
    root = Path(run_root if run_root is not None else manifest.get("output_root", "")).resolve(strict=True)
    if (root / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json").exists():
        raise _fail("systemic stop marker exists")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 10:
        raise _fail("job-count closure drift")
    job_by_outer: dict[str, Mapping[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, Mapping):
            raise _fail("job record invalid")
        _validate_job_identity(job, manifest_sha, lock_sha)
        outer = str(job["outer_key"])
        if outer in job_by_outer:
            raise _fail("duplicate outer_key+scene+arm job")
        job_by_outer[outer] = job
    expected_outers = {str(row["outer_key"]) for row in matrix.HARD9_K1_ROWS}
    if set(job_by_outer) != expected_outers:
        raise _fail("missing/extra frozen outer_key closure")
    if sum(str(job["outer_role"]) == "performance" for job in jobs) != 9 or sum(str(job["outer_role"]) == "liveness" for job in jobs) != 1:
        raise _fail("9 performance+1 liveness closure drift")
    baseline_path = _baseline_path_arg(baseline_paired_rows_path, matrix.HISTORICAL_BASELINE_PATH)
    per_old_path = _baseline_path_arg(per_old_class_rows_path, matrix.HISTORICAL_PER_OLD_CLASS_PATH)
    baseline_expected = matrix.HISTORICAL_BASELINE_SHA256 if baseline_paired_rows_path is None else None
    per_old_expected = matrix.HISTORICAL_PER_OLD_CLASS_SHA256 if per_old_class_rows_path is None else None
    baseline_rows = _read_csv(baseline_path, expected_sha256=baseline_expected)
    per_old_source_rows = _read_csv(per_old_path, expected_sha256=per_old_expected)
    baseline_by_outer = _baseline_lookup(baseline_rows)
    per_old_by_outer: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in per_old_source_rows:
        per_old_by_outer[str(row.get("outer_key", ""))].append(row)
    raw_paths = _raw_score_lookup(manifest, lock, jobs)

    paired_rows: list[dict[str, Any]] = []
    per_old_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    for job in jobs:
        outer = str(job["outer_key"])
        if outer not in baseline_by_outer or outer not in raw_paths:
            raise _fail(f"job/baseline identity mismatch: {outer}")
        raw_score = _read_json(raw_paths[outer])
        candidate_score, job_resources, candidate_artifacts = _job_artifacts(
            job,
            root,
            manifest_sha=manifest_sha,
            lock_sha=lock_sha,
            truth_sidecar_root=truth_sidecar_root,
        )
        candidate_surface, baseline_surface = _validate_candidate_e0_surface(
            candidate_score,
            raw_score,
            job,
            outer_key=outer,
            candidate_before_path=candidate_artifacts["before_path"],
            candidate_after_path=candidate_artifacts["after_path"],
            raw_score_path=raw_paths[outer],
            truth_path=candidate_artifacts["truth_path"],
        )
        baseline_metrics = _metrics_from_surface_pair(baseline_surface)
        candidate_metrics = _metrics_from_surface_pair(candidate_surface)
        validate_paired_e0_row(
            baseline_by_outer[outer],
            raw_score,
            job,
            outer_key=outer,
            metrics=baseline_metrics,
        )
        if job_resources:
            candidate_metrics["query_macs"] = int(job_resources[0]["candidate_query_macs"])
            candidate_metrics["state_bytes"] = int(job_resources[0]["candidate_state_bytes"])
            baseline_metrics["query_macs"] = int(job_resources[0]["e0_query_macs"])
            baseline_metrics["state_bytes"] = int(job_resources[0]["e0_state_bytes"])
        old_join = validate_per_old_class_join(per_old_by_outer.get(outer, []), raw_score, outer_key=outer)
        row: dict[str, Any] = {
            "outer_key": outer,
            "outer_role": str(job["outer_role"]),
            "arm_id": ARM_ID,
            "receiver": str(job["receiver"]),
            "seed": int(job["seed"]),
            "k_shot": int(job["k_shot"]),
            "new_class_count": int(job["new_class_count"]),
            "slice": f"K{job['k_shot']}_new{job['new_class_count']}",
            "query_macs": candidate_metrics["query_macs"],
            "state_bytes": candidate_metrics["state_bytes"],
            "e0_query_macs": baseline_metrics["query_macs"],
            "e0_state_bytes": baseline_metrics["state_bytes"],
            "DA1_REG0_old_acc": candidate_metrics["da1_reg0_old_acc"],
            "DA1_REG0_old_floor": candidate_metrics["da1_reg0_old_floor"],
        }
        for metric in EIGHT_PARETO_METRICS:
            row[f"candidate_{metric}"] = candidate_metrics[metric]
            row[f"e0_{metric}"] = baseline_metrics[metric]
            row[f"delta_{metric}_vs_e0"] = strict_pareto_deltas(candidate_metrics, baseline_metrics)[metric]
        row["resource_scene_count"] = len(job_resources)
        row["resource_query_state_exact"] = all(item["query_state_exact"] for item in job_resources)
        row["candidate_peak_max_bytes"] = max(item["candidate_peak_bytes"] for item in job_resources)
        row["candidate_wall_p90_ns"] = _p90([item["candidate_wall_ns"] for item in job_resources])
        row["candidate_ratio_p90"] = _p90([item["wall_ratio"] for item in job_resources])
        paired_rows.append(row)
        resource_rows.extend(job_resources)
        for tx, values in sorted(old_join.items()):
            per_old_rows.append({
                "outer_key": outer,
                "arm_id": ARM_ID,
                "tx": tx,
                "candidate_accuracy": candidate_metrics["old_class_accuracy"][tx],
                "e0_accuracy": values["e0_accuracy"],
                "delta_accuracy": candidate_metrics["old_class_accuracy"][tx] - values["e0_accuracy"],
                "historical_baseline_accuracy": values["historical_baseline_accuracy"],
                "historical_delta_accuracy": values["historical_delta_accuracy"],
            })
        scenario_rows.extend(_scenario_rows(outer, job, candidate_surface, baseline_surface))

    if len(paired_rows) != 10 or len(per_old_rows) != 60 or len(scenario_rows) != 30:
        raise _fail("Hard9 result row closure drift")
    performance = [row for row in paired_rows if row["outer_role"] == "performance"]
    liveness = [row for row in paired_rows if row["outer_role"] == "liveness"]
    performance_scenes = [row for row in scenario_rows if row["outer_key"] in {item["outer_key"] for item in performance}]
    deltas_by_metric = {metric: [_finite(row[f"delta_{metric}_vs_e0"], f"delta {metric}") for row in performance] for metric in EIGHT_PARETO_METRICS}
    mean_deltas = {metric: _mean(values) for metric, values in deltas_by_metric.items()}
    direction_counts = {metric: sum((value > _TOLERANCE if metric in EIGHT_PARETO_METRICS[:5] else value < -_TOLERANCE) for value in values) for metric, values in deltas_by_metric.items()}
    all_strict = all(_strict_row_ok({metric: deltas_by_metric[metric][index] for metric in EIGHT_PARETO_METRICS}) for index in range(len(performance)))
    all_magnitude = _magnitude_ok(mean_deltas)

    by_tx: dict[str, list[float]] = defaultdict(list)
    by_outer: dict[str, list[float]] = defaultdict(list)
    for item in per_old_rows:
        if item["outer_key"] in {row["outer_key"] for row in performance}:
            by_tx[str(item["tx"])].append(float(item["delta_accuracy"]))
            by_outer[str(item["outer_key"])].append(float(item["delta_accuracy"]))
    per_old_summary = {
        tx: {
            "row_count": len(values),
            "mean_delta_accuracy": _mean(values),
            "min_delta_accuracy": min(values),
            "nondecrease_count": sum(value >= -0.01 for value in values),
        }
        for tx, values in sorted(by_tx.items())
    }
    per_outer_summary = {
        outer: {
            "row_count": len(values),
            "min_delta_accuracy": min(values),
            "nondecrease_count": sum(value >= -_TOLERANCE for value in values),
            "passed": len(values) == 6 and min(values) >= -0.01 and sum(value >= -_TOLERANCE for value in values) >= 5,
        }
        for outer, values in sorted(by_outer.items())
    }
    per_old_stability = len(per_old_summary) == 6 and all(item["row_count"] == 9 and item["min_delta_accuracy"] >= -0.01 for item in per_old_summary.values()) and len(per_outer_summary) == 9 and all(item["passed"] for item in per_outer_summary.values())
    by_receiver = _group_stability(performance_scenes, "receiver")
    by_slice = _group_stability(performance_scenes, "slice")
    by_scene = _group_stability(performance_scenes, "scenario")
    scenario_row_stability = all(
        _strict_row_ok(
            {
                metric: _finite(row[f"delta_{metric}"], f"scenario delta {metric}")
                for metric in EIGHT_PARETO_METRICS
            }
        )
        for row in performance_scenes
    )
    group_stability = bool(
        scenario_row_stability
        and all(
            item["passed"]
            for groups in (by_receiver, by_slice, by_scene)
            for item in groups.values()
        )
    )
    stability = bool(all_strict and per_old_stability and group_stability)
    resource_eval = evaluate_resource_gate(resource_rows, query_state_exact=True)
    # Query/state integrity is a hard gate; K1 is not included in the
    # performance metric aggregation but remains part of artifact closure.
    query_exact = all(bool(row["resource_query_state_exact"]) for row in paired_rows)
    resource_hard = bool(resource_eval["hard_passed"] and query_exact)
    resource_target = bool(resource_eval["target_passed"] and query_exact)
    gates = {
        "complete_artifact_closure": {"passed": True, "observed": {"paired": 10, "per_old": 60, "scenario": 30, "performance_scene": 27}, "threshold": "10/60/30;9+1;27 performance scenes"},
        "performance_outer_closure": {"passed": len(performance) == 9 and len(liveness) == 1 and all(int(row["k_shot"]) > 1 for row in performance), "observed": {"performance": len(performance), "liveness": len(liveness), "performance_k": sorted({int(row["k_shot"]) for row in performance})}, "threshold": "9 performance+1 K1 liveness"},
        "all_strict_pareto": {"passed": all_strict, "observed": {"direction_counts": direction_counts, "mean_deltas": mean_deltas}, "threshold": "all 8 metrics strict on every performance outer"},
        "all_magnitude": {"passed": all_magnitude, "observed": mean_deltas, "threshold": STRICT_PARETO_THRESHOLDS},
        "stability": {"passed": stability, "observed": {"per_old_class": per_old_stability, "per_outer_old": per_outer_summary, "by_receiver": by_receiver, "by_slice": by_slice, "by_scene": by_scene, "scenario_row_stability": scenario_row_stability}, "threshold": "receiver/K-new/scene all-8 strict direction and per-old stability"},
        "resource_integrity": {"passed": query_exact, "observed": {"query_state_exact": query_exact}, "threshold": "query MAC/state exact per scene"},
        "resource_hard": {"passed": resource_hard, "observed": resource_eval, "threshold": {"absolute_peak_bytes": matrix.RESOURCE_GATE["candidate_peak_hard_max_bytes"], "wall_ns": 150_000_000, "ratio": 1.50}},
        "resource_target": {"passed": resource_target, "observed": resource_eval, "threshold": {"absolute_peak_bytes": matrix.RESOURCE_GATE["candidate_peak_target_max_bytes"], "wall_p90_ns": 120_000_000, "ratio_p90": 1.25}},
    }
    gate_state = {name: bool(value["passed"]) for name, value in gates.items()}
    verdict = decide_verdict(gate_state)
    aggregate = {
        "row_count": 10,
        "performance_row_count": 9,
        "liveness_row_count": 1,
        "candidate_mean_da1_reg0_old_acc": _mean(
            row["DA1_REG0_old_acc"] for row in performance
        ),
        "candidate_mean_da1_reg0_old_floor": _mean(
            row["DA1_REG0_old_floor"] for row in performance
        ),
        **{f"candidate_mean_{metric}": _mean(row[f"candidate_{metric}"] for row in performance) for metric in EIGHT_PARETO_METRICS},
        **{f"e0_mean_{metric}": _mean(row[f"e0_{metric}"] for row in performance) for metric in EIGHT_PARETO_METRICS},
        **{f"mean_delta_{metric}_vs_e0": mean_deltas[metric] for metric in EIGHT_PARETO_METRICS},
        "resource_wall_p90_ns": resource_eval["wall_p90_ns"],
        "resource_ratio_p90": resource_eval["wall_ratio_p90"],
        "candidate_peak_max_bytes": resource_eval["candidate_peak_max_bytes"],
        "candidate_peak_target_pass": resource_eval["candidate_peak_target_pass"],
    }
    return {
        "schema": "cvs.phase2.d92_ccoc_hard9_k1.analysis.v1",
        "status": "ANALYZED",
        "claim_scope": manifest.get("claim_scope", matrix.CLAIM_SCOPE),
        "matrix_manifest_sha256": manifest_sha,
        "method_lock_sha256": lock_sha,
        "selection_sha256": matrix.CANONICAL_SELECTION_SHA256,
        "baseline": {"paired_rows_path": str(baseline_path), "paired_rows_sha256": _sha(baseline_path), "per_old_class_rows_path": str(per_old_path), "per_old_class_rows_sha256": _sha(per_old_path), "raw_score_paths": {outer: str(path) for outer, path in raw_paths.items()}},
        "aggregate": aggregate,
        "paired_rows": paired_rows,
        "per_old_class_rows": per_old_rows,
        "per_old_class_summary": per_old_summary,
        "scenario_rows": scenario_rows,
        "liveness_rows": liveness,
        "resource_rows": resource_rows,
        "by_receiver": by_receiver,
        "by_slice": by_slice,
        "by_scene": by_scene,
        "gates": gates,
        "gate_state": gate_state,
        "all_gates_pass": verdict == "ADVANCE_TO_TARGET125_CANDIDATE",
        "verdict": verdict,
    }


def _prepare_output_root(output_root: str | Path) -> Path:
    root = Path(output_root).resolve()
    if root.exists() and (not root.is_dir() or root.is_symlink() or any(root.iterdir())):
        raise D92CCOCHard9K1AnalysisError(
            f"output overwrite refused: existing output root {root}"
        )
    return root


def _controlled_reject_result(error: Exception) -> dict[str, Any]:
    """Materialize evidence failures as a self-contained reject package."""

    gate_names = (
        "complete_artifact_closure",
        "performance_outer_closure",
        "all_strict_pareto",
        "all_magnitude",
        "stability",
        "resource_integrity",
        "resource_hard",
        "resource_target",
    )
    gates = {
        name: {
            "passed": False,
            "observed": {"error": str(error)} if name == "complete_artifact_closure" else "not evaluated after evidence failure",
            "threshold": "immutable evidence closure required",
        }
        for name in gate_names
    }
    return {
        "schema": "cvs.phase2.d92_ccoc_hard9_k1.analysis.v1",
        "status": "REJECTED_EVIDENCE_CLOSURE",
        "claim_scope": matrix.CLAIM_SCOPE,
        "aggregate": {},
        "paired_rows": [],
        "per_old_class_rows": [],
        "per_old_class_summary": {},
        "scenario_rows": [],
        "liveness_rows": [],
        "resource_rows": [],
        "by_receiver": {},
        "by_slice": {},
        "by_scene": {},
        "gates": gates,
        "gate_state": {name: False for name in gate_names},
        "all_gates_pass": False,
        "verdict": "REJECT_ROUTE",
    }


def analyze_d92_ccoc_hard9_k1(
    matrix_manifest_path: str | Path,
    *,
    run_root: str | Path | None,
    method_lock_path: str | Path,
    baseline_paired_rows_path: str | Path | None = None,
    per_old_class_rows_path: str | Path | None = None,
    truth_sidecar_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze a frozen run; audit/evidence faults always return REJECT_ROUTE.

    Invalid call shapes and output-root overwrite requests still raise before
    analysis.  Missing or unreadable frozen evidence is a controlled verdict,
    whether or not a seven-file destination was requested.
    """

    prepared_output = _prepare_output_root(output_root) if output_root is not None else None
    try:
        result = _analyze_d92_ccoc_hard9_k1(
            matrix_manifest_path,
            run_root=run_root,
            method_lock_path=method_lock_path,
            baseline_paired_rows_path=baseline_paired_rows_path,
            per_old_class_rows_path=per_old_class_rows_path,
            truth_sidecar_root=truth_sidecar_root,
        )
    except (D92CCOCHard9K1AnalysisError, OSError) as error:
        result = _controlled_reject_result(error)
        if prepared_output is not None:
            result["output_paths"] = write_analysis_outputs(result, prepared_output)
        return result
    if prepared_output is not None:
        result["output_paths"] = write_analysis_outputs(result, prepared_output)
    return result


analyze_ccoc_hard9_k1 = analyze_d92_ccoc_hard9_k1


def _csv_rows(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in value]


def render_analysis_markdown(result: Mapping[str, Any]) -> str:
    verdict = str(result.get("verdict", "REJECT_ROUTE"))
    aggregate = result.get("aggregate", {})
    lines = [
        "# D92 CCOC Hard9+K1 truth-last分析",
        "",
        f"- 唯一裁决：`{verdict}`",
        f"- 状态：`{result.get('status', 'ANALYZED')}`",
        "- 证据边界：prediction先封存，truth-last独立评分；K1仅作liveness，不进入性能聚合。",
        "",
        "## 四态指标表",
        "",
        "| 状态 | old accuracy | old floor | H_old_new | seen-new accuracy | 说明 |",
        "|---|---:|---:|---:|---:|---|",
        "| DA0_REG0 | N/A | N/A | N/A | N/A | 未提供该状态证据 |",
        f"| DA1_REG0 | {aggregate.get('candidate_mean_da1_reg0_old_acc', 'N/A')} | {aggregate.get('candidate_mean_da1_reg0_old_floor', 'N/A')} | N/A | N/A | 注册前 |",
        "| DA0_REG1 | N/A | N/A | N/A | N/A | 未提供该状态证据 |",
        f"| DA1_REG1 | {aggregate.get('candidate_mean_c_old_acc', 'N/A')} | {aggregate.get('candidate_mean_old_floor', 'N/A')} | {aggregate.get('candidate_mean_h_old_new', 'N/A')} | {aggregate.get('candidate_mean_seen_new_acc', 'N/A')} | 注册后 |",
        "",
        "## 冻结门",
        "",
        "| 门 | 结果 |",
        "|---|---|",
    ]
    for name, gate in (result.get("gates") or {}).items():
        lines.append(f"| {name} | {'PASS' if gate.get('passed') else 'FAIL'} |")
    lines.extend(["", "## 结果边界", "", "本文件只记录同排、同outer、同scene、同arm的机械证据，不构成科学解释或性能推广声明。", ""])
    return "\n".join(lines)


def write_analysis_outputs(result: Mapping[str, Any], output_root: str | Path) -> dict[str, str]:
    """Write the exclusive seven-file analysis package; never overwrite."""

    root = Path(output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise D92CCOCHard9K1AnalysisError(f"output overwrite refused: existing output root {root}")
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "summary.json": result,
        "gates.json": {"schema": result.get("schema"), "gate_state": result.get("gate_state"), "gates": result.get("gates"), "verdict": result.get("verdict")},
        "paired_rows.csv": result.get("paired_rows", []),
        "per_old_class_rows.csv": result.get("per_old_class_rows", []),
        "scenario_rows.csv": result.get("scenario_rows", []),
        "liveness_rows.csv": result.get("liveness_rows", []),
        "analysis.md": render_analysis_markdown(result),
    }
    for name, value in files.items():
        path = root / name
        if path.exists() or path.is_symlink():
            raise _fail(f"output overwrite refused: {path}")
        if name.endswith(".json"):
            path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        elif name.endswith(".csv"):
            rows = list(value)
            fieldnames: list[str] = []
            for row in rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(str(key))
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        else:
            path.write_text(str(value), encoding="utf-8", newline="\n")
    return {name: str(root / name) for name in files}


__all__ = [
    "ARM_ID",
    "CANDIDATE_ID",
    "DIRECTION_GATE",
    "D92CCOCHard9K1AnalysisError",
    "EIGHT_PARETO_METRICS",
    "PARETO_METRICS",
    "SCENES",
    "STRICT_PARETO_THRESHOLDS",
    "VERDICTS",
    "analyze_ccoc_hard9_k1",
    "analyze_d92_ccoc_hard9_k1",
    "compute_confusion_rates",
    "compute_old_balanced_accuracy",
    "compute_score_metrics",
    "decide_verdict",
    "evaluate_resource_gate",
    "render_analysis_markdown",
    "strict_pareto_deltas",
    "validate_per_old_class_join",
    "validate_truth_binding",
    "write_analysis_outputs",
]

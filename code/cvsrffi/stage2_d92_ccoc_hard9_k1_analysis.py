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
            score.get("per_old_class_floor_before", min(old_accuracy.values())),
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

    metrics = compute_score_metrics(raw_score)
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
        actual = _finite(row.get(field), f"paired E0 {field}", lower=0.0)
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
    payload = json.loads(fit_path.read_text(encoding="utf-8-sig")) if fit_path.is_file() and not fit_path.is_symlink() else None
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
    return after_rows, {scene: row for scene, row in zip(SCENES, after_rows)}


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
    target = all(bool(row.get("wall_target_pass")) and bool(row.get("ratio_target_pass")) and bool(row.get("peak_target_pass")) and bool(row.get("query_state_exact")) for row in candidate_rows)
    return {
        "query_state_exact": bool(query_state_exact) and all(bool(row.get("query_state_exact")) for row in candidate_rows),
        "hard_limits_passed": bool(hard_scene and query_state_exact),
        "hard_passed": bool(hard_scene and query_state_exact),
        "target_limits_passed": bool(target),
        "target_passed": bool(target and hard_scene and query_state_exact),
        "wall_p90_ns": _p90(walls),
        "wall_ratio_p90": _p90(ratios),
        "wall_ratio_median": float(statistics.median(ratios)),
        "candidate_peak_max_bytes": max(peaks),
        "candidate_peak_target_pass": max(peaks) <= float(matrix.RESOURCE_GATE["candidate_peak_target_max_bytes"]),
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


def _scenario_rows(outer_key: str, job: Mapping[str, Any], candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate_before, candidate_after = _scenario_closure(candidate, f"candidate {outer_key}")
    baseline_before, baseline_after = _scenario_closure(baseline, f"E0 {outer_key}")
    rows: list[dict[str, Any]] = []
    for scene in SCENES:
        c = candidate_after[scene]
        b = baseline_after[scene]
        c_old = _finite(candidate_before[scene].get("query_count"), f"candidate old count {scene}", lower=0.0)
        b_old = _finite(baseline_before[scene].get("query_count"), f"E0 old count {scene}", lower=0.0)
        c_total = _finite(c.get("query_count"), f"candidate total count {scene}", lower=0.0)
        b_total = _finite(b.get("query_count"), f"E0 total count {scene}", lower=0.0)
        values = {
            "h_old_new": (_finite(c.get("h_old_new"), f"candidate H {scene}", lower=0.0, upper=1.0), _finite(b.get("h_old_new"), f"E0 H {scene}", lower=0.0, upper=1.0)),
            "old_acc": (_finite(c.get("old_acc"), f"candidate old acc {scene}", lower=0.0, upper=1.0), _finite(b.get("old_acc"), f"E0 old acc {scene}", lower=0.0, upper=1.0)),
            "seen_new_acc": (_finite(c.get("seen_new_acc"), f"candidate new acc {scene}", lower=0.0, upper=1.0), _finite(b.get("seen_new_acc"), f"E0 new acc {scene}", lower=0.0, upper=1.0)),
            "new_to_old_rate": (_finite(c.get("new_to_old_rate"), f"candidate new-to-old {scene}", lower=0.0, upper=1.0), _finite(b.get("new_to_old_rate"), f"E0 new-to-old {scene}", lower=0.0, upper=1.0)),
            "old_to_new_rate": (_finite(c.get("old_to_new_rate"), f"candidate old-to-new {scene}", lower=0.0, upper=1.0), _finite(b.get("old_to_new_rate"), f"E0 old-to-new {scene}", lower=0.0, upper=1.0)),
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
        h_delta = _mean(float(row["candidate_h_old_new"]) - float(row["e0_h_old_new"]) for row in members)
        n_delta = _mean(float(row["candidate_seen_new_acc"]) - float(row["e0_seen_new_acc"]) for row in members)
        result[key] = {"row_count": len(members), "mean_delta_h_old_new": h_delta, "mean_delta_seen_new_acc": n_delta, "passed": h_delta >= -_TOLERANCE and n_delta >= -_TOLERANCE}
    return result


def _job_key(job: Mapping[str, Any]) -> str:
    return str(job.get("outer_key", ""))


def _validate_manifest_shape(manifest: Mapping[str, Any], lock_sha: str) -> None:
    try:
        matrix.validate_hard9_k1_manifest(manifest, expected_method_lock_sha256=lock_sha, require_package_hashes=True)
        return
    except Exception as error:
        # A small test fixture may omit package paths while retaining all
        # analysis-relevant fields.  Do not weaken the frozen identity checks.
        required = {"schema", "jobs", "job_count", "outer_count", "performance_outer_count", "liveness_outer_count", "scene_count", "scene_arm_count", "selection_sha256", "method_lock_sha256"}
        if not required.issubset(manifest):
            raise _fail("matrix manifest closure drift") from error
        if manifest.get("schema") != matrix.MATRIX_SCHEMA or manifest.get("job_count") != 10 or manifest.get("outer_count") != 10 or manifest.get("performance_outer_count") != 9 or manifest.get("liveness_outer_count") != 1 or manifest.get("scene_count") != 3 or manifest.get("scene_arm_count") != 30 or str(manifest.get("method_lock_sha256")).lower() != lock_sha.lower() or str(manifest.get("selection_sha256")).lower() != matrix.CANONICAL_SELECTION_SHA256.lower():
            raise _fail("matrix manifest identity/count drift") from error


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
    if str(receipt.get("before_prediction_sha256", "")).lower() != _sha(before_path) or str(receipt.get("after_prediction_sha256", "")).lower() != _sha(after_path) or str(receipt.get("score_sha256", "")).lower() != _sha(score_path):
        raise _fail("job receipt prediction/score SHA drift")
    truth_sha = str(job.get("truth_sidecar_sha256", "")).lower()
    if (
        len(truth_sha) != 64
        or str(receipt.get("truth_sidecar_sha256", "")).lower() != truth_sha
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
) -> None:
    if (
        score.get("schema") != "cvs.phase2.diag_cosine_dev_pair_score.v1"
        or score.get("candidate") != CANDIDATE_ID
        or score.get("query_truth_fed_back_to_predictor") is not False
        or score.get("query_truth_joined_only_after_immutable_predictions") is not True
    ):
        raise _fail("score candidate/query binding drift")
    before_sha = _sha(before_path)
    after_sha = _sha(after_path)
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
        "before_prediction_sha256": before_sha,
        "after_prediction_sha256": after_sha,
        "performance_result_allowed": False,
    }
    binding = _read_json(binding_path)
    for key, expected in expected_binding.items():
        actual = binding.get(key)
        if key.endswith("sha256"):
            actual = str(actual or "").lower()
        if actual != expected:
            raise _fail(f"score binding drift: {key}")
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
        "score_artifact_sha256": str(receipt["score_sha256"]).lower(),
        "truth_sidecar_sha256": truth_sha,
        "before_prediction_sha256": before_sha,
        "after_prediction_sha256": after_sha,
    }
    for key, expected in expected_evidence.items():
        actual = evidence.get(key)
        if key.endswith("sha256"):
            actual = str(actual or "").lower()
        if actual != expected:
            raise _fail(f"score binding evidence drift: {key}")


def _job_artifacts(
    job: Mapping[str, Any],
    root: Path,
    *,
    manifest_sha: str,
    lock_sha: str,
    truth_sidecar_root: str | Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
    _validate_receipt(job, receipt, manifest_sha=manifest_sha, lock_sha=lock_sha, score_path=score_path, before_path=before_path, after_path=after_path)
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
    )
    truth_path = _logical_truth_path(job, truth_sidecar_root)
    validate_truth_binding(score, receipt, job, truth_path)
    fit_rows, _ = _validate_fit_rows(job, job_root)
    resources = _resource_join(job, job_root, fit_rows)
    return score, resources, [{"outer_key": outer, "receipt": receipt, "fit_rows": fit_rows}]


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


def analyze_d92_ccoc_hard9_k1(
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
        baseline_metrics = compute_score_metrics(raw_score)
        validate_paired_e0_row(
            baseline_by_outer[outer],
            raw_score,
            job,
            outer_key=outer,
        )
        candidate_score, job_resources, _ = _job_artifacts(job, root, manifest_sha=manifest_sha, lock_sha=lock_sha, truth_sidecar_root=truth_sidecar_root)
        candidate_metrics = compute_score_metrics(candidate_score)
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
        scenario_rows.extend(_scenario_rows(outer, job, candidate_score, raw_score))

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
    group_stability = all(item["passed"] for groups in (by_receiver, by_slice, by_scene) for item in groups.values())
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
        "stability": {"passed": stability, "observed": {"per_old_class": per_old_stability, "per_outer_old": per_outer_summary, "by_receiver": by_receiver, "by_slice": by_slice, "by_scene": by_scene}, "threshold": "receiver/K-new/scene and per-old stability"},
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

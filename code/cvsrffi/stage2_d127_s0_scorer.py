"""Fail-closed, truth-side scorer for the frozen D127 S0 before/after screen.

The predictor is deliberately truth-free.  This module first validates its
sealed 18-row pair closure and opens an independently held query-ID truth
catalog only afterwards.  It never returns labels or roles to the predictor.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PREDICTION_SCHEMA = "cvs.stage2.d127.s0.truth_free_prediction.v1"
PAIR_MANIFEST_SCHEMA = "cvs.stage2.d127.s0.before_after_pair_manifest.v1"
TRUTH_CATALOG_SCHEMA = "cvs.stage2.d127.s0.truth_catalog.v1"
FORMAL_D92_REFERENCE_SCHEMA = "cvs.stage2.d127.s0.formal_d92_same_row_reference.v1"
SCORE_SCHEMA = "cvs.stage2.d127.s0.score_manifest.v1"
PROTOCOL_SCHEMA = "p2_min_v1"

ROW_COUNT = 18
CANDIDATE_IDS = (
    "DA-A-FSRG-time_fuse",
    "DA-B-FSRG-t2norm",
    "DA-C-RDHA-joint_proj",
)
ARM_IDS = ("M0", "M_DA", "M_L92", "M_JOINT")
COMMON_ARM_IDS = ("M0", "M_L92")
ADAPTED_ARM_IDS = ("M_DA", "M_JOINT")
STATES = ("before", "after")
_FORBIDDEN_KEYS = frozenset(
    {"truth", "querytruth", "role", "roles", "quota", "classquota", "globalreassignment"}
)


class D127S0ScorerError(ValueError):
    """Raised when an S0 prediction/truth/reference closure drifts."""


def canonical_bytes(value: Any) -> bytes:
    """Canonical UTF-8 bytes used by D127 prediction and score receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D127S0ScorerError(message)


def _text(value: Any, name: str) -> str:
    _require(type(value) is str and bool(value) and value.strip() == value, f"{name} must be nonempty trimmed text")
    return value


def _sha(value: Any, name: str) -> str:
    text = _text(value, name)
    _require(len(text) == 64 and all(ch in "0123456789abcdef" for ch in text), f"{name} must be lowercase SHA256")
    return text


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def _exact(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == expected, f"{name} field closure drift")
    return value


def _mapping_with_exact_keys(
    value: Any, expected: Sequence[str], name: str
) -> Mapping[str, Any]:
    """Validate mapping closure without depending on JSON object key order."""

    _require(isinstance(value, Mapping), f"{name} must be an object")
    _require(
        len(value) == len(expected) and set(value) == set(expected),
        f"{name} closure drift",
    )
    return value


def _strings(value: Any, name: str, *, unique: bool = True) -> tuple[str, ...]:
    _require(isinstance(value, list) and bool(value), f"{name} must be a nonempty list")
    result = tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))
    _require(not unique or len(result) == len(set(result)), f"{name} contains duplicates")
    return result


def _receipt(document: Mapping[str, Any], field: str, name: str) -> str:
    actual = _sha(document.get(field), f"{name}.{field}")
    unsigned = dict(document)
    unsigned.pop(field, None)
    _require(canonical_sha256(unsigned) == actual, f"{name} canonical receipt mismatch")
    return actual


def _read_mapping(value: Mapping[str, Any] | str | Path, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    path = Path(value)
    _require(path.is_file() and not path.is_symlink(), f"{name} must be a regular JSON file")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D127S0ScorerError(f"{name} is not valid UTF-8 JSON") from exc
    _require(isinstance(parsed, Mapping), f"{name} must contain a JSON object")
    return parsed


def _normalized_key(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _reject_prediction_truth(value: Any, name: str) -> None:
    """Make truth/role/quota fields impossible in either sealed prediction."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalized_key(key)
            if normalized in _FORBIDDEN_KEYS:
                raise D127S0ScorerError(f"{name} contains forbidden truth/role/quota field: {key}")
            _reject_prediction_truth(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_prediction_truth(item, f"{name}[{index}]")


def _payload_arm(
    value: Any,
    *,
    arm_id: str,
    expected_classes: tuple[str, ...],
    query_count: int,
    name: str,
) -> tuple[str, ...]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    _require(value.get("arm_id") == arm_id, f"{name} arm identity drift")
    classes = _strings(value.get("classes"), f"{name}.classes")
    _require(classes == expected_classes, f"{name} registered-class closure drift")
    predictions = _strings(value.get("predictions"), f"{name}.predictions", unique=False)
    _require(len(predictions) == query_count, f"{name} prediction/query length drift")
    _require(all(label in classes for label in predictions), f"{name} prediction outside registered classes")
    return predictions


def _prediction_row(
    value: Any,
    *,
    manifest_row: Mapping[str, Any],
    state: str,
    index: int,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{state} prediction row[{index}] must be an object")
    raw = dict(value)
    receipt = _sha(raw.pop("row_sha256", None), f"{state} prediction row[{index}].row_sha256")
    _require(canonical_sha256(raw) == receipt, f"{state} prediction row[{index}] canonical receipt mismatch")
    expected_identity = {
        "row_id": manifest_row["row_id"],
        "receiver_id": manifest_row["receiver_id"],
        "k_shot": manifest_row["k_shot"],
        "scene": manifest_row["scene"],
    }
    _require(all(value.get(key) == expected for key, expected in expected_identity.items()), f"{state} prediction row[{index}] identity drift")
    query_ids = _strings(value.get("opaque_query_ids"), f"{state} prediction row[{index}].opaque_query_ids")
    classes = tuple(manifest_row["old_classes"] if state == "before" else (*manifest_row["old_classes"], *manifest_row["new_classes"]))
    common = _mapping_with_exact_keys(
        value.get("common_arms"), COMMON_ARM_IDS,
        f"{state} prediction row[{index}] common-arm",
    )
    candidates = _mapping_with_exact_keys(
        value.get("candidates"), CANDIDATE_IDS,
        f"{state} prediction row[{index}] candidate",
    )
    arms_by_candidate: dict[str, dict[str, tuple[str, ...]]] = {}
    shared = {
        arm_id: _payload_arm(common.get(arm_id), arm_id=arm_id, expected_classes=classes, query_count=len(query_ids), name=f"{state} prediction row[{index}].common_arms.{arm_id}")
        for arm_id in COMMON_ARM_IDS
    }
    for candidate_id in CANDIDATE_IDS:
        candidate = candidates[candidate_id]
        _require(isinstance(candidate, Mapping), f"{state} prediction row[{index}].{candidate_id} must be an object")
        candidate_arms = _mapping_with_exact_keys(
            candidate.get("arms"), ADAPTED_ARM_IDS,
            f"{state} prediction row[{index}].{candidate_id} adapted-arm",
        )
        arms = dict(shared)
        arms.update(
            {
                arm_id: _payload_arm(candidate_arms.get(arm_id), arm_id=arm_id, expected_classes=classes, query_count=len(query_ids), name=f"{state} prediction row[{index}].{candidate_id}.arms.{arm_id}")
                for arm_id in ADAPTED_ARM_IDS
            }
        )
        arms_by_candidate[candidate_id] = arms
    return {"query_ids": query_ids, "arms_by_candidate": arms_by_candidate}


def _manifest_rows(value: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    _require(isinstance(value, list) and len(value) == ROW_COUNT, "pair manifest must contain exactly 18 rows")
    expected_fields = {
        "row_id", "receiver_id", "k_shot", "scene", "old_classes", "new_classes",
        "before_query_ids_sha256", "after_query_ids_sha256", "formal_d92_row_key", "formal_d92_score_row_sha256",
    }
    rows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    coverage: set[tuple[str, int, str]] = set()
    for index, raw in enumerate(value):
        row = _exact(raw, expected_fields, f"pair manifest.rows[{index}]")
        row_id = _text(row["row_id"], f"pair manifest.rows[{index}].row_id")
        receiver = _text(row["receiver_id"], f"pair manifest.rows[{index}].receiver_id")
        k_shot = _integer(row["k_shot"], f"pair manifest.rows[{index}].k_shot", minimum=1)
        _require(k_shot in {1, 5}, "pair manifest allows only K1/K5")
        scene = _text(row["scene"], f"pair manifest.rows[{index}].scene")
        old_classes = _strings(row["old_classes"], f"pair manifest.rows[{index}].old_classes")
        new_classes = _strings(row["new_classes"], f"pair manifest.rows[{index}].new_classes")
        _require(not set(old_classes) & set(new_classes), "pair manifest old/new class overlap")
        normalized = {
            "row_id": row_id, "receiver_id": receiver, "k_shot": k_shot, "scene": scene,
            "old_classes": old_classes, "new_classes": new_classes,
            "before_query_ids_sha256": _sha(row["before_query_ids_sha256"], "before_query_ids_sha256"),
            "after_query_ids_sha256": _sha(row["after_query_ids_sha256"], "after_query_ids_sha256"),
            "formal_d92_row_key": _text(row["formal_d92_row_key"], "formal_d92_row_key"),
            "formal_d92_score_row_sha256": _sha(row["formal_d92_score_row_sha256"], "formal_d92_score_row_sha256"),
        }
        _require(row_id not in by_id and (receiver, k_shot, scene) not in coverage, "pair manifest duplicate row identity")
        rows.append(normalized)
        by_id[row_id] = normalized
        coverage.add((receiver, k_shot, scene))
    expected_order = sorted(rows, key=lambda row: (row["receiver_id"], row["k_shot"], row["scene"], row["row_id"]))
    _require(rows == expected_order, "pair manifest row order drift")
    receivers = {row["receiver_id"] for row in rows}
    scenes = {row["scene"] for row in rows}
    expected_coverage = {(receiver, k, scene) for receiver in receivers for k in (1, 5) for scene in scenes}
    _require(len(receivers) == 3 and len(scenes) == 3 and coverage == expected_coverage, "pair manifest 3 receiver x K1/K5 x 3 scene coverage drift")
    return rows, by_id


def _validate_prediction_payload(
    payload: Mapping[str, Any], *, state: str, manifest_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    _reject_prediction_truth(payload, f"{state} prediction")
    _require(payload.get("schema") == PREDICTION_SCHEMA, f"{state} prediction schema drift")
    _require(payload.get("truth_loaded") is False, f"{state} prediction must remain truth-closed")
    _require(payload.get("candidate_ids") == list(CANDIDATE_IDS), f"{state} prediction candidate order drift")
    _require(payload.get("row_count") == ROW_COUNT and payload.get("rows_complete") is True, f"{state} prediction 18-row closure drift")
    _require(all(payload.get(field) == 0 for field in ("query_rows_used_for_fit", "query_state_updates", "query_selection_count")), f"{state} prediction query access drift")
    receipt = _sha(payload.get("prediction_sha256"), f"{state} prediction.prediction_sha256")
    unsigned = dict(payload)
    unsigned.pop("prediction_sha256", None)
    _require(canonical_sha256(unsigned) == receipt, f"{state} prediction canonical receipt mismatch")
    raw_rows = payload.get("rows")
    _require(isinstance(raw_rows, list) and len(raw_rows) == ROW_COUNT, f"{state} prediction row coverage drift")
    by_id: dict[str, dict[str, Any]] = {}
    all_query_ids: list[str] = []
    for index, (raw, manifest_row) in enumerate(zip(raw_rows, manifest_rows, strict=True)):
        row = _prediction_row(raw, manifest_row=manifest_row, state=state, index=index)
        _require(raw.get("row_id") not in by_id, f"{state} prediction duplicate row")
        by_id[raw["row_id"]] = row
        all_query_ids.extend(row["query_ids"])
    _require(len(all_query_ids) == len(set(all_query_ids)), f"{state} prediction query IDs must be globally unique")
    return {"prediction_sha256": receipt, "rows": by_id, "query_ids": tuple(all_query_ids)}


def validate_d127_s0_prediction_pairs(
    *,
    before_prediction: Mapping[str, Any] | str | Path,
    after_prediction: Mapping[str, Any] | str | Path,
    pair_manifest: Mapping[str, Any] | str | Path,
    expected_method_lock_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the complete truth-free before/after S0 closure.

    This function intentionally cannot receive a truth catalog and is safe to
    call before the independent scorer opens query labels or old/new roles.
    """

    manifest = _read_mapping(pair_manifest, "D127 pair manifest")
    fields = {
        "schema", "pair_id", "protocol_schema", "truth_open", "method_lock_sha256", "capsule_id", "split_id",
        "query_id_root_sha256", "candidate_ids", "arm_ids", "row_count", "before_prediction_sha256", "after_prediction_sha256",
        "rows", "pair_manifest_sha256",
    }
    document = _exact(manifest, fields, "D127 pair manifest")
    _require(document["schema"] == PAIR_MANIFEST_SCHEMA and document["protocol_schema"] == PROTOCOL_SCHEMA, "pair manifest schema/protocol drift")
    _require(document["truth_open"] is False, "pair manifest must remain truth-closed")
    _text(document["pair_id"], "pair manifest.pair_id")
    method_lock = _sha(document["method_lock_sha256"], "pair manifest.method_lock_sha256")
    if expected_method_lock_sha256 is not None:
        _require(method_lock == _sha(expected_method_lock_sha256, "expected_method_lock_sha256"), "method lock drift")
    _text(document["capsule_id"], "pair manifest.capsule_id")
    _text(document["split_id"], "pair manifest.split_id")
    query_root = _sha(document["query_id_root_sha256"], "pair manifest.query_id_root_sha256")
    _require(document["candidate_ids"] == list(CANDIDATE_IDS) and document["arm_ids"] == list(ARM_IDS), "pair manifest candidate/arm order drift")
    _require(document["row_count"] == ROW_COUNT, "pair manifest row count drift")
    pair_receipt = _receipt(document, "pair_manifest_sha256", "pair manifest")
    rows, _ = _manifest_rows(document["rows"])
    before = _validate_prediction_payload(_read_mapping(before_prediction, "before prediction"), state="before", manifest_rows=rows)
    after = _validate_prediction_payload(_read_mapping(after_prediction, "after prediction"), state="after", manifest_rows=rows)
    _require(before["prediction_sha256"] == _sha(document["before_prediction_sha256"], "pair manifest.before_prediction_sha256"), "before prediction/pair manifest hash drift")
    _require(after["prediction_sha256"] == _sha(document["after_prediction_sha256"], "pair manifest.after_prediction_sha256"), "after prediction/pair manifest hash drift")
    _require(canonical_sha256(sorted(after["query_ids"])) == query_root, "pair manifest after-query root drift")
    for row in rows:
        row_id = row["row_id"]
        before_ids = before["rows"][row_id]["query_ids"]
        after_ids = after["rows"][row_id]["query_ids"]
        _require(canonical_sha256(list(before_ids)) == row["before_query_ids_sha256"], "before query receipt drift")
        _require(canonical_sha256(list(after_ids)) == row["after_query_ids_sha256"], "after query receipt drift")
        _require(set(before_ids).issubset(set(after_ids)), "before query IDs must be an after-query subset")
    return {
        "pair_manifest_sha256": pair_receipt,
        "method_lock_sha256": method_lock,
        "rows": rows,
        "before": before,
        "after": after,
        "query_id_root_sha256": query_root,
    }


def _open_truth_catalog(
    value: Mapping[str, Any] | str | Path, *, prediction: Mapping[str, Any]
) -> tuple[dict[str, dict[str, str]], str]:
    catalog = _read_mapping(value, "D127 independent truth catalog")
    fields = {"schema", "truth_open", "pair_manifest_sha256", "query_count", "queries", "truth_catalog_sha256"}
    document = _exact(catalog, fields, "D127 independent truth catalog")
    _require(document["schema"] == TRUTH_CATALOG_SCHEMA and document["truth_open"] is True, "truth catalog opening/schema drift")
    _require(_sha(document["pair_manifest_sha256"], "truth catalog.pair_manifest_sha256") == prediction["pair_manifest_sha256"], "truth catalog/pair manifest binding drift")
    receipt = _receipt(document, "truth_catalog_sha256", "truth catalog")
    raw_queries = document["queries"]
    _require(isinstance(raw_queries, list) and document["query_count"] == len(raw_queries), "truth catalog query count drift")
    expected_ids = set(prediction["after"]["query_ids"])
    truth: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(raw_queries):
        item = _exact(raw, {"opaque_query_id", "label", "role"}, f"truth catalog.queries[{index}]")
        query_id = _text(item["opaque_query_id"], f"truth catalog.queries[{index}].opaque_query_id")
        _require(query_id not in truth, "truth catalog duplicate opaque query ID")
        truth[query_id] = {
            "label": _text(item["label"], f"truth catalog.queries[{index}].label"),
            "role": _text(item["role"], f"truth catalog.queries[{index}].role"),
        }
    _require(set(truth) == expected_ids, "truth catalog/query coverage drift")
    for row in prediction["rows"]:
        row_id = row["row_id"]
        old = set(row["old_classes"])
        new = set(row["new_classes"])
        after_ids = prediction["after"]["rows"][row_id]["query_ids"]
        before_ids = prediction["before"]["rows"][row_id]["query_ids"]
        expected_before = tuple(query_id for query_id in after_ids if truth[query_id]["role"] == "old")
        _require(before_ids == expected_before, "before/after old query IDs or truth roles drift")
        labels = [truth[query_id]["label"] for query_id in after_ids]
        roles = [truth[query_id]["role"] for query_id in after_ids]
        _require(all(role in {"old", "new"} for role in roles), "truth catalog role drift")
        _require(all((role == "old" and label in old) or (role == "new" and label in new) for label, role in zip(labels, roles, strict=True)), "truth catalog label/role registry drift")
        _require(old.issubset({truth[qid]["label"] for qid in before_ids}), "before old-class coverage drift")
        _require(new.issubset({truth[qid]["label"] for qid in after_ids if truth[qid]["role"] == "new"}), "after new-class coverage drift")
    return truth, receipt


def _open_formal_d92_reference(
    value: Mapping[str, Any] | str | Path, *, prediction: Mapping[str, Any]
) -> str:
    reference = _read_mapping(value, "formal D92 same-row reference")
    fields = {"schema", "pair_manifest_sha256", "pipeline_receipt_sha256", "row_count", "rows", "formal_d92_reference_sha256"}
    document = _exact(reference, fields, "formal D92 same-row reference")
    _require(document["schema"] == FORMAL_D92_REFERENCE_SCHEMA, "formal D92 reference schema drift")
    _require(_sha(document["pair_manifest_sha256"], "formal D92 pair hash") == prediction["pair_manifest_sha256"], "formal D92/pair manifest binding drift")
    _sha(document["pipeline_receipt_sha256"], "formal D92 pipeline receipt")
    _require(document["row_count"] == ROW_COUNT and isinstance(document["rows"], list) and len(document["rows"]) == ROW_COUNT, "formal D92 18-row coverage drift")
    receipt = _receipt(document, "formal_d92_reference_sha256", "formal D92 reference")
    by_id: dict[str, Mapping[str, Any]] = {}
    expected_fields = {"row_id", "receiver_id", "k_shot", "scene", "formal_d92_row_key", "formal_d92_score_row_sha256"}
    for index, raw in enumerate(document["rows"]):
        item = _exact(raw, expected_fields, f"formal D92 rows[{index}]")
        row_id = _text(item["row_id"], f"formal D92 rows[{index}].row_id")
        _require(row_id not in by_id, "formal D92 duplicate row")
        by_id[row_id] = item
    _require(set(by_id) == {row["row_id"] for row in prediction["rows"]}, "formal D92 row-key coverage drift")
    for row in prediction["rows"]:
        item = by_id[row["row_id"]]
        _require(all(item[field] == row[field] for field in ("row_id", "receiver_id", "k_shot", "scene", "formal_d92_row_key", "formal_d92_score_row_sha256")), "formal D92 same-row key/hash drift")
    return receipt


def _rate(correct: int, total: int, name: str) -> float:
    _require(type(correct) is int and type(total) is int and total > 0, f"{name} denominator drift")
    return 100.0 * correct / total


def _harmonic(old_accuracy: float, new_accuracy: float) -> float:
    return 0.0 if old_accuracy + new_accuracy == 0.0 else 2.0 * old_accuracy * new_accuracy / (old_accuracy + new_accuracy)


def _class_counts(pairs: Sequence[tuple[str, str]], classes: Sequence[str], name: str) -> dict[str, dict[str, int]]:
    counts = {label: {"correct_count": 0, "query_count": 0} for label in classes}
    for prediction, label in pairs:
        _require(label in counts, f"{name} includes a class outside the old registry")
        counts[label]["query_count"] += 1
        counts[label]["correct_count"] += int(prediction == label)
    _require(all(value["query_count"] > 0 for value in counts.values()), f"{name} old-class floor coverage drift")
    return counts


def _floor(counts: Mapping[str, Mapping[str, int]], name: str) -> float:
    return min(_rate(item["correct_count"], item["query_count"], f"{name}.{label}") for label, item in counts.items())


def _metric_row(
    *,
    manifest_row: Mapping[str, Any],
    candidate_id: str,
    arm_id: str,
    before_predictions: Sequence[str],
    after_predictions: Sequence[str],
    before_ids: Sequence[str],
    after_ids: Sequence[str],
    truth: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    old = set(manifest_row["old_classes"])
    new = set(manifest_row["new_classes"])
    before_pairs = [(prediction, truth[qid]["label"]) for qid, prediction in zip(before_ids, before_predictions, strict=True)]
    after_pairs = [(prediction, truth[qid]["label"], truth[qid]["role"]) for qid, prediction in zip(after_ids, after_predictions, strict=True)]
    _require(all(label in old for _, label in before_pairs), "before truth role drift")
    after_old = [(prediction, label) for prediction, label, role in after_pairs if role == "old"]
    after_new = [(prediction, label) for prediction, label, role in after_pairs if role == "new"]
    _require(len(after_old) + len(after_new) == len(after_pairs), "after truth role partition drift")
    before_counts = _class_counts(before_pairs, manifest_row["old_classes"], "before")
    after_counts = _class_counts(after_old, manifest_row["old_classes"], "after")
    before_correct = sum(prediction == label for prediction, label in before_pairs)
    after_old_correct = sum(prediction == label for prediction, label in after_old)
    new_correct = sum(prediction == label for prediction, label in after_new)
    before_old = _rate(before_correct, len(before_pairs), "before old")
    after_old_acc = _rate(after_old_correct, len(after_old), "after old")
    seen_new = _rate(new_correct, len(after_new), "seen new")
    result: dict[str, Any] = {
        "row_id": manifest_row["row_id"], "receiver_id": manifest_row["receiver_id"], "k_shot": manifest_row["k_shot"], "scene": manifest_row["scene"],
        "candidate_id": candidate_id, "arm_id": arm_id,
        "B_old": before_old, "A_old": after_old_acc, "seen_new": seen_new, "H_old_new": _harmonic(after_old_acc, seen_new),
        "old_per_class_floor": _floor(after_counts, "old_per_class_floor"), "B_old_per_class_floor": _floor(before_counts, "B_old_per_class_floor"),
        "forgetting": before_old - after_old_acc,
        "before_old_correct_count": before_correct, "before_old_query_count": len(before_pairs),
        "after_old_correct_count": after_old_correct, "after_old_query_count": len(after_old),
        "new_correct_count": new_correct, "new_query_count": len(after_new),
        "total_correct_count": before_correct + after_old_correct + new_correct,
        "total_query_count": len(before_pairs) + len(after_pairs),
        "before_old_by_class": before_counts, "after_old_by_class": after_counts,
        "formal_d92_row_key": manifest_row["formal_d92_row_key"], "formal_d92_score_row_sha256": manifest_row["formal_d92_score_row_sha256"],
    }
    _require(all(math.isfinite(float(result[field])) for field in ("B_old", "A_old", "seen_new", "H_old_new", "old_per_class_floor", "forgetting")), "nonfinite metric result")
    result["metric_row_sha256"] = canonical_sha256(result)
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]], *, group_key: str, group_value: Any, candidate_id: str, arm_id: str) -> dict[str, Any]:
    _require(bool(rows), "aggregate cannot be empty")
    fields = (
        "before_old_correct_count", "before_old_query_count", "after_old_correct_count", "after_old_query_count",
        "new_correct_count", "new_query_count", "total_correct_count", "total_query_count",
    )
    totals = {field: sum(int(row[field]) for row in rows) for field in fields}
    before_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct_count": 0, "query_count": 0})
    after_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct_count": 0, "query_count": 0})
    for row in rows:
        for label, counts in row["before_old_by_class"].items():
            before_counts[label]["correct_count"] += int(counts["correct_count"])
            before_counts[label]["query_count"] += int(counts["query_count"])
        for label, counts in row["after_old_by_class"].items():
            after_counts[label]["correct_count"] += int(counts["correct_count"])
            after_counts[label]["query_count"] += int(counts["query_count"])
    before_old = _rate(totals["before_old_correct_count"], totals["before_old_query_count"], "aggregate B_old")
    after_old = _rate(totals["after_old_correct_count"], totals["after_old_query_count"], "aggregate A_old")
    seen_new = _rate(totals["new_correct_count"], totals["new_query_count"], "aggregate seen_new")
    result: dict[str, Any] = {
        "group_by": group_key, "group_value": group_value, "candidate_id": candidate_id, "arm_id": arm_id,
        "row_count": len(rows), "aggregation": "micro_average_same_row_counts",
        "B_old": before_old, "A_old": after_old, "seen_new": seen_new, "H_old_new": _harmonic(after_old, seen_new),
        "old_per_class_floor": _floor(after_counts, "aggregate old_per_class_floor"), "B_old_per_class_floor": _floor(before_counts, "aggregate B_old_per_class_floor"),
        "forgetting": before_old - after_old, **totals,
        "source_metric_row_sha256s": [row["metric_row_sha256"] for row in rows],
    }
    result["aggregate_sha256"] = canonical_sha256(result)
    return result


def _direction_decisions(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        by_arm = {arm_id: [row for row in metric_rows if row["candidate_id"] == candidate_id and row["arm_id"] == arm_id] for arm_id in ARM_IDS}
        all_rows = {arm_id: _aggregate(by_arm[arm_id], group_key="scope", group_value="S0_18", candidate_id=candidate_id, arm_id=arm_id) for arm_id in ARM_IDS}
        k5 = {arm_id: _aggregate([row for row in by_arm[arm_id] if row["k_shot"] == 5], group_key="k_shot", group_value=5, candidate_id=candidate_id, arm_id=arm_id) for arm_id in ("M_DA", "M_JOINT")}
        da_delta_h = all_rows["M_DA"]["H_old_new"] - all_rows["M0"]["H_old_new"]
        lite_k5_delta_h = k5["M_JOINT"]["H_old_new"] - k5["M_DA"]["H_old_new"]
        joint_delta_h = all_rows["M_JOINT"]["H_old_new"] - all_rows["M0"]["H_old_new"]
        joint_total_correct_delta = all_rows["M_JOINT"]["total_correct_count"] - all_rows["M0"]["total_correct_count"]
        result: dict[str, Any] = {
            "candidate_id": candidate_id,
            "DA_delta_H_old_new": da_delta_h, "DA_direction_pass": da_delta_h > 0.0,
            "K5_Lite_after_DA_delta_H_old_new": lite_k5_delta_h, "K5_Lite_after_DA_direction_pass": lite_k5_delta_h > 0.0,
            "joint_delta_H_old_new": joint_delta_h, "joint_total_correct_delta": joint_total_correct_delta,
            "joint_direction_pass": joint_delta_h > 0.0 and joint_total_correct_delta > 0,
        }
        result["all_three_direction_pass"] = bool(result["DA_direction_pass"] and result["K5_Lite_after_DA_direction_pass"] and result["joint_direction_pass"])
        result["direction_receipt_sha256"] = canonical_sha256(result)
        decisions.append(result)
    return decisions


def score_d127_s0(
    *,
    before_prediction: Mapping[str, Any] | str | Path,
    after_prediction: Mapping[str, Any] | str | Path,
    pair_manifest: Mapping[str, Any] | str | Path,
    truth_catalog: Mapping[str, Any] | str | Path,
    formal_d92_reference: Mapping[str, Any] | str | Path,
    expected_method_lock_sha256: str | None = None,
) -> dict[str, Any]:
    """Open truth only after prediction closure and compute D127 S0 metrics."""

    prediction = validate_d127_s0_prediction_pairs(
        before_prediction=before_prediction, after_prediction=after_prediction,
        pair_manifest=pair_manifest, expected_method_lock_sha256=expected_method_lock_sha256,
    )
    truth, truth_receipt = _open_truth_catalog(truth_catalog, prediction=prediction)
    formal_receipt = _open_formal_d92_reference(formal_d92_reference, prediction=prediction)
    metric_rows: list[dict[str, Any]] = []
    same_rows: list[dict[str, Any]] = []
    for manifest_row in prediction["rows"]:
        row_id = manifest_row["row_id"]
        before = prediction["before"]["rows"][row_id]
        after = prediction["after"]["rows"][row_id]
        arms: list[dict[str, Any]] = []
        for candidate_id in CANDIDATE_IDS:
            for arm_id in ARM_IDS:
                metric = _metric_row(
                    manifest_row=manifest_row, candidate_id=candidate_id, arm_id=arm_id,
                    before_predictions=before["arms_by_candidate"][candidate_id][arm_id],
                    after_predictions=after["arms_by_candidate"][candidate_id][arm_id],
                    before_ids=before["query_ids"], after_ids=after["query_ids"], truth=truth,
                )
                arms.append(metric)
                metric_rows.append(metric)
        same: dict[str, Any] = {
            "row_id": row_id, "receiver_id": manifest_row["receiver_id"], "k_shot": manifest_row["k_shot"], "scene": manifest_row["scene"],
            "formal_d92_row_key": manifest_row["formal_d92_row_key"], "formal_d92_score_row_sha256": manifest_row["formal_d92_score_row_sha256"],
            "candidate_arm_metrics": arms,
        }
        same["same_row_sha256"] = canonical_sha256(same)
        same_rows.append(same)
    _require(len(same_rows) == ROW_COUNT and len(metric_rows) == ROW_COUNT * len(CANDIDATE_IDS) * len(ARM_IDS), "score metric row coverage drift")
    aggregation: dict[str, list[dict[str, Any]]] = {"receiver_id": [], "scene": [], "k_shot": []}
    for group_key in tuple(aggregation):
        values = sorted({row[group_key] for row in metric_rows}, key=str)
        for candidate_id in CANDIDATE_IDS:
            for arm_id in ARM_IDS:
                for group_value in values:
                    aggregation[group_key].append(_aggregate([row for row in metric_rows if row["candidate_id"] == candidate_id and row["arm_id"] == arm_id and row[group_key] == group_value], group_key=group_key, group_value=group_value, candidate_id=candidate_id, arm_id=arm_id))
    score: dict[str, Any] = {
        "schema": SCORE_SCHEMA, "pair_manifest_sha256": prediction["pair_manifest_sha256"], "method_lock_sha256": prediction["method_lock_sha256"],
        "truth_catalog_sha256": truth_receipt, "formal_d92_reference_sha256": formal_receipt,
        "row_count": ROW_COUNT, "metric_row_count": len(metric_rows), "same_row_results": same_rows,
        "aggregates": aggregation, "s0_direction_decisions": _direction_decisions(metric_rows),
        "truth_never_returned_to_predictor": True, "formal_d92_is_same_row_reference_only": True,
    }
    score["score_manifest_sha256"] = canonical_sha256(score)
    return score


# ---------------------------------------------------------------------------
# Release integration for the package adapter's one-file paired prediction.
# These contracts intentionally live beside the scoring core rather than in
# the predictor.  The paired artifact is still truth-free at every point
# before ``write_d127_s0_truth_open_event_exclusive`` succeeds.

PAIRED_NORMALIZED_SCHEMA = "cvs.stage2.d127.s0.normalized_paired_prediction.v1"
TRUTH_OPEN_EVENT_SCHEMA = "cvs.stage2.d127.s0.truth_open_event.v1"
PAIRED_TRUTH_CATALOG_SCHEMA = "cvs.stage2.d127.s0.paired_truth_catalog.v1"
PAIRED_FORMAL_D92_REFERENCE_SCHEMA = "cvs.stage2.d127.s0.paired_formal_d92_reference.v1"
PAIRED_SCORE_SCHEMA = "cvs.stage2.d127.s0.paired_score_manifest.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pinned_json(path: str | Path, *, expected_sha256: str, name: str) -> tuple[dict[str, Any], str]:
    candidate = Path(path)
    expected = _sha(expected_sha256, f"expected {name} SHA256")
    _require(candidate.is_file() and not candidate.is_symlink(), f"{name} must be a regular non-symlink file")
    actual = _sha256_file(candidate)
    _require(actual == expected, f"{name} SHA mismatch")
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D127S0ScorerError(f"{name} is not valid UTF-8 JSON") from exc
    _require(isinstance(document, dict), f"{name} must contain a JSON object")
    return document, actual


def _write_exclusive_json(path: str | Path, payload: Mapping[str, Any], *, name: str) -> Path:
    target = Path(path)
    _require(not target.is_symlink(), f"{name} output cannot be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as stream:
            stream.write(canonical_bytes(dict(payload)) + b"\n")
    except FileExistsError as exc:
        raise D127S0ScorerError(f"{name} output already exists") from exc
    return target


def _paired_arm_predictions(value: Any, *, arm_id: str, classes: tuple[str, ...], count: int, name: str) -> tuple[str, ...]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    observed_classes = _strings(value.get("classes"), f"{name}.classes")
    _require(observed_classes == classes, f"{name} registry drift")
    predictions = _strings(value.get("predictions"), f"{name}.predictions", unique=False)
    _require(len(predictions) == count and all(label in classes for label in predictions), f"{name} prediction closure drift")
    if "arm_id" in value:
        _require(value["arm_id"] == arm_id, f"{name} arm identity drift")
    return predictions


def normalize_d127_s0_paired_prediction(
    *,
    paired_prediction: Mapping[str, Any],
    prepared_plan: Mapping[str, Any],
    method_lock_sha256: str,
) -> dict[str, Any]:
    """Normalize an already validated package-adapter paired artifact.

    The package adapter owns the source-side target-package validation.  This
    function preserves only the truth-free prediction values and the binding
    fields needed by the independent scorer; it does not call any model.
    """

    from . import stage2_d127_s0_package_adapter as adapter

    adapter.validate_d127_s0_prediction_pairs(paired_prediction, prepared_plan=prepared_plan)
    method_lock = _sha(method_lock_sha256, "method_lock_sha256")
    _require(prepared_plan.get("method_lock_sha256") == method_lock, "prepared plan/method-lock drift")
    _require(paired_prediction.get("method_lock_sha256") == method_lock, "paired prediction/method-lock drift")
    _require(paired_prediction.get("truth_loaded") is False, "paired prediction must remain truth-closed")
    _require(tuple(paired_prediction.get("candidate_ids", ())) == CANDIDATE_IDS, "paired prediction candidate order drift")
    pair_manifest = paired_prediction.get("pair_manifest")
    _require(isinstance(pair_manifest, Mapping), "paired prediction scorer pair manifest is missing")
    pair_manifest_sha = _sha(pair_manifest.get("pair_manifest_sha256"), "paired scorer pair manifest SHA256")
    bindings = paired_prediction.get("pair_bindings")
    states = _mapping_with_exact_keys(
        paired_prediction.get("states"), STATES, "paired prediction state"
    )
    _require(isinstance(bindings, list) and len(bindings) == ROW_COUNT, "paired prediction pair-binding coverage drift")
    normalized_rows: list[dict[str, Any]] = []
    seen_after: set[str] = set()
    for index, binding in enumerate(bindings):
        _require(isinstance(binding, Mapping), "paired prediction pair binding must be an object")
        before_row = states["before"].get("rows", [])[index]
        after_row = states["after"].get("rows", [])[index]
        _require(isinstance(before_row, Mapping) and isinstance(after_row, Mapping), "paired prediction state row is missing")
        row_id = _text(binding.get("row_id"), f"paired binding[{index}].row_id")
        receiver = _text(binding.get("receiver"), f"paired binding[{index}].receiver")
        k_shot = _integer(binding.get("k_shot"), f"paired binding[{index}].k_shot", minimum=1)
        scene = _text(binding.get("scene"), f"paired binding[{index}].scene")
        _require(all(row.get("row_id") == row_id and row.get("receiver") == receiver and row.get("k_shot") == k_shot and row.get("scene") == scene for row in (before_row, after_row)), "paired prediction row/binding identity drift")
        before_ids = _strings(before_row.get("opaque_query_ids"), f"paired before[{index}].opaque_query_ids")
        after_ids = _strings(after_row.get("opaque_query_ids"), f"paired after[{index}].opaque_query_ids")
        _require(not seen_after.intersection(after_ids), "paired prediction after opaque query IDs are reused")
        seen_after.update(after_ids)
        before_common = _mapping_with_exact_keys(
            before_row.get("common_arms"), COMMON_ARM_IDS, "paired before common-arm"
        )
        after_common = _mapping_with_exact_keys(
            after_row.get("common_arms"), COMMON_ARM_IDS, "paired after common-arm"
        )
        before_candidates = _mapping_with_exact_keys(
            before_row.get("candidates"), CANDIDATE_IDS, "paired before candidate"
        )
        after_candidates = _mapping_with_exact_keys(
            after_row.get("candidates"), CANDIDATE_IDS, "paired after candidate"
        )
        old_classes = _strings(before_common["M0"].get("classes"), f"paired before[{index}].M0.classes")
        after_classes = _strings(after_common["M0"].get("classes"), f"paired after[{index}].M0.classes")
        _require(after_classes[: len(old_classes)] == old_classes and len(after_classes) > len(old_classes), "paired old/new registry prefix drift")
        new_classes = after_classes[len(old_classes):]
        arms_by_state: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {"before": {}, "after": {}}
        for state, state_row, common, candidates, ids, classes in (
            ("before", before_row, before_common, before_candidates, before_ids, old_classes),
            ("after", after_row, after_common, after_candidates, after_ids, after_classes),
        ):
            shared = {
                arm_id: _paired_arm_predictions(common[arm_id], arm_id=arm_id, classes=classes, count=len(ids), name=f"paired {state}[{index}].common.{arm_id}")
                for arm_id in COMMON_ARM_IDS
            }
            for candidate_id in CANDIDATE_IDS:
                candidate = candidates[candidate_id]
                _require(isinstance(candidate, Mapping), f"paired {state}[{index}].{candidate_id} must be an object")
                candidate_arms_input = _mapping_with_exact_keys(
                    candidate.get("arms"), ADAPTED_ARM_IDS,
                    f"paired {state}[{index}].{candidate_id} adapted-arm",
                )
                candidate_arms = dict(shared)
                candidate_arms.update(
                    {
                        arm_id: _paired_arm_predictions(candidate_arms_input[arm_id], arm_id=arm_id, classes=classes, count=len(ids), name=f"paired {state}[{index}].{candidate_id}.{arm_id}")
                        for arm_id in ADAPTED_ARM_IDS
                    }
                )
                arms_by_state[state][candidate_id] = candidate_arms
        formal = binding.get("formal_d92_reference")
        _require(isinstance(formal, Mapping), "paired formal D92 locator is missing")
        source_job = _text(formal.get("source_d92_job_id"), "paired formal D92 source job")
        retry2_manifest = _sha(formal.get("d92_retry2_manifest_sha256"), "paired formal D92 retry2 manifest SHA256")
        normalized_rows.append(
            {
                "row_id": row_id, "receiver_id": receiver, "k_shot": k_shot, "scene": scene,
                "old_classes": list(old_classes), "new_classes": list(new_classes),
                "before_query_ids": list(before_ids), "after_query_ids": list(after_ids),
                "before_query_ids_sha256": canonical_sha256(list(before_ids)), "after_query_ids_sha256": canonical_sha256(list(after_ids)),
                "formal_d92_source_job_id": source_job, "formal_d92_retry2_manifest_sha256": retry2_manifest,
                "arms_by_state": {
                    state: {candidate: {arm: list(predictions) for arm, predictions in arms.items()} for candidate, arms in candidate_map.items()}
                    for state, candidate_map in arms_by_state.items()
                },
            }
        )
    expected_order = sorted(normalized_rows, key=lambda row: (row["receiver_id"], row["k_shot"], row["scene"], row["row_id"]))
    _require(normalized_rows == expected_order, "normalized paired row order drift")
    normalized: dict[str, Any] = {
        "schema": PAIRED_NORMALIZED_SCHEMA,
        "paired_prediction_sha256": _sha(paired_prediction.get("paired_prediction_sha256"), "paired prediction SHA256"),
        "prepared_plan_sha256": _sha(prepared_plan.get("prepared_plan_sha256"), "prepared plan SHA256"),
        "method_lock_sha256": method_lock,
        "pair_manifest_sha256": pair_manifest_sha,
        "row_count": ROW_COUNT,
        "candidate_ids": list(CANDIDATE_IDS), "arm_ids": list(ARM_IDS),
        "after_query_id_root_sha256": canonical_sha256(sorted(seen_after)),
        "rows": normalized_rows,
    }
    normalized["normalized_prediction_sha256"] = canonical_sha256(normalized)
    return normalized


def prepare_d127_s0_scoring_inputs(
    *,
    paired_prediction_path: str | Path,
    expected_paired_prediction_sha256: str,
    prepared_plan_path: str | Path,
    expected_prepared_plan_sha256: str,
    method_lock_path: str | Path,
    expected_method_lock_sha256: str,
) -> dict[str, Any]:
    """Perform every truth-free release check required before a truth open."""

    from . import stage2_d127_s0_package_adapter as adapter

    plan, plan_file_sha = adapter.load_d127_s0_prepared_plan(prepared_plan_path, expected_sha256=expected_prepared_plan_sha256)
    lock, lock_file_sha, _locks = adapter.load_d127_s0_method_lock(method_lock_path, expected_sha256=expected_method_lock_sha256)
    _require(lock_file_sha == plan["method_lock_sha256"], "method-lock/prepared-plan file SHA drift")
    _require(lock.get("checkpoint", {}).get("sha256") == plan.get("checkpoint_sha256"), "method-lock/prepared-plan checkpoint drift")
    paired, paired_file_sha = _load_pinned_json(paired_prediction_path, expected_sha256=expected_paired_prediction_sha256, name="paired prediction")
    normalized = normalize_d127_s0_paired_prediction(paired_prediction=paired, prepared_plan=plan, method_lock_sha256=lock_file_sha)
    return {
        "normalized_prediction": normalized,
        "paired_prediction_file_sha256": paired_file_sha,
        "prepared_plan_file_sha256": plan_file_sha,
        "method_lock_file_sha256": lock_file_sha,
    }


def build_d127_s0_truth_open_event(normalized_prediction: Mapping[str, Any]) -> dict[str, Any]:
    _require(normalized_prediction.get("schema") == PAIRED_NORMALIZED_SCHEMA, "normalized paired schema drift")
    normalized_sha = _sha(normalized_prediction.get("normalized_prediction_sha256"), "normalized prediction SHA256")
    unsigned = dict(normalized_prediction)
    unsigned.pop("normalized_prediction_sha256", None)
    _require(canonical_sha256(unsigned) == normalized_sha, "normalized paired receipt drift")
    event: dict[str, Any] = {
        "schema": TRUTH_OPEN_EVENT_SCHEMA, "truth_open": True,
        "paired_prediction_sha256": _sha(normalized_prediction.get("paired_prediction_sha256"), "paired prediction SHA256"),
        "prepared_plan_sha256": _sha(normalized_prediction.get("prepared_plan_sha256"), "prepared plan SHA256"),
        "method_lock_sha256": _sha(normalized_prediction.get("method_lock_sha256"), "method lock SHA256"),
        "pair_manifest_sha256": _sha(normalized_prediction.get("pair_manifest_sha256"), "pair manifest SHA256"),
        "normalized_prediction_sha256": normalized_sha,
        "truth_catalog_permitted": True, "formal_d92_reference_permitted": True,
    }
    event["truth_open_event_sha256"] = canonical_sha256(event)
    return event


def write_d127_s0_truth_open_event_exclusive(path: str | Path, event: Mapping[str, Any]) -> Path:
    _validate_truth_open_event(event, normalized_prediction=None)
    return _write_exclusive_json(path, event, name="D127 truth-open event")


def _validate_truth_open_event(event: Mapping[str, Any], *, normalized_prediction: Mapping[str, Any] | None) -> str:
    fields = {
        "schema", "truth_open", "paired_prediction_sha256", "prepared_plan_sha256", "method_lock_sha256",
        "pair_manifest_sha256", "normalized_prediction_sha256", "truth_catalog_permitted",
        "formal_d92_reference_permitted", "truth_open_event_sha256",
    }
    document = _exact(event, fields, "D127 truth-open event")
    _require(document["schema"] == TRUTH_OPEN_EVENT_SCHEMA and document["truth_open"] is True, "truth-open event schema/open drift")
    _require(document["truth_catalog_permitted"] is True and document["formal_d92_reference_permitted"] is True, "truth-open event permission drift")
    receipt = _receipt(document, "truth_open_event_sha256", "truth-open event")
    for field in ("paired_prediction_sha256", "prepared_plan_sha256", "method_lock_sha256", "pair_manifest_sha256", "normalized_prediction_sha256"):
        _sha(document[field], f"truth-open event.{field}")
    if normalized_prediction is not None:
        _require(all(document[field] == normalized_prediction[field] for field in ("paired_prediction_sha256", "prepared_plan_sha256", "method_lock_sha256", "pair_manifest_sha256", "normalized_prediction_sha256")), "truth-open event/prediction binding drift")
    return receipt


def _open_paired_truth_catalog(catalog: Mapping[str, Any], *, normalized_prediction: Mapping[str, Any]) -> tuple[dict[str, dict[str, str]], str]:
    fields = {
        "schema", "truth_open", "paired_prediction_sha256", "prepared_plan_sha256", "method_lock_sha256",
        "pair_manifest_sha256", "normalized_prediction_sha256", "query_count", "queries", "truth_catalog_sha256",
    }
    document = _exact(catalog, fields, "D127 paired truth catalog")
    _require(document["schema"] == PAIRED_TRUTH_CATALOG_SCHEMA and document["truth_open"] is True, "paired truth catalog opening/schema drift")
    _require(all(document[field] == normalized_prediction[field] for field in ("paired_prediction_sha256", "prepared_plan_sha256", "method_lock_sha256", "pair_manifest_sha256", "normalized_prediction_sha256")), "paired truth catalog binding drift")
    receipt = _receipt(document, "truth_catalog_sha256", "paired truth catalog")
    raw = document["queries"]
    _require(isinstance(raw, list) and document["query_count"] == len(raw), "paired truth catalog query count drift")
    truth: dict[str, dict[str, str]] = {}
    for index, item in enumerate(raw):
        record = _exact(item, {"opaque_query_id", "label", "role"}, f"paired truth catalog.queries[{index}]")
        query_id = _text(record["opaque_query_id"], f"paired truth catalog query ID[{index}]")
        _require(query_id not in truth, "paired truth catalog duplicate query ID")
        truth[query_id] = {"label": _text(record["label"], "paired truth label"), "role": _text(record["role"], "paired truth role")}
    expected = {query_id for row in normalized_prediction["rows"] for query_id in row["after_query_ids"]}
    _require(set(truth) == expected, "paired truth catalog/query coverage drift")
    for row in normalized_prediction["rows"]:
        old, new = set(row["old_classes"]), set(row["new_classes"])
        after_ids, before_ids = tuple(row["after_query_ids"]), tuple(row["before_query_ids"])
        _require(before_ids == tuple(query_id for query_id in after_ids if truth[query_id]["role"] == "old"), "paired truth old-ID subset/role drift")
        _require(all((truth[qid]["role"] == "old" and truth[qid]["label"] in old) or (truth[qid]["role"] == "new" and truth[qid]["label"] in new) for qid in after_ids), "paired truth label/role registry drift")
        _require(old.issubset({truth[qid]["label"] for qid in before_ids}), "paired truth before old-class coverage drift")
        _require(new.issubset({truth[qid]["label"] for qid in after_ids if truth[qid]["role"] == "new"}), "paired truth after new-class coverage drift")
    return truth, receipt


def _open_paired_formal_d92_reference(reference: Mapping[str, Any], *, normalized_prediction: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], str]:
    fields = {
        "schema", "paired_prediction_sha256", "prepared_plan_sha256", "method_lock_sha256", "pair_manifest_sha256",
        "normalized_prediction_sha256", "pipeline_receipt_sha256", "row_count", "rows", "formal_d92_reference_sha256",
    }
    document = _exact(reference, fields, "D127 paired formal D92 reference")
    _require(document["schema"] == PAIRED_FORMAL_D92_REFERENCE_SCHEMA, "paired formal D92 reference schema drift")
    _require(all(document[field] == normalized_prediction[field] for field in ("paired_prediction_sha256", "prepared_plan_sha256", "method_lock_sha256", "pair_manifest_sha256", "normalized_prediction_sha256")), "paired formal D92 reference binding drift")
    _sha(document["pipeline_receipt_sha256"], "paired formal D92 pipeline receipt")
    _require(document["row_count"] == ROW_COUNT and isinstance(document["rows"], list) and len(document["rows"]) == ROW_COUNT, "paired formal D92 reference row coverage drift")
    receipt = _receipt(document, "formal_d92_reference_sha256", "paired formal D92 reference")
    expected_fields = {
        "row_id", "receiver_id", "k_shot", "scene", "source_d92_job_id", "d92_retry2_manifest_sha256",
        "formal_d92_score_row_key", "formal_d92_score_row_sha256",
    }
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(document["rows"]):
        row = _exact(raw, expected_fields, f"paired formal D92 rows[{index}]")
        row_id = _text(row["row_id"], "paired formal D92 row ID")
        _require(row_id not in by_id, "paired formal D92 duplicate row")
        by_id[row_id] = row
    _require(set(by_id) == {row["row_id"] for row in normalized_prediction["rows"]}, "paired formal D92 row IDs drift")
    for row in normalized_prediction["rows"]:
        reference_row = by_id[row["row_id"]]
        _require(
            reference_row["receiver_id"] == row["receiver_id"] and reference_row["k_shot"] == row["k_shot"] and reference_row["scene"] == row["scene"]
            and reference_row["source_d92_job_id"] == row["formal_d92_source_job_id"]
            and reference_row["d92_retry2_manifest_sha256"] == row["formal_d92_retry2_manifest_sha256"],
            "paired formal D92 same-row locator/hash drift",
        )
        _text(reference_row["formal_d92_score_row_key"], "paired formal D92 score row key")
        _sha(reference_row["formal_d92_score_row_sha256"], "paired formal D92 score row SHA256")
    return by_id, receipt


def score_d127_s0_paired(
    *,
    normalized_prediction: Mapping[str, Any],
    truth_open_event: Mapping[str, Any],
    truth_catalog: Mapping[str, Any],
    formal_d92_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Score a validated adapter paired prediction after the durable open event."""

    _require(normalized_prediction.get("schema") == PAIRED_NORMALIZED_SCHEMA, "normalized paired schema drift")
    _validate_truth_open_event(truth_open_event, normalized_prediction=normalized_prediction)
    # The following two calls are intentionally after the event validation.
    truth, truth_receipt = _open_paired_truth_catalog(truth_catalog, normalized_prediction=normalized_prediction)
    formal_by_id, formal_receipt = _open_paired_formal_d92_reference(formal_d92_reference, normalized_prediction=normalized_prediction)
    metric_rows: list[dict[str, Any]] = []
    same_rows: list[dict[str, Any]] = []
    for row in normalized_prediction["rows"]:
        formal = formal_by_id[row["row_id"]]
        metric_manifest = {
            "row_id": row["row_id"], "receiver_id": row["receiver_id"], "k_shot": row["k_shot"], "scene": row["scene"],
            "old_classes": tuple(row["old_classes"]), "new_classes": tuple(row["new_classes"]),
            "formal_d92_row_key": formal["formal_d92_score_row_key"], "formal_d92_score_row_sha256": formal["formal_d92_score_row_sha256"],
        }
        details: list[dict[str, Any]] = []
        for candidate_id in CANDIDATE_IDS:
            for arm_id in ARM_IDS:
                metric = _metric_row(
                    manifest_row=metric_manifest, candidate_id=candidate_id, arm_id=arm_id,
                    before_predictions=row["arms_by_state"]["before"][candidate_id][arm_id],
                    after_predictions=row["arms_by_state"]["after"][candidate_id][arm_id],
                    before_ids=row["before_query_ids"], after_ids=row["after_query_ids"], truth=truth,
                )
                details.append(metric)
                metric_rows.append(metric)
        same = {
            "row_id": row["row_id"], "receiver_id": row["receiver_id"], "k_shot": row["k_shot"], "scene": row["scene"],
            "formal_d92_score_row_key": formal["formal_d92_score_row_key"], "formal_d92_score_row_sha256": formal["formal_d92_score_row_sha256"],
            "candidate_arm_metrics": details,
        }
        same["same_row_sha256"] = canonical_sha256(same)
        same_rows.append(same)
    _require(len(same_rows) == ROW_COUNT and len(metric_rows) == ROW_COUNT * len(CANDIDATE_IDS) * len(ARM_IDS), "paired score metric coverage drift")
    aggregation: dict[str, list[dict[str, Any]]] = {"receiver_id": [], "scene": [], "k_shot": []}
    for group_key in tuple(aggregation):
        for group_value in sorted({row[group_key] for row in metric_rows}, key=str):
            for candidate_id in CANDIDATE_IDS:
                for arm_id in ARM_IDS:
                    subset = [row for row in metric_rows if row["candidate_id"] == candidate_id and row["arm_id"] == arm_id and row[group_key] == group_value]
                    aggregation[group_key].append(_aggregate(subset, group_key=group_key, group_value=group_value, candidate_id=candidate_id, arm_id=arm_id))
    score: dict[str, Any] = {
        "schema": PAIRED_SCORE_SCHEMA,
        "paired_prediction_sha256": normalized_prediction["paired_prediction_sha256"], "prepared_plan_sha256": normalized_prediction["prepared_plan_sha256"],
        "method_lock_sha256": normalized_prediction["method_lock_sha256"], "pair_manifest_sha256": normalized_prediction["pair_manifest_sha256"],
        "normalized_prediction_sha256": normalized_prediction["normalized_prediction_sha256"],
        "truth_open_event_sha256": truth_open_event["truth_open_event_sha256"], "truth_catalog_sha256": truth_receipt,
        "formal_d92_reference_sha256": formal_receipt, "row_count": ROW_COUNT, "metric_row_count": len(metric_rows),
        "same_row_results": same_rows, "aggregates": aggregation, "s0_direction_decisions": _direction_decisions(metric_rows),
        "truth_never_returned_to_predictor": True, "formal_d92_is_same_row_reference_only": True,
    }
    score["score_manifest_sha256"] = canonical_sha256(score)
    return score


def write_d127_s0_paired_score_exclusive(path: str | Path, score: Mapping[str, Any]) -> Path:
    _require(score.get("schema") == PAIRED_SCORE_SCHEMA, "paired score schema drift")
    _receipt(score, "score_manifest_sha256", "paired score")
    return _write_exclusive_json(path, score, name="D127 paired score")


__all__ = [
    "ADAPTED_ARM_IDS", "ARM_IDS", "CANDIDATE_IDS", "COMMON_ARM_IDS", "D127S0ScorerError",
    "FORMAL_D92_REFERENCE_SCHEMA", "PAIR_MANIFEST_SCHEMA", "PREDICTION_SCHEMA", "PROTOCOL_SCHEMA",
    "PAIRED_FORMAL_D92_REFERENCE_SCHEMA", "PAIRED_NORMALIZED_SCHEMA", "PAIRED_SCORE_SCHEMA",
    "PAIRED_TRUTH_CATALOG_SCHEMA", "ROW_COUNT", "SCORE_SCHEMA", "TRUTH_CATALOG_SCHEMA",
    "TRUTH_OPEN_EVENT_SCHEMA", "build_d127_s0_truth_open_event", "canonical_sha256",
    "normalize_d127_s0_paired_prediction", "prepare_d127_s0_scoring_inputs", "score_d127_s0",
    "score_d127_s0_paired", "validate_d127_s0_prediction_pairs",
    "write_d127_s0_paired_score_exclusive", "write_d127_s0_truth_open_event_exclusive",
]

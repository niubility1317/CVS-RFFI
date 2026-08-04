"""Independent truth-side scorer for the frozen NEXT-R4 proxy matrix.

The predictor publishes a truth-free 24-row artifact.  This module first
validates the complete four-state Q/H closure, the binding receipts and K1
alias identities; only then does it open the opaque ``truth_by_query_id``
catalog.  Metrics are always computed on the same row, receiver and K keys.
There is no best-row selection, tuning, promotion decision or runtime import.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from . import stage2_next_r4_matrix as matrix


PREDICTION_SCHEMA = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.proxy_prediction.v1"
SCORE_SCHEMA = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.proxy_score.v1"
ROW_SCORE_SCHEMA = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.row_score.v1"

_FORBIDDEN_KEYS = frozenset(
    {
        "truth", "truth_label", "query_truth", "query_label", "query_labels",
        "query_role", "query_roles", "class_quota", "batch_class_count",
        "true_batch_class_count", "true_batch_class_counts", "global_reassignment",
        "hungarian", "optimal_transport",
    }
)
_REG_STATES = {
    "REG0": ("DA0_REG0", "DA1_REG0"),
    "REG1": ("DA0_REG1", "DA1_REG1"),
}
_STATE_TO_REG = {state: registration for registration, states in _REG_STATES.items() for state in states}
_N_A = "N/A"


class NextR4ScoreError(ValueError):
    """Raised when prediction closure, receipts, truth join or metrics drift."""


def _reject_forbidden(value: Any, *, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_KEYS:
                raise NextR4ScoreError(f"{name} contains forbidden field {key}")
            _reject_forbidden(item, name=f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_forbidden(item, name=f"{name}[{index}]")


def _strings(value: Any, *, name: str, unique: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise NextR4ScoreError(f"{name} must be a sequence")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise NextR4ScoreError(f"{name} contains an empty value")
    if unique and len(set(result)) != len(result):
        raise NextR4ScoreError(f"{name} must be unique")
    return result


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NextR4ScoreError("non-finite score output")
        return value
    return value


def _harmonic(old: float, new: float) -> float:
    if old + new <= 0.0:
        return 0.0
    return 2.0 * old * new / (old + new)


def _binding_query_ids(binding: Mapping[str, Any], *, observation: bool = False) -> tuple[str, ...]:
    field = "query_observation_ids_by_class" if observation else "query_ids_by_class"
    classes = _strings(binding.get("registered_classes"), name="binding.registered_classes")
    values = binding.get(field)
    if not isinstance(values, Mapping) or tuple(sorted(values)) != tuple(sorted(classes)):
        raise NextR4ScoreError(f"binding.{field} class registry drift")
    flattened: list[str] = []
    for class_id in classes:
        flattened.extend(_strings(values[class_id], name=f"binding.{field}.{class_id}"))
    if len(flattened) != len(set(flattened)):
        raise NextR4ScoreError(f"binding.{field} contains duplicate IDs")
    return tuple(flattened)


def _validated_binding(row: Mapping[str, Any], planned: Mapping[str, Any]) -> Mapping[str, Any]:
    binding = row.get("binding_receipt")
    if not isinstance(binding, Mapping):
        raise NextR4ScoreError(f"prediction row {planned['row_id']} requires binding_receipt")
    try:
        validated = matrix.validate_next_r4_binding(binding)
    except Exception as error:
        raise NextR4ScoreError(f"prediction row {planned['row_id']} binding receipt drift") from error
    expected_key = "k1_row_id" if int(planned["active_k"]) == 1 else "k5_row_id"
    if binding.get(expected_key) != planned["row_id"]:
        raise NextR4ScoreError(f"prediction row {planned['row_id']} binding row id drift")
    if tuple(binding.get("registered_classes", ())) != tuple(planned["all_registered_classes"]):
        raise NextR4ScoreError(f"prediction row {planned['row_id']} binding class registry drift")
    _binding_query_ids(validated)
    _binding_query_ids(validated, observation=True)
    reuse = row.get("fa_state_reuse_receipt")
    if not isinstance(reuse, Mapping):
        raise NextR4ScoreError(f"prediction row {planned['row_id']} requires fa_state_reuse_receipt")
    try:
        checked_reuse = matrix.validate_fa_state_reuse(
            {"DA1_REG0": reuse.get("source_sha256"), "DA1_REG1": reuse.get("target_sha256")}
        )
    except Exception as error:
        raise NextR4ScoreError(f"prediction row {planned['row_id']} FA reuse receipt drift") from error
    if dict(checked_reuse) != dict(reuse):
        raise NextR4ScoreError(f"prediction row {planned['row_id']} FA reuse receipt is not canonical")
    return MappingProxyType({"binding": validated, "fa_reuse": checked_reuse})


def _state_map(registration: Mapping[str, Any], *, name: str, states_expected: Sequence[str]) -> Mapping[str, Any]:
    states = registration.get("states")
    if states is None:
        states = registration.get("state_predictions")
    if states is None:
        states = {key: value for key, value in registration.items() if key in matrix.STATE_IDS}
    if not isinstance(states, Mapping) or set(states) != set(states_expected):
        raise NextR4ScoreError(f"{name} must contain exactly {', '.join(states_expected)}")
    return MappingProxyType({state: states[state] for state in states_expected})


def _state_parts(
    state: Mapping[str, Any], *, row: Mapping[str, Any], registration_id: str,
    state_id: str, binding: Mapping[str, Any], name: str,
) -> Mapping[str, Any]:
    if not isinstance(state, Mapping):
        raise NextR4ScoreError(f"{name} must be an object")
    if state.get("state_id") != state_id:
        raise NextR4ScoreError(f"{name}.state_id drift")
    if state.get("state_name_zh") != matrix.STATE_NAMES_ZH[state_id]:
        raise NextR4ScoreError(f"{name}.state_name_zh drift")
    planned_classes = tuple(
        row["retained_classes"] if registration_id == "REG0" else row["all_registered_classes"]
    )
    if tuple(state.get("registered_classes", ())) != planned_classes:
        raise NextR4ScoreError(f"{name}.registered_classes drift")
    qids = _strings(state.get("query_physical_ids"), name=f"{name}.query_physical_ids")
    obs = _strings(state.get("query_observation_ids"), name=f"{name}.query_observation_ids")
    expected_qids = _binding_query_ids(binding)
    expected_obs = _binding_query_ids(binding, observation=True)
    if qids != expected_qids or obs != expected_obs or len(qids) != len(obs):
        raise NextR4ScoreError(f"{name} common query physical/observation binding drift")
    arms = state.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"Q", "H"}:
        raise NextR4ScoreError(f"{name} must contain exactly Q/H arms")
    normalized: dict[str, Any] = {}
    for arm_id in ("Q", "H"):
        arm = arms[arm_id]
        if not isinstance(arm, Mapping):
            raise NextR4ScoreError(f"{name}.{arm_id} must be an object")
        predictions = arm.get("predictions", arm.get("values"))
        predictions = _strings(predictions, name=f"{name}.{arm_id}.predictions", unique=False)
        if len(predictions) != len(qids):
            raise NextR4ScoreError(f"{name}.{arm_id} query/prediction length drift")
        if any(value not in planned_classes for value in predictions):
            raise NextR4ScoreError(f"{name}.{arm_id} predicts an unregistered class")
        receipt = arm.get("receipt")
        if not isinstance(receipt, Mapping):
            raise NextR4ScoreError(f"{name}.{arm_id} requires receipt")
        normalized[arm_id] = {"predictions": predictions, "receipt": receipt}
    # K1 H must be an exact object/value alias of Q, never a second prediction.
    if int(row["active_k"]) == 1:
        if normalized["H"]["predictions"] != normalized["Q"]["predictions"]:
            raise NextR4ScoreError(f"{name} K1 H predictions are not an exact Q alias")
        h_receipt = normalized["H"]["receipt"]
        if h_receipt.get("exact_qknn_alias") is not True:
            raise NextR4ScoreError(f"{name} K1 H alias receipt is missing exact_qknn_alias=true")
        if h_receipt.get("alias_target_arm") not in (None, "Q"):
            raise NextR4ScoreError(f"{name} K1 H alias target drift")
    else:
        if normalized["H"]["receipt"].get("exact_qknn_alias") is True:
            raise NextR4ScoreError(f"{name} K5 H cannot carry a K1 alias receipt")
        if normalized["H"]["receipt"].get("unique_prediction") is False:
            raise NextR4ScoreError(f"{name} K5 H receipt is not a unique prediction")
    return MappingProxyType({"query_ids": qids, "observation_ids": obs, "arms": MappingProxyType(normalized)})


def _support_pair_checks(records: Mapping[str, Mapping[str, Any]], *, receiver: str, held_class: str) -> None:
    pair = {
        int(item["row"]["active_k"]): item
        for item in records.values()
        if item["row"]["held_receiver"] == receiver and item["row"]["held_class"] == held_class
    }
    if set(pair) != set(matrix.K_VALUES):
        raise NextR4ScoreError(f"{receiver}/{held_class} K1/K5 row coverage drift")
    b1 = pair[1]["binding"]
    b5 = pair[5]["binding"]
    # The canonical binding carries both row IDs and the exact K1 prefix.  It
    # must be byte-identical on both K rows; no cross-row substitute is allowed.
    if dict(b1) != dict(b5):
        raise NextR4ScoreError(f"{receiver}/{held_class} K1/K5 binding receipt drift")
    for registration_id in matrix.REGISTRATION_IDS:
        q1 = pair[1]["registrations"][registration_id]["query_ids"]
        q5 = pair[5]["registrations"][registration_id]["query_ids"]
        if q1 != q5:
            raise NextR4ScoreError(f"{receiver}/{held_class} K1/K5 query physical IDs drift")


def _validate_prediction(prediction: Mapping[str, Any], plan: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(prediction, Mapping):
        raise NextR4ScoreError("prediction must be a mapping")
    _reject_forbidden(prediction, name="prediction")
    try:
        frozen = matrix.validate_next_r4_proxy24_plan(plan)
    except Exception as error:
        raise NextR4ScoreError("invalid NEXT-R4 plan") from error
    if (
        prediction.get("schema") not in (PREDICTION_SCHEMA, SCORE_SCHEMA)
        or prediction.get("candidate_id") != matrix.CANDIDATE_ID
        or prediction.get("protocol_schema") != matrix.PROTOCOL_SCHEMA
        or prediction.get("matrix_sha256") != frozen["matrix_sha256"]
        or prediction.get("evaluation_semantics", prediction.get("artifact_semantics")) != matrix.PROXY_SEMANTICS
        or prediction.get("formal_new_registration_claim") is not False
        or prediction.get("truth_loaded") is not False
        or prediction.get("rows_complete") is not True
    ):
        raise NextR4ScoreError("NEXT-R4 prediction header drift")
    for field in ("query_rows_used_for_fit", "query_state_updates", "query_selection_count"):
        if prediction.get(field) != 0:
            raise NextR4ScoreError("prediction query-isolation counters drift")
    raw_rows = prediction.get("rows")
    if not isinstance(raw_rows, (tuple, list)) or len(raw_rows) != matrix.ROW_COUNT:
        raise NextR4ScoreError("NEXT-R4 prediction must contain exactly 24 rows")
    records: dict[str, Mapping[str, Any]] = {}
    for index, (raw, planned) in enumerate(zip(raw_rows, frozen["rows"], strict=True)):
        if not isinstance(raw, Mapping):
            raise NextR4ScoreError(f"prediction row[{index}] must be an object")
        if str(raw.get("row_id", "")) != planned["row_id"] or planned["row_id"] in records:
            raise NextR4ScoreError(f"prediction row[{index}] identity drift")
        for field in ("held_receiver", "held_class", "active_k"):
            if raw.get(field) != planned[field]:
                raise NextR4ScoreError(f"prediction row[{index}] plan binding drift")
        if raw.get("evaluation_semantics", raw.get("artifact_semantics")) != matrix.PROXY_SEMANTICS or raw.get("formal_new_registration_claim") is not False:
            raise NextR4ScoreError(f"prediction row[{index}] proxy semantics drift")
        receipt_bundle = _validated_binding(raw, planned)
        registrations = raw.get("registrations")
        if not isinstance(registrations, Mapping) or set(registrations) != set(matrix.REGISTRATION_IDS):
            raise NextR4ScoreError(f"prediction row[{index}] REG0/REG1 closure drift")
        normalized_regs: dict[str, Any] = {}
        for registration_id, expected_states in _REG_STATES.items():
            registration = registrations[registration_id]
            if not isinstance(registration, Mapping):
                raise NextR4ScoreError(f"prediction row[{index}] {registration_id} must be an object")
            expected_classes = tuple(planned["retained_classes"] if registration_id == "REG0" else planned["all_registered_classes"])
            if tuple(registration.get("registered_classes", ())) != expected_classes:
                raise NextR4ScoreError(f"prediction row[{index}] {registration_id} class registry drift")
            states = _state_map(registration, name=f"prediction row[{index}] {registration_id}", states_expected=expected_states)
            parsed_states: dict[str, Any] = {}
            for state_id in expected_states:
                parsed_states[state_id] = _state_parts(
                    states[state_id], row=planned, registration_id=registration_id,
                    state_id=state_id, binding=receipt_bundle["binding"],
                    name=f"prediction row[{index}] {registration_id}.{state_id}",
                )
            if len({parsed_states[s]["query_ids"] for s in expected_states}) != 1 or len({parsed_states[s]["observation_ids"] for s in expected_states}) != 1:
                raise NextR4ScoreError(f"prediction row[{index}] {registration_id} state query binding drift")
            normalized_regs[registration_id] = {
                "registered_classes": expected_classes,
                "query_ids": parsed_states[expected_states[0]]["query_ids"],
                "query_observation_ids": parsed_states[expected_states[0]]["observation_ids"],
                "states": parsed_states,
            }
        if normalized_regs["REG0"]["query_ids"] != normalized_regs["REG1"]["query_ids"]:
            raise NextR4ScoreError(f"prediction row[{index}] REG0/REG1 query physical IDs drift")
        records[planned["row_id"]] = {
            "row": planned,
            "binding": receipt_bundle["binding"],
            "fa_state_reuse_receipt": receipt_bundle["fa_reuse"],
            "registrations": normalized_regs,
        }
    if tuple(records) != tuple(str(item["row_id"]) for item in frozen["rows"]):
        raise NextR4ScoreError("NEXT-R4 row matrix incomplete or reordered")
    for receiver in matrix.HELD_RECEIVERS:
        for held_class in frozen["held_classes"]:
            _support_pair_checks(records, receiver=receiver, held_class=held_class)
    if len(records) != matrix.ROW_COUNT:
        raise NextR4ScoreError("NEXT-R4 row matrix incomplete")
    return tuple(records[str(item["row_id"])] for item in frozen["rows"])


def _metric_row(
    *, row: Mapping[str, Any], registration_id: str, state_id: str, arm_id: str,
    query_ids: Sequence[str], predictions: Sequence[str], truth: Mapping[str, str],
) -> Mapping[str, Any]:
    old_classes = tuple(row["retained_classes"])
    classes = tuple(old_classes if registration_id == "REG0" else row["all_registered_classes"])
    pairs = tuple((str(prediction), str(truth[query_id]), query_id) for query_id, prediction in zip(query_ids, predictions, strict=True))
    if any(query_id not in truth for query_id in query_ids):
        raise NextR4ScoreError("truth catalog lacks a prediction physical_id")
    if any(label not in row["all_registered_classes"] for _, label, _ in pairs):
        raise NextR4ScoreError("truth label is outside frozen class registry")
    if any(prediction not in classes for prediction, _, _ in pairs):
        raise NextR4ScoreError("prediction is outside this registration class registry")
    old_pairs = tuple((prediction, label) for prediction, label, _ in pairs if label in old_classes)
    if not old_pairs or any(not any(label == cls for _, label in old_pairs) for cls in old_classes):
        raise NextR4ScoreError("old-class truth coverage is incomplete")
    per_class: dict[str, dict[str, Any]] = {}
    for cls in row["all_registered_classes"]:
        class_pairs = tuple((prediction, label) for prediction, label, _ in pairs if label == cls)
        correct = sum(int(prediction == label) for prediction, label in class_pairs)
        per_class[cls] = {"correct": correct, "count": len(class_pairs), "accuracy": correct / len(class_pairs) if class_pairs else None}
    old_correct = sum(int(prediction == label) for prediction, label in old_pairs)
    old_ba = old_correct / len(old_pairs)
    old_floor = min(float(per_class[cls]["accuracy"]) for cls in old_classes)
    total_correct = sum(int(prediction == label) for prediction, label, _ in pairs)
    result: dict[str, Any] = {
        "schema": ROW_SCORE_SCHEMA,
        "row_id": row["row_id"], "held_receiver": row["held_receiver"], "held_class": row["held_class"], "active_k": row["active_k"],
        "registration_id": registration_id, "state_id": state_id, "state_name_zh": matrix.STATE_NAMES_ZH[state_id], "arm_id": arm_id,
        "evaluation_semantics": matrix.PROXY_SEMANTICS, "formal_new_registration_claim": False,
        "registered_classes": classes, "retained_classes": old_classes, "all_registered_classes": tuple(row["all_registered_classes"]),
        "old_ba": old_ba, "old_floor": old_floor, "all_floor": None,
        "A_old": old_ba, "A_retained": old_ba, "F_old": old_floor, "F_retained": old_floor, "F_registered": None,
        "seen_new_acc": _N_A, "N_seen_new": _N_A, "H_old_new": _N_A, "forgetting": _N_A,
        "total_correct_count": total_correct, "total_query_count": len(pairs),
        "retained_correct_count": old_correct, "retained_query_count": len(old_pairs),
        "per_class": per_class, "per_class_old_accuracy": {cls: per_class[cls]["accuracy"] for cls in old_classes},
        "query_physical_ids": tuple(query_ids),
        "registration_metric_status": "NA_BEFORE_REGISTRATION" if registration_id == "REG0" else "DEFINED_AFTER_REGISTRATION",
    }
    if registration_id == "REG1":
        if any(per_class[cls]["count"] < 1 for cls in classes):
            raise NextR4ScoreError("REG1 truth coverage is incomplete")
        new_pairs = tuple((prediction, label) for prediction, label, _ in pairs if label == row["held_class"])
        if not new_pairs:
            raise NextR4ScoreError("REG1 held-class truth coverage is incomplete")
        new_correct = sum(int(prediction == label) for prediction, label in new_pairs)
        seen_new = new_correct / len(new_pairs)
        all_floor = min(float(per_class[cls]["accuracy"]) for cls in classes)
        result.update({
            "all_floor": all_floor, "F_registered": all_floor, "seen_new_acc": seen_new, "N_seen_new": seen_new,
            "H_old_new": _harmonic(old_ba, seen_new), "new_correct_count": new_correct, "new_query_count": len(new_pairs),
            "registered_per_class": {cls: per_class[cls] for cls in classes},
        })
    else:
        result.update({"all_floor": _N_A, "new_correct_count": _N_A, "new_query_count": _N_A})
    return MappingProxyType(_plain(result))


def _attach_forgetting(metrics: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Attach REG1 DA-forgetting using the paired DA0_REG1 row/state/arm."""
    by_key = {(item["row_id"], item["arm_id"], item["state_id"]): item for item in metrics}
    output: list[Mapping[str, Any]] = []
    for item in metrics:
        mutable = dict(item)
        if item["registration_id"] == "REG1":
            baseline = by_key.get((item["row_id"], item["arm_id"], "DA0_REG1"))
            if baseline is None:
                raise NextR4ScoreError("REG1 forgetting baseline is incomplete")
            mutable["forgetting"] = float(baseline["old_ba"]) - float(item["old_ba"])
            mutable["forgetting_definition"] = "old_ba_DA0_REG1_minus_old_ba_current"
        output.append(MappingProxyType(_plain(mutable)))
    return output


def _aggregate(metrics: Sequence[Mapping[str, Any]], *, active_k: int, state_id: str, arm_id: str, receiver: str | None = None) -> Mapping[str, Any]:
    registration_id = _STATE_TO_REG[state_id]
    selected = [item for item in metrics if int(item["active_k"]) == active_k and item["state_id"] == state_id and item["arm_id"] == arm_id and (receiver is None or item["held_receiver"] == receiver)]
    expected = matrix.CLASS_COUNT * (matrix.SELECTED_RECEIVER_COUNT if receiver is None else 1)
    if len(selected) != expected:
        raise NextR4ScoreError("aggregate requires complete same-row receiver/class coverage")
    old_correct = sum(int(item["retained_correct_count"]) for item in selected)
    old_query = sum(int(item["retained_query_count"]) for item in selected)
    total_correct = sum(int(item["total_correct_count"]) for item in selected)
    total_query = sum(int(item["total_query_count"]) for item in selected)
    classes = tuple(selected[0]["all_registered_classes"])
    pooled_old = {cls: {"correct": 0, "count": 0} for cls in classes}
    pooled_all = {cls: {"correct": 0, "count": 0} for cls in classes}
    for item in selected:
        for cls in item["retained_classes"]:
            value = item["per_class"][cls]
            pooled_old[cls]["correct"] += int(value["correct"]); pooled_old[cls]["count"] += int(value["count"])
        for cls in item["registered_classes"]:
            value = item["per_class"][cls]
            pooled_all[cls]["correct"] += int(value["correct"]); pooled_all[cls]["count"] += int(value["count"])
    if old_query <= 0 or any(value["count"] <= 0 for cls, value in pooled_old.items() if cls in selected[0]["retained_classes"]):
        raise NextR4ScoreError("pooled old-class coverage is incomplete")
    old_per_class = {cls: {**value, "accuracy": value["correct"] / value["count"]} for cls, value in sorted(pooled_old.items()) if value["count"] > 0}
    result: dict[str, Any] = {
        "active_k": active_k, "state_id": state_id, "state_name_zh": matrix.STATE_NAMES_ZH[state_id], "arm_id": arm_id,
        "registration_id": registration_id, "receiver": receiver or "ALL_RECEIVERS", "row_count": len(selected), "outer_key_count": len(selected),
        "old_ba": old_correct / old_query, "old_floor": min(float(value["accuracy"]) for value in old_per_class.values()),
        "all_floor": _N_A, "A_old": old_correct / old_query, "A_retained": old_correct / old_query,
        "F_old": min(float(value["accuracy"]) for value in old_per_class.values()), "F_retained": min(float(value["accuracy"]) for value in old_per_class.values()), "F_registered": None,
        "seen_new_acc": _N_A, "N_seen_new": _N_A, "H_old_new": _N_A, "forgetting": _N_A,
        "retained_correct_count": old_correct, "retained_query_count": old_query, "total_correct_count": total_correct, "total_query_count": total_query,
        "per_class_old_accuracy": {cls: value["accuracy"] for cls, value in old_per_class.items()}, "retained_per_class": old_per_class, "same_row_complete": True,
        "registration_metric_status": "NA_BEFORE_REGISTRATION" if registration_id == "REG0" else "DEFINED_AFTER_REGISTRATION",
    }
    if registration_id == "REG1":
        all_per_class = {cls: {**value, "accuracy": value["correct"] / value["count"]} for cls, value in sorted(pooled_all.items())}
        if any(value["count"] <= 0 for value in all_per_class.values()):
            raise NextR4ScoreError("pooled REG1 class coverage is incomplete")
        new_correct = sum(int(item.get("new_correct_count") or 0) for item in selected)
        new_query = sum(int(item.get("new_query_count") or 0) for item in selected)
        # The held class differs by row; summing all row-held new counts is the
        # matched new-class aggregate, not a cross-row best selection.
        if new_query <= 0:
            raise NextR4ScoreError("pooled REG1 held-class coverage is incomplete")
        seen_new = new_correct / new_query
        all_floor = min(float(value["accuracy"]) for value in all_per_class.values())
        result.update({"all_floor": all_floor, "F_registered": all_floor, "seen_new_acc": seen_new, "N_seen_new": seen_new, "H_old_new": _harmonic(result["old_ba"], seen_new), "registered_per_class": all_per_class, "new_correct_count": new_correct, "new_query_count": new_query})
    return MappingProxyType(_plain(result))


def _delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> Mapping[str, Any]:
    result: dict[str, Any] = {"left_state": left["state_id"], "right_state": right["state_id"], "left_arm": left["arm_id"], "right_arm": right["arm_id"]}
    for field in ("old_ba", "old_floor", "all_floor", "seen_new_acc", "H_old_new", "forgetting"):
        lv, rv = left.get(field), right.get(field)
        result[f"delta_{field}"] = _N_A if isinstance(lv, str) or isinstance(rv, str) or lv is None or rv is None else float(lv) - float(rv)
    result["delta_total_correct_count"] = int(left["total_correct_count"]) - int(right["total_correct_count"])
    result["delta_total_query_count"] = int(left["total_query_count"]) - int(right["total_query_count"])
    return MappingProxyType(_plain(result))


def _comparisons(aggregates: Mapping[str, Mapping[str, Mapping[str, Any]]], *, active_k: int) -> Mapping[str, Any]:
    q = aggregates[str(active_k)]
    def state(state_id: str, arm: str = "Q") -> Mapping[str, Any]:
        return q[state_id][arm]
    result: dict[str, Any] = {}
    for left, right, key in (("DA1_REG0", "DA0_REG0", "DA1_REG0-DA0_REG0"), ("DA1_REG1", "DA0_REG1", "DA1_REG1-DA0_REG1"), ("DA0_REG1", "DA0_REG0", "DA0_REG1-DA0_REG0"), ("DA1_REG1", "DA1_REG0", "DA1_REG1-DA1_REG0")):
        result[key] = _delta(state(left), state(right))
    result["K5_H-K5_Q"] = {sid: _delta(state(sid, "H"), state(sid, "Q")) for sid in matrix.STATE_IDS} if active_k == 5 else _N_A
    if active_k == 5:
        h_da1 = _delta(state("DA1_REG1", "H"), state("DA1_REG1", "Q"))
        h_da0 = _delta(state("DA0_REG1", "H"), state("DA0_REG1", "Q"))
        did: dict[str, Any] = {}
        for key in h_da1:
            if key.startswith("delta_"):
                lv, rv = h_da1[key], h_da0[key]
                did[key] = _N_A if isinstance(lv, str) or isinstance(rv, str) else float(lv) - float(rv)
        did.update({"formula": "(DA1_REG1_H-DA1_REG1_Q)-(DA0_REG1_H-DA0_REG1_Q)", "left_state": "DA1_REG1/DA0_REG1", "right_state": None})
        result["(DA1_H-DA1_Q)-(DA0_H-DA0_Q)"] = did
    else:
        result["(DA1_H-DA1_Q)-(DA0_H-DA0_Q)"] = _N_A
    return MappingProxyType(_plain(result))


def score_next_r4_proxy24(*, prediction: Mapping[str, Any], plan: Mapping[str, Any], truth_by_query_id: Mapping[str, str]) -> Mapping[str, Any]:
    """Validate complete prediction closure, then score against independent truth."""
    frozen = matrix.validate_next_r4_proxy24_plan(plan)
    records = _validate_prediction(prediction, frozen)
    # Truth is intentionally touched only after every row, state, arm, binding,
    # alias and FA-reuse receipt has passed validation above.
    if not isinstance(truth_by_query_id, Mapping):
        raise NextR4ScoreError("truth_by_query_id must be a mapping")
    truth = {str(key): str(value) for key, value in truth_by_query_id.items()}
    if any(not key or not value for key, value in truth.items()):
        raise NextR4ScoreError("truth catalog contains empty IDs or labels")
    query_ids = {state["query_ids"] for record in records for registration in matrix.REGISTRATION_IDS for state in record["registrations"][registration]["states"].values()}
    all_query_ids = set().union(*query_ids) if query_ids else set()
    if set(truth) != all_query_ids:
        raise NextR4ScoreError("truth catalog/query identity coverage drift")
    if any(label not in tuple(frozen["held_classes"]) for label in truth.values()):
        raise NextR4ScoreError("truth label is outside frozen class registry")
    row_scores: list[Mapping[str, Any]] = []
    flat: list[Mapping[str, Any]] = []
    for record in records:
        row = record["row"]
        row_context: dict[str, Any] = {"row_id": row["row_id"], "held_receiver": row["held_receiver"], "held_class": row["held_class"], "active_k": row["active_k"], "evaluation_semantics": matrix.PROXY_SEMANTICS, "formal_new_registration_claim": False, "fa_state_reuse_receipt": record["fa_state_reuse_receipt"], "registrations": {}}
        for registration_id in matrix.REGISTRATION_IDS:
            reg = record["registrations"][registration_id]
            state_outputs: dict[str, Any] = {}
            for state_id in _REG_STATES[registration_id]:
                state = reg["states"][state_id]
                arms_out: dict[str, Any] = {}
                for arm_id in ("Q", "H"):
                    metric = _metric_row(row=row, registration_id=registration_id, state_id=state_id, arm_id=arm_id, query_ids=state["query_ids"], predictions=state["arms"][arm_id]["predictions"], truth=truth)
                    arms_out[arm_id] = metric; flat.append(metric)
                state_outputs[state_id] = {"registration_id": registration_id, "state_id": state_id, "state_name_zh": matrix.STATE_NAMES_ZH[state_id], "query_physical_ids": state["query_ids"], "query_observation_ids": state["observation_ids"], "arms": arms_out}
            row_context["registrations"][registration_id] = {"registered_classes": reg["registered_classes"], "query_physical_ids": reg["query_ids"], "query_observation_ids": reg["query_observation_ids"], "states": state_outputs}
        row_scores.append(MappingProxyType(_plain(row_context)))
    flat = _attach_forgetting(flat)
    # Replace row arm metric references with the finalized forgetting values.
    by_key = {(item["row_id"], item["state_id"], item["arm_id"]): item for item in flat}
    for row in row_scores:
        for reg in row["registrations"].values():
            for state in reg["states"].values():
                for arm_id in ("Q", "H"):
                    state["arms"][arm_id] = by_key[(state["arms"][arm_id]["row_id"], state["state_id"], arm_id)]
    aggregates: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {}
    receiver_aggregates: dict[str, dict[str, dict[str, dict[str, Mapping[str, Any]]]]] = {}
    for active_k in matrix.K_VALUES:
        aggregates[str(active_k)] = {state_id: {arm_id: _aggregate(flat, active_k=active_k, state_id=state_id, arm_id=arm_id) for arm_id in ("Q", "H")} for state_id in matrix.STATE_IDS}
        receiver_aggregates[str(active_k)] = {}
        for receiver in matrix.HELD_RECEIVERS:
            receiver_aggregates[str(active_k)][receiver] = {state_id: {arm_id: _aggregate(flat, active_k=active_k, state_id=state_id, arm_id=arm_id, receiver=receiver) for arm_id in ("Q", "H")} for state_id in matrix.STATE_IDS}
    comparisons = {str(active_k): _comparisons(aggregates, active_k=active_k) for active_k in matrix.K_VALUES}
    return MappingProxyType(_plain({
        "schema": SCORE_SCHEMA, "candidate_id": matrix.CANDIDATE_ID, "protocol_schema": matrix.PROTOCOL_SCHEMA, "matrix_sha256": frozen["matrix_sha256"], "evaluation_semantics": matrix.PROXY_SEMANTICS, "formal_new_registration_claim": False, "formal_target_claim": False,
        "truth_opened_after_complete_prediction": True, "partial_scoring_used": False, "rows_complete": True, "row_count": matrix.ROW_COUNT,
        "unique_prediction_count": matrix.UNIQUE_PREDICTION_COUNT, "artifact_arm_count": matrix.ARTIFACT_ARM_COUNT, "row_scores": row_scores, "state_scores": flat,
        "aggregates_by_k_and_state_and_arm": aggregates, "aggregates_by_receiver_k_state_and_arm": receiver_aggregates, "causal_comparisons_by_k": comparisons, "comparisons_by_k": comparisons,
        "promotion_eligible": False, "decision": "SOURCE_HELD_PROXY_SCORED_ONLY", "cross_row_best_selection_used": False, "truth_query_count_opened": len(truth), "truth_label_join_only": True,
    }))


score_next_r4 = score_next_r4_proxy24
score_run = score_next_r4_proxy24


__all__ = ["NextR4ScoreError", "PREDICTION_SCHEMA", "ROW_SCORE_SCHEMA", "SCORE_SCHEMA", "score_next_r4", "score_next_r4_proxy24", "score_run"]

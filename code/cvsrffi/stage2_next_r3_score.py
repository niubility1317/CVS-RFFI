"""Truth-side scorer for the frozen NEXT-R3 source-held proxy matrix.

The predictor is expected to publish one immutable, truth-free mapping with
24 rows.  Each row carries REG0/REG1 and the six logical arms for all four
explicit DA/registration states.  This module validates that complete closure
first, then joins an opaque ``truth_by_query_id`` mapping and computes only
same-row/pooled metrics.  It never selects a best row and never emits a
promotion decision or a formal new-registration claim.

For compatibility with small local tools, both a pure mapping API
(:func:`score_next_r3_proxy24`) and a D129-shaped alias
(:func:`score_joint6_screen`) are provided.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from . import stage2_next_r3_matrix as matrix


PREDICTION_SCHEMA = "cvs.stage2.next_r3.rdce_tsl160.proxy_prediction.v1"
SCORE_SCHEMA = "cvs.stage2.next_r3.rdce_tsl160.proxy_score.v1"
ROW_SCORE_SCHEMA = "cvs.stage2.next_r3.rdce_tsl160.row_score.v1"

_FORBIDDEN_KEYS = frozenset(
    {
        "truth",
        "truth_label",
        "query_truth",
        "query_label",
        "query_labels",
        "query_role",
        "query_roles",
        "class_quota",
        "batch_class_count",
        "true_batch_class_count",
        "true_batch_class_counts",
        "global_reassignment",
        "hungarian",
        "optimal_transport",
    }
)
_METRIC_FIELDS = (
    "A_old",
    "A_retained",
    "F_old",
    "F_retained",
    "F_registered",
    "N_seen_new",
    "H_old_new",
    "H_retained_new",
    "total_correct_count",
    "total_query_count",
)


class NextR3ScoreError(ValueError):
    """Raised when prediction closure, truth join, or metrics drift."""


def _reject_forbidden(value: Any, *, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_KEYS:
                raise NextR3ScoreError(f"{name} contains forbidden field {key}")
            _reject_forbidden(item, name=f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_forbidden(item, name=f"{name}[{index}]")


def _strings(value: object, *, name: str, unique: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise NextR3ScoreError(f"{name} must be a sequence")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise NextR3ScoreError(f"{name} contains an empty value")
    if unique and len(set(result)) != len(result):
        raise NextR3ScoreError(f"{name} must be unique")
    return result


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NextR3ScoreError("non-finite score output")
        return value
    return value


def _harmonic(old_accuracy: float, new_accuracy: float) -> float:
    if old_accuracy + new_accuracy <= 0.0:
        return 0.0
    return 2.0 * old_accuracy * new_accuracy / (old_accuracy + new_accuracy)


def _state_map(registration: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    """Normalize the two runner spellings used for a registration bundle."""

    states = registration.get("states")
    if states is None:
        states = registration.get("state_predictions")
    if states is None:
        # A compact row may put states directly under the registration.
        candidate = {
            key: value for key, value in registration.items() if key in matrix.STATE_IDS
        }
        states = candidate or None
    if not isinstance(states, Mapping) or set(states) != set(matrix.STATE_IDS):
        raise NextR3ScoreError(f"{name} must contain all four state IDs")
    return states


def _state_parts(
    state: Mapping[str, Any], *, registration: Mapping[str, Any], name: str
) -> tuple[tuple[str, ...], Mapping[str, tuple[str, ...]]]:
    if not isinstance(state, Mapping):
        raise NextR3ScoreError(f"{name} must be an object")
    query_ids_value = state.get("query_physical_ids")
    if query_ids_value is None:
        query_ids_value = state.get("opaque_query_ids")
    if query_ids_value is None:
        query_ids_value = registration.get("query_physical_ids")
    query_ids = _strings(query_ids_value, name=f"{name}.query_physical_ids")
    arms = state.get("arms")
    if arms is None:
        arms = state.get("predictions")
    if isinstance(arms, Mapping) and set(arms) == {"Q", "F", "L"}:
        # ``NextR3RuntimeResult.four_state`` exposes Q/F/L under each
        # causal state.  Expand that compact view to the frozen six-arm names.
        prefix = "R0" if name.rsplit(".", 1)[-1].startswith("DA0") else "R1"
        arms = {f"{prefix}{head}": value for head, value in arms.items()}
    if not isinstance(arms, Mapping) or set(arms) != set(matrix.ARM_IDS):
        raise NextR3ScoreError(f"{name} must contain all six arm IDs")
    normalized: dict[str, tuple[str, ...]] = {}
    for arm_id in matrix.ARM_IDS:
        value = arms[arm_id]
        if isinstance(value, Mapping):
            value = value.get("predictions", value.get("values"))
        predictions = _strings(value, name=f"{name}.{arm_id}", unique=False)
        if len(predictions) != len(query_ids):
            raise NextR3ScoreError(f"{name}.{arm_id} query/prediction length drift")
        normalized[arm_id] = predictions
    return query_ids, MappingProxyType(normalized)


def _support_identity_checks(
    pair_rows: Mapping[int, Mapping[str, Any]], *, name: str
) -> None:
    """Check optional support/query fields without choosing or changing data."""

    for registration in matrix.REGISTRATION_IDS:
        k1 = pair_rows[1]["registrations"][registration]
        k5 = pair_rows[5]["registrations"][registration]
        s1 = k1.get("support_physical_ids")
        s5 = k5.get("support_physical_ids")
        q1 = k1.get("query_physical_ids")
        q5 = k5.get("query_physical_ids")
        if s1 is not None or s5 is not None:
            ids1 = _strings(s1 or (), name=f"{name}.{registration}.K1.support")
            ids5 = _strings(s5 or (), name=f"{name}.{registration}.K5.support")
            expected1 = len(tuple(k1["registered_classes"]))
            expected5 = len(tuple(k5["registered_classes"])) * 5
            if len(ids1) != expected1 or len(ids5) != expected5:
                raise NextR3ScoreError(f"{name} support count drift")
            # Exact prefix is checked class-wise when labels are available.
            labels1 = tuple(k1.get("support_labels", ()))
            labels5 = tuple(k5.get("support_labels", ()))
            if labels1 and labels5:
                if labels1 != labels5[::5] and labels1 != labels5[: len(labels1)]:
                    # Runners commonly group support by class, in which case
                    # the first item of each class block is the exact prefix.
                    expected_labels = tuple(
                        cls
                        for cls in tuple(k1["registered_classes"])
                    )
                    if labels1 != expected_labels:
                        raise NextR3ScoreError(f"{name} K1 support label prefix drift")
            if set(ids1) - set(ids5):
                raise NextR3ScoreError(f"{name} K1 support must be K5 prefix")
        if q1 is not None or q5 is not None:
            ids1 = _strings(q1 or (), name=f"{name}.{registration}.K1.query")
            ids5 = _strings(q5 or (), name=f"{name}.{registration}.K5.query")
            if ids1 != ids5:
                raise NextR3ScoreError(f"{name} K1/K5 query IDs must be shared")
        for payload, label in ((k1, "K1"), (k5, "K5")):
            support = set(_strings(payload.get("support_physical_ids", ()), name=f"{name}.{registration}.{label}.support")) if payload.get("support_physical_ids") is not None else set()
            query = set(_strings(payload.get("query_physical_ids", ()), name=f"{name}.{registration}.{label}.query")) if payload.get("query_physical_ids") is not None else set()
            if support & query:
                raise NextR3ScoreError(f"{name} support/query physical IDs overlap")


def _validate_prediction(
    prediction: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(prediction, Mapping):
        raise NextR3ScoreError("prediction must be a mapping")
    _reject_forbidden(prediction, name="prediction")
    frozen = matrix.validate_next_r3_proxy24_plan(plan)
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
        raise NextR3ScoreError("NEXT-R3 prediction header drift")
    for field in (
        "query_rows_used_for_fit",
        "query_state_updates",
        "query_selection_count",
    ):
        if prediction.get(field) != 0:
            raise NextR3ScoreError("prediction query-isolation counters drift")
    raw_rows = prediction.get("rows")
    if not isinstance(raw_rows, (tuple, list)) or len(raw_rows) != matrix.ROW_COUNT:
        raise NextR3ScoreError("NEXT-R3 prediction must contain exactly 24 rows")
    plan_rows = tuple(frozen["rows"])
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, (raw, planned) in enumerate(zip(raw_rows, plan_rows, strict=True)):
        if not isinstance(raw, Mapping):
            raise NextR3ScoreError(f"prediction row[{index}] must be an object")
        row_id = str(raw.get("row_id", ""))
        if row_id != planned["row_id"] or row_id in by_id:
            raise NextR3ScoreError(f"prediction row[{index}] identity drift")
        for field in ("held_receiver", "held_class", "active_k"):
            if raw.get(field) != planned[field]:
                raise NextR3ScoreError(f"prediction row[{index}] plan binding drift")
        if (
            raw.get("evaluation_semantics", raw.get("artifact_semantics"))
            != matrix.PROXY_SEMANTICS
            or raw.get("formal_new_registration_claim") is not False
        ):
            raise NextR3ScoreError(f"prediction row[{index}] proxy semantics drift")
        registrations = raw.get("registrations")
        if not isinstance(registrations, Mapping) or set(registrations) != set(matrix.REGISTRATION_IDS):
            raise NextR3ScoreError(f"prediction row[{index}] REG0/REG1 closure drift")
        normalized_regs: dict[str, Any] = {}
        for registration_id in matrix.REGISTRATION_IDS:
            registration = registrations[registration_id]
            if not isinstance(registration, Mapping):
                raise NextR3ScoreError(f"prediction row[{index}] {registration_id} must be an object")
            expected_classes = tuple(
                planned["retained_classes"]
                if registration_id == "REG0"
                else planned["all_registered_classes"]
            )
            if tuple(registration.get("registered_classes", ())) != expected_classes:
                raise NextR3ScoreError(f"prediction row[{index}] {registration_id} class registry drift")
            states = _state_map(registration, name=f"prediction row[{index}] {registration_id}")
            normalized_states: dict[str, Any] = {}
            for state_id in matrix.STATE_IDS:
                state = states[state_id]
                qids, arms = _state_parts(state, registration=registration, name=f"prediction row[{index}] {registration_id}.{state_id}")
                # All six arms for a state share one opaque query stream.
                normalized_states[state_id] = {"query_ids": qids, "arms": arms}
            # The same registration query IDs must be reused across DA states.
            state_queries = {value["query_ids"] for value in normalized_states.values()}
            if len(state_queries) != 1:
                raise NextR3ScoreError(f"prediction row[{index}] {registration_id} query root drift")
            normalized_regs[registration_id] = {
                "registered_classes": expected_classes,
                "query_ids": next(iter(state_queries)),
                "states": normalized_states,
                **{key: registration[key] for key in ("support_physical_ids", "support_labels", "query_physical_ids") if key in registration},
            }
        q0 = set(normalized_regs["REG0"]["query_ids"])
        q1 = set(normalized_regs["REG1"]["query_ids"])
        if not q0.issubset(q1):
            raise NextR3ScoreError(f"prediction row[{index}] REG0 query must be REG1 subset")
        by_id[row_id] = {
            "row": planned,
            "registrations": normalized_regs,
            "row_index": index,
        }
    expected_ids = {str(item["row_id"]) for item in plan_rows}
    if set(by_id) != expected_ids:
        raise NextR3ScoreError("NEXT-R3 row matrix incomplete")
    # K1/K5 nesting is checked per held receiver/class pair after all rows are
    # present, preventing an accidental cross-row prefix substitution.
    for receiver in matrix.HELD_RECEIVERS:
        for held_class in frozen["held_classes"]:
            pair = {
                int(item["active_k"]): by_id[item["row_id"]]
                for item in plan_rows
                if item["held_receiver"] == receiver and item["held_class"] == held_class
            }
            if set(pair) != set(matrix.K_VALUES):
                raise NextR3ScoreError("NEXT-R3 K1/K5 pair coverage drift")
            _support_identity_checks(pair, name=f"{receiver}/{held_class}")
    return tuple(by_id[item["row_id"]] for item in plan_rows)


def _metric_row(
    *,
    row: Mapping[str, Any],
    registration_id: str,
    state_id: str,
    arm_id: str,
    query_ids: Sequence[str],
    predictions: Sequence[str],
    truth: Mapping[str, str],
) -> Mapping[str, Any]:
    classes = tuple(row["retained_classes"] if registration_id == "REG0" else row["all_registered_classes"])
    old_classes = tuple(row["retained_classes"])
    if any(query_id not in truth for query_id in query_ids):
        raise NextR3ScoreError("truth catalog lacks a prediction physical_id")
    pairs = tuple((str(prediction), str(truth[query_id]), query_id) for query_id, prediction in zip(query_ids, predictions, strict=True))
    if any(label not in row["all_registered_classes"] for _, label, _ in pairs):
        raise NextR3ScoreError("truth label is outside frozen class registry")
    if any(prediction not in classes for prediction, _, _ in pairs):
        raise NextR3ScoreError("prediction is outside this registration class registry")
    old_pairs = tuple((prediction, label) for prediction, label, _ in pairs if label in old_classes)
    if not old_pairs or any(not any(label == cls for _, label in old_pairs) for cls in old_classes):
        raise NextR3ScoreError("old-class truth coverage is incomplete")
    per_class: dict[str, dict[str, Any]] = {}
    for cls in row["all_registered_classes"]:
        class_pairs = tuple((prediction, label) for prediction, label, _ in pairs if label == cls)
        correct = sum(int(prediction == label) for prediction, label in class_pairs)
        per_class[cls] = {
            "correct": correct,
            "count": len(class_pairs),
            "accuracy": correct / len(class_pairs) if class_pairs else None,
        }
    old_correct = sum(int(prediction == label) for prediction, label in old_pairs)
    a_old = old_correct / len(old_pairs)
    f_old = min(float(per_class[cls]["accuracy"]) for cls in old_classes)
    total_correct = sum(int(prediction == label) for prediction, label, _ in pairs)
    result: dict[str, Any] = {
        "schema": ROW_SCORE_SCHEMA,
        "row_id": row["row_id"],
        "held_receiver": row["held_receiver"],
        "held_class": row["held_class"],
        "active_k": row["active_k"],
        "registration_id": registration_id,
        "state_id": state_id,
        "arm_id": arm_id,
        "evaluation_semantics": matrix.PROXY_SEMANTICS,
        "formal_new_registration_claim": False,
        "registered_classes": classes,
        "retained_classes": old_classes,
        "all_registered_classes": tuple(row["all_registered_classes"]),
        "A_old": a_old,
        "A_retained": a_old,
        "F_old": f_old,
        "F_retained": f_old,
        "F_registered": None,
        "total_correct_count": total_correct,
        "total_query_count": len(pairs),
        "retained_correct_count": old_correct,
        "retained_query_count": len(old_pairs),
        "per_class": per_class,
        "query_physical_ids": tuple(query_ids),
    }
    if registration_id == "REG1":
        all_pairs = tuple((prediction, label) for prediction, label, _ in pairs)
        if any(not any(label == cls for _, label in all_pairs) for cls in classes):
            raise NextR3ScoreError("REG1 truth coverage is incomplete")
        new_pairs = tuple((prediction, label) for prediction, label, _ in pairs if label == row["held_class"])
        if not new_pairs:
            raise NextR3ScoreError("REG1 held-class truth coverage is incomplete")
        new_correct = sum(int(prediction == label) for prediction, label in new_pairs)
        n_seen_new = new_correct / len(new_pairs)
        f_registered = min(float(per_class[cls]["accuracy"]) for cls in classes)
        result.update(
            {
                "N_seen_new": n_seen_new,
                "H_old_new": _harmonic(a_old, n_seen_new),
                "H_retained_new": _harmonic(a_old, n_seen_new),
                "F_registered": f_registered,
                "new_correct_count": new_correct,
                "new_query_count": len(new_pairs),
                "registration_metric_status": "DEFINED_AFTER_REGISTRATION",
            }
        )
    else:
        # N/H are intentionally None (rendered as N/A by reports), never zero.
        result.update(
            {
                "N_seen_new": None,
                "H_old_new": None,
                "H_retained_new": None,
                "new_correct_count": None,
                "new_query_count": None,
                "registration_metric_status": "NA_BEFORE_REGISTRATION",
            }
        )
    return MappingProxyType(_plain(result))


def _aggregate(rows: Sequence[Mapping[str, Any]], *, active_k: int, state_id: str, arm_id: str) -> Mapping[str, Any]:
    registration_id = matrix.registration_for_state(state_id)
    selected = [
        item
        for item in rows
        if item["active_k"] == active_k
        and item["state_id"] == state_id
        and item["registration_id"] == registration_id
        and item["arm_id"] == arm_id
    ]
    if len(selected) != matrix.SELECTED_RECEIVER_COUNT * matrix.CLASS_COUNT:
        raise NextR3ScoreError("aggregate requires all 12 same-row receiver/class keys")
    retained_correct = sum(int(item["retained_correct_count"]) for item in selected)
    retained_query = sum(int(item["retained_query_count"]) for item in selected)
    total_correct = sum(int(item["total_correct_count"]) for item in selected)
    total_query = sum(int(item["total_query_count"]) for item in selected)
    classes = tuple(sorted({cls for item in selected for cls in item["all_registered_classes"]}))
    pooled_old: dict[str, dict[str, int]] = {cls: {"correct": 0, "count": 0} for cls in classes}
    pooled_all: dict[str, dict[str, int]] = {cls: {"correct": 0, "count": 0} for cls in classes}
    for item in selected:
        for cls in item["retained_classes"]:
            value = item["per_class"][cls]
            pooled_old[cls]["correct"] += int(value["correct"])
            pooled_old[cls]["count"] += int(value["count"])
        for cls in item["registered_classes"]:
            value = item["per_class"][cls]
            pooled_all[cls]["correct"] += int(value["correct"])
            pooled_all[cls]["count"] += int(value["count"])
    if any(value["count"] < 1 for value in pooled_old.values()):
        raise NextR3ScoreError("pooled old-class floor lacks six-class coverage")
    old_per_class = {
        cls: {**value, "accuracy": value["correct"] / value["count"]}
        for cls, value in sorted(pooled_old.items())
    }
    result: dict[str, Any] = {
        "active_k": active_k,
        "state_id": state_id,
        "arm_id": arm_id,
        "row_count": len(selected),
        "outer_key_count": len(selected),
        "A_old": retained_correct / retained_query,
        "A_retained": retained_correct / retained_query,
        "F_old": min(float(value["accuracy"]) for value in old_per_class.values()),
        "F_retained": min(float(value["accuracy"]) for value in old_per_class.values()),
        "F_registered": None,
        "retained_correct_count": retained_correct,
        "retained_query_count": retained_query,
        "total_correct_count": total_correct,
        "total_query_count": total_query,
        "retained_per_class": old_per_class,
        "same_row_complete": True,
    }
    if state_id in matrix.REG1_STATES:
        new_correct = sum(int(item.get("new_correct_count") or 0) for item in selected)
        new_query = sum(int(item.get("new_query_count") or 0) for item in selected)
        if new_query <= 0 or any(value["count"] < 1 for value in pooled_all.values()):
            raise NextR3ScoreError("pooled REG1 new/full-class coverage is incomplete")
        all_per_class = {
            cls: {**value, "accuracy": value["correct"] / value["count"]}
            for cls, value in sorted(pooled_all.items())
        }
        n_seen_new = new_correct / new_query
        result.update(
            {
                "N_seen_new": n_seen_new,
                "H_old_new": _harmonic(result["A_old"], n_seen_new),
                "H_retained_new": _harmonic(result["A_old"], n_seen_new),
                "F_registered": min(float(value["accuracy"]) for value in all_per_class.values()),
                "registered_per_class": all_per_class,
                "new_correct_count": new_correct,
                "new_query_count": new_query,
                "registration_metric_status": "DEFINED_AFTER_REGISTRATION",
            }
        )
    else:
        result.update(
            {
                "N_seen_new": None,
                "H_old_new": None,
                "H_retained_new": None,
                "new_correct_count": None,
                "new_query_count": None,
                "registration_metric_status": "NA_BEFORE_REGISTRATION",
            }
        )
    return MappingProxyType(_plain(result))


def _difference(
    metrics: Mapping[str, Mapping[str, Any]],
    left: str,
    right: str,
    *,
    fields: Sequence[str] = ("A_old", "F_old", "F_registered", "N_seen_new", "H_old_new"),
) -> Mapping[str, Any]:
    result: dict[str, Any] = {"left": left, "right": right}
    for field in fields:
        left_value = metrics[left].get(field)
        right_value = metrics[right].get(field)
        if left_value is None or right_value is None:
            result[f"delta_{field}"] = None
        else:
            result[f"delta_{field}"] = float(left_value) - float(right_value)
    result["delta_total_correct_count"] = int(metrics[left]["total_correct_count"]) - int(metrics[right]["total_correct_count"])
    result["delta_total_query_count"] = int(metrics[left]["total_query_count"]) - int(metrics[right]["total_query_count"])
    return MappingProxyType(_plain(result))


def _with_aliases(value: Mapping[str, Any]) -> Mapping[str, Any]:
    result = dict(value)
    # Both project spellings are kept in the same row; no metric is selected
    # from a different row to populate an alias.
    if "delta_A_old" in result:
        result.setdefault("delta_A_retained", result["delta_A_old"])
    if "delta_F_old" in result:
        result.setdefault("delta_F_retained", result["delta_F_old"])
    return MappingProxyType(_plain(result))


def _build_comparisons(
    aggregates: Mapping[str, Mapping[str, Mapping[str, Any]]], *, active_k: int
) -> Mapping[str, Any]:
    """Compute the six frozen contrasts for both REG0 and REG1.

    The direct keys expose the post-registration (REG1) contrast for compact
    consumers; ``by_registration`` retains both REG0 and REG1 contexts so no
    cross-row or cross-state maximization is possible.
    """

    by_reg: dict[str, dict[str, Mapping[str, Any]]] = {"REG0": {}, "REG1": {}}
    for registration, (pre_state, post_state) in {
        "REG0": ("DA0_REG0", "DA1_REG0"),
        "REG1": ("DA0_REG1", "DA1_REG1"),
    }.items():
        q = aggregates[str(active_k)]
        # A state aggregate has one entry per arm.  Use arm labels as the
        # source of every arithmetic operation, preserving same-K context.
        def m(state: str, arm: str) -> Mapping[str, Any]:
            return q[state][arm]

        da_q = _with_aliases(_difference({"L": m(post_state, "R1Q"), "R": m(pre_state, "R0Q")}, "L", "R"))
        lite_q = _with_aliases(_difference({"L": m(pre_state, "R0L"), "R": m(pre_state, "R0Q")}, "L", "R"))
        r1_l_q = _with_aliases(_difference({"L": m(post_state, "R1L"), "R": m(post_state, "R1Q")}, "L", "R"))
        r0_l_f = _with_aliases(_difference({"L": m(pre_state, "R0L"), "R": m(pre_state, "R0F")}, "L", "R"))
        r1_l_f = _with_aliases(_difference({"L": m(post_state, "R1L"), "R": m(post_state, "R1F")}, "L", "R"))
        q_l_did = {
            key: (None if lite_q.get(key) is None or r1_l_q.get(key) is None else float(r1_l_q[key]) - float(lite_q[key]))
            for key in lite_q
            if key.startswith("delta_")
        }
        f_interaction = {
            key: (None if r0_l_f.get(key) is None or r1_l_f.get(key) is None else float(r1_l_f[key]) - float(r0_l_f[key]))
            for key in r0_l_f
            if key.startswith("delta_")
        }
        q_l_did.update({"left": "(R1L-R1Q)-(R0L-R0Q)", "right": None})
        f_interaction.update({"left": "(R1L-R1F)-(R0L-R0F)", "right": None})
        by_reg[registration] = {
            "R1Q-R0Q": da_q,
            "R0L-R0Q": lite_q,
            "R1L-R1Q": r1_l_q,
            "Q_L_DID": _with_aliases(q_l_did),
            "DID_Q_L": _with_aliases(q_l_did),
            "Q/L_DID": _with_aliases(q_l_did),
            "(R1L-R1Q)-(R0L-R0Q)": _with_aliases(q_l_did),
            "R0L-R0F": r0_l_f,
            "R1L-R1F": r1_l_f,
            "F_INTERACTION": _with_aliases(f_interaction),
            "F_DID": _with_aliases(f_interaction),
            "DID_F": _with_aliases(f_interaction),
            "(R1L-R1F)-(R0L-R0F)": _with_aliases(f_interaction),
        }
    # Canonical direct keys use REG1; explicit nested context is always kept.
    direct = dict(by_reg["REG1"])
    direct["by_registration"] = by_reg
    direct["pre_registration"] = by_reg["REG0"]
    direct["post_registration"] = by_reg["REG1"]
    return MappingProxyType(_plain(direct))


def score_next_r3_proxy24(
    *,
    prediction: Mapping[str, Any],
    plan: Mapping[str, Any],
    truth_by_query_id: Mapping[str, str],
) -> Mapping[str, Any]:
    """Validate complete truth-free predictions, then score the 24-row proxy.

    ``truth_by_query_id`` is intentionally a separate argument.  The function
    does not inspect it until the prediction matrix and all state/arm fields
    have passed validation.
    """

    frozen = matrix.validate_next_r3_proxy24_plan(plan)
    records = _validate_prediction(prediction, frozen)
    if not isinstance(truth_by_query_id, Mapping):
        raise NextR3ScoreError("truth_by_query_id must be a mapping")
    truth = {str(key): str(value) for key, value in truth_by_query_id.items()}
    if any(not key or not value for key, value in truth.items()):
        raise NextR3ScoreError("truth catalog contains empty IDs or labels")
    all_query_ids = {
        query_id
        for record in records
        for registration in matrix.REGISTRATION_IDS
        for query_id in record["registrations"][registration]["query_ids"]
    }
    if set(truth) != all_query_ids:
        raise NextR3ScoreError("truth catalog/query identity coverage drift")
    if any(label not in tuple(frozen["held_classes"]) for label in truth.values()):
        raise NextR3ScoreError("truth label is outside frozen class registry")
    row_scores: list[Mapping[str, Any]] = []
    flat_scores: list[Mapping[str, Any]] = []
    for record in records:
        row = record["row"]
        context: dict[str, Any] = {
            "row_id": row["row_id"],
            "held_receiver": row["held_receiver"],
            "held_class": row["held_class"],
            "active_k": row["active_k"],
            "evaluation_semantics": matrix.PROXY_SEMANTICS,
            "formal_new_registration_claim": False,
            "registrations": {},
        }
        for registration_id in matrix.REGISTRATION_IDS:
            reg = record["registrations"][registration_id]
            state_output: dict[str, Any] = {}
            for state_id in matrix.STATE_IDS:
                state = reg["states"][state_id]
                arm_output: dict[str, Any] = {}
                for arm_id in matrix.ARM_IDS:
                    metric = _metric_row(
                        row=row,
                        registration_id=registration_id,
                        state_id=state_id,
                        arm_id=arm_id,
                        query_ids=state["query_ids"],
                        predictions=state["arms"][arm_id],
                        truth=truth,
                    )
                    arm_output[arm_id] = metric
                    flat_scores.append(metric)
                state_output[state_id] = {
                    "registration_id": registration_id,
                    "query_physical_ids": state["query_ids"],
                    "arms": arm_output,
                }
            context["registrations"][registration_id] = {
                "registered_classes": reg["registered_classes"],
                "query_physical_ids": reg["query_ids"],
                "states": state_output,
            }
        row_scores.append(MappingProxyType(_plain(context)))
    aggregates: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {}
    for active_k in matrix.K_VALUES:
        aggregates[str(active_k)] = {}
        for state_id in matrix.STATE_IDS:
            aggregates[str(active_k)][state_id] = {}
            for arm_id in matrix.ARM_IDS:
                aggregates[str(active_k)][state_id][arm_id] = _aggregate(
                    flat_scores, active_k=active_k, state_id=state_id, arm_id=arm_id
                )
    comparisons = {
        str(active_k): _build_comparisons(aggregates, active_k=active_k)
        for active_k in matrix.K_VALUES
    }
    # Keep explicit N/A states and all row context in the output.  No value is
    # called "best" and no cross-row selection/ranking is performed.
    return MappingProxyType(
        _plain(
            {
                "schema": SCORE_SCHEMA,
                "candidate_id": matrix.CANDIDATE_ID,
                "protocol_schema": matrix.PROTOCOL_SCHEMA,
                "matrix_sha256": frozen["matrix_sha256"],
                "evaluation_semantics": matrix.PROXY_SEMANTICS,
                "formal_new_registration_claim": False,
                "formal_target_claim": False,
                "truth_opened_after_complete_prediction": True,
                "partial_scoring_used": False,
                "rows_complete": True,
                "row_count": matrix.ROW_COUNT,
                "state_prediction_count": matrix.STATE_PREDICTION_COUNT,
                "arm_prediction_count": matrix.ARM_PREDICTION_COUNT,
                "row_scores": row_scores,
                "state_scores": flat_scores,
                "aggregates_by_k_and_state_and_arm": aggregates,
                "causal_comparisons_by_k": comparisons,
                "comparisons_by_k": comparisons,
                "promotion_eligible": False,
                "decision": "SOURCE_HELD_PROXY_SCORED_ONLY",
                "cross_row_best_selection_used": False,
                "truth_query_count_opened": len(truth),
                "truth_label_join_only": True,
            }
        )
    )


def score_joint6_screen(
    *, prediction: Mapping[str, Any], plan: Mapping[str, Any], truth_by_query_id: Mapping[str, str]
) -> Mapping[str, Any]:
    """D129-shaped alias used by compact local scorer scripts."""

    return score_next_r3_proxy24(
        prediction=prediction, plan=plan, truth_by_query_id=truth_by_query_id
    )


score_next_r3 = score_next_r3_proxy24
score_run = score_next_r3_proxy24


__all__ = [
    "NextR3ScoreError",
    "PREDICTION_SCHEMA",
    "ROW_SCORE_SCHEMA",
    "SCORE_SCHEMA",
    "score_joint6_screen",
    "score_next_r3",
    "score_next_r3_proxy24",
    "score_run",
]

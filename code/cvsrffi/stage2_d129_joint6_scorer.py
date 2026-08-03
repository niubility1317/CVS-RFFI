"""Independent truth-side scorer for the D129 seen-class LOCO proxy screen.

The sealed Phase1 checkpoint has already seen all six TX classes.  Therefore
this scorer deliberately emits retained/held-proxy metrics and cannot emit or
claim formal Stage2-C new-registration metrics.
"""

from __future__ import annotations

import math
import hashlib
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from . import stage2_d129_joint6_matrix as matrix


PREDICTION_SCHEMA = "cvs.stage2.d129.joint6.proxy_prediction.v2"
SCORE_SCHEMA = "cvs.stage2.d129.joint6.proxy_score.v2"
_FORBIDDEN_PREDICTION_KEYS = frozenset(
    {
        "truth",
        "truth_label",
        "query_label",
        "query_labels",
        "query_role",
        "query_roles",
        "class_quota",
        "batch_class_count",
    }
)
_COMPARISONS = (
    ("DA_EFFECT", "R1Q", "R0Q"),
    ("LITE_BASE", "R0L", "R0F"),
    ("JOINT_REPLACE", "R1L", "R1F"),
)


class D129Joint6ScorerError(ValueError):
    """Raised when prediction closure or truth-side scoring drifts."""


def _reject_forbidden(value: Any, *, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_PREDICTION_KEYS:
                raise D129Joint6ScorerError(
                    f"{name} contains forbidden prediction field {key}"
                )
            _reject_forbidden(item, name=f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_forbidden(item, name=f"{name}[{index}]")


def _strings(value: Any, *, name: str, unique: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise D129Joint6ScorerError(f"{name} must be a sequence")
    result = tuple(str(item) for item in value)
    if not result or any(not item for item in result):
        raise D129Joint6ScorerError(f"{name} contains empty values")
    if unique and len(set(result)) != len(result):
        raise D129Joint6ScorerError(f"{name} must be unique")
    return result


def _sha256(value: Any, *, name: str) -> str:
    text = str(value)
    if (
        len(text) != 64
        or text.lower() != text
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise D129Joint6ScorerError(f"{name} must be a lowercase SHA256")
    return text


def _harmonic(retained_accuracy: float, held_accuracy: float) -> float:
    return (
        0.0
        if retained_accuracy + held_accuracy <= 0.0
        else 2.0 * retained_accuracy * held_accuracy
        / (retained_accuracy + held_accuracy)
    )


def _metrics(pairs: Sequence[tuple[str, str, bool]], classes: Sequence[str]) -> dict[str, Any]:
    retained_pairs = [
        (prediction, truth) for prediction, truth, is_held in pairs if not is_held
    ]
    held_pairs = [
        (prediction, truth) for prediction, truth, is_held in pairs if is_held
    ]
    if not retained_pairs or not held_pairs:
        raise D129Joint6ScorerError(
            "each aggregate requires retained and held-proxy query coverage"
        )
    retained_correct = sum(
        prediction == truth for prediction, truth in retained_pairs
    )
    held_correct = sum(prediction == truth for prediction, truth in held_pairs)
    retained_accuracy = retained_correct / len(retained_pairs)
    held_accuracy = held_correct / len(held_pairs)
    per_class: dict[str, dict[str, int]] = {
        class_id: {"correct": 0, "count": 0} for class_id in classes
    }
    for prediction, truth in retained_pairs:
        per_class[truth]["count"] += 1
        per_class[truth]["correct"] += int(prediction == truth)
    covered = [value for value in per_class.values() if value["count"] > 0]
    if len(covered) != len(classes):
        raise D129Joint6ScorerError("pooled retained-class floor lacks class coverage")
    floor = min(value["correct"] / value["count"] for value in covered)
    result = {
        "A_retained": retained_accuracy,
        "A_held_proxy": held_accuracy,
        "H_retained_held_proxy": _harmonic(retained_accuracy, held_accuracy),
        "F_retained": floor,
        "retained_correct_count": retained_correct,
        "retained_query_count": len(retained_pairs),
        "held_proxy_correct_count": held_correct,
        "held_proxy_query_count": len(held_pairs),
        "total_correct_count": retained_correct + held_correct,
        "total_query_count": len(retained_pairs) + len(held_pairs),
        "retained_per_class": MappingProxyType(
            {
                key: MappingProxyType(value)
                for key, value in sorted(per_class.items())
            }
        ),
    }
    if not all(
        math.isfinite(float(result[field]))
        for field in (
            "A_retained",
            "A_held_proxy",
            "H_retained_held_proxy",
            "F_retained",
        )
    ):
        raise D129Joint6ScorerError("non-finite Joint6 metric")
    return result


def _validate_prediction(
    prediction: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    _reject_forbidden(prediction, name="prediction")
    if (
        prediction.get("schema") != PREDICTION_SCHEMA
        or prediction.get("truth_loaded") is not False
        or prediction.get("matrix_sha256") != plan.get("matrix_sha256")
        or prediction.get("protocol_schema") != "p2_min_v1"
        or not str(prediction.get("capsule_id", ""))
        or not str(prediction.get("split_id", ""))
        or tuple(prediction.get("candidate_ids", ())) != matrix.CANDIDATE_IDS
        or tuple(prediction.get("arm_ids", ())) != matrix.ARM_IDS
        or prediction.get("rows_complete") is not True
        or any(
            int(prediction.get(field, -1)) != 0
            for field in (
                "query_rows_used_for_fit",
                "query_state_updates",
                "query_selection_count",
            )
        )
    ):
        raise D129Joint6ScorerError("Joint6 prediction header drift")
    for field in (
        "checkpoint_sha256",
        "archive_sha256",
        "method_lock_sha256",
        "query_catalog_root_sha256",
    ):
        _sha256(prediction.get(field), name=f"prediction.{field}")
    raw_rows = prediction.get("rows")
    expected_count = matrix.ROW_COUNT_PER_CANDIDATE * len(matrix.CANDIDATE_IDS)
    if not isinstance(raw_rows, list) or len(raw_rows) != expected_count:
        raise D129Joint6ScorerError("Joint6 prediction row coverage drift")
    plan_by_id = {row["row_id"]: row for row in plan["rows"]}
    validated: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise D129Joint6ScorerError(f"prediction row[{index}] must be a mapping")
        candidate_id = str(raw.get("candidate_id", ""))
        row_id = str(raw.get("row_id", ""))
        key = (candidate_id, row_id)
        if candidate_id not in matrix.CANDIDATE_IDS or row_id not in plan_by_id or key in validated:
            raise D129Joint6ScorerError(f"prediction row[{index}] identity drift")
        planned = plan_by_id[row_id]
        if (
            raw.get("active_k") != planned["active_k"]
            or raw.get("held_receiver") != planned["held_receiver"]
            or raw.get("held_class") != planned["held_class"]
            or raw.get("registered_classes") != planned["registered_classes"]
        ):
            raise D129Joint6ScorerError(f"prediction row[{index}] plan binding drift")
        query_ids = _strings(raw.get("opaque_query_ids"), name=f"row[{index}].opaque_query_ids")
        if (
            raw.get("evaluation_semantics")
            != "phase1_seen_class_loco_directional_proxy"
            or raw.get("formal_new_registration_claim") is not False
        ):
            raise D129Joint6ScorerError(
                f"prediction row[{index}] proxy semantics drift"
            )
        for field in (
            "binding_sha256",
            "phase1_seal_sha256",
            "query_physical_root_sha256",
            "checkpoint_sha256",
            "asset_sha256",
            "common_r0_sha256",
        ):
            _sha256(raw.get(field), name=f"row[{index}].{field}")
        observed_query_root = hashlib.sha256(
            "\n".join(query_ids).encode("utf-8")
        ).hexdigest()
        if (
            raw["query_physical_root_sha256"] != observed_query_root
            or raw["checkpoint_sha256"] != prediction["checkpoint_sha256"]
        ):
            raise D129Joint6ScorerError(
                f"prediction row[{index}] query/checkpoint binding drift"
            )
        arms = raw.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(matrix.ARM_IDS):
            raise D129Joint6ScorerError(f"prediction row[{index}] arm closure drift")
        arm_predictions: dict[str, tuple[str, ...]] = {}
        registry = tuple(planned["registered_classes"])
        for arm_id in matrix.ARM_IDS:
            values = _strings(
                arms[arm_id], name=f"row[{index}].{arm_id}", unique=False
            )
            if len(values) != len(query_ids) or any(value not in registry for value in values):
                raise D129Joint6ScorerError(f"prediction row[{index}] {arm_id} drift")
            arm_predictions[arm_id] = values
        if planned["active_k"] == 1 and not (
            arm_predictions["R0F"] == arm_predictions["R0Q"]
            and arm_predictions["R0L"] == arm_predictions["R0Q"]
            and arm_predictions["R1F"] == arm_predictions["R1Q"]
            and arm_predictions["R1L"] == arm_predictions["R1Q"]
        ):
            raise D129Joint6ScorerError("K1 F/L predictions must alias Q")
        validated[key] = {
            "plan": planned,
            "query_ids": query_ids,
            "arms": arm_predictions,
            "binding_sha256": raw["binding_sha256"],
            "phase1_seal_sha256": raw["phase1_seal_sha256"],
            "query_physical_root_sha256": raw["query_physical_root_sha256"],
            "checkpoint_sha256": raw["checkpoint_sha256"],
            "common_r0_sha256": raw["common_r0_sha256"],
        }
    expected = {
        (candidate_id, row["row_id"])
        for candidate_id in matrix.CANDIDATE_IDS
        for row in plan["rows"]
    }
    if set(validated) != expected:
        raise D129Joint6ScorerError("Joint6 candidate/row matrix incomplete")
    for row in plan["rows"]:
        left = validated[(matrix.CANDIDATE_IDS[0], row["row_id"])]
        right = validated[(matrix.CANDIDATE_IDS[1], row["row_id"])]
        if (
            left["query_ids"] != right["query_ids"]
            or any(
            left["arms"][arm_id] != right["arms"][arm_id]
            for arm_id in matrix.COMMON_ARM_IDS
            )
            or any(
                left[field] != right[field]
                for field in (
                    "binding_sha256",
                    "phase1_seal_sha256",
                    "query_physical_root_sha256",
                    "checkpoint_sha256",
                    "common_r0_sha256",
                )
            )
        ):
            raise D129Joint6ScorerError("common R0 arms differ across candidates")
    return validated


def score_joint6_screen(
    *,
    prediction: Mapping[str, Any],
    plan: Mapping[str, Any],
    truth_by_query_id: Mapping[str, str],
) -> Mapping[str, Any]:
    """Open truth after complete prediction closure and score frozen contrasts."""

    rows = _validate_prediction(prediction, plan)
    truth = {str(key): str(value) for key, value in truth_by_query_id.items()}
    all_query_ids = {qid for row in rows.values() for qid in row["query_ids"]}
    if set(truth) != all_query_ids or any(not key or not value for key, value in truth.items()):
        raise D129Joint6ScorerError("truth catalog/query identity coverage drift")
    candidate_scores: dict[str, Any] = {}
    for candidate_id in matrix.CANDIDATE_IDS:
        metrics_by_k: dict[int, dict[str, Any]] = {}
        for active_k in matrix.K_VALUES:
            pairs_by_arm: dict[str, list[tuple[str, str, bool]]] = {
                arm_id: [] for arm_id in matrix.ARM_IDS
            }
            for (row_candidate, _row_id), row in rows.items():
                if row_candidate != candidate_id or row["plan"]["active_k"] != active_k:
                    continue
                held_class = row["plan"]["held_class"]
                registry = set(row["plan"]["registered_classes"])
                for qid in row["query_ids"]:
                    if truth[qid] not in registry:
                        raise D129Joint6ScorerError("truth label outside registered classes")
                for arm_id in matrix.ARM_IDS:
                    pairs_by_arm[arm_id].extend(
                        (prediction_value, truth[qid], truth[qid] == held_class)
                        for qid, prediction_value in zip(
                            row["query_ids"], row["arms"][arm_id], strict=True
                        )
                    )
            metrics_by_k[active_k] = {
                arm_id: _metrics(pairs, plan["class_ids"])
                for arm_id, pairs in pairs_by_arm.items()
            }
        k5 = metrics_by_k[5]
        decisions: dict[str, Any] = {}
        for name, treatment, control in _COMPARISONS:
            deltas = {
                field: k5[treatment][field] - k5[control][field]
                for field in (
                    "H_retained_held_proxy",
                    "A_retained",
                    "A_held_proxy",
                    "F_retained",
                )
            }
            total_delta = (
                k5[treatment]["total_correct_count"]
                - k5[control]["total_correct_count"]
            )
            decisions[name] = MappingProxyType(
                {
                    "treatment": treatment,
                    "control": control,
                    **{f"delta_{key}": value for key, value in deltas.items()},
                    "delta_total_correct_count": total_delta,
                    "pass": (
                        deltas["H_retained_held_proxy"] > 0.0
                        and total_delta > 0
                        and deltas["A_retained"] >= 0.0
                        and deltas["A_held_proxy"] >= 0.0
                        and deltas["F_retained"] >= 0.0
                    ),
                }
            )
        candidate_scores[candidate_id] = MappingProxyType(
            {
                "metrics_by_k": MappingProxyType(
                    {
                        key: MappingProxyType(value)
                        for key, value in metrics_by_k.items()
                    }
                ),
                "k1_head_gain_claim_allowed": False,
                "k1_da_contrast": MappingProxyType(
                    {
                        field: metrics_by_k[1]["R1Q"][field]
                        - metrics_by_k[1]["R0Q"][field]
                        for field in (
                            "H_retained_held_proxy",
                            "A_retained",
                            "A_held_proxy",
                            "F_retained",
                        )
                    }
                ),
                "k5_primary_comparisons": MappingProxyType(decisions),
                "candidate_pass": all(value["pass"] for value in decisions.values()),
            }
        )
    return MappingProxyType(
        {
            "schema": SCORE_SCHEMA,
            "matrix_sha256": plan["matrix_sha256"],
            "candidate_scores": MappingProxyType(candidate_scores),
            "truth_opened_after_complete_prediction": True,
            "partial_performance_selection_used": False,
            "evaluation_semantics": "phase1_seen_class_loco_directional_proxy",
            "formal_new_registration_claim": False,
            "formal_H_old_new_emitted": False,
            "promotion_rule": "all_three_K5_primary_comparisons_pass",
        }
    )


__all__ = [
    "D129Joint6ScorerError",
    "PREDICTION_SCHEMA",
    "SCORE_SCHEMA",
    "score_joint6_screen",
]

"""Truth-separated D104 63-row/four-arm held execution primitives."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .rxid_metabias4_bundle import RXIDMetaBias4Bundle
from .rxid_metabias4_held_execution import (
    _d103_int8_receipt,
    canonical_sha256,
    frozen_qknn,
    leave_day_stability,
)
from .stage2_d104_rxid_angq import (
    ARMS,
    D104RXIDANGQError,
    audit_d104_four_arm_int8,
    build_d104_prediction_artifact,
    fit_d104_four_arm_state,
)


SCHEMA = "cvs.d104_r1.rxid_angq.held_execution.v1"


class D104HeldExecutionError(ValueError):
    """Raised when D104 held prediction or scoring closure drifts."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(member) for key, member in value.items()}
    if isinstance(value, tuple):
        return [_plain(member) for member in value]
    if isinstance(value, list):
        return [_plain(member) for member in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def predict_d104_matched_row(
    *,
    held_receiver: str,
    held_class: str | None,
    k_shot: int,
    support_pre_relu: np.ndarray,
    support_zdom: np.ndarray,
    support_labels: Sequence[str],
    support_physical_ids: Sequence[str],
    query_pre_relu: np.ndarray,
    query_physical_ids: Sequence[str],
    registered_classes: Sequence[str],
    d103_outer_bundle: RXIDMetaBias4Bundle,
    d103_day_bundles: Sequence[RXIDMetaBias4Bundle],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Produce one immutable truth-free D104 four-arm row."""

    classes = tuple(str(value) for value in registered_classes)
    labels = tuple(str(value) for value in support_labels)
    support_ids = tuple(str(value) for value in support_physical_ids)
    query_ids = tuple(str(value) for value in query_physical_ids)
    if (
        k_shot not in (1, 5, 10)
        or len(classes) != 6
        or len(set(classes)) != 6
        or len(support_ids) != len(labels)
        or len(support_ids) != len(support_pre_relu)
        or len(support_ids) != len(support_zdom)
        or len(query_ids) != len(query_pre_relu)
        or len(set(support_ids)) != len(support_ids)
        or len(set(query_ids)) != len(query_ids)
        or set(support_ids).intersection(query_ids)
    ):
        raise D104HeldExecutionError("D104 matched row metadata closure drift")
    qknn = frozen_qknn(k_shot)
    stability = None
    k1_receipt = None
    k1_view_audit = None
    if k_shot == 1:
        stability = leave_day_stability(
            d103_outer_bundle,
            d103_day_bundles,
            support_zdom,
            labels,
            classes,
        )
        k1_receipt, k1_view_audit = _d103_int8_receipt(
            d103_outer_bundle,
            support_pre_relu,
            support_zdom,
            labels,
            classes,
            qknn,
            float(stability["direction_cosine_median"]),
        )
    elif len(d103_day_bundles) != 4:
        raise D104HeldExecutionError("D104 row requires four matched day bundles")
    support_receipt = canonical_sha256(
        {
            "schema": SCHEMA,
            "held_receiver": held_receiver,
            "held_class": held_class,
            "K": k_shot,
            "support_physical_ids": list(support_ids),
            "query_physical_ids_not_persisted_in_fit_receipt": True,
        }
    )
    state = fit_d104_four_arm_state(
        d103_outer_bundle,
        np.asarray(support_pre_relu, dtype=np.float32),
        np.asarray(support_zdom, dtype=np.float32),
        labels,
        classes,
        qknn_config=qknn,
        stage="S_C",
        support_receipt_sha256=support_receipt,
        k1_identifiability_receipt=k1_receipt,
    )
    prediction = _plain(
        build_d104_prediction_artifact(
            state,
            np.asarray(query_pre_relu, dtype=np.float32),
            query_ids,
        )
    )
    int8 = _plain(
        audit_d104_four_arm_int8(
            state,
            np.asarray(support_pre_relu, dtype=np.float32),
            labels,
            np.asarray(query_pre_relu, dtype=np.float32),
        )
    )
    artifact: dict[str, Any] = {
        "schema": SCHEMA + ".prediction",
        "held_receiver": str(held_receiver),
        "held_class": None if held_class is None else str(held_class),
        "K": k_shot,
        "registered_classes": list(classes),
        "query_physical_ids": list(query_ids),
        "arm_predictions": prediction["arm_predictions"],
        "arm_prediction_receipts": {
            arm: canonical_sha256(
                {
                    "held_receiver": str(held_receiver),
                    "held_class": None if held_class is None else str(held_class),
                    "K": k_shot,
                    "arm": arm,
                    "query_physical_ids": list(query_ids),
                    "predictions": prediction["arm_predictions"][arm],
                }
            )
            for arm in ARMS
        },
        "state_receipt_sha256": state.state_receipt_sha256,
        "method_lock_sha256": state.method_lock["method_lock_sha256"],
        "d103_bundle_content_root_sha256": (
            d103_outer_bundle.content_root_sha256
        ),
        "d103_fit_audit": _plain(state.d103_state.fit_audit),
        "resource_receipts": _plain(state.resource_receipts),
        "int8_audit": int8,
        "k1_view_audit": _plain(k1_view_audit),
        "k1_stability": _plain(stability),
        "all_four_arms_present": tuple(prediction["arm_predictions"]) == ARMS,
        "query_truth_present": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "all_registered_classes_compete": True,
        "per_query_independent": True,
        "target25_authorized": False,
    }
    if (
        set(artifact["arm_predictions"]) != set(ARMS)
        or set(artifact["arm_prediction_receipts"]) != set(ARMS)
        or any(
            len(artifact["arm_predictions"][arm]) != len(query_ids)
            for arm in ARMS
        )
    ):
        raise D104HeldExecutionError("D104 row prediction closure failed")
    artifact["prediction_receipt_sha256"] = canonical_sha256(artifact)
    return artifact, stability


def _metrics(
    truth: np.ndarray,
    predicted: Sequence[str],
    classes: Sequence[str],
) -> dict[str, Any]:
    prediction = np.asarray(predicted).astype(str)
    if len(prediction) != len(truth):
        raise D104HeldExecutionError("D104 scoring row length drift")
    counts = []
    correct = []
    for class_id in classes:
        local = truth == class_id
        denominator = int(np.sum(local))
        if denominator < 1:
            raise D104HeldExecutionError("D104 truth lacks a registered class")
        counts.append(denominator)
        correct.append(int(np.sum(prediction[local] == class_id)))
    per_class = [
        numerator / denominator
        for numerator, denominator in zip(correct, counts, strict=True)
    ]
    return {
        "balanced_accuracy": float(np.mean(per_class)),
        "per_class_floor": float(np.min(per_class)),
        "joint_score": float((np.mean(per_class) + np.min(per_class)) / 2.0),
        "correct_count": int(np.sum(prediction == truth)),
        "query_count": len(truth),
        "per_class_correct": correct,
        "per_class_count": counts,
    }


def score_d104_prediction_artifact(
    prediction: Mapping[str, Any],
    truth_labels: Sequence[str],
) -> dict[str, Any]:
    """Score one already-sealed four-arm row on the independent truth side."""

    truth = np.asarray(tuple(str(value) for value in truth_labels), dtype=str)
    classes = tuple(str(value) for value in prediction.get("registered_classes", ()))
    arms = prediction.get("arm_predictions")
    if (
        prediction.get("schema") != SCHEMA + ".prediction"
        or prediction.get("query_truth_present") is not False
        or prediction.get("target25_authorized") is not False
        or len(truth) != len(prediction.get("query_physical_ids", ()))
        or len(classes) != 6
        or not isinstance(arms, Mapping)
        or tuple(arms) != ARMS
        or canonical_sha256(
            {
                key: value
                for key, value in prediction.items()
                if key != "prediction_receipt_sha256"
            }
        )
        != prediction.get("prediction_receipt_sha256")
    ):
        raise D104HeldExecutionError("D104 truth-side prediction closure drift")
    metrics = {
        arm: _metrics(truth, arms[arm], classes)
        for arm in ARMS
    }
    effects: dict[str, dict[str, float | int]] = {}
    pairs = {
        "H0_HEAD_at_base": ("M_HEAD", "M0"),
        "H1_HEAD_at_DA": ("M_JOINT", "M_DA"),
        "D0_DA_at_legacy": ("M_DA", "M0"),
        "D1_DA_at_ANGQ": ("M_JOINT", "M_HEAD"),
    }
    for name, (left, right) in pairs.items():
        effects[name] = {
            metric: (
                int(metrics[left][metric]) - int(metrics[right][metric])
                if metric == "correct_count"
                else float(metrics[left][metric]) - float(metrics[right][metric])
            )
            for metric in (
                "balanced_accuracy",
                "per_class_floor",
                "joint_score",
                "correct_count",
            )
        }
    result = {
        "held_receiver": prediction["held_receiver"],
        "held_class": prediction["held_class"],
        "K": int(prediction["K"]),
        "registered_classes": list(classes),
        "arm_metrics": metrics,
        "simple_effects": effects,
        "head_main_effect": {
            metric: (
                float(effects["H0_HEAD_at_base"][metric])
                + float(effects["H1_HEAD_at_DA"][metric])
            )
            / 2.0
            for metric in (
                "balanced_accuracy",
                "per_class_floor",
                "joint_score",
                "correct_count",
            )
        },
        "da_main_effect": {
            metric: (
                float(effects["D0_DA_at_legacy"][metric])
                + float(effects["D1_DA_at_ANGQ"][metric])
            )
            / 2.0
            for metric in (
                "balanced_accuracy",
                "per_class_floor",
                "joint_score",
                "correct_count",
            )
        },
        "interaction": {
            metric: (
                float(metrics["M_JOINT"][metric])
                - float(metrics["M_HEAD"][metric])
                - float(metrics["M_DA"][metric])
                + float(metrics["M0"][metric])
            )
            for metric in (
                "balanced_accuracy",
                "per_class_floor",
                "joint_score",
                "correct_count",
            )
        },
        "prediction_receipt_sha256": prediction["prediction_receipt_sha256"],
        "truth_row_count": len(truth),
        "target_access": False,
    }
    audit = prediction.get("d103_fit_audit")
    stability = prediction.get("k1_stability")
    view = prediction.get("k1_view_audit")
    if int(prediction["K"]) == 1:
        if (
            not isinstance(audit, Mapping)
            or not isinstance(stability, Mapping)
            or not isinstance(view, Mapping)
        ):
            raise D104HeldExecutionError("D104 K1 evidence missing")
        result["k1_evidence"] = {
            "active": audit.get("status") == "ACTIVE",
            "information_rank": int(audit["data_information_rank"]),
            "minimum_singular_value": float(
                audit["data_minimum_singular_value"]
            ),
            "condition_number": float(audit["system_condition_number"]),
            "prior_fraction": float(audit["prior_fraction"]),
            "coefficient_norm": float(audit["coefficient_norm"]),
            "view_top1_agreement": float(view["top1_agreement"]),
            "view_margin_flip_count": int(view["margin_sign_flip_count"]),
            "direction_cosine_median": float(
                stability["direction_cosine_median"]
            ),
        }
    else:
        result["k1_evidence"] = None
    result["score_receipt_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "D104HeldExecutionError",
    "SCHEMA",
    "predict_d104_matched_row",
    "score_d104_prediction_artifact",
]

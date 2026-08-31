"""Independent truth-last scoring and paired deltas for WISER-RF probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_PROBE_MEMBERS = {
    "P1_SOURCE_HEAD": "p1_predictions",
    "P2_SOURCE_PROTOTYPE": "p2_predictions",
    "P3_OLD_D92": "p3_predictions",
}
_PROBE_LOGIT_MEMBERS = {
    "P1_SOURCE_HEAD": "p1_logits",
    "P2_SOURCE_PROTOTYPE": "p2_logits",
    "P3_OLD_D92": "p3_logits",
}
_CLASS_REGISTRY = tuple(str(index) for index in range(6))
_DETAILED_BINDING_FIELDS = (
    "outer_key",
    "capsule_id",
    "split_id",
    "receiver",
    "scenario",
    "arm",
)


def _class_counts(truth: np.ndarray) -> dict[str, int]:
    return {str(class_id): int(np.sum(truth == class_id)) for class_id in range(6)}


def _validate_indices(values: np.ndarray, *, label: str, size: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != size:
        raise ValueError(f"WISER {label} geometry drift")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"WISER {label} indices must be integers")
    result = array.astype(np.int64, copy=False)
    if bool(((result < 0) | (result >= 6)).any()):
        raise ValueError(f"WISER {label} index is outside the six-class registry")
    return result


def _probe_metrics(
    prediction: np.ndarray, logits: np.ndarray, truth: np.ndarray
) -> dict[str, Any]:
    """Return strict detailed metrics without deriving confidence from labels."""

    truth_values = _validate_indices(truth, label="truth", size=len(truth))
    prediction_values = _validate_indices(
        prediction, label="prediction", size=len(truth_values)
    )
    logit_values = np.asarray(logits, dtype=np.float64)
    if logit_values.shape != (len(truth_values), 6):
        raise ValueError("WISER probe logit geometry drift")
    if not bool(np.isfinite(logit_values).all()):
        raise ValueError("WISER probe logits must be finite")
    if not np.array_equal(prediction_values, logit_values.argmax(axis=1).astype(np.int64)):
        raise ValueError("WISER probe prediction/argmax drift")
    per_class = {
        str(class_id): float(
            np.mean(prediction_values[truth_values == class_id] == class_id)
        )
        for class_id in range(6)
    }
    values = np.asarray(list(per_class.values()), dtype=np.float64)
    maximum = logit_values.max(axis=1)
    logsumexp = maximum + np.log(np.exp(logit_values - maximum[:, None]).sum(axis=1))
    nll = float(np.mean(logsumexp - logit_values[np.arange(len(truth_values)), truth_values]))
    return {
        "balanced_accuracy": float(values.mean()),
        "floor": float(values.min()),
        "accuracy": float(np.mean(prediction_values == truth_values)),
        "nll": nll,
        "per_class_accuracy": per_class,
    }


def _legacy_probe_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    prediction_values = _validate_indices(prediction, label="prediction", size=len(truth))
    truth_values = _validate_indices(truth, label="truth", size=len(truth))
    classes = np.unique(truth_values)
    per_class = {
        str(int(class_id)): float(
            np.mean(prediction_values[truth_values == class_id] == class_id)
        )
        for class_id in classes
    }
    values = np.asarray(list(per_class.values()), dtype=np.float64)
    return {
        "balanced_accuracy": float(values.mean()),
        "floor": float(values.min()),
        "accuracy": float(np.mean(prediction_values == truth_values)),
        "per_class_accuracy": per_class,
    }


def _geometry(features: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if features.ndim != 2 or features.shape[0] != truth.shape[0]:
        raise ValueError("query feature/truth geometry drift")
    classes = np.unique(truth)
    centers = np.stack([features[truth == class_id].mean(axis=0) for class_id in classes])
    center_by_row = np.stack(
        [centers[int(np.where(classes == class_id)[0][0])] for class_id in truth]
    )
    within = float(np.mean(np.sum((features - center_by_row) ** 2, axis=1)))
    global_center = centers.mean(axis=0)
    between = float(np.mean(np.sum((centers - global_center) ** 2, axis=1)))
    return {
        "within_trace": within,
        "between_trace": between,
        "between_within_ratio": between / max(within, 1.0e-12),
    }


def _require_detailed_receipt_fields(receipt: Mapping[str, Any]) -> None:
    missing = [
        field
        for field in _DETAILED_BINDING_FIELDS
        if field not in receipt or receipt[field] is None or str(receipt[field]) == ""
    ]
    if missing:
        raise ValueError(f"WISER detailed receipt binding missing: {', '.join(missing)}")


def score_wiser_predictions(
    predictions_path: str | Path,
    receipt_path: str | Path,
    truth_path: str | Path,
) -> Mapping[str, Any]:
    """Join opaque tokens only after a complete truth-blind prediction receipt."""

    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8-sig"))
    if (
        receipt.get("status") != "PREDICTIONS_COMPLETE"
        or receipt.get("query_truth_opened") is not False
        or receipt.get("query_role_opened") is not False
        or receipt.get("support_state_frozen_before_query") is not True
    ):
        raise ValueError("WISER prediction is not truth-last eligible")
    truth_payload = json.loads(Path(truth_path).read_text(encoding="utf-8-sig"))
    if str(truth_payload.get("receiver")) != str(receipt.get("receiver")):
        raise ValueError("WISER truth receiver binding drift")
    rows = truth_payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("WISER truth rows are missing")
    truth_lookup: dict[str, int] = {}
    for row in rows:
        token = str(row["query_token"])
        if token in truth_lookup:
            raise ValueError("WISER duplicate truth token drift")
        try:
            class_id = int(row["true_class_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("WISER truth index is invalid") from error
        if class_id < 0 or class_id >= 6:
            raise ValueError("WISER truth index is outside the six-class registry")
        truth_lookup[token] = class_id

    with np.load(Path(predictions_path), allow_pickle=False) as arrays:
        required = {"query_tokens", "query_z_id", *_PROBE_MEMBERS.values()}
        if not required.issubset(arrays.files):
            raise ValueError("WISER prediction members are incomplete")
        tokens = np.asarray(arrays["query_tokens"]).astype(str)
        expected_tokens = receipt.get("expected_query_tokens")
        if not isinstance(expected_tokens, list):
            raise ValueError("WISER frozen query-token registry is missing")
        expected = tuple(str(token) for token in expected_tokens)
        if (
            tokens.ndim != 1
            or len(tokens) != int(receipt.get("query_rows", -1))
            or tuple(tokens.tolist()) != expected
            or len(set(expected)) != len(expected)
            or len(set(tokens.tolist())) != len(tokens)
            or not set(expected).issubset(truth_lookup)
        ):
            raise ValueError("WISER truth token join drift")
        truth = np.asarray([truth_lookup[token] for token in tokens], dtype=np.int64)
        if set(truth.tolist()) != set(range(6)):
            raise ValueError("WISER representation probe needs complete six-old-class coverage")
        features = np.asarray(arrays["query_z_id"], dtype=np.float64)
        if features.shape[0] != len(tokens) or not np.isfinite(features).all():
            raise ValueError("WISER query feature closure drift")

        detailed_members = set(_PROBE_LOGIT_MEMBERS.values())
        present_logits = detailed_members.intersection(arrays.files)
        if present_logits and present_logits != detailed_members:
            raise ValueError("WISER detailed probe logits are incomplete")
        detailed = present_logits == detailed_members
        if detailed:
            _require_detailed_receipt_fields(receipt)

        probes: dict[str, dict[str, Any]] = {}
        pairing_predictions: dict[str, list[int]] = {}
        for probe_name, member in _PROBE_MEMBERS.items():
            prediction = _validate_indices(
                arrays[member], label=f"{probe_name} prediction", size=len(truth)
            )
            if detailed:
                probes[probe_name] = _probe_metrics(
                    prediction, arrays[_PROBE_LOGIT_MEMBERS[probe_name]], truth
                )
                pairing_predictions[probe_name] = prediction.tolist()
            else:
                probes[probe_name] = _legacy_probe_metrics(prediction, truth)
        geometry = _geometry(features, truth)

    result: dict[str, Any] = {
        "schema": (
            "cvs.phase2.wiser_rf.truth_last_score.v2"
            if detailed
            else "cvs.phase2.wiser_rf.truth_last_score.v1"
        ),
        "status": "ANALYZED",
        "arm": str(receipt["arm"]),
        "receiver": str(receipt["receiver"]),
        "scenario": str(receipt["scenario"]),
        "query_rows": int(len(tokens)),
        "old_query_rows": int(len(tokens)),
        "probes": probes,
        "geometry": geometry,
        "truth_join_after_prediction_only": True,
        "truth_handle_alignment_verified": True,
    }
    if detailed:
        result.update(
            {
                "outer_key": str(receipt["outer_key"]),
                "capsule_id": str(receipt["capsule_id"]),
                "split_id": str(receipt["split_id"]),
                "query_tokens": tokens.tolist(),
                "class_registry": list(_CLASS_REGISTRY),
                "per_class_query_rows": _class_counts(truth),
                "pairing_payload": {
                    "query_tokens": tokens.tolist(),
                    "truth": truth.tolist(),
                    "true_class_indices": truth.tolist(),
                    "predictions": pairing_predictions,
                },
            }
        )
    return result


def _comparison_payload(
    row: Mapping[str, Any], *, side: str
) -> tuple[list[str], np.ndarray, Mapping[str, Any]]:
    if row.get("schema") != "cvs.phase2.wiser_rf.truth_last_score.v2":
        raise ValueError(f"WISER {side} row lacks detailed pairing evidence")
    payload = row.get("pairing_payload")
    if not isinstance(payload, Mapping):
        raise ValueError(f"WISER {side} row lacks detailed pairing evidence")
    tokens = payload.get("query_tokens")
    truth = payload.get("truth")
    true_class_indices = payload.get("true_class_indices")
    predictions = payload.get("predictions")
    if (
        not isinstance(tokens, list)
        or not isinstance(truth, list)
        or not isinstance(true_class_indices, list)
        or not isinstance(predictions, Mapping)
    ):
        raise ValueError(f"WISER {side} detailed pairing payload is malformed")
    if row.get("query_tokens") != tokens or len(tokens) != int(row.get("query_rows", -1)):
        raise ValueError("WISER pairing binding drift")
    if len(set(tokens)) != len(tokens):
        raise ValueError("WISER pairing binding drift")
    truth_values = _validate_indices(np.asarray(truth), label="paired truth", size=len(tokens))
    class_index_values = _validate_indices(
        np.asarray(true_class_indices), label="paired true class", size=len(tokens)
    )
    if not np.array_equal(truth_values, class_index_values):
        raise ValueError("WISER pairing binding drift")
    if set(truth_values.tolist()) != set(range(6)):
        raise ValueError("WISER pairing binding drift")
    if row.get("per_class_query_rows") != _class_counts(truth_values):
        raise ValueError("WISER pairing binding drift")
    return tokens, truth_values, predictions


def _validated_probe_metrics(row: Mapping[str, Any], probe: str) -> Mapping[str, Any]:
    metrics = row.get("probes", {}).get(probe) if isinstance(row.get("probes"), Mapping) else None
    if not isinstance(metrics, Mapping):
        raise ValueError("WISER probe registry drift")
    required = {"accuracy", "balanced_accuracy", "floor", "nll", "per_class_accuracy"}
    if not required.issubset(metrics):
        raise ValueError("WISER detailed pairing metric evidence is incomplete")
    if not all(np.isfinite(float(metrics[key])) for key in required - {"per_class_accuracy"}):
        raise ValueError("WISER detailed pairing metric evidence is nonfinite")
    per_class = metrics["per_class_accuracy"]
    if not isinstance(per_class, Mapping) or tuple(per_class.keys()) != _CLASS_REGISTRY:
        raise ValueError("WISER class registry drift")
    if not all(np.isfinite(float(per_class[key])) for key in _CLASS_REGISTRY):
        raise ValueError("WISER detailed pairing metric evidence is nonfinite")
    return metrics


def compare_wiser_score_rows(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Compare two detailed truth-last rows on their exact frozen query pairing."""

    if str(control.get("arm")) not in {"B0", "N0"}:
        raise ValueError("WISER control arm must be B0 or N0")
    if str(control.get("arm")) == str(candidate.get("arm")):
        raise ValueError("WISER paired arms must be distinct")
    control_tokens, control_truth, control_predictions = _comparison_payload(control, side="control")
    candidate_tokens, candidate_truth, candidate_predictions = _comparison_payload(
        candidate, side="candidate"
    )
    for field in (
        "outer_key",
        "capsule_id",
        "split_id",
        "receiver",
        "scenario",
        "query_rows",
        "query_tokens",
        "class_registry",
        "per_class_query_rows",
    ):
        if control.get(field) != candidate.get(field):
            raise ValueError("WISER pairing binding drift")
    if control_tokens != candidate_tokens or not np.array_equal(control_truth, candidate_truth):
        raise ValueError("WISER pairing binding drift")
    control_probe_names = tuple(control.get("probes", {}).keys()) if isinstance(control.get("probes"), Mapping) else ()
    candidate_probe_names = tuple(candidate.get("probes", {}).keys()) if isinstance(candidate.get("probes"), Mapping) else ()
    if control_probe_names != tuple(_PROBE_MEMBERS) or candidate_probe_names != tuple(_PROBE_MEMBERS):
        raise ValueError("WISER probe registry drift")
    if set(control_predictions) != set(_PROBE_MEMBERS) or set(candidate_predictions) != set(_PROBE_MEMBERS):
        raise ValueError("WISER probe registry drift")

    comparisons: dict[str, dict[str, Any]] = {}
    for probe in _PROBE_MEMBERS:
        control_metrics = _validated_probe_metrics(control, probe)
        candidate_metrics = _validated_probe_metrics(candidate, probe)
        control_prediction = _validate_indices(
            np.asarray(control_predictions[probe]),
            label=f"{probe} control prediction",
            size=len(control_truth),
        )
        candidate_prediction = _validate_indices(
            np.asarray(candidate_predictions[probe]),
            label=f"{probe} candidate prediction",
            size=len(control_truth),
        )
        control_correct = control_prediction == control_truth
        candidate_correct = candidate_prediction == control_truth
        help_mask = ~control_correct & candidate_correct
        harm_mask = control_correct & ~candidate_correct
        changed = control_prediction != candidate_prediction
        neutral_flip = ~control_correct & ~candidate_correct & changed
        per_class_delta = {
            class_id: 100.0
            * (
                float(candidate_metrics["per_class_accuracy"][class_id])
                - float(control_metrics["per_class_accuracy"][class_id])
            )
            for class_id in _CLASS_REGISTRY
        }
        comparisons[probe] = {
            "control_metrics": dict(control_metrics),
            "candidate_metrics": dict(candidate_metrics),
            "accuracy_delta_pp": 100.0
            * (float(candidate_metrics["accuracy"]) - float(control_metrics["accuracy"])),
            "balanced_accuracy_delta_pp": 100.0
            * (
                float(candidate_metrics["balanced_accuracy"])
                - float(control_metrics["balanced_accuracy"])
            ),
            "floor_delta_pp": 100.0
            * (float(candidate_metrics["floor"]) - float(control_metrics["floor"])),
            "per_class_accuracy_delta_pp": per_class_delta,
            "nll_delta": float(candidate_metrics["nll"]) - float(control_metrics["nll"]),
            "help_count": int(help_mask.sum()),
            "harm_count": int(harm_mask.sum()),
            "unchanged_count": int(len(control_truth) - help_mask.sum() - harm_mask.sum()),
            "net_help_minus_harm": int(help_mask.sum() - harm_mask.sum()),
            "prediction_flip_count": int(changed.sum()),
            "neutral_flip_count": int(neutral_flip.sum()),
            "same_prediction_count": int((~changed).sum()),
        }
    p3 = comparisons["P3_OLD_D92"]
    return {
        "schema": "cvs.phase2.wiser_rf.paired_query_delta.v1",
        "comparison_state": "DA1_REG0-DA0_REG0",
        "control_arm": str(control["arm"]),
        "candidate_arm": str(candidate["arm"]),
        "outer_key": str(control["outer_key"]),
        "capsule_id": str(control["capsule_id"]),
        "split_id": str(control["split_id"]),
        "receiver": str(control["receiver"]),
        "scenario": str(control["scenario"]),
        "query_rows": int(control["query_rows"]),
        "probes": comparisons,
        "p3": p3,
        **p3,
    }


__all__ = ["compare_wiser_score_rows", "score_wiser_predictions"]

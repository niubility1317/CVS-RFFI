"""Independent truth-last scoring for immutable MARC-OT predictions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_PROBES = {
    "P1_SOURCE_HEAD": ("p1_predictions", "p1_logits"),
    "P2_SUPPORT_PROTOTYPE": ("p2_predictions", "p2_logits"),
    "P3_OLD_D92": ("p3_predictions", "p3_logits"),
}
_RECEIPT_SCHEMA = "cvs.phase2.marc_ot.prediction_receipt.v1"
_BINDING_FIELDS = ("outer_key", "capsule_id", "split_id", "receiver", "scenario")
_RESOURCE_FIELDS = (
    "training_seconds",
    "inference_seconds",
    "peak_rss_bytes",
    "peak_cuda_bytes",
    "trainable_parameter_count",
)
_RESOURCE_STATUS_FIELDS = ("peak_rss_status", "peak_cuda_status")


@dataclass(frozen=True)
class MARCOTPredictionPreflight:
    """Truth-blind, fully validated prediction material for one immutable unit."""

    receipt: Mapping[str, Any]
    registry: tuple[str, ...]
    tokens: tuple[str, ...]
    arrays: Mapping[str, np.ndarray]
    resources: Mapping[str, Any]


def _read_object(path: Path, *, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {context}: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _validate_receipt(value: Mapping[str, Any]) -> None:
    if value.get("schema") != _RECEIPT_SCHEMA:
        raise ValueError("MARC-OT prediction receipt schema drift")
    if value.get("status") != "PREDICTIONS_COMPLETE":
        raise ValueError("MARC-OT prediction is not complete")
    if (
        value.get("query_truth_opened") is not False
        or value.get("query_role_opened") is not False
    ):
        raise ValueError("MARC-OT prediction is not truth-last eligible")
    if value.get("support_state_frozen_before_query") is not True:
        raise ValueError("MARC-OT support state was not frozen before query")
    if value.get("protocol_schema") != "p2_min_v1" or value.get("phase2_data_status") != "VALIDATED_ONCE":
        raise ValueError("MARC-OT prediction protocol binding drift")
    for field in (*_BINDING_FIELDS, "arm"):
        if not isinstance(value.get(field), str) or not str(value[field]):
            raise ValueError(f"MARC-OT prediction receipt binding missing: {field}")


def _truth_lookup(
    truth_payload: Mapping[str, Any], receipt: Mapping[str, Any], registry_size: int
) -> dict[str, tuple[int, str]]:
    for field in ("receiver", "capsule_id", "split_id"):
        if str(truth_payload.get(field)) != str(receipt.get(field)):
            raise ValueError("MARC-OT truth/prediction binding drift")
    rows = truth_payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("MARC-OT truth rows are missing")
    result: dict[str, tuple[int, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("MARC-OT truth row is malformed")
        token = str(row.get("query_token", ""))
        if not token or token in result:
            raise ValueError("MARC-OT truth token registry is invalid")
        value = row.get("true_class_index")
        role = row.get("evaluation_role", "target_old")
        if role not in {"target_old", "target_new"}:
            raise ValueError("MARC-OT truth evaluation role is invalid")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("MARC-OT truth class index is outside the full registry")
        if (role == "target_old" and value >= registry_size) or (
            role == "target_new" and value < registry_size
        ):
            raise ValueError("MARC-OT truth class index/evaluation role drift")
        result[token] = (value, role)
    return result


def _validate_predictions(values: Any, *, rows: int, classes: int, label: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (rows,) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"MARC-OT {label} prediction geometry drift")
    result = raw.astype(np.int64, copy=False)
    if bool(((result < 0) | (result >= classes)).any()):
        raise ValueError(f"MARC-OT {label} prediction is outside the full registry")
    return result


def _nll_contributions(logits: np.ndarray, truth: np.ndarray) -> np.ndarray:
    maximum = logits.max(axis=1)
    logsumexp = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
    return logsumexp - logits[np.arange(len(truth)), truth]


def _metrics(
    prediction: np.ndarray,
    logits: np.ndarray,
    truth: np.ndarray,
    registry: tuple[str, ...],
) -> tuple[dict[str, Any], np.ndarray]:
    if logits.shape != (len(truth), len(registry)) or not bool(np.isfinite(logits).all()):
        raise ValueError("MARC-OT probe logit geometry or finiteness drift")
    if not np.array_equal(prediction, logits.argmax(axis=1).astype(np.int64)):
        raise ValueError("MARC-OT probe prediction/argmax receipt drift")
    if set(truth.tolist()) != set(range(len(registry))):
        raise ValueError("MARC-OT score needs complete registered-class coverage")
    per_class_accuracy: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    per_class_rows: dict[str, int] = {}
    for class_id, class_name in enumerate(registry):
        truth_mask = truth == class_id
        predicted_mask = prediction == class_id
        true_positive = int(np.sum(truth_mask & predicted_mask))
        false_positive = int(np.sum(~truth_mask & predicted_mask))
        false_negative = int(np.sum(truth_mask & ~predicted_mask))
        per_class_rows[class_name] = int(truth_mask.sum())
        per_class_accuracy[class_name] = float(true_positive / int(truth_mask.sum()))
        denominator = 2 * true_positive + false_positive + false_negative
        per_class_f1[class_name] = 0.0 if denominator == 0 else float(2 * true_positive / denominator)
    accuracy_values = np.asarray(tuple(per_class_accuracy.values()), dtype=np.float64)
    f1_values = np.asarray(tuple(per_class_f1.values()), dtype=np.float64)
    contributions = _nll_contributions(logits, truth)
    return (
        {
            "accuracy": float(np.mean(prediction == truth)),
            "balanced_accuracy": float(accuracy_values.mean()),
            "floor": float(accuracy_values.min()),
            "macro_f1": float(f1_values.mean()),
            "nll": float(contributions.mean()),
            "per_class_accuracy": per_class_accuracy,
            "per_class_f1": per_class_f1,
            "per_class_query_rows": per_class_rows,
        },
        contributions,
    )


def _resources(value: Any) -> Mapping[str, Any]:
    required = frozenset((*_RESOURCE_FIELDS, *_RESOURCE_STATUS_FIELDS))
    if not isinstance(value, Mapping) or frozenset(value) != required:
        raise ValueError("MARC-OT resource receipt is incomplete")
    result: dict[str, Any] = {}
    for field in _RESOURCE_FIELDS:
        current = value[field]
        is_peak = field in {"peak_rss_bytes", "peak_cuda_bytes"}
        status = value[field.replace("_bytes", "_status")] if is_peak else None
        if is_peak and current == "N/A":
            if status not in {"UNAVAILABLE", "NOT_APPLICABLE"}:
                raise ValueError("MARC-OT unavailable resource lacks an explicit status")
            result[field] = "N/A"
            continue
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            raise ValueError("MARC-OT resource receipt is malformed")
        numeric = float(current)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("MARC-OT resource receipt is malformed")
        if is_peak and (status != "MEASURED" or numeric <= 0.0):
            raise ValueError("MARC-OT measured resource peak must be positive")
        result[field] = int(current) if field.endswith(("_bytes", "_count")) else numeric
    for field in _RESOURCE_STATUS_FIELDS:
        if field in value:
            status = value[field]
            if status not in {"MEASURED", "UNAVAILABLE", "NOT_APPLICABLE"}:
                raise ValueError("MARC-OT resource status is malformed")
            result[field] = status
    return result


def preflight_marc_ot_prediction(
    prediction_root: str | Path,
) -> MARCOTPredictionPreflight:
    """Validate receipt and every prediction member without opening truth."""

    root = Path(prediction_root)
    receipt = _read_object(root / "prediction_receipt.json", context="prediction receipt")
    _validate_receipt(receipt)
    registry_raw = receipt.get("class_registry")
    if not isinstance(registry_raw, list):
        raise ValueError("MARC-OT full class registry is missing")
    registry = tuple(str(value) for value in registry_raw)
    if not registry or len(set(registry)) != len(registry):
        raise ValueError("MARC-OT full class registry is invalid")
    try:
        arrays = np.load(root / "predictions.npz", allow_pickle=False)
    except OSError as error:
        raise ValueError("MARC-OT prediction artifact is missing") from error
    with arrays:
        required = {"query_tokens"}
        for prediction_member, logit_member in _PROBES.values():
            required.update((prediction_member, logit_member))
        allowed = required | {"query_z_id"}
        if not required.issubset(arrays.files) or not set(arrays.files).issubset(allowed):
            raise ValueError("MARC-OT prediction members are incomplete")
        tokens = np.asarray(arrays["query_tokens"]).astype(str)
        expected_raw = receipt.get("expected_query_tokens")
        if not isinstance(expected_raw, list):
            raise ValueError("MARC-OT receipt query-token registry is missing")
        expected = tuple(str(value) for value in expected_raw)
        if (
            tokens.ndim != 1
            or len(tokens) != int(receipt.get("query_rows", -1))
            or tuple(tokens.tolist()) != expected
            or len(set(expected)) != len(expected)
        ):
            raise ValueError("MARC-OT prediction/receipt token binding drift")
        material = {"query_tokens": tokens.copy()}
        if "query_z_id" in arrays.files:
            features = np.asarray(arrays["query_z_id"], dtype=np.float64)
            if features.ndim != 2 or features.shape[0] != len(tokens) or not np.isfinite(features).all():
                raise ValueError("MARC-OT query feature geometry or finiteness drift")
            material["query_z_id"] = features.copy()
        for probe, (prediction_member, logit_member) in _PROBES.items():
            prediction = _validate_predictions(
                arrays[prediction_member], rows=len(tokens), classes=len(registry), label=probe
            )
            logits = np.asarray(arrays[logit_member], dtype=np.float64)
            if logits.shape != (len(tokens), len(registry)) or not bool(np.isfinite(logits).all()):
                raise ValueError("MARC-OT probe logit geometry or finiteness drift")
            if not np.array_equal(prediction, logits.argmax(axis=1).astype(np.int64)):
                raise ValueError("MARC-OT probe prediction/argmax receipt drift")
            material[prediction_member] = prediction.copy()
            material[logit_member] = logits.copy()
    resources = _resources(receipt.get("resources"))
    return MARCOTPredictionPreflight(
        receipt=dict(receipt),
        registry=registry,
        tokens=expected,
        arrays=material,
        resources=dict(resources),
    )


def load_marc_ot_truth(truth_sidecar: str | Path) -> Mapping[str, Any]:
    """Open one truth sidecar after the caller has completed prediction preflight."""

    return _read_object(Path(truth_sidecar), context="truth sidecar")


def score_preflighted_marc_ot_prediction(
    preflight: MARCOTPredictionPreflight,
    truth_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Join one already-preflighted prediction unit to an already-opened truth object."""

    if not isinstance(preflight, MARCOTPredictionPreflight):
        raise ValueError("MARC-OT prediction preflight is required")
    receipt = preflight.receipt
    registry = preflight.registry
    expected = preflight.tokens
    arrays = preflight.arrays
    lookup = _truth_lookup(truth_payload, receipt, len(registry))
    if not set(expected).issubset(lookup):
        raise ValueError("MARC-OT prediction/receipt token binding drift")
    roles = np.asarray([lookup[token][1] for token in expected])
    old_mask = roles == "target_old"
    truth = np.asarray([lookup[token][0] for token in expected], dtype=np.int64)[old_mask]
    scored_tokens = np.asarray(expected)[old_mask]
    probes: dict[str, Mapping[str, Any]] = {}
    pairing_predictions: dict[str, list[int]] = {}
    pairing_nll: dict[str, list[float]] = {}
    for probe, (prediction_member, logit_member) in _PROBES.items():
        prediction = arrays[prediction_member][old_mask]
        logits = arrays[logit_member][old_mask]
        metrics, contributions = _metrics(prediction, logits, truth, registry)
        probes[probe] = metrics
        pairing_predictions[probe] = prediction.tolist()
        pairing_nll[probe] = contributions.tolist()
    arm = str(receipt["arm"])
    return {
        "schema": "cvs.phase2.marc_ot.truth_last_score.v1",
        "status": "ANALYZED",
        "arm": arm,
        "adaptation_state": "DA0" if arm == "R0" else "DA1",
        "registration_state": "REG0",
        **{field: str(receipt[field]) for field in _BINDING_FIELDS},
        "query_rows": len(scored_tokens),
        "total_query_rows": len(expected),
        "old_query_rows": len(scored_tokens),
        "ignored_non_old_query_rows": int(len(expected) - len(scored_tokens)),
        "scored_evaluation_role": "target_old",
        "query_tokens": scored_tokens.tolist(),
        "class_registry": list(registry),
        "probes": probes,
        "resources": dict(preflight.resources),
        "pairing_payload": {
            "query_tokens": scored_tokens.tolist(),
            "truth": truth.tolist(),
            "predictions": pairing_predictions,
            "nll_contributions": pairing_nll,
        },
        "truth_join_after_prediction_only": True,
        "truth_handle_alignment_verified": True,
    }


def score_marc_ot_predictions(
    prediction_root: str | Path, truth_sidecar: str | Path
) -> Mapping[str, Any]:
    """Preflight a complete frozen prediction root, then open truth exactly once."""

    preflight = preflight_marc_ot_prediction(prediction_root)
    truth_payload = load_marc_ot_truth(truth_sidecar)
    return score_preflighted_marc_ot_prediction(preflight, truth_payload)


def _pairing(row: Mapping[str, Any], *, side: str):
    if row.get("schema") != "cvs.phase2.marc_ot.truth_last_score.v1":
        raise ValueError(f"MARC-OT {side} row is not a detailed truth-last score")
    payload = row.get("pairing_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("MARC-OT pairing payload is missing")
    tokens = payload.get("query_tokens")
    truth = payload.get("truth")
    predictions = payload.get("predictions")
    nll = payload.get("nll_contributions")
    if not isinstance(tokens, list) or not isinstance(truth, list) or not isinstance(predictions, Mapping) or not isinstance(nll, Mapping):
        raise ValueError("MARC-OT pairing payload is malformed")
    truth_values = np.asarray(truth)
    if truth_values.shape != (len(tokens),) or not np.issubdtype(truth_values.dtype, np.integer):
        raise ValueError("MARC-OT pairing truth is malformed")
    return tokens, truth_values.astype(np.int64), predictions, nll


def _validated_pairing_metrics(
    stored: Any,
    prediction: np.ndarray,
    truth: np.ndarray,
    registry: tuple[str, ...],
    nll_evidence: Any,
) -> Mapping[str, Any]:
    if not isinstance(stored, Mapping):
        raise ValueError("MARC-OT stored probe metric is missing")
    per_class_accuracy: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    per_class_rows: dict[str, int] = {}
    for class_id, class_name in enumerate(registry):
        truth_mask = truth == class_id
        prediction_mask = prediction == class_id
        true_positive = int(np.sum(truth_mask & prediction_mask))
        false_positive = int(np.sum(~truth_mask & prediction_mask))
        false_negative = int(np.sum(truth_mask & ~prediction_mask))
        row_count = int(truth_mask.sum())
        if row_count == 0:
            raise ValueError("MARC-OT paired truth lacks full class coverage")
        per_class_rows[class_name] = row_count
        per_class_accuracy[class_name] = float(true_positive / row_count)
        denominator = 2 * true_positive + false_positive + false_negative
        per_class_f1[class_name] = 0.0 if denominator == 0 else float(2 * true_positive / denominator)
    contributions = np.asarray(nll_evidence, dtype=np.float64)
    if contributions.shape != (len(truth),) or not np.isfinite(contributions).all():
        raise ValueError("MARC-OT paired NLL evidence is malformed")
    accuracies = np.asarray(tuple(per_class_accuracy.values()))
    recomputed: Mapping[str, Any] = {
        "accuracy": float(np.mean(prediction == truth)),
        "balanced_accuracy": float(accuracies.mean()),
        "floor": float(accuracies.min()),
        "macro_f1": float(np.mean(tuple(per_class_f1.values()))),
        "nll": float(contributions.mean()),
        "per_class_accuracy": per_class_accuracy,
        "per_class_f1": per_class_f1,
        "per_class_query_rows": per_class_rows,
    }
    for field in ("accuracy", "balanced_accuracy", "floor", "macro_f1", "nll"):
        try:
            agrees = np.isclose(float(stored[field]), float(recomputed[field]), rtol=0.0, atol=1.0e-12)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("MARC-OT stored metric disagrees with pairing evidence") from error
        if not bool(agrees):
            raise ValueError("MARC-OT stored metric disagrees with pairing evidence")
    for field in ("per_class_accuracy", "per_class_f1", "per_class_query_rows"):
        if stored.get(field) != recomputed[field]:
            raise ValueError("MARC-OT stored metric disagrees with pairing evidence")
    return recomputed


def compare_marc_ot_score_rows(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Pair one R1/R2/R4/R6/R8 row against exact same-row R0 evidence."""

    if str(control.get("arm")) != "R0" or str(candidate.get("arm")) == "R0":
        raise ValueError("MARC-OT paired comparison requires R0 control and non-R0 candidate")
    control_tokens, control_truth, control_predictions, control_nll = _pairing(control, side="control")
    candidate_tokens, candidate_truth, candidate_predictions, candidate_nll = _pairing(candidate, side="candidate")
    for field in (*_BINDING_FIELDS, "query_rows", "query_tokens", "class_registry"):
        if control.get(field) != candidate.get(field):
            raise ValueError("MARC-OT paired score binding drift")
    if control_tokens != candidate_tokens or not np.array_equal(control_truth, candidate_truth):
        raise ValueError("MARC-OT paired score token/truth binding drift")
    registry = tuple(str(value) for value in control["class_registry"])
    comparisons: dict[str, dict[str, Any]] = {}
    for probe in _PROBES:
        if probe not in control_predictions or probe not in candidate_predictions:
            raise ValueError("MARC-OT paired probe registry drift")
        control_prediction = _validate_predictions(
            np.asarray(control_predictions[probe]), rows=len(control_truth), classes=len(registry), label=probe
        )
        candidate_prediction = _validate_predictions(
            np.asarray(candidate_predictions[probe]), rows=len(control_truth), classes=len(registry), label=probe
        )
        control_metrics = _validated_pairing_metrics(
            control["probes"][probe],
            control_prediction,
            control_truth,
            registry,
            control_nll[probe],
        )
        candidate_metrics = _validated_pairing_metrics(
            candidate["probes"][probe],
            candidate_prediction,
            control_truth,
            registry,
            candidate_nll[probe],
        )
        control_correct = control_prediction == control_truth
        candidate_correct = candidate_prediction == control_truth
        help_mask = ~control_correct & candidate_correct
        harm_mask = control_correct & ~candidate_correct
        control_nll_values = np.asarray(control_nll[probe], dtype=np.float64)
        candidate_nll_values = np.asarray(candidate_nll[probe], dtype=np.float64)
        comparisons[probe] = {
            "control_metrics": dict(control_metrics),
            "candidate_metrics": dict(candidate_metrics),
            "accuracy_delta_pp": 100.0 * (float(candidate_metrics["accuracy"]) - float(control_metrics["accuracy"])),
            "balanced_accuracy_delta_pp": 100.0 * (
                float(candidate_metrics["balanced_accuracy"]) - float(control_metrics["balanced_accuracy"])
            ),
            "floor_delta_pp": 100.0 * (float(candidate_metrics["floor"]) - float(control_metrics["floor"])),
            "macro_f1_delta_pp": 100.0 * (
                float(candidate_metrics["macro_f1"]) - float(control_metrics["macro_f1"])
            ),
            "nll_delta": float(candidate_nll_values.mean() - control_nll_values.mean()),
            "per_class_accuracy_delta_pp": {
                class_name: 100.0
                * (
                    float(candidate_metrics["per_class_accuracy"][class_name])
                    - float(control_metrics["per_class_accuracy"][class_name])
                )
                for class_name in registry
            },
            "help_count": int(help_mask.sum()),
            "harm_count": int(harm_mask.sum()),
            "net_help_minus_harm": int(help_mask.sum() - harm_mask.sum()),
            "unchanged_correctness_count": int(len(control_truth) - help_mask.sum() - harm_mask.sum()),
            "prediction_flip_count": int((control_prediction != candidate_prediction).sum()),
        }
    resource_delta = {}
    for field in _RESOURCE_FIELDS:
        control_value = control["resources"][field]
        candidate_value = candidate["resources"][field]
        resource_delta[field] = (
            float(candidate_value) - float(control_value)
            if isinstance(control_value, (int, float))
            and not isinstance(control_value, bool)
            and isinstance(candidate_value, (int, float))
            and not isinstance(candidate_value, bool)
            else "N/A"
        )
    return {
        "schema": "cvs.phase2.marc_ot.paired_query_delta.v1",
        "comparison_state": "DA1_REG0-DA0_REG0",
        "control_arm": "R0",
        "candidate_arm": str(candidate["arm"]),
        **{field: control[field] for field in _BINDING_FIELDS},
        "query_rows": int(control["query_rows"]),
        "probes": comparisons,
        "resources": {
            "control": dict(control["resources"]),
            "candidate": dict(candidate["resources"]),
            "delta": resource_delta,
        },
    }


__all__ = [
    "MARCOTPredictionPreflight",
    "compare_marc_ot_score_rows",
    "load_marc_ot_truth",
    "preflight_marc_ot_prediction",
    "score_marc_ot_predictions",
    "score_preflighted_marc_ot_prediction",
]

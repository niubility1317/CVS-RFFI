"""Truth-side same-row scoring for the frozen Phase2 full-ablation design.

The module deliberately has no predictor, dataset, training, Torch, or
scheduler imports.  It first verifies the immutable ``.cvspred`` container,
then opens the scorer-only truth sidecar.  Performance values produced here
are terminal evidence and must never feed a predictor or run scheduler.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_metric_scorer import (
    Stage2ScoringError,
    canonical_json_bytes,
    load_verified_scoring_sidecar,
    load_verified_sealed_prediction,
    score_prediction_arrays,
)


SAME_ROW_SCORE_SCHEMA = "cvs.full_ablation.phase2.same_row_score.v1"
FAILED_ROW_SCHEMA = "cvs.full_ablation.phase2.failed_row.v1"
BEHAVIOR_RECEIPT_SCHEMA = "cvs.full_ablation.phase2.behavior_receipt.v1"
QUANTIZATION_RECEIPT_SCHEMA = "cvs.full_ablation.phase2.quantization_receipt.v1"
RESOURCE_RECEIPT_SCHEMA = "cvs.full_ablation.phase2.resource_receipt.v1"
ABLATION_SCORER_RECEIPT_SCHEMA = "cvs.full_ablation.phase2.scorer_receipt.v1"
FAILURE_RECEIPT_SCHEMA = "cvs.full_ablation.phase2.failure_receipt.v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_KEYS = {
    "logical_row_key",
    "ablation_id",
    "physical_execution_id",
    "effective_config_hash",
    "alias_of",
}
_BEHAVIOR_KEYS = {
    "schema",
    "fallback_counts",
    "full_block_weights",
    "fisher_gate_accept_counts",
    "atomic_rollback_counts",
    "failure_closure_count",
}
_QUANTIZATION_KEYS = {
    "schema",
    "max_logit_abs_error",
    "mean_logit_abs_error",
    "argmax_flip_rate",
    "prediction_agreement_rate",
}
_RESOURCE_KEYS = {
    "schema",
    "feature_cache_bytes",
    "deployment_state_bytes",
    "state_bytes",
    "registration_time_ms",
    "row_peak_rss_bytes",
    "row_peak_vram_bytes",
    "candidate_peak_memory_isolated",
    "closed_form_fit_count",
    "mac_equivalent_upper_bound",
    "query_head_mac",
    "candidate_head_batch_query_latency_ms_per_row",
    "end_to_end_query_latency_available",
    "end_to_end_query_latency_ms",
    "batch1_head_resource",
    "row_orchestration_time_ms",
    "auxiliary_state_cost_in_candidate_resource",
    "auxiliary_prediction_cost_in_candidate_latency",
}
_PRIMARY_FIELDS = (
    "A_o_pre",
    "A_o_post",
    "A_n",
    "H",
    "F",
    "min_old",
    "min_new",
)


class FullAblationScoringError(Stage2ScoringError):
    """Raised when a full-ablation scorer contract fails closed."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_object(payload: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(payload))


def _exact_mapping(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FullAblationScoringError(f"{context} exact schema drift")
    return dict(value)


def _text(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
    ):
        raise FullAblationScoringError(f"{context} must be nonempty trimmed text")
    return value


def _sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FullAblationScoringError(f"{context} must be a lowercase SHA256")
    return value


def _finite_number(value: Any, *, context: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FullAblationScoringError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise FullAblationScoringError(
            f"{context} must be finite and >= {minimum}"
        )
    return result


def _count(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FullAblationScoringError(f"{context} must be a nonnegative integer")
    return value


def _count_mapping(value: Any, *, context: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise FullAblationScoringError(f"{context} must be a nonempty object")
    result: dict[str, int] = {}
    for key, item in value.items():
        result[_text(key, context=f"{context} key")] = _count(
            item, context=f"{context}.{key}"
        )
    return dict(sorted(result.items()))


def _validate_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    identity = _exact_mapping(value, _IDENTITY_KEYS, context="row identity")
    for key in ("logical_row_key", "ablation_id", "physical_execution_id"):
        identity[key] = _text(identity[key], context=f"row identity.{key}")
    identity["effective_config_hash"] = _sha256(
        identity["effective_config_hash"],
        context="row identity.effective_config_hash",
    )
    alias = identity["alias_of"]
    if alias is not None:
        alias = _text(alias, context="row identity.alias_of")
        if alias == identity["logical_row_key"]:
            raise FullAblationScoringError("an alias cannot refer to itself")
    identity["alias_of"] = alias
    return identity


def _validate_behavior(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _exact_mapping(value, _BEHAVIOR_KEYS, context="behavior receipt")
    if receipt["schema"] != BEHAVIOR_RECEIPT_SCHEMA:
        raise FullAblationScoringError("behavior receipt schema drift")
    receipt["fallback_counts"] = _count_mapping(
        receipt["fallback_counts"], context="fallback_counts"
    )
    receipt["fisher_gate_accept_counts"] = _count_mapping(
        receipt["fisher_gate_accept_counts"],
        context="fisher_gate_accept_counts",
    )
    receipt["atomic_rollback_counts"] = _count_mapping(
        receipt["atomic_rollback_counts"], context="atomic_rollback_counts"
    )
    fisher = receipt["fisher_gate_accept_counts"]
    if set(fisher) != {"attempted", "accepted"}:
        raise FullAblationScoringError(
            "fisher_gate_accept_counts exact schema drift"
        )
    if fisher["accepted"] > fisher["attempted"]:
        raise FullAblationScoringError(
            "Fisher accepted count cannot exceed attempted count"
        )
    rollback = receipt["atomic_rollback_counts"]
    if set(rollback) != {"attempted", "rolled_back"}:
        raise FullAblationScoringError("atomic_rollback_counts exact schema drift")
    if rollback["rolled_back"] > rollback["attempted"]:
        raise FullAblationScoringError(
            "atomic rollback count cannot exceed attempted count"
        )
    weights = _exact_mapping(
        receipt["full_block_weights"],
        {"full", "block3"},
        context="full_block_weights",
    )
    weights = {
        key: _finite_number(value, context=f"full_block_weights.{key}")
        for key, value in weights.items()
    }
    if any(value > 1.0 for value in weights.values()):
        raise FullAblationScoringError("full/block weights must be <= 1")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-8):
        raise FullAblationScoringError("full/block weights must sum to one")
    receipt["full_block_weights"] = weights
    receipt["failure_closure_count"] = _count(
        receipt["failure_closure_count"], context="failure_closure_count"
    )
    return receipt


def _validate_quantization(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _exact_mapping(
        value, _QUANTIZATION_KEYS, context="quantization receipt"
    )
    if receipt["schema"] != QUANTIZATION_RECEIPT_SCHEMA:
        raise FullAblationScoringError("quantization receipt schema drift")
    for field in ("max_logit_abs_error", "mean_logit_abs_error"):
        receipt[field] = _finite_number(receipt[field], context=field)
    if receipt["mean_logit_abs_error"] > receipt["max_logit_abs_error"] + 1e-12:
        raise FullAblationScoringError(
            "mean logit error cannot exceed maximum logit error"
        )
    for field in ("argmax_flip_rate", "prediction_agreement_rate"):
        receipt[field] = _finite_number(receipt[field], context=field)
        if receipt[field] > 1.0:
            raise FullAblationScoringError(f"{field} must be <= 1")
    if not math.isclose(
        receipt["argmax_flip_rate"] + receipt["prediction_agreement_rate"],
        1.0,
        abs_tol=1e-8,
    ):
        raise FullAblationScoringError(
            "argmax flip and prediction agreement rates must sum to one"
        )
    return receipt


def _validate_resource(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _exact_mapping(value, _RESOURCE_KEYS, context="resource receipt")
    if receipt["schema"] != RESOURCE_RECEIPT_SCHEMA:
        raise FullAblationScoringError("resource receipt schema drift")
    integer_fields = {
        "feature_cache_bytes",
        "deployment_state_bytes",
        "state_bytes",
        "row_peak_rss_bytes",
        "row_peak_vram_bytes",
        "closed_form_fit_count",
        "mac_equivalent_upper_bound",
        "query_head_mac",
    }
    for field in integer_fields:
        receipt[field] = _count(receipt[field], context=field)
    for field in (
        "registration_time_ms",
        "candidate_head_batch_query_latency_ms_per_row",
        "row_orchestration_time_ms",
    ):
        receipt[field] = _finite_number(receipt[field], context=field)
    for field in (
        "candidate_peak_memory_isolated",
        "end_to_end_query_latency_available",
        "auxiliary_state_cost_in_candidate_resource",
        "auxiliary_prediction_cost_in_candidate_latency",
    ):
        if not isinstance(receipt[field], bool):
            raise FullAblationScoringError(
                f"{field} must be boolean"
            )
    if (
        receipt["candidate_peak_memory_isolated"] is not False
        or receipt["end_to_end_query_latency_available"] is not False
        or receipt["end_to_end_query_latency_ms"] is not None
        or receipt["auxiliary_state_cost_in_candidate_resource"] is not False
        or receipt["auxiliary_prediction_cost_in_candidate_latency"] is not False
    ):
        raise FullAblationScoringError(
            "resource scope declaration drift"
        )
    batch1 = receipt["batch1_head_resource"]
    if batch1 is not None and (
        not isinstance(batch1, Mapping)
        or not isinstance(batch1.get("decode_cost"), Mapping)
    ):
        raise FullAblationScoringError(
            "batch1 head resource receipt drift"
        )
    return receipt


def _prediction_column_hash(
    arrays: Mapping[str, np.ndarray], column: str
) -> str:
    payload = [
        {
            "scenario": scenario,
            "query_token": token,
            "prediction": prediction,
        }
        for scenario, token, prediction in zip(
            np.asarray(arrays["scenarios"]).astype(str).tolist(),
            np.asarray(arrays["query_tokens"]).astype(str).tolist(),
            np.asarray(arrays[column]).astype(str).tolist(),
        )
    ]
    return _sha256_object(payload)


def _per_class_metrics(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    handle_role: dict[str, str] = {}
    handle_tx: dict[str, str] = {}
    for row in predictions:
        handle = row["true_class_handle"]
        if handle is None:
            continue
        role = row["evaluation_role"]
        tx = row["transmitter_label"]
        if handle in handle_role and handle_role[handle] != role:
            raise FullAblationScoringError("class handle maps to multiple roles")
        if handle in handle_tx and handle_tx[handle] != tx:
            raise FullAblationScoringError(
                "class handle maps to multiple transmitter labels"
            )
        handle_role[handle] = role
        handle_tx[handle] = tx

    confusion: dict[str, dict[str, int]] = {}
    class_correct: dict[str, int] = {}
    class_count: dict[str, int] = {}
    old_to_new = 0
    new_to_old = 0
    old_count = 0
    new_count = 0
    for row in predictions:
        true_handle = row["true_class_handle"]
        if true_handle is None:
            continue
        predicted_handle = row["candidate_after"]
        if predicted_handle not in handle_role:
            raise FullAblationScoringError(
                "predicted class handle has no truth-side registered role"
            )
        true_role = row["evaluation_role"]
        predicted_role = handle_role[predicted_handle]
        true_tx = row["transmitter_label"]
        predicted_tx = handle_tx[predicted_handle]
        confusion.setdefault(true_tx, {})
        confusion[true_tx][predicted_tx] = (
            confusion[true_tx].get(predicted_tx, 0) + 1
        )
        class_count[true_tx] = class_count.get(true_tx, 0) + 1
        class_correct[true_tx] = class_correct.get(true_tx, 0) + int(
            predicted_handle == true_handle
        )
        if true_role == "target_old":
            old_count += 1
            old_to_new += int(predicted_role == "target_new")
        elif true_role == "target_new":
            new_count += 1
            new_to_old += int(predicted_role == "target_old")
        else:
            raise FullAblationScoringError("unsupported truth-side class role")

    per_class_accuracy = {
        tx: class_correct.get(tx, 0) / count
        for tx, count in sorted(class_count.items())
    }
    new_txs = {
        row["transmitter_label"]
        for row in predictions
        if row["evaluation_role"] == "target_new"
        and row["true_class_handle"] is not None
    }
    min_new = (
        min(per_class_accuracy[tx] for tx in new_txs) if new_txs else None
    )
    return {
        "min_new": min_new,
        "old_to_new_count": old_to_new,
        "old_to_new_rate": old_to_new / old_count if old_count else None,
        "new_to_old_count": new_to_old,
        "new_to_old_rate": new_to_old / new_count if new_count else None,
        "per_class_accuracy": per_class_accuracy,
        "per_class_confusion": {
            true_tx: dict(sorted(predicted.items()))
            for true_tx, predicted in sorted(confusion.items())
        },
    }


def score_full_ablation_row(
    prediction_artifact_path: str | Path,
    scoring_manifest_path: str | Path,
    *,
    expected_prediction_artifact_sha256: str,
    expected_prediction_seal_sha256: str,
    expected_scoring_manifest_sha256: str,
    row_identity: Mapping[str, Any],
    behavior_receipt: Mapping[str, Any],
    quantization_receipt: Mapping[str, Any],
    resource_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one logical row while preserving its physical/alias identity."""

    identity = _validate_identity(row_identity)

    # P0 ordering boundary: this call validates the immutable prediction
    # container and detached seal before any truth-side file is opened.
    binding, arrays, prediction_audit = load_verified_sealed_prediction(
        prediction_artifact_path,
        expected_prediction_artifact_sha256=expected_prediction_artifact_sha256,
        expected_prediction_seal_sha256=expected_prediction_seal_sha256,
    )
    if binding["row_id"] != identity["physical_execution_id"]:
        raise FullAblationScoringError(
            "physical_execution_id does not bind the sealed prediction row"
        )

    behavior = _validate_behavior(behavior_receipt)
    quantization = _validate_quantization(quantization_receipt)
    resources = _validate_resource(resource_receipt)

    # This is the first truth-side open in this function and occurs only after
    # the prediction artifact has successfully verified.
    truth, scoring_manifest, scoring_audit = load_verified_scoring_sidecar(
        scoring_manifest_path,
        expected_scoring_manifest_sha256=expected_scoring_manifest_sha256,
    )
    if (
        binding["predictor_package_root_sha256"]
        != scoring_manifest["predictor_package_root_sha256"]
        or binding["predictor_package_seal_sha256"]
        != scoring_manifest["predictor_package_seal_sha256"]
    ):
        raise FullAblationScoringError(
            "prediction/scoring predictor package binding mismatch"
        )

    base_rows, scored_predictions = score_prediction_arrays(
        binding=binding, arrays=arrays, truth=truth
    )
    scenario_rows: list[dict[str, Any]] = []
    for base in base_rows:
        scenario_predictions = [
            row
            for row in scored_predictions
            if row["scenario"] == base["scenario"]
        ]
        detailed = _per_class_metrics(scenario_predictions)
        if binding["stage"] == "stage2c":
            primary = {
                "A_o_pre": base["old_acc_before_increment"],
                "A_o_post": base["old_acc_after_increment"],
                "A_n": base["seen_new_acc"],
                "H": base["H_old_new"],
                "F": base["candidate_average_forgetting"],
                "min_old": base["min_old_class_acc"],
                "min_new": detailed["min_new"],
            }
        else:
            # Stage2-B is the registered-old adaptation state.  It defines
            # A_o_pre for a later paired Stage2-C row, but cannot claim
            # post-registration/new-class metrics.
            primary = {
                "A_o_pre": base["old_acc_after_increment"],
                "A_o_post": None,
                "A_n": None,
                "H": None,
                "F": None,
                "min_old": base["min_old_class_acc"],
                "min_new": None,
            }
        row = {
            "scenario": base["scenario"],
            "query_count": base["query_count"],
            "target_old_query_count": base["target_old_query_count"],
            "target_new_query_count": base["target_new_query_count"],
            **primary,
            "old_to_new_count": detailed["old_to_new_count"],
            "old_to_new_rate": detailed["old_to_new_rate"],
            "new_to_old_count": detailed["new_to_old_count"],
            "new_to_old_rate": detailed["new_to_old_rate"],
            "per_class_accuracy": detailed["per_class_accuracy"],
            "per_class_confusion": detailed["per_class_confusion"],
        }
        if binding["stage"] == "stage2c" and any(
            row[field] is None for field in _PRIMARY_FIELDS
        ):
            raise FullAblationScoringError(
                "Stage2-C same-row primary metric closure is incomplete"
            )
        scenario_rows.append(row)

    before_hash = _prediction_column_hash(arrays, "candidate_before")
    after_hash = _prediction_column_hash(arrays, "candidate_after")
    behavior_sha256 = _sha256_object(behavior)
    quantization_sha256 = _sha256_object(quantization)
    resource_sha256 = _sha256_object(resources)
    same_row_metrics_sha256 = _sha256_object(scenario_rows)
    scorer_receipt = {
        "schema": ABLATION_SCORER_RECEIPT_SCHEMA,
        "status": "PASS",
        "logical_row_key": identity["logical_row_key"],
        "ablation_id": identity["ablation_id"],
        "physical_execution_id": identity["physical_execution_id"],
        "effective_config_hash": identity["effective_config_hash"],
        "alias_of": identity["alias_of"],
        "independent_observation": identity["alias_of"] is None,
        "stage": binding["stage"],
        "receiver": binding["receiver"],
        "k_shot": int(binding["k_shot"]),
        "prediction_artifact_sha256": prediction_audit[
            "prediction_artifact_sha256"
        ],
        "prediction_seal_sha256": prediction_audit["prediction_seal_sha256"],
        "before_prediction_hash": before_hash,
        "after_prediction_hash": after_hash,
        "behavior_receipt_sha256": behavior_sha256,
        "quantization_receipt_sha256": quantization_sha256,
        "resource_receipt_sha256": resource_sha256,
        "same_row_metrics_sha256": same_row_metrics_sha256,
        "scoring_manifest_sha256": scoring_audit["scoring_manifest_sha256"],
        "truth_sidecar_sha256": scoring_audit["truth_sidecar_sha256"],
        "truth_opened_after_prediction_commit": True,
        "scorer_output_must_not_feed_predictor": True,
        "join_policy": "exact_scenario_query_token",
    }
    scorer_receipt_sha256 = _sha256_object(scorer_receipt)
    return {
        "schema": SAME_ROW_SCORE_SCHEMA,
        "status": "PASS",
        **identity,
        "independent_observation": identity["alias_of"] is None,
        "stage": binding["stage"],
        "receiver": binding["receiver"],
        "k_shot": int(binding["k_shot"]),
        "candidate_lock_sha256": binding["candidate_lock_sha256"],
        "predictor_package_root_sha256": binding[
            "predictor_package_root_sha256"
        ],
        "prediction_artifact_sha256": prediction_audit[
            "prediction_artifact_sha256"
        ],
        "prediction_seal_sha256": prediction_audit["prediction_seal_sha256"],
        "before_prediction_hash": before_hash,
        "after_prediction_hash": after_hash,
        "behavior_receipt_sha256": behavior_sha256,
        "quantization_receipt_sha256": quantization_sha256,
        "resource_receipt_sha256": resource_sha256,
        "same_row_metrics_sha256": same_row_metrics_sha256,
        "truth_opened_after_prediction_commit": True,
        "scenario_rows": scenario_rows,
        "behavior": behavior,
        "quantization": quantization,
        "resource": resources,
        "scorer_receipt": scorer_receipt,
        "scorer_receipt_sha256": scorer_receipt_sha256,
    }


def build_failed_row_record(
    *,
    row_identity: Mapping[str, Any],
    stage: str,
    receiver: str,
    k_shot: int,
    failure_code: str,
    failure_fingerprint: str,
    zero_prediction: bool,
) -> dict[str, Any]:
    """Create an explicit non-performance record for a failed logical row."""

    identity = _validate_identity(row_identity)
    if stage not in {"stage2a", "stage2b", "stage2c"}:
        raise FullAblationScoringError("failed row stage is unsupported")
    if (
        isinstance(k_shot, bool)
        or not isinstance(k_shot, int)
        or (stage == "stage2a" and k_shot != 0)
        or (stage != "stage2a" and k_shot <= 0)
    ):
        raise FullAblationScoringError(
            "failed row k_shot does not match the stage"
        )
    if not isinstance(zero_prediction, bool):
        raise FullAblationScoringError("zero_prediction must be boolean")
    failure_receipt = {
        "schema": FAILURE_RECEIPT_SCHEMA,
        "logical_row_key": identity["logical_row_key"],
        "ablation_id": identity["ablation_id"],
        "physical_execution_id": identity["physical_execution_id"],
        "effective_config_hash": identity["effective_config_hash"],
        "alias_of": identity["alias_of"],
        "independent_observation": identity["alias_of"] is None,
        "stage": stage,
        "receiver": _text(receiver, context="failed row receiver"),
        "k_shot": k_shot,
        "failure_code": _text(failure_code, context="failure_code"),
        "failure_fingerprint": _text(
            failure_fingerprint, context="failure_fingerprint"
        ),
        "zero_prediction": zero_prediction,
    }
    return {
        "schema": FAILED_ROW_SCHEMA,
        "status": "FAILED",
        **identity,
        "independent_observation": identity["alias_of"] is None,
        "stage": stage,
        "receiver": failure_receipt["receiver"],
        "k_shot": k_shot,
        "failure_code": failure_receipt["failure_code"],
        "failure_fingerprint": failure_receipt["failure_fingerprint"],
        "zero_prediction": zero_prediction,
        "scenario_rows": [],
        "failure_receipt": failure_receipt,
        "failure_receipt_sha256": _sha256_object(failure_receipt),
        **{field: None for field in _PRIMARY_FIELDS},
    }


def write_row_record_exclusive(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a scorer record with ``O_EXCL``; existing evidence is immutable."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(destination, flags, 0o444)
    try:
        data = canonical_json_bytes(payload) + b"\n"
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing scorer record")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ABLATION_SCORER_RECEIPT_SCHEMA",
    "BEHAVIOR_RECEIPT_SCHEMA",
    "FAILED_ROW_SCHEMA",
    "FAILURE_RECEIPT_SCHEMA",
    "FullAblationScoringError",
    "QUANTIZATION_RECEIPT_SCHEMA",
    "RESOURCE_RECEIPT_SCHEMA",
    "SAME_ROW_SCORE_SCHEMA",
    "build_failed_row_record",
    "score_full_ablation_row",
    "write_row_record_exclusive",
]

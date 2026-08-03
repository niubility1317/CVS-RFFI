"""Minimal support-only integration of D129 DA candidates and six heads."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage2_d129_joint6_da as da
from . import stage2_d129_joint6_heads as heads
from . import stage2_d129_joint6_matrix as matrix
from . import stage2_zid_student_t_qknn as qknn


RUNTIME_SCHEMA = "cvs.stage2.d129.joint6.proxy_runtime.v2"
SMOKE_SCHEMA = "cvs.stage2.d129.joint6.no_truth_smoke.v1"


class D129Joint6RuntimeError(ValueError):
    """Raised when the frozen DA-to-six-head runtime contract drifts."""


def _balanced_support(
    support_zid160: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], int]:
    rows = np.asarray(support_zid160)
    labels = tuple(str(value) for value in support_labels)
    classes = tuple(str(value) for value in registered_classes)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != heads.Z_DIM
        or rows.shape[0] < 2
        or not np.isfinite(rows).all()
        or len(labels) != len(rows)
        or len(classes) < 2
        or len(set(classes)) != len(classes)
        or any(label not in classes for label in labels)
    ):
        raise D129Joint6RuntimeError("balanced support input closure drift")
    grouped = [rows[np.asarray([label == class_id for label in labels])] for class_id in classes]
    counts = tuple(len(value) for value in grouped)
    if any(count < 1 for count in counts) or len(set(counts)) != 1 or counts[0] not in {1, 5}:
        raise D129Joint6RuntimeError("Joint6 runtime requires balanced K1/K5 support")
    support3d = np.stack(grouped).astype(np.float32, copy=False)
    return support3d, labels, classes, counts[0]


def _margin(logits: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(logits, dtype=np.float64), axis=1)
    if values.ndim != 2 or values.shape[1] < 2:
        raise D129Joint6RuntimeError("qKNN margin requires at least two classes")
    return values[:, -1] - values[:, -2]


def _state_fingerprint(state: da.CSPAR2State | da.SRDH2State) -> tuple[Any, ...]:
    if isinstance(state, da.CSPAR2State):
        return (
            state.alpha_fp16.tobytes(),
            state.support_root_sha256,
            state.receipt.as_dict()["resource_receipt_sha256"],
        )
    if isinstance(state, da.SRDH2State):
        return (
            state.response_fp16.tobytes(),
            state.summary_fp16.tobytes(),
            state.support_root_sha256,
            state.receipt.as_dict()["resource_receipt_sha256"],
        )
    raise D129Joint6RuntimeError("unsupported DA state fingerprint")


def _nonidentity_smoke(result: heads.D129Joint6Result) -> Mapping[str, Any]:
    r0_support = np.asarray(result.r0_cache.support_zid160, dtype=np.float64)
    r1_support = np.asarray(result.r1_cache.support_zid160, dtype=np.float64)
    r0_query = np.asarray(result.r0_cache.query_zid160, dtype=np.float64)
    r1_query = np.asarray(result.r1_cache.query_zid160, dtype=np.float64)
    gram0 = r0_query @ r0_support.T
    gram1 = r1_query @ r1_support.T
    gram_delta = float(np.max(np.abs(gram1 - gram0)))
    neighbor0 = np.argmax(gram0, axis=1)
    neighbor1 = np.argmax(gram1, axis=1)
    neighbor_change_count = int(np.sum(neighbor0 != neighbor1))
    margin_delta = float(
        np.max(np.abs(_margin(result.r1q.logits) - _margin(result.r0q.logits)))
    )
    tolerance = float(64.0 * np.finfo(np.float32).eps)
    gram_changed = gram_delta > tolerance
    margin_changed = margin_delta > tolerance
    pass_value = gram_changed and (neighbor_change_count > 0 or margin_changed)
    return MappingProxyType(
        {
            "schema": SMOKE_SCHEMA,
            "truth_loaded": False,
            "support_query_gram_max_abs_delta": gram_delta,
            "nearest_support_change_count": neighbor_change_count,
            "qknn_margin_max_abs_delta": margin_delta,
            "numeric_noise_tolerance": tolerance,
            "gram_changed_beyond_numeric_noise": gram_changed,
            "neighbor_or_margin_changed": neighbor_change_count > 0 or margin_changed,
            "smoke_pass": pass_value,
            "feature_change_alone_is_sufficient": False,
        }
    )


@dataclass(frozen=True)
class D129CandidateJoint6Result:
    candidate_id: str
    da_state: da.CSPAR2State | da.SRDH2State
    six_arm: heads.D129Joint6Result
    query_read_only_receipt: Mapping[str, Any]
    smoke_receipt: Mapping[str, Any]
    runtime_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.candidate_id not in matrix.CANDIDATE_IDS:
            raise D129Joint6RuntimeError("runtime candidate identity drift")
        if tuple(arm.arm_id for arm in self.six_arm.arms) != matrix.ARM_IDS:
            raise D129Joint6RuntimeError("runtime six-arm closure drift")
        if self.query_read_only_receipt.get("protocol_closed") is not True:
            raise D129Joint6RuntimeError("runtime query read-only closure failed")
        object.__setattr__(self, "query_read_only_receipt", MappingProxyType(dict(self.query_read_only_receipt)))
        object.__setattr__(self, "smoke_receipt", MappingProxyType(dict(self.smoke_receipt)))
        object.__setattr__(self, "runtime_receipt", MappingProxyType(dict(self.runtime_receipt)))


def run_d129_candidate_joint6(
    *,
    asset: da.CSPAR2Asset | da.SRDH2Asset,
    base_support_zid160: np.ndarray,
    base_query_zid160: np.ndarray,
    support_labels: Sequence[str],
    support_physical_ids: Sequence[str],
    registered_classes: Sequence[str],
    retained_class_count: int,
    opaque_query_ids: Sequence[str],
    qknn_lock: qknn.Phase1ZIDStudentTLock,
    fold_binding: Mapping[str, Any],
    common_r0: heads.D129CommonR0,
) -> D129CandidateJoint6Result:
    """Fit one support-only DA state, cache R1 once, and execute six heads.

    There is intentionally no query label, role, quota, source path, clean path,
    scorer result, optimizer, or checkpoint hook argument.
    """

    support3d, labels, classes, active_k = _balanced_support(
        base_support_zid160, support_labels, registered_classes
    )
    try:
        binding = dict(matrix.validate_joint6_binding(fold_binding))
    except matrix.D129Joint6MatrixError as exc:
        raise D129Joint6RuntimeError("fold binding digest drift") from exc
    support_ids = tuple(str(value) for value in support_physical_ids)
    query_ids = tuple(str(value) for value in opaque_query_ids)
    expected_support_key = (
        "k1_support_ids_by_class" if active_k == 1 else "k5_support_ids_by_class"
    )
    expected_row_key = "k1_row_id" if active_k == 1 else "k5_row_id"
    expected_support = tuple(
        physical_id
        for class_id in classes
        for physical_id in binding.get(expected_support_key, {}).get(class_id, ())
    )
    expected_query = tuple(
        physical_id
        for class_id in classes
        for physical_id in binding.get("query_ids_by_class", {}).get(class_id, ())
    )
    if (
        binding.get("schema") != matrix.ROW_BINDING_SCHEMA
        or binding.get("registered_classes") != list(classes)
        or binding.get("evaluation_semantics")
        != "phase1_seen_class_loco_directional_proxy"
        or binding.get("formal_new_registration_claim") is not False
        or binding.get("k1_is_exact_k5_prefix") is not True
        or binding.get("support_query_physical_ids_disjoint") is not True
        or not binding.get(expected_row_key)
        or support_ids != expected_support
        or query_ids != expected_query
        or getattr(asset, "phase1_seal_sha256", None)
        != binding.get("phase1_seal_sha256")
    ):
        raise D129Joint6RuntimeError("fold/asset/physical-ID binding drift")
    query = np.asarray(base_query_zid160)
    if (
        query.dtype != np.float32
        or query.ndim != 2
        or query.shape[1] != heads.Z_DIM
        or len(query) < 1
        or not np.isfinite(query).all()
    ):
        raise D129Joint6RuntimeError("base query must be finite float32 [N,160]")
    if int(retained_class_count) != len(classes) - 1:
        raise D129Joint6RuntimeError("retained/held-proxy registry split drift")
    if isinstance(asset, da.CSPAR2Asset):
        candidate_id = da.CSPAR2_CANDIDATE_ID
        state = da.fit_cspar2_support(asset, support3d)
        adapted_support = da.transform_cspar2(asset, state, base_support_zid160)
        state_before_query = _state_fingerprint(state)
        adapted_query = da.transform_cspar2(asset, state, query)
    elif isinstance(asset, da.SRDH2Asset):
        candidate_id = da.SRDH2_CANDIDATE_ID
        state = da.fit_srdh2_support(asset, support3d)
        adapted_support = da.transform_srdh2(asset, state, base_support_zid160)
        state_before_query = _state_fingerprint(state)
        adapted_query = da.transform_srdh2(asset, state, query)
    else:
        raise D129Joint6RuntimeError("runtime requires a frozen D129 asset")
    if state_before_query != _state_fingerprint(state) or not state.receipt.protocol_closed:
        raise D129Joint6RuntimeError("query transform mutated support-only DA state")
    query_audit = {
        "schema": da.STATE_SCHEMA,
        "candidate_id": candidate_id,
        "query_output_shape": list(adapted_query.shape),
        "query_rows_used_for_fit": state.receipt.query_rows_used_for_fit,
        "query_state_updates": state.receipt.query_state_updates,
        "query_selection_count": state.receipt.query_selection_count,
        "query_gradient_calls": state.receipt.query_gradient_calls,
        "truth_role_quota_inputs": state.receipt.truth_role_quota_inputs,
        "global_reassignment_calls": state.receipt.global_reassignment_calls,
        "protocol_closed": state.receipt.protocol_closed,
    }
    resource = state.receipt.as_dict()
    da_numeric_bytes = int(
        resource["asset_numeric_payload_bytes"] + resource["dynamic_numeric_bytes"]
    )
    six_arm = heads.run_d129_joint6_heads(
        base_support_zid=base_support_zid160,
        adapted_support_zid=adapted_support,
        base_query_zid=query,
        adapted_query_zid=adapted_query,
        support_labels=labels,
        registered_classes=classes,
        old_class_count=int(retained_class_count),
        partition_semantics="phase1_seen_class_loco_directional_proxy",
        opaque_query_ids=query_ids,
        qknn_lock=qknn_lock,
        common_r0=common_r0,
        da_numeric_state_bytes=da_numeric_bytes,
    )
    if (
        six_arm.row_receipt.get("common_r0_supplied_by_caller") is not True
        or six_arm.row_receipt.get(
            "common_r0_head_fit_calls_in_this_candidate_call"
        )
        != 0
    ):
        raise D129Joint6RuntimeError("candidate runtime recomputed common R0 heads")
    smoke = _nonidentity_smoke(six_arm)
    runtime_receipt = {
        "schema": RUNTIME_SCHEMA,
        "candidate_id": candidate_id,
        "active_k": active_k,
        "registered_classes": list(classes),
        "retained_class_count": int(retained_class_count),
        "held_proxy_class_count": len(classes) - int(retained_class_count),
        "evaluation_semantics": "phase1_seen_class_loco_directional_proxy",
        "formal_new_registration_claim": False,
        "held_receiver": binding["held_receiver"],
        "held_class": binding["held_class"],
        "phase1_seal_sha256": binding["phase1_seal_sha256"],
        "binding_sha256": binding["binding_sha256"],
        "query_physical_root_sha256": binding["query_physical_root_sha256"],
        "checkpoint_sha256": asset.checkpoint_sha256,
        "asset_sha256": da.d129_joint6_asset_sha256(asset),
        "row_id": binding[expected_row_key],
        "representation_cache_count": 2,
        "heads_per_representation": 3,
        "same_row_six_arm_binding": True,
        "common_r0_sha256": six_arm.row_receipt["common_r0_sha256"],
        "common_r0_head_fit_calls_in_candidate_runtime": 0,
        "adaptation_support_fit_calls": 1,
        "adaptation_support_transform_calls": 1,
        "adaptation_query_transform_calls": 1,
        "backbone_forward_calls_in_runtime": 0,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "truth_input_exists": False,
        "query_role_input_exists": False,
        "class_quota_input_exists": False,
        "source_runtime_access": False,
        "clean_runtime_access": False,
        "phase2_optimizer_steps": 0,
        "phase2_backward_calls": 0,
        "da_numeric_state_bytes": da_numeric_bytes,
    }
    return D129CandidateJoint6Result(
        candidate_id=candidate_id,
        da_state=state,
        six_arm=six_arm,
        query_read_only_receipt=query_audit,
        smoke_receipt=smoke,
        runtime_receipt=runtime_receipt,
    )


def build_joint6_prediction_row(
    row: matrix.Joint6LocoRow,
    result: D129CandidateJoint6Result,
) -> Mapping[str, Any]:
    """Create one scorer-facing truth-free prediction row."""

    receipt = result.six_arm.row_receipt
    if (
        row.active_k != receipt.get("active_k")
        or list(row.registered_classes) != receipt.get("registered_classes")
        or len(row.retained_classes) != receipt.get("old_class_count")
        or result.runtime_receipt.get("row_id") != row.row_id
        or result.runtime_receipt.get("held_receiver") != row.held_receiver
        or result.runtime_receipt.get("held_class") != row.held_class
    ):
        raise D129Joint6RuntimeError("prediction row/method result binding drift")
    return MappingProxyType(
        {
            "candidate_id": result.candidate_id,
            "row_id": row.row_id,
            "held_receiver": row.held_receiver,
            "held_class": row.held_class,
            "active_k": row.active_k,
            "registered_classes": list(row.registered_classes),
            "binding_sha256": result.runtime_receipt["binding_sha256"],
            "phase1_seal_sha256": result.runtime_receipt[
                "phase1_seal_sha256"
            ],
            "query_physical_root_sha256": result.runtime_receipt[
                "query_physical_root_sha256"
            ],
            "checkpoint_sha256": result.runtime_receipt["checkpoint_sha256"],
            "asset_sha256": result.runtime_receipt["asset_sha256"],
            "common_r0_sha256": result.runtime_receipt["common_r0_sha256"],
            "evaluation_semantics": "phase1_seen_class_loco_directional_proxy",
            "formal_new_registration_claim": False,
            "opaque_query_ids": list(receipt["opaque_query_ids"]),
            "arms": {
                arm.arm_id: list(arm.predictions) for arm in result.six_arm.arms
            },
        }
    )


__all__ = [
    "D129CandidateJoint6Result",
    "D129Joint6RuntimeError",
    "RUNTIME_SCHEMA",
    "SMOKE_SCHEMA",
    "build_joint6_prediction_row",
    "run_d129_candidate_joint6",
]

"""Formal local D104 four-arm Stage2 state over D103 MetaBias4 and ANGQ.

This module is prediction-complete but truth-blind. It builds the frozen
M0/M_DA/M_HEAD/M_JOINT arms from matched support and publishes immutable
per-query predictions over every opaque registered class.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .rxid_metabias4_bundle import RXIDMetaBias4Bundle
from .stage2_d104_angq_qknn import (
    ANGQ_SCHEMA,
    CANDIDATE_ID,
    FACTORS,
    RESOURCE_SCHEMA,
    audit_d104_angq_resource_delta,
    build_d104_angq_support_bank,
)
from .stage2_rb_metabias4_qknn import (
    baseline_zid_from_pre_relu,
    build_d102_baseline_bank,
)
from .stage2_rxid_metabias4 import (
    D103Stage2State,
    K1IdentifiabilityReceipt,
    fit_d103_stage2_state,
    stable_first_argmax,
    transform_d103_query,
)
from .stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    TypedINT8ZIDSupportBank,
    TypedSharedPSDMetric,
    _canonical_sha256,
    _identity_class_scales,
    _score_with_support,
    identity_shared_psd_metric,
    normalize_zid_rows,
    score_zid_student_t_logits,
)


SCHEMA = "cvs.phase2.d104_r1.rxid_angq.four_arm_state.v1"
METHOD_LOCK_SCHEMA = "cvs.phase2.d104_r1.rxid_angq.method_lock.v1"
PREDICTION_SCHEMA = "cvs.phase2.d104_r1.rxid_angq.prediction.v1"
INT8_AUDIT_SCHEMA = "cvs.phase2.d104_r1.rxid_angq.int8_audit.v1"
ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")


class D104RXIDANGQError(ValueError):
    """Raised when D104 four-arm state, prediction, or audit drifts."""


def _require_sha256(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise D104RXIDANGQError(f"{name} must be a lowercase SHA256")
    return text


def _method_lock(
    bundle: RXIDMetaBias4Bundle,
    qknn_config: Phase1ZIDStudentTLock,
) -> Mapping[str, Any]:
    payload = {
        "schema": METHOD_LOCK_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "arms": list(ARMS),
        "d103_bundle_content_root_sha256": bundle.content_root_sha256,
        "qknn_config_lock_digest": qknn_config.lock_digest,
        "angq_schema": ANGQ_SCHEMA,
        "factor_grid": [float(value) for value in FACTORS],
        "factor_enumeration": "ascending",
        "factor_tie_break": "stable_first_strict_greater",
        "rounding": "numpy.rint_ties_to_even",
        "scale_dtype": np.dtype(np.float16).str,
        "code_dtype": np.dtype(np.int8).str,
        "input_normalization_count": 1,
        "candidate_input_renormalization_count": 0,
        "k1_bandwidth": "phase1_locked_shared_h0",
        "k5_k10_bandwidth": (
            "angq_decoded_support_same_class_symmetric_formula_fp16"
        ),
        "query_formula": "typed_student_t_all_registered_classes",
        "query_state_updates": 0,
        "query_truth_read": False,
        "old_new_role_access": False,
        "class_quota_access": False,
        "per_query_independent": True,
        "target25_authorized": False,
    }
    payload["method_lock_sha256"] = _canonical_sha256(payload)
    return MappingProxyType(payload)


def _state_receipt(
    *,
    method_lock: Mapping[str, Any],
    d103_state: D103Stage2State,
    m0_bank: TypedINT8ZIDSupportBank,
    m_head_bank: TypedINT8ZIDSupportBank,
    m_joint_bank: TypedINT8ZIDSupportBank,
    metric: TypedSharedPSDMetric,
    support_receipt_sha256: str,
    resource_receipts: Mapping[str, Mapping[str, Any]],
) -> str:
    return _canonical_sha256(
        {
            "schema": SCHEMA,
            "method_lock": method_lock,
            "d103_state_receipt_sha256": d103_state.state_receipt_sha256,
            "arm_bank_receipts": {
                "M0": m0_bank.bank_receipt_sha256,
                "M_DA": d103_state.bank.bank_receipt_sha256,
                "M_HEAD": m_head_bank.bank_receipt_sha256,
                "M_JOINT": m_joint_bank.bank_receipt_sha256,
            },
            "metric_receipt_sha256": metric.metric_receipt_sha256,
            "support_receipt_sha256": support_receipt_sha256,
            "resource_receipts": resource_receipts,
            "query_state_updates": 0,
            "query_rows_used_for_fit": 0,
        }
    )


@dataclass(frozen=True, slots=True)
class D104FourArmState:
    d103_state: D103Stage2State
    m0_bank: TypedINT8ZIDSupportBank
    m_head_bank: TypedINT8ZIDSupportBank
    m_joint_bank: TypedINT8ZIDSupportBank
    metric: TypedSharedPSDMetric
    method_lock: Mapping[str, Any]
    resource_receipts: Mapping[str, Mapping[str, Any]]
    support_receipt_sha256: str
    state_receipt_sha256: str
    query_state_updates: int = 0
    query_rows_used_for_fit: int = 0
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != SCHEMA
            or type(self.d103_state) is not D103Stage2State
            or type(self.m0_bank) is not TypedINT8ZIDSupportBank
            or type(self.m_head_bank) is not TypedINT8ZIDSupportBank
            or type(self.m_joint_bank) is not TypedINT8ZIDSupportBank
            or type(self.metric) is not TypedSharedPSDMetric
            or not self.metric.exact_identity
            or type(self.query_state_updates) is not int
            or self.query_state_updates != 0
            or type(self.query_rows_used_for_fit) is not int
            or self.query_rows_used_for_fit != 0
        ):
            raise D104RXIDANGQError("D104 four-arm state lifecycle/type drift")
        banks = (
            self.m0_bank,
            self.d103_state.bank,
            self.m_head_bank,
            self.m_joint_bank,
        )
        reference = self.m0_bank
        if any(
            bank.classes != reference.classes
            or bank.support_counts != reference.support_counts
            or bank.active_k != reference.active_k
            or bank.config_lock_digest != reference.config_lock_digest
            for bank in banks
        ):
            raise D104RXIDANGQError("D104 four-arm matched bank closure drift")
        if (
            dict(self.m_head_bank.quantization_audit).get("schema") != ANGQ_SCHEMA
            or dict(self.m_joint_bank.quantization_audit).get("schema") != ANGQ_SCHEMA
            or self.metric.config_lock_digest != reference.config_lock_digest
        ):
            raise D104RXIDANGQError("D104 ANGQ head/metric schema drift")
        lock = dict(self.method_lock)
        if (
            lock.get("schema") != METHOD_LOCK_SCHEMA
            or lock.get("candidate_id") != CANDIDATE_ID
            or tuple(lock.get("arms", ())) != ARMS
            or lock.get("qknn_config_lock_digest") != reference.config_lock_digest
            or lock.get("d103_bundle_content_root_sha256")
            != self.d103_state.bundle.content_root_sha256
            or lock.get("query_state_updates") != 0
            or lock.get("query_truth_read") is not False
            or lock.get("target25_authorized") is not False
        ):
            raise D104RXIDANGQError("D104 method lock drift")
        lock_sha256 = lock.pop("method_lock_sha256", None)
        if lock_sha256 != _canonical_sha256(lock):
            raise D104RXIDANGQError("D104 method lock receipt drift")
        resources = {
            str(name): MappingProxyType(dict(receipt))
            for name, receipt in dict(self.resource_receipts).items()
        }
        if set(resources) != {"head_effect", "joint_effect"}:
            raise D104RXIDANGQError("D104 resource receipt pair drift")
        for receipt in resources.values():
            if (
                receipt.get("schema") != RESOURCE_SCHEMA
                or receipt.get("passes_d104_resource_gate") is not True
                or receipt.get("numeric_bank_array_bytes_delta") != 0
                or receipt.get("query_mac_delta") != 0
            ):
                raise D104RXIDANGQError("D104 resource receipt gate drift")
        _require_sha256(self.support_receipt_sha256, "support_receipt_sha256")
        _require_sha256(self.state_receipt_sha256, "state_receipt_sha256")
        expected = _state_receipt(
            method_lock=self.method_lock,
            d103_state=self.d103_state,
            m0_bank=self.m0_bank,
            m_head_bank=self.m_head_bank,
            m_joint_bank=self.m_joint_bank,
            metric=self.metric,
            support_receipt_sha256=self.support_receipt_sha256,
            resource_receipts=resources,
        )
        if expected != self.state_receipt_sha256:
            raise D104RXIDANGQError("D104 state receipt verification failed")
        object.__setattr__(self, "method_lock", MappingProxyType(dict(self.method_lock)))
        object.__setattr__(self, "resource_receipts", MappingProxyType(resources))

    @property
    def classes(self) -> tuple[str, ...]:
        return self.m0_bank.classes

    @property
    def active_k(self) -> int:
        return self.m0_bank.active_k


def fit_d104_four_arm_state(
    bundle: RXIDMetaBias4Bundle,
    support_pre_relu: np.ndarray,
    support_zdom: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    qknn_config: Phase1ZIDStudentTLock,
    stage: str,
    support_receipt_sha256: str,
    k1_identifiability_receipt: K1IdentifiabilityReceipt | None = None,
) -> D104FourArmState:
    """Build all four matched arms using support only."""

    _require_sha256(support_receipt_sha256, "support_receipt_sha256")
    m0_bank, metric, base_support = build_d102_baseline_bank(
        support_pre_relu,
        support_labels,
        registered_classes,
        qknn_config=qknn_config,
    )
    d103_state = fit_d103_stage2_state(
        bundle,
        support_pre_relu,
        support_zdom,
        support_labels,
        registered_classes,
        qknn_config=qknn_config,
        stage=stage,
        support_receipt_sha256=support_receipt_sha256,
        k1_identifiability_receipt=k1_identifiability_receipt,
    )
    m_head_bank = build_d104_angq_support_bank(
        base_support,
        support_labels,
        registered_classes,
        config=qknn_config,
    )
    joint_support = transform_d103_query(d103_state, support_pre_relu)
    m_joint_bank = build_d104_angq_support_bank(
        joint_support,
        support_labels,
        registered_classes,
        config=qknn_config,
    )
    resources = MappingProxyType(
        {
            "head_effect": audit_d104_angq_resource_delta(
                m0_bank,
                m_head_bank,
            ),
            "joint_effect": audit_d104_angq_resource_delta(
                d103_state.bank,
                m_joint_bank,
            ),
        }
    )
    lock = _method_lock(bundle, qknn_config)
    receipt = _state_receipt(
        method_lock=lock,
        d103_state=d103_state,
        m0_bank=m0_bank,
        m_head_bank=m_head_bank,
        m_joint_bank=m_joint_bank,
        metric=metric,
        support_receipt_sha256=support_receipt_sha256,
        resource_receipts=resources,
    )
    return D104FourArmState(
        d103_state=d103_state,
        m0_bank=m0_bank,
        m_head_bank=m_head_bank,
        m_joint_bank=m_joint_bank,
        metric=metric,
        method_lock=lock,
        resource_receipts=resources,
        support_receipt_sha256=support_receipt_sha256,
        state_receipt_sha256=receipt,
    )


def predict_d104_four_arm_logits(
    state: D104FourArmState,
    query_pre_relu: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Score each query independently against every class in all four arms."""

    if type(state) is not D104FourArmState:
        raise D104RXIDANGQError("prediction requires an exact D104 state")
    base_query = baseline_zid_from_pre_relu(query_pre_relu)
    da_query = transform_d103_query(state.d103_state, query_pre_relu)
    result = {
        "M0": score_zid_student_t_logits(
            state.m0_bank,
            base_query,
            metric=state.metric,
        ),
        "M_DA": score_zid_student_t_logits(
            state.d103_state.bank,
            da_query,
            metric=state.metric,
        ),
        "M_HEAD": score_zid_student_t_logits(
            state.m_head_bank,
            base_query,
            metric=state.metric,
        ),
        "M_JOINT": score_zid_student_t_logits(
            state.m_joint_bank,
            da_query,
            metric=state.metric,
        ),
    }
    if any(
        logits.dtype != np.float32
        or logits.ndim != 2
        or logits.shape[1] != len(state.classes)
        or not np.isfinite(logits).all()
        for logits in result.values()
    ):
        raise D104RXIDANGQError("D104 four-arm logit closure drift")
    row_counts = {len(logits) for logits in result.values()}
    if len(row_counts) != 1:
        raise D104RXIDANGQError("D104 four-arm query row-count drift")
    return MappingProxyType(result)


def build_d104_prediction_artifact(
    state: D104FourArmState,
    query_pre_relu: np.ndarray,
    query_physical_ids: Sequence[str],
) -> Mapping[str, Any]:
    """Publish truth-free immutable predictions; no scorer input exists."""

    if type(state) is not D104FourArmState:
        raise D104RXIDANGQError("prediction artifact requires an exact D104 state")
    physical_ids = tuple(str(value) for value in query_physical_ids)
    if len(physical_ids) < 1 or len(set(physical_ids)) != len(physical_ids):
        raise D104RXIDANGQError("query physical IDs must be unique and non-empty")
    logits = predict_d104_four_arm_logits(state, query_pre_relu)
    if any(len(value) != len(physical_ids) for value in logits.values()):
        raise D104RXIDANGQError("query physical ID/logit alignment drift")
    predictions = {
        arm: [
            state.classes[int(index)]
            for index in stable_first_argmax(value, axis=1)
        ]
        for arm, value in logits.items()
    }
    artifact: dict[str, Any] = {
        "schema": PREDICTION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "state_receipt_sha256": state.state_receipt_sha256,
        "method_lock_sha256": state.method_lock["method_lock_sha256"],
        "registered_classes": list(state.classes),
        "active_k": state.active_k,
        "query_physical_ids": list(physical_ids),
        "arm_predictions": predictions,
        "arm_prediction_counts": {
            arm: len(values) for arm, values in predictions.items()
        },
        "all_four_arms_present": tuple(predictions) == ARMS,
        "all_registered_classes_compete": True,
        "stable_class_tie_break": "opaque_registry_index_ascending",
        "per_query_independent": True,
        "query_truth_present": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "old_new_role_access": False,
        "class_quota_access": False,
        "target25_authorized": False,
    }
    artifact["prediction_receipt_sha256"] = _canonical_sha256(artifact)
    return MappingProxyType(artifact)


def _stable_winner_runner(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(logits, dtype=np.float64)
    if (
        scores.ndim != 2
        or scores.shape[0] < 1
        or scores.shape[1] < 2
        or not np.isfinite(scores).all()
    ):
        raise D104RXIDANGQError("winner/runner audit requires finite class scores")
    winner = np.argmax(scores, axis=1)
    masked = scores.copy()
    masked[np.arange(len(masked)), winner] = -np.inf
    runner = np.argmax(masked, axis=1)
    return winner, runner


def _quantization_audit(
    *,
    bank: TypedINT8ZIDSupportBank,
    full_precision_support: np.ndarray,
    support_labels: Sequence[str],
    validation: np.ndarray,
) -> Mapping[str, Any]:
    support = normalize_zid_rows(full_precision_support).astype(np.float64)
    query = normalize_zid_rows(validation).astype(np.float64)
    labels = tuple(str(value) for value in support_labels)
    if len(labels) != len(support) or any(label not in bank.classes for label in labels):
        raise D104RXIDANGQError("D104 INT8 audit support label drift")
    class_map = {label: index for index, label in enumerate(bank.classes)}
    indices = np.asarray([class_map[label] for label in labels], dtype=np.int16)
    counts = tuple(int(np.sum(indices == index)) for index in range(len(bank.classes)))
    if counts != bank.support_counts:
        raise D104RXIDANGQError("D104 INT8 audit support count drift")
    teacher_scales = _identity_class_scales(
        support,
        indices,
        len(bank.classes),
        bank.config,
    )
    teacher_logits = _score_with_support(
        support=support,
        class_indices=indices,
        support_counts=counts,
        class_scales=teacher_scales,
        query=query,
        config=bank.config,
        metric=identity_shared_psd_metric(config=bank.config),
    ).astype(np.float64)
    deployed_logits = score_zid_student_t_logits(
        bank,
        np.asarray(validation),
        metric=identity_shared_psd_metric(config=bank.config),
    ).astype(np.float64)
    winner, runner = _stable_winner_runner(teacher_logits)
    row = np.arange(len(teacher_logits))
    deployed_margin = (
        deployed_logits[row, winner] - deployed_logits[row, runner]
    )
    teacher_top1 = stable_first_argmax(teacher_logits, axis=1)
    deployed_top1 = stable_first_argmax(deployed_logits, axis=1)

    shared_bandwidth_teacher_logits = _score_with_support(
        support=support,
        class_indices=indices,
        support_counts=counts,
        class_scales=bank.class_scales_fp16.astype(np.float64),
        query=query,
        config=bank.config,
        metric=identity_shared_psd_metric(config=bank.config),
    ).astype(np.float64)
    shared_winner, shared_runner = _stable_winner_runner(
        shared_bandwidth_teacher_logits
    )
    shared_margin = (
        deployed_logits[row, shared_winner]
        - deployed_logits[row, shared_runner]
    )
    shared_teacher_top1 = stable_first_argmax(
        shared_bandwidth_teacher_logits,
        axis=1,
    )
    result = {
        "schema": INT8_AUDIT_SCHEMA,
        "validation_row_count": int(len(query)),
        "top1_agreement": float(np.mean(teacher_top1 == deployed_top1)),
        "teacher_winner_margin_flip_count": int(
            np.sum(deployed_margin <= 0.0)
        ),
        "required_top1_agreement": 0.995,
        "passes_end_to_end_gate": bool(
            float(np.mean(teacher_top1 == deployed_top1)) >= 0.995
            and int(np.sum(deployed_margin <= 0.0)) == 0
        ),
        "teacher_bandwidth_source": (
            "fp32_support_same_class_symmetric_formula_fp32"
        ),
        "deployed_bandwidth_source": (
            "angq_decoded_support_same_class_symmetric_formula_fp16"
        ),
        "shared_angq_fp16_bandwidth_direction_audit": {
            "top1_agreement": float(
                np.mean(shared_teacher_top1 == deployed_top1)
            ),
            "teacher_winner_margin_flip_count": int(
                np.sum(shared_margin <= 0.0)
            ),
            "promotion_gate": False,
        },
        "stable_winner_runner_tie_break": "opaque_registry_index_ascending",
        "query_rows_used_for_fit": 0,
        "query_truth_read": False,
        "query_state_updates": 0,
    }
    return MappingProxyType(result)


def audit_d104_four_arm_int8(
    state: D104FourArmState,
    support_pre_relu: np.ndarray,
    support_labels: Sequence[str],
    validation_pre_relu: np.ndarray,
) -> Mapping[str, Any]:
    """Publish independent M_HEAD and M_JOINT teacher/deployed audits."""

    if type(state) is not D104FourArmState:
        raise D104RXIDANGQError("INT8 audit requires an exact D104 state")
    base_support = baseline_zid_from_pre_relu(support_pre_relu)
    base_validation = baseline_zid_from_pre_relu(validation_pre_relu)
    joint_support = transform_d103_query(state.d103_state, support_pre_relu)
    joint_validation = transform_d103_query(
        state.d103_state,
        validation_pre_relu,
    )
    result = {
        "schema": INT8_AUDIT_SCHEMA + ".four_arm",
        "M_HEAD": _quantization_audit(
            bank=state.m_head_bank,
            full_precision_support=base_support,
            support_labels=support_labels,
            validation=base_validation,
        ),
        "M_JOINT": _quantization_audit(
            bank=state.m_joint_bank,
            full_precision_support=joint_support,
            support_labels=support_labels,
            validation=joint_validation,
        ),
        "query_truth_read": False,
        "query_state_updates": 0,
        "target25_authorized": False,
    }
    result["passes_d104_int8_gate"] = bool(
        result["M_HEAD"]["passes_end_to_end_gate"]
        and result["M_JOINT"]["passes_end_to_end_gate"]
    )
    result["receipt_sha256"] = _canonical_sha256(result)
    return MappingProxyType(result)


__all__ = [
    "ARMS",
    "D104FourArmState",
    "D104RXIDANGQError",
    "INT8_AUDIT_SCHEMA",
    "METHOD_LOCK_SCHEMA",
    "PREDICTION_SCHEMA",
    "SCHEMA",
    "audit_d104_four_arm_int8",
    "build_d104_prediction_artifact",
    "fit_d104_four_arm_state",
    "predict_d104_four_arm_logits",
]

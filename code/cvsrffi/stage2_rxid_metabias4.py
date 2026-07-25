"""D103-R1 Stage2 wrapper over the D102 closed form and typed qKNN.

The D102 solver is used only as a transient support-only calculation boundary.
The returned state retains the D103 INT8 bundle, the four binary16
coefficients, and the existing typed INT8 qKNN state.  No decoded learned
array is persisted.  An unidentifiable K1 row remains prediction-complete by
publishing the exact M0 bank and transform under INACTIVE_NON_PROMOTABLE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.rxid_metabias4_bundle import (
    AMAX_FP16,
    CODE_DIM,
    DOMAIN_DIM,
    LAMBDA0_FP16,
    RXIDMetaBias4Bundle,
    Z_DIM,
    serialize_rxid_metabias4_bundle,
)
from cvsrffi.stage2_rb_metabias4_qknn import (
    baseline_zid_from_pre_relu,
    build_d102_baseline_bank,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    TypedINT8ZIDSupportBank,
    TypedSharedPSDMetric,
    audit_int8_margin,
    audit_runtime_state,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    normalize_zid_rows,
    score_zid_student_t_logits,
    serialize_typed_zid_runtime_state,
)


SCHEMA = "cvs.phase2.rxid_dualsplit_metabias4.state.v1"
K1_RECEIPT_SCHEMA = "cvs.phase1.rxid_dualsplit_metabias4.k1_gate.v1"
FIT_AUDIT_SCHEMA = "cvs.phase2.rxid_dualsplit_metabias4.fit_audit.v1"
RESOURCE_AUDIT_SCHEMA = "cvs.phase2.rxid_dualsplit_metabias4.resource_audit.v1"
WIRE_MAGIC = b"CVSD103R1\x00\x01"
ALLOWED_STAGES = ("S_B", "S_C")
ALLOWED_K = (1, 5, 10)
ACTIVE = "ACTIVE"
INACTIVE_NON_PROMOTABLE = "INACTIVE_NON_PROMOTABLE"
MAX_STATE_BYTES = 80 * 1024
MAX_POST_BACKBONE_MAC_PER_QUERY = 262_144


class RXIDMetaBias4Stage2Error(ValueError):
    """Raised when D103-R1 Stage2 protocol or numeric closure drifts."""


def _canonical_bytes(value: Any) -> bytes:
    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): convert(member) for key, member in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(member) for member in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(
        convert(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise RXIDMetaBias4Stage2Error(f"{name} must be a lowercase SHA256")
    return text


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _d102_roundtrip_rows(value: np.ndarray, *, normalize: bool = False) -> np.ndarray:
    """Reproduce D102's reviewed INT8/FP16 conversion boundary exactly."""

    rows = np.asarray(value, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(rows.astype(np.float64), axis=1, keepdims=True)
        if np.any(norms <= 0.0):
            raise RXIDMetaBias4Stage2Error("D102 roundtrip normalization drift")
        rows = np.asarray(rows.astype(np.float64) / norms, dtype=np.float32)
    maximum = np.max(np.abs(rows.astype(np.float64)), axis=1)
    if np.any(maximum <= 1.0e-12):
        raise RXIDMetaBias4Stage2Error("D102 roundtrip contains zero row")
    scales = maximum / 127.0
    codes = np.clip(
        np.rint(rows.astype(np.float64) / scales[:, None]),
        -127.0,
        127.0,
    ).astype(np.int8)
    decoded = np.asarray(
        codes.astype(np.float32) * scales.astype(np.float16).astype(np.float32)[:, None],
        dtype=np.float32,
    )
    if normalize:
        norms = np.linalg.norm(decoded.astype(np.float64), axis=1, keepdims=True)
        decoded = np.asarray(decoded.astype(np.float64) / norms, dtype=np.float32)
    return decoded


def _array_receipt(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


@dataclass(frozen=True, slots=True)
class K1IdentifiabilityReceipt:
    """Frozen Phase1-held non-query evidence for the non-numeric K1 gates."""

    view_top1_agreement: float
    large_margin_flip_count: int
    independent_direction_cosine_median: float
    independent_episode_count: int
    receipt_sha256: str
    query_rows_used_for_fit: int = 0
    evidence_scope: str = "support_only_no_held_query"
    schema: str = K1_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != K1_RECEIPT_SCHEMA
            or not math.isfinite(float(self.view_top1_agreement))
            or not 0.0 <= float(self.view_top1_agreement) <= 1.0
            or type(self.large_margin_flip_count) is not int
            or self.large_margin_flip_count < 0
            or type(self.independent_episode_count) is not int
            or self.independent_episode_count < 2
            or not math.isfinite(float(self.independent_direction_cosine_median))
            or not -1.0 <= float(self.independent_direction_cosine_median) <= 1.0
            or type(self.query_rows_used_for_fit) is not int
            or self.query_rows_used_for_fit != 0
            or self.evidence_scope != "support_only_no_held_query"
        ):
            raise RXIDMetaBias4Stage2Error("K1 identifiability receipt drift")
        _require_sha256(self.receipt_sha256, "K1 receipt_sha256")

    @property
    def passes(self) -> bool:
        return bool(
            float(self.view_top1_agreement) >= 0.995
            and self.large_margin_flip_count == 0
            and float(self.independent_direction_cosine_median) >= 0.80
        )


@dataclass(frozen=True, slots=True)
class D103CoefficientSolution:
    """Support-only analytic solution exposed for leave-day falsification."""

    coefficient_fp16: np.ndarray
    fit_audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        coefficient = np.asarray(self.coefficient_fp16)
        audit = dict(self.fit_audit)
        if (
            coefficient.dtype != np.float16
            or coefficient.shape != (CODE_DIM,)
            or not np.isfinite(coefficient).all()
            or audit.get("query_rows_used_for_fit") != 0
            or audit.get("support_only") is not True
        ):
            raise RXIDMetaBias4Stage2Error("coefficient solution contract drift")
        object.__setattr__(
            self, "coefficient_fp16", _readonly(coefficient, np.float16)
        )
        object.__setattr__(self, "fit_audit", MappingProxyType(audit))


def _validate_frozen_qknn(config: Phase1ZIDStudentTLock) -> None:
    if type(config) is not Phase1ZIDStudentTLock:
        raise RXIDMetaBias4Stage2Error("D103 requires an exact typed qKNN lock")
    exact = (
        config.active_k in ALLOWED_K
        and float(config.student_nu) == 3.0
        and int(config.kernel_effective_dim) == 160
        and float(config.kernel_volume_gamma) == 1.0
        and float(config.shared_h0) == 0.2
        and float(config.scale_prior_strength) == 2.0
        and float(config.scale_min_ratio) == 0.5
        and float(config.scale_max_ratio) == 2.0
        and float(config.temperature) == 1.0
    )
    if not exact:
        raise RXIDMetaBias4Stage2Error("typed qKNN constants drift from D103 §7")


def solve_d103_support_coefficient(
    bundle: RXIDMetaBias4Bundle,
    support_zdom: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    active_k: int,
) -> D103CoefficientSolution:
    """Solve the frozen D102 analytic equation without a deployment-gate shortcut.

    This support-only calculation is deliberately available even when the
    Phase1 TX probe fails.  It permits a prediction-complete held diagnostic,
    while deployment serialization remains fail-closed.
    """

    if type(bundle) is not RXIDMetaBias4Bundle:
        raise RXIDMetaBias4Stage2Error("coefficient solve requires exact bundle")
    if type(active_k) is not int or active_k not in ALLOWED_K:
        raise RXIDMetaBias4Stage2Error("active_k drift")
    zdom = np.asarray(support_zdom)
    if (
        zdom.dtype != np.float32
        or zdom.ndim != 2
        or zdom.shape[1] != Z_DIM
        or len(zdom) < 1
        or not np.isfinite(zdom).all()
    ):
        raise RXIDMetaBias4Stage2Error("support z_dom contract drift")
    labels = tuple(str(value) for value in support_labels)
    classes = tuple(str(value) for value in registered_classes)
    if (
        len(classes) < 2
        or len(set(classes)) != len(classes)
        or len(labels) != len(zdom)
        or any(label not in classes for label in labels)
        or any(labels.count(label) != active_k for label in classes)
    ):
        raise RXIDMetaBias4Stage2Error(
            "support labels require balanced K-shot opaque registry"
        )

    u = _d102_roundtrip_rows(bundle.decode_u()).astype(np.float64)
    representation = zdom.astype(np.float64) @ u.T
    norms = np.linalg.norm(representation, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise RXIDMetaBias4Stage2Error("encoded support domain norm drift")
    representation /= norms
    bank_g = _d102_roundtrip_rows(
        bundle.decode_bank_g(), normalize=True
    ).astype(np.float64)
    similarity = np.clip(representation @ bank_g.T, -1.0, 1.0)
    scaled = similarity / float(bundle.temperature_fp16)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    weights = np.exp(scaled)
    weights /= np.sum(weights, axis=1, keepdims=True)
    sigma = np.asarray(bundle.decode_bank_sigma(), dtype=np.float16).astype(
        np.float64
    )
    coverage = np.sum(
        weights * np.exp(-(1.0 - similarity) / (sigma[None, :] ** 2)),
        axis=1,
    )
    precision = coverage[:, None] * (
        weights
        @ np.asarray(bundle.decode_bank_precision(), dtype=np.float16).astype(
            np.float64
        )
    )
    means = weights @ np.asarray(
        bundle.decode_bank_t(), dtype=np.float16
    ).astype(np.float64)
    if (
        not np.isfinite(weights).all()
        or not np.isfinite(coverage).all()
        or np.any(coverage <= 0.0)
        or np.any(coverage > 1.0 + 1.0e-12)
        or not np.isfinite(precision).all()
        or np.any(precision <= 0.0)
    ):
        raise RXIDMetaBias4Stage2Error("support domain encoding closure failed")

    class_precision = []
    class_weighted_mean = []
    for label in classes:
        mask = np.asarray([value == label for value in labels])
        class_precision.append(np.mean(precision[mask], axis=0))
        class_weighted_mean.append(
            np.mean(precision[mask] * means[mask], axis=0)
        )
    a_data = np.mean(np.stack(class_precision), axis=0)
    b_data = np.mean(np.stack(class_weighted_mean), axis=0)
    lambda0 = LAMBDA0_FP16.astype(np.float64)
    system = lambda0 + a_data
    if np.any(system <= 0.0) or not np.isfinite(system).all():
        raise RXIDMetaBias4Stage2Error("analytic support system drift")
    a_tilde = b_data / system
    limits = AMAX_FP16.astype(np.float64)
    a_box = np.clip(a_tilde, -limits, limits)
    radius = float(bundle.radius_fp16)
    quadratic = float(np.sum(lambda0 * a_box * a_box))
    if quadratic > radius * radius:
        coefficient = radius * a_box / math.sqrt(quadratic)
    else:
        coefficient = a_box
    coefficient_fp16 = np.asarray(coefficient, dtype=np.float16)
    deployed = coefficient_fp16.astype(np.float64)
    if (
        not np.isfinite(deployed).all()
        or np.any(np.abs(deployed) > limits + 1.0e-6)
        or float(np.sum(lambda0 * deployed * deployed))
        > radius * radius + 1.0e-5
    ):
        raise RXIDMetaBias4Stage2Error("deployed coefficient constraint drift")
    audit = {
        "support_only": True,
        "query_rows_used_for_fit": 0,
        "registered_class_count": len(classes),
        "support_row_count": len(zdom),
        "active_k": active_k,
        "data_information_rank": int(
            np.linalg.matrix_rank(np.diag(a_data), tol=1.0e-10)
        ),
        "system_eigenvalue_min": float(np.min(system)),
        "system_eigenvalue_max": float(np.max(system)),
        "system_condition_number": float(np.max(system) / np.min(system)),
        "prior_fraction": float(np.sum(lambda0) / np.sum(system)),
        "a_deployed_norm": float(np.linalg.norm(deployed)),
        "coverage_min": float(np.min(coverage)),
        "coverage_mean": float(np.mean(coverage)),
        "coverage_max": float(np.max(coverage)),
        "tx_probe_gate_pass": bundle.tx_probe_gate_pass,
        "deployment_serialization_authorized": bundle.tx_probe_gate_pass,
    }
    return D103CoefficientSolution(
        coefficient_fp16=coefficient_fp16,
        fit_audit=audit,
    )


def _apply_d103_metabias(
    bundle: RXIDMetaBias4Bundle,
    pre_relu: np.ndarray,
    coefficient_fp16: np.ndarray,
) -> np.ndarray:
    pre = np.asarray(pre_relu)
    code = np.asarray(coefficient_fp16)
    if (
        pre.dtype != np.float32
        or pre.ndim != 2
        or pre.shape[1] != Z_DIM
        or len(pre) < 1
        or not np.isfinite(pre).all()
        or code.dtype != np.float16
        or code.shape != (CODE_DIM,)
        or not np.isfinite(code).all()
    ):
        raise RXIDMetaBias4Stage2Error("D103 pre-ReLU/coefficient contract drift")
    bias = bundle.decode_b().astype(np.float64) @ code.astype(np.float64)
    shifted = np.maximum(pre.astype(np.float64) + bias[None, :], 0.0)
    return normalize_zid_rows(np.ascontiguousarray(shifted, dtype=np.float32))


def stable_first_argmax(value: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Return the first maximum, implementing the frozen ascending-index tie rule."""

    array = np.asarray(value)
    if array.size < 1 or not np.isfinite(array).all():
        raise RXIDMetaBias4Stage2Error("stable argmax requires finite non-empty values")
    return np.argmax(array, axis=axis)


@dataclass(frozen=True, slots=True)
class D103Stage2State:
    bundle: RXIDMetaBias4Bundle
    stage: str
    coefficient_fp16: np.ndarray
    bank: TypedINT8ZIDSupportBank
    metric: TypedSharedPSDMetric
    support_receipt_sha256: str
    status: str
    fit_audit: Mapping[str, Any]
    state_receipt_sha256: str
    query_state_updates: int = 0
    query_rows_used_for_fit: int = 0
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        coefficient = np.asarray(self.coefficient_fp16)
        audit = dict(self.fit_audit)
        if (
            self.schema != SCHEMA
            or type(self.bundle) is not RXIDMetaBias4Bundle
            or self.stage not in ALLOWED_STAGES
            or coefficient.dtype != np.float16
            or coefficient.shape != (CODE_DIM,)
            or not np.isfinite(coefficient).all()
            or type(self.bank) is not TypedINT8ZIDSupportBank
            or type(self.metric) is not TypedSharedPSDMetric
            or not self.metric.exact_identity
            or self.bank.config_lock_digest != self.metric.config_lock_digest
            or self.status not in (ACTIVE, INACTIVE_NON_PROMOTABLE)
            or type(self.query_state_updates) is not int
            or self.query_state_updates != 0
            or type(self.query_rows_used_for_fit) is not int
            or self.query_rows_used_for_fit != 0
            or audit.get("status") != self.status
            or audit.get("query_rows_used_for_fit") != 0
        ):
            raise RXIDMetaBias4Stage2Error("D103 state lifecycle/type drift")
        if self.status == INACTIVE_NON_PROMOTABLE and np.any(coefficient != 0.0):
            raise RXIDMetaBias4Stage2Error("inactive D103 state must carry zero code")
        _require_sha256(self.support_receipt_sha256, "support_receipt_sha256")
        _require_sha256(self.state_receipt_sha256, "state_receipt_sha256")
        expected = _state_receipt(
            self.bundle,
            self.stage,
            coefficient,
            self.bank,
            self.metric,
            self.support_receipt_sha256,
            self.status,
            audit,
        )
        if expected != self.state_receipt_sha256:
            raise RXIDMetaBias4Stage2Error("D103 state receipt verification failed")
        object.__setattr__(
            self, "coefficient_fp16", _readonly(coefficient, np.float16)
        )
        object.__setattr__(self, "fit_audit", MappingProxyType(audit))

    @property
    def active(self) -> bool:
        return self.status == ACTIVE


def _state_receipt(
    bundle: RXIDMetaBias4Bundle,
    stage: str,
    coefficient: np.ndarray,
    bank: TypedINT8ZIDSupportBank,
    metric: TypedSharedPSDMetric,
    support_receipt_sha256: str,
    status: str,
    fit_audit: Mapping[str, Any],
) -> str:
    qwire = serialize_typed_zid_runtime_state(bank, metric)
    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema": SCHEMA,
                "bundle_content_root_sha256": bundle.content_root_sha256,
                "stage": stage,
                "coefficient_fp16": _array_receipt(coefficient),
                "qknn_wire_sha256": _sha256_bytes(qwire),
                "support_receipt_sha256": support_receipt_sha256,
                "status": status,
                "fit_audit": fit_audit,
                "query_state_updates": 0,
                "query_rows_used_for_fit": 0,
            }
        )
    )


def fit_d103_stage2_state(
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
) -> D103Stage2State:
    """Fit support-only D103; an inactive K1 row returns a complete M0 state."""

    if type(bundle) is not RXIDMetaBias4Bundle:
        raise RXIDMetaBias4Stage2Error("fit requires an exact D103 bundle")
    if stage not in ALLOWED_STAGES:
        raise RXIDMetaBias4Stage2Error("stage must be S_B or S_C")
    _require_sha256(support_receipt_sha256, "support_receipt_sha256")
    _validate_frozen_qknn(qknn_config)
    if (
        k1_identifiability_receipt is not None
        and type(k1_identifiability_receipt) is not K1IdentifiabilityReceipt
    ):
        raise RXIDMetaBias4Stage2Error("K1 gate requires an exact typed receipt")

    baseline_bank, baseline_metric, _ = build_d102_baseline_bank(
        support_pre_relu,
        support_labels,
        registered_classes,
        qknn_config=qknn_config,
    )
    solution = solve_d103_support_coefficient(
        bundle,
        support_zdom,
        support_labels,
        registered_classes,
        active_k=qknn_config.active_k,
    )
    d102_audit = dict(solution.fit_audit)
    data_minimum_singular = (
        float(d102_audit["system_eigenvalue_min"])
        - float(np.min(LAMBDA0_FP16.astype(np.float64)))
    )
    numeric_gates = {
        "rank_4": int(d102_audit["data_information_rank"]) == CODE_DIM,
        "min_singular_ge_0_05": data_minimum_singular >= 0.05,
        "condition_le_10": float(d102_audit["system_condition_number"]) <= 10.0,
        "prior_fraction_le_0_80": float(d102_audit["prior_fraction"]) <= 0.80,
        "coefficient_norm_ge_1e_4": float(d102_audit["a_deployed_norm"]) >= 1.0e-4,
    }
    external_gates = {
        "receipt_present": k1_identifiability_receipt is not None,
        "view_top1_agreement_ge_0_995": bool(
            k1_identifiability_receipt is not None
            and k1_identifiability_receipt.view_top1_agreement >= 0.995
        ),
        "large_margin_flip_zero": bool(
            k1_identifiability_receipt is not None
            and k1_identifiability_receipt.large_margin_flip_count == 0
        ),
        "direction_cosine_median_ge_0_80": bool(
            k1_identifiability_receipt is not None
            and k1_identifiability_receipt.independent_direction_cosine_median
            >= 0.80
        ),
    }
    k1_active = bool(all(numeric_gates.values()) and all(external_gates.values()))
    active = qknn_config.active_k != 1 or k1_active

    if active:
        coefficient = _readonly(solution.coefficient_fp16, np.float16)
        adapted = _apply_d103_metabias(bundle, support_pre_relu, coefficient)
        bank = build_typed_zid_support_bank(
            adapted,
            support_labels,
            registered_classes,
            config=qknn_config,
        )
        metric = identity_shared_psd_metric(config=qknn_config)
        status = ACTIVE
    else:
        coefficient = _readonly(np.zeros(CODE_DIM), np.float16)
        bank = baseline_bank
        metric = baseline_metric
        status = INACTIVE_NON_PROMOTABLE

    audit: dict[str, Any] = {
        "schema": FIT_AUDIT_SCHEMA,
        "stage": stage,
        "active_k": qknn_config.active_k,
        "status": status,
        "d102_closed_form_reused": True,
        "typed_qknn_reused": True,
        "data_information_rank": int(d102_audit["data_information_rank"]),
        "data_minimum_singular_value": data_minimum_singular,
        "system_condition_number": float(d102_audit["system_condition_number"]),
        "prior_fraction": float(d102_audit["prior_fraction"]),
        "coefficient_norm": (
            float(d102_audit["a_deployed_norm"]) if active else 0.0
        ),
        "k1_numeric_gates": numeric_gates,
        "k1_external_gates": external_gates,
        "k1_receipt_evidence_scope": (
            k1_identifiability_receipt.evidence_scope
            if k1_identifiability_receipt is not None
            else None
        ),
        "inactive_prediction_path": (
            "exact_M0_full_all_registered_classes"
            if status == INACTIVE_NON_PROMOTABLE
            else None
        ),
        "inactive_fold_counts_as_success": False,
        "d103_instance_rejected": status == INACTIVE_NON_PROMOTABLE,
        "query_rows_used_for_fit": 0,
        "query_truth_read": False,
        "target_rows_read": 0,
        "target25_authorized": False,
        "target25_gate_eligible_claimed": False,
        "old_new_role_access": False,
        "class_quota_access": False,
        "all_registered_classes_compete": True,
        "stable_bank_tie_break": "sealed_bank_index_ascending",
        "stable_support_tie_break": "support_enrollment_index_ascending",
        "stable_class_tie_break": "opaque_registry_index_ascending",
        "persistent_decoded_learning_arrays": False,
    }
    receipt = _state_receipt(
        bundle,
        stage,
        coefficient,
        bank,
        metric,
        support_receipt_sha256,
        status,
        audit,
    )
    return D103Stage2State(
        bundle=bundle,
        stage=stage,
        coefficient_fp16=coefficient,
        bank=bank,
        metric=metric,
        support_receipt_sha256=support_receipt_sha256,
        status=status,
        fit_audit=audit,
        state_receipt_sha256=receipt,
    )


def transform_d103_query(
    state: D103Stage2State, query_pre_relu: np.ndarray
) -> np.ndarray:
    """Read-only per-query transform with exact M0 closure when inactive."""

    if type(state) is not D103Stage2State:
        raise RXIDMetaBias4Stage2Error("query transform requires an exact D103 state")
    if state.status == INACTIVE_NON_PROMOTABLE:
        return baseline_zid_from_pre_relu(query_pre_relu)
    return _apply_d103_metabias(
        state.bundle, query_pre_relu, state.coefficient_fp16
    )


def predict_d103_logits(
    state: D103Stage2State, query_pre_relu: np.ndarray
) -> np.ndarray:
    """Return read-only logits for every opaque registered class."""

    query = transform_d103_query(state, query_pre_relu)
    return score_zid_student_t_logits(state.bank, query, metric=state.metric)


def predict_d103_class_indices(
    state: D103Stage2State, query_pre_relu: np.ndarray
) -> np.ndarray:
    """Return the lowest opaque registry index on an exact class-score tie."""

    return stable_first_argmax(predict_d103_logits(state, query_pre_relu), axis=1)


def serialize_d103_runtime_state(state: D103Stage2State) -> bytes:
    if type(state) is not D103Stage2State:
        raise RXIDMetaBias4Stage2Error("serialization requires an exact D103 state")
    bundle_wire = serialize_rxid_metabias4_bundle(state.bundle)
    qknn_wire = serialize_typed_zid_runtime_state(state.bank, state.metric)
    header = _canonical_bytes(
        {
            "schema": state.schema,
            "state_receipt_sha256": state.state_receipt_sha256,
            "support_receipt_sha256": state.support_receipt_sha256,
            "status": state.status,
            "bundle_wire_sha256": _sha256_bytes(bundle_wire),
            "qknn_wire_sha256": _sha256_bytes(qknn_wire),
            "coefficient_fp16": _array_receipt(state.coefficient_fp16),
            "fit_audit": state.fit_audit,
            "query_state_updates": 0,
            "query_rows_used_for_fit": 0,
            "persistent_fp16_or_fp32_learned_sidecar": False,
        }
    )
    return b"".join(
        (
            WIRE_MAGIC,
            struct.pack("<Q", len(header)),
            header,
            struct.pack("<Q", len(bundle_wire)),
            bundle_wire,
            struct.pack("<Q", len(qknn_wire)),
            qknn_wire,
            state.coefficient_fp16.tobytes(order="C"),
        )
    )


def audit_d103_int8(
    state: D103Stage2State,
    full_precision_support_pre_relu: np.ndarray,
    support_labels: Sequence[str],
    validation_pre_relu: np.ndarray,
) -> dict[str, Any]:
    """Apply the frozen 99.5%/zero-flip teacher-student gate without truth."""

    if type(state) is not D103Stage2State:
        raise RXIDMetaBias4Stage2Error("INT8 audit requires an exact D103 state")
    if state.status == INACTIVE_NON_PROMOTABLE:
        support = baseline_zid_from_pre_relu(full_precision_support_pre_relu)
        validation = baseline_zid_from_pre_relu(validation_pre_relu)
    else:
        support = _apply_d103_metabias(
            state.bundle,
            full_precision_support_pre_relu,
            state.coefficient_fp16,
        )
        validation = _apply_d103_metabias(
            state.bundle, validation_pre_relu, state.coefficient_fp16
        )
    base = audit_int8_margin(
        state.bank,
        support,
        support_labels,
        validation,
        metric=state.metric,
    )
    result = dict(base)
    result.update(
        {
            "required_top1_agreement": 0.995,
            "large_margin_flip_count": int(base["margin_sign_flip_count"]),
            "passes_d103_int8_gate": (
                float(base["top1_agreement"]) >= 0.995
                and int(base["margin_sign_flip_count"]) == 0
            ),
            "query_truth_read": False,
            "target25_authorized": False,
        }
    )
    return result


def audit_d103_resources(state: D103Stage2State) -> dict[str, Any]:
    if type(state) is not D103Stage2State:
        raise RXIDMetaBias4Stage2Error("resource audit requires an exact D103 state")
    qknn = audit_runtime_state(state.bank, state.metric)
    query_mac = Z_DIM * CODE_DIM + int(
        qknn["score_query_variable_matmul_mac_per_query"]
    )
    if state.status == INACTIVE_NON_PROMOTABLE:
        query_mac -= Z_DIM * CODE_DIM
    deployment_authorized = state.bundle.tx_probe_gate_pass
    state_bytes = (
        len(serialize_d103_runtime_state(state))
        if deployment_authorized
        else None
    )
    return {
        "schema": RESOURCE_AUDIT_SCHEMA,
        "status": state.status,
        "actual_serialized_state_bytes": state_bytes,
        "numeric_bundle_state_bytes": state.bundle.numeric_state_bytes,
        "fp32_persistent_sidecar_bytes": 0,
        "fp16_persistent_learned_sidecar_bytes": 0,
        "trainable_parameters_stage2": 0,
        "optimizer_steps_stage2": 0,
        "post_backbone_mac_per_query": query_mac,
        "state_gate_bytes": MAX_STATE_BYTES,
        "query_gate_mac": MAX_POST_BACKBONE_MAC_PER_QUERY,
        "passes_state_gate": bool(
            deployment_authorized
            and state_bytes is not None
            and state_bytes < MAX_STATE_BYTES
        ),
        "passes_query_mac_gate": query_mac <= MAX_POST_BACKBONE_MAC_PER_QUERY,
        "tx_probe_gate_pass": state.bundle.tx_probe_gate_pass,
        "deployment_serialization_authorized": deployment_authorized,
        "query_state_updates": 0,
        "query_batch_dependency": False,
    }


__all__ = [
    "ACTIVE",
    "ALLOWED_K",
    "ALLOWED_STAGES",
    "D103Stage2State",
    "D103CoefficientSolution",
    "INACTIVE_NON_PROMOTABLE",
    "K1IdentifiabilityReceipt",
    "MAX_POST_BACKBONE_MAC_PER_QUERY",
    "MAX_STATE_BYTES",
    "RXIDMetaBias4Stage2Error",
    "audit_d103_int8",
    "audit_d103_resources",
    "fit_d103_stage2_state",
    "predict_d103_class_indices",
    "predict_d103_logits",
    "serialize_d103_runtime_state",
    "solve_d103_support_coefficient",
    "stable_first_argmax",
    "transform_d103_query",
]

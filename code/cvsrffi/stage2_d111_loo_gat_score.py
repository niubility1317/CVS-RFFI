"""Query-independent D111 LOO domain-anchor transport over the frozen M0 head."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from cvsrffi.stage2_d111_loo_gat_bundle import D111Bundle, FEATURE_DIM
from cvsrffi.stage2_zid_student_t_qknn import (
    TypedINT8ZIDSupportBank,
    decode_zid_support_bank,
    identity_shared_psd_metric,
    normalize_zid_rows,
    score_zid_student_t_logits,
)


SCHEMA = "cvs.stage2.d111.loo_gat.score_state.v1"
WEISZFELD_STEPS = 32
WEISZFELD_DAMPING = 0.5
NUMERIC_EPSILON = 1.0e-12
EXPECTED_OLD_CLASS_COUNT = 6


class D111ScoreError(ValueError):
    """Raised when D111 scoring cannot preserve its frozen semantics."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    frozen.setflags(write=False)
    return frozen


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _normalize_vector(value: np.ndarray, field: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (FEATURE_DIM,) or not np.isfinite(vector).all() or norm <= NUMERIC_EPSILON:
        raise D111ScoreError(f"{field} must be one finite non-zero z_id vector")
    return vector / norm


def _canonical_rows(value: np.ndarray) -> np.ndarray:
    rows = [np.asarray(row, dtype=np.float64).copy() for row in value]
    rows.sort(key=lambda row: np.ascontiguousarray(row).tobytes(order="C"))
    return np.asarray(rows, dtype=np.float64)


def _weiszfeld_certificate(points: np.ndarray) -> tuple[np.ndarray, float, float, float, bool]:
    """Return a fixed-step estimate and a feasible geometric-median dual bound."""

    rows = _canonical_rows(points)
    if rows.shape != (EXPECTED_OLD_CLASS_COUNT - 1, 3) or not np.isfinite(rows).all():
        raise D111ScoreError("LOO residual set must be finite [5,3]")
    estimate = np.mean(rows, axis=0)
    for _ in range(WEISZFELD_STEPS):
        distance = np.linalg.norm(rows - estimate[None, :], axis=1)
        weight = 1.0 / np.maximum(distance, NUMERIC_EPSILON)
        target = np.sum(weight[:, None] * rows, axis=0) / np.sum(weight)
        estimate = (
            (1.0 - WEISZFELD_DAMPING) * estimate
            + WEISZFELD_DAMPING * target
        )

    difference = rows - estimate[None, :]
    distance = np.linalg.norm(difference, axis=1)
    primal = float(np.sum(distance))
    direction = difference / np.maximum(distance[:, None], NUMERIC_EPSILON)
    dual_vector = direction - np.mean(direction, axis=0, keepdims=True)
    maximum_norm = float(np.max(np.linalg.norm(dual_vector, axis=1)))
    dual_vector /= max(1.0, maximum_norm)
    dual = float(np.sum(dual_vector * rows))
    sum_constraint = float(np.linalg.norm(np.sum(dual_vector, axis=0)))
    norm_constraint = float(np.max(np.linalg.norm(dual_vector, axis=1)))
    feasible = bool(sum_constraint <= 1.0e-10 and norm_constraint <= 1.0 + 1.0e-12)
    gap = primal - dual
    if not all(math.isfinite(value) for value in (primal, dual, gap)) or gap < -1.0e-9:
        feasible = False
        gap = math.inf
    return estimate, primal, dual, max(0.0, gap), feasible


@dataclass(frozen=True, slots=True)
class D111LOOGATState:
    classes: tuple[str, ...]
    old_class_indices: tuple[int, ...]
    anchors: np.ndarray
    rho: np.ndarray
    qualified: np.ndarray
    iterations: np.ndarray
    primal: np.ndarray
    dual: np.ndarray
    gap: np.ndarray
    consensus_count: np.ndarray
    bank_receipt_sha256: str
    config_lock_digest: str
    bundle_content_root_sha256: str
    resource_receipt: Mapping[str, int]
    state_receipt_sha256: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        class_count = len(self.classes)
        expected_vector = (class_count,)
        if (
            self.schema != SCHEMA
            or len(self.old_class_indices) != EXPECTED_OLD_CLASS_COUNT
            or self.anchors.shape != (class_count, FEATURE_DIM)
            or self.rho.shape != expected_vector
            or self.qualified.shape != expected_vector
            or self.iterations.shape != expected_vector
            or self.primal.shape != expected_vector
            or self.dual.shape != expected_vector
            or self.gap.shape != expected_vector
            or self.consensus_count.shape != expected_vector
        ):
            raise D111ScoreError("D111 state shape/schema drift")
        if any(array.flags.writeable for array in (
            self.anchors,
            self.rho,
            self.qualified,
            self.iterations,
            self.primal,
            self.dual,
            self.gap,
            self.consensus_count,
        )):
            raise D111ScoreError("D111 state arrays must be deeply readonly")


def _state_payload(state: D111LOOGATState) -> dict[str, Any]:
    return {
        "schema": state.schema,
        "classes": list(state.classes),
        "old_class_indices": list(state.old_class_indices),
        "bank_receipt_sha256": state.bank_receipt_sha256,
        "config_lock_digest": state.config_lock_digest,
        "bundle_content_root_sha256": state.bundle_content_root_sha256,
        "arrays": {
            name: _array_receipt(getattr(state, name))
            for name in (
                "anchors",
                "rho",
                "qualified",
                "iterations",
                "primal",
                "dual",
                "gap",
                "consensus_count",
            )
        },
        "resource_receipt": dict(state.resource_receipt),
        "query_rows_used_for_fit": 0,
        "truth_role_quota_inputs": 0,
    }


def _verify_state(state: D111LOOGATState) -> None:
    if _canonical_sha256(_state_payload(state)) != state.state_receipt_sha256:
        raise D111ScoreError("D111 state receipt verification failed")
    old = np.asarray(state.old_class_indices, dtype=np.int64)
    if (
        len(set(state.classes)) != len(state.classes)
        or len(set(state.old_class_indices)) != EXPECTED_OLD_CLASS_COUNT
        or np.any(old < 0)
        or np.any(old >= len(state.classes))
        or np.any(state.rho < 0.0)
        or np.any(state.rho >= 1.0)
        or np.any((state.rho > 0.0) != state.qualified)
        or np.any(state.iterations[old] != WEISZFELD_STEPS)
        or np.any(state.consensus_count[old] < 0)
        or np.any(state.consensus_count[old] > EXPECTED_OLD_CLASS_COUNT - 1)
    ):
        raise D111ScoreError("D111 state semantic invariant drift")


def fit_d111_loo_gat_state(
    bundle: D111Bundle, bank: TypedINT8ZIDSupportBank
) -> D111LOOGATState:
    """Fit one immutable support-only state; no query or truth enters this API."""

    if type(bundle) is not D111Bundle or type(bank) is not TypedINT8ZIDSupportBank:
        raise D111ScoreError("D111 fit requires exact bundle and qKNN bank types")
    if float(bank.config.kernel_volume_gamma) != 1.0:
        raise D111ScoreError("D111 unit-mass density requires kernel_volume_gamma=1")
    if (
        bundle.manifest.get("effective_formal_phase2_eligible") is not True
        or bundle.manifest.get("effective_bundle_state")
        != "FORMAL_D111_OUTER_JOINT_SEALED"
    ):
        raise D111ScoreError("D111 fit requires a verified formal outer-sealed bundle")
    old_classes = tuple(bundle.class_registry)
    if len(old_classes) != EXPECTED_OLD_CLASS_COUNT:
        raise D111ScoreError("D111 LOO certificate is frozen for exactly six old classes")
    if any(name not in bank.classes for name in old_classes):
        raise D111ScoreError("D111 old registry is not contained in the support bank")
    old_indices = tuple(bank.classes.index(name) for name in old_classes)
    support = normalize_zid_rows(decode_zid_support_bank(bank).astype(np.float32)).astype(
        np.float64
    )
    means: list[np.ndarray] = []
    class_support: list[np.ndarray] = []
    for bank_index in old_indices:
        local = support[bank.class_indices_int16 == bank_index]
        if len(local) != bank.active_k:
            raise D111ScoreError("D111 support count drift")
        class_support.append(local)
        means.append(_normalize_vector(np.mean(local, axis=0), "old support mean"))
    mean_matrix = np.asarray(means)
    basis = np.asarray(bundle.basis, dtype=np.float64)
    source_anchor = np.asarray(bundle.anchors, dtype=np.float64)
    source_variance = np.asarray(bundle.v_g, dtype=np.float64)
    if basis.shape != (3, FEATURE_DIM) or source_anchor.shape != (
        EXPECTED_OLD_CLASS_COUNT,
        FEATURE_DIM,
    ) or source_variance.shape != (EXPECTED_OLD_CLASS_COUNT,):
        raise D111ScoreError("D111 bundle geometry drift")
    if (
        not np.isfinite(basis).all()
        or not np.isfinite(source_anchor).all()
        or not np.isfinite(source_variance).all()
        or np.any(source_variance <= 0.0)
        or not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in (bundle.v_s, bundle.envelope_b, bundle.epsilon)
        )
    ):
        raise D111ScoreError("D111 bundle numeric receipt drift")
    residual = (mean_matrix - source_anchor) @ basis.T

    class_count = len(bank.classes)
    anchors = np.zeros((class_count, FEATURE_DIM), dtype=np.float64)
    rho = np.zeros(class_count, dtype=np.float64)
    qualified = np.zeros(class_count, dtype=np.bool_)
    iterations = np.zeros(class_count, dtype=np.int16)
    primal = np.zeros(class_count, dtype=np.float64)
    dual = np.zeros(class_count, dtype=np.float64)
    gap = np.zeros(class_count, dtype=np.float64)
    consensus = np.zeros(class_count, dtype=np.int16)

    for old_position, bank_index in enumerate(old_indices):
        loo = np.delete(residual, old_position, axis=0)
        centre, local_primal, local_dual, local_gap, dual_feasible = _weiszfeld_certificate(loo)
        transported = _normalize_vector(
            source_anchor[old_position] + basis.T @ centre,
            "transported D111 anchor",
        )
        local_consensus = int(
            np.sum(np.linalg.norm(loo - centre[None, :], axis=1) <= bundle.envelope_b)
        )
        is_qualified = bool(
            dual_feasible
            and local_gap <= bundle.epsilon
            and local_consensus >= 3
        )
        local = class_support[old_position]
        if bank.active_k == 1:
            target_variance = float(bundle.v_s)
        else:
            # Frozen S_c^2: unbiased per-coordinate chord scatter around the
            # normalized support mean m_c used by the transport equation.
            sample_variance = float(
                np.sum(np.square(local - mean_matrix[old_position]))
                / ((bank.active_k - 1) * FEATURE_DIM)
            )
            target_variance = max(
                float(bundle.v_s) / bank.active_k,
                sample_variance / bank.active_k,
            )
        anchor_variance = float(source_variance[old_position]) + (
            6.0 * float(bundle.envelope_b) + float(bundle.epsilon)
        ) ** 2 / FEATURE_DIM
        discrepancy = float(
            np.sum(np.square(mean_matrix[old_position] - transported)) / FEATURE_DIM
        )
        denominator = target_variance + anchor_variance + discrepancy
        local_rho = target_variance / denominator if is_qualified and denominator > 0.0 else 0.0
        if not math.isfinite(local_rho) or not 0.0 <= local_rho < 1.0:
            raise D111ScoreError("D111 mixture mass became invalid")
        anchors[bank_index] = transported
        rho[bank_index] = local_rho
        qualified[bank_index] = is_qualified
        iterations[bank_index] = WEISZFELD_STEPS
        primal[bank_index] = local_primal
        dual[bank_index] = local_dual
        gap[bank_index] = local_gap
        consensus[bank_index] = local_consensus

    arrays = {
        "anchors": _readonly(anchors, np.float32),
        "rho": _readonly(rho, np.float32),
        "qualified": _readonly(qualified, np.bool_),
        "iterations": _readonly(iterations, np.int16),
        "primal": _readonly(primal, np.float64),
        "dual": _readonly(dual, np.float64),
        "gap": _readonly(gap, np.float64),
        "consensus_count": _readonly(consensus, np.int16),
    }
    resource = {
        "persistent_numeric_bytes": int(sum(array.nbytes for array in arrays.values())),
        "enrollment_projection_macs": EXPECTED_OLD_CLASS_COUNT * FEATURE_DIM * 3,
        "weiszfeld_scalar_steps": EXPECTED_OLD_CLASS_COUNT
        * (EXPECTED_OLD_CLASS_COUNT - 1)
        * 3
        * WEISZFELD_STEPS,
        "extra_query_macs_per_row_upper_bound": EXPECTED_OLD_CLASS_COUNT * FEATURE_DIM,
        "query_dependent_state_bytes": 0,
    }
    bundle_root = str(bundle.manifest.get("content_root_sha256", ""))
    state = D111LOOGATState(
        classes=tuple(bank.classes),
        old_class_indices=old_indices,
        bank_receipt_sha256=bank.bank_receipt_sha256,
        config_lock_digest=bank.config_lock_digest,
        bundle_content_root_sha256=bundle_root,
        resource_receipt=MappingProxyType(resource),
        state_receipt_sha256="0" * 64,
        **arrays,
    )
    object.__setattr__(state, "state_receipt_sha256", _canonical_sha256(_state_payload(state)))
    _verify_state(state)
    return state


def score_d111_loo_gat_logits(
    state: D111LOOGATState,
    bank: TypedINT8ZIDSupportBank,
    query_zid: np.ndarray,
) -> np.ndarray:
    """Score independent queries; the state is never updated by query rows."""

    if type(state) is not D111LOOGATState or type(bank) is not TypedINT8ZIDSupportBank:
        raise D111ScoreError("D111 score requires exact state and bank types")
    _verify_state(state)
    if (
        state.classes != tuple(bank.classes)
        or state.bank_receipt_sha256 != bank.bank_receipt_sha256
        or state.config_lock_digest != bank.config_lock_digest
    ):
        raise D111ScoreError("D111 state/support bank binding drift")
    metric = identity_shared_psd_metric(config=bank.config)
    baseline = score_zid_student_t_logits(bank, query_zid, metric=metric)
    active = np.flatnonzero(state.rho > 0.0)
    if len(active) == 0:
        return baseline
    query = normalize_zid_rows(query_zid).astype(np.float64)
    output = np.asarray(baseline, dtype=np.float64).copy()
    dimension = bank.config.kernel_effective_dim
    nu = float(bank.config.student_nu)
    common_log_normalizer = (
        math.lgamma((nu + dimension) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * dimension * math.log(nu * math.pi)
    )
    for class_index in active:
        local_rho = float(state.rho[class_index])
        anchor = np.asarray(state.anchors[class_index], dtype=np.float64)
        cosine = np.clip(query @ anchor, -1.0, 1.0)
        distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
        h = float(bank.class_scales_fp16[class_index])
        anchor_log_kernel = (
            -dimension * math.log(h)
            - 0.5
            * (nu + dimension)
            * np.log1p(distance / (nu * h * h))
        )
        support_log_density = (
            np.asarray(baseline[:, class_index], dtype=np.float64)
            + common_log_normalizer
        )
        anchor_log_density = anchor_log_kernel + common_log_normalizer
        # Return to M0's global logit origin after forming the normalized
        # unit-mass mixture. The subtracted constant is common to every class.
        output[:, class_index] = (
            np.logaddexp(
                math.log1p(-local_rho) + support_log_density,
                math.log(local_rho) + anchor_log_density,
            )
            - common_log_normalizer
        )
    if not np.isfinite(output).all():
        raise D111ScoreError("D111 logits became non-finite")
    return _readonly(output, np.float32)


def predict_d111_loo_gat(
    state: D111LOOGATState,
    bank: TypedINT8ZIDSupportBank,
    query_zid: np.ndarray,
) -> tuple[str, ...]:
    logits = score_d111_loo_gat_logits(state, bank, query_zid)
    return tuple(state.classes[index] for index in np.argmax(logits, axis=1))


def audit_d111_loo_gat_state(state: D111LOOGATState) -> Mapping[str, Any]:
    if type(state) is not D111LOOGATState:
        raise D111ScoreError("D111 audit requires an exact state")
    _verify_state(state)
    return MappingProxyType(
        {
            "schema": state.schema,
            "state_receipt_sha256": state.state_receipt_sha256,
            "old_class_count": len(state.old_class_indices),
            "qualified_old_count": int(np.sum(state.qualified[list(state.old_class_indices)])),
            "positive_rho_count": int(np.sum(state.rho > 0.0)),
            "max_rho": float(np.max(state.rho)),
            "max_primal_dual_gap": float(np.max(state.gap[list(state.old_class_indices)])),
            "min_consensus_count": int(
                np.min(state.consensus_count[list(state.old_class_indices)])
            ),
            "resource_receipt": state.resource_receipt,
            "query_rows_used_for_fit": 0,
            "truth_role_quota_inputs": 0,
        }
    )


__all__ = [
    "D111LOOGATState",
    "D111ScoreError",
    "SCHEMA",
    "WEISZFELD_STEPS",
    "audit_d111_loo_gat_state",
    "fit_d111_loo_gat_state",
    "predict_d111_loo_gat",
    "score_d111_loo_gat_logits",
]

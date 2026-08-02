"""D122 RDCE-coordinate static-ground head for the source-held G1 surface.

This module composes two *already frozen* factors only: D106's support-only
RDCE state and D112's immutable source-held G1 ground aggregate.  It accepts
no query while fitting.  During scoring every query is independent, all
registered classes use the original Student-t kernel, and only a valid old
class replaces a fraction of its support density with one transformed ground
anchor.  Invalid local geometry resolves exactly to that class's M_DA column.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_d112_seam_bundle import (
    D112Bundle,
    FEATURE_DIM,
    G1_COMPONENT_STATE,
    G1_EVALUATION_SCOPE,
)
from cvsrffi.stage2_d112_seam_qknn import (
    fit_d112_ground_head_source_held_g1_state,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    TypedINT8ZIDSupportBank,
    build_typed_zid_support_bank,
    decode_zid_support_bank,
    identity_shared_psd_metric,
    normalize_zid_rows,
    score_zid_student_t_logits,
)


SCHEMA = "cvs.stage2.d122.rdce_ground_head.score_state.v1"
EXPECTED_OLD_CLASS_COUNT = 6
RDCE_RANK = 3
EPSILON = 64.0 * float(np.finfo(np.float32).eps)


class D122RDCEGroundHeadError(ValueError):
    """Raised when D122 cannot preserve its frozen source-held semantics."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _require_sha256(value: Any, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise D122RDCEGroundHeadError(f"{field} must be a lowercase SHA256")
    return value


def _normalized_raw_rows(value: np.ndarray, *, field: str) -> np.ndarray:
    rows = np.ascontiguousarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != FEATURE_DIM or not np.isfinite(rows).all():
        raise D122RDCEGroundHeadError(f"{field} must be finite [N,{FEATURE_DIM}]")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= EPSILON):
        raise D122RDCEGroundHeadError(f"{field} contains a zero-norm row")
    return np.ascontiguousarray(rows / norms, dtype=np.float64)


def _unit(value: np.ndarray, *, field: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (FEATURE_DIM,) or not np.isfinite(vector).all():
        raise D122RDCEGroundHeadError(f"{field} must be finite [{FEATURE_DIM}]")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= EPSILON:
        raise D122RDCEGroundHeadError(f"{field} is geometrically degenerate")
    return vector / norm


def _d106_like_transform(
    rows: np.ndarray,
    basis: np.ndarray,
    attenuation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply D106's source-held ``A`` then L2 normalization without a new fit.

    The returned transformed rows have D106's float32 boundary.  The second
    return value is the pre-normalization ``||A x||^2`` for the corresponding
    normalized input rows, retained only during enrollment.
    """

    values = _normalized_raw_rows(rows, field="RDCE transform rows")
    directions = np.asarray(basis, dtype=np.float64)
    attenuation64 = np.asarray(attenuation, dtype=np.float64)
    if (
        directions.shape != (RDCE_RANK, FEATURE_DIM)
        or attenuation64.shape != (RDCE_RANK,)
        or not np.isfinite(directions).all()
        or not np.isfinite(attenuation64).all()
        or np.any(attenuation64 <= 0.0)
        or np.any(attenuation64 >= 1.0)
    ):
        raise D122RDCEGroundHeadError("D106 RDCE basis/attenuation shape or range drift")
    coefficient = 1.0 - np.sqrt(1.0 - attenuation64)
    applied = values - ((values @ directions.T) * coefficient[None, :]) @ directions
    squared_norm = np.sum(np.square(applied), axis=1)
    norms = np.sqrt(squared_norm)
    if not np.isfinite(applied).all() or not np.isfinite(norms).all() or np.any(norms <= EPSILON):
        raise D122RDCEGroundHeadError("D106 RDCE transform became degenerate")
    transformed = applied / norms[:, None]
    return np.ascontiguousarray(transformed, dtype=np.float32), np.ascontiguousarray(
        squared_norm, dtype=np.float64
    )


def _low_rank_jacobian_multiplier(
    *,
    basis: np.ndarray,
    attenuation: np.ndarray,
    transformed: np.ndarray,
    ax_squared_norm: float,
) -> float:
    """Evaluate the frozen rank-three D122 delta-method multiplier.

    This intentionally uses ``G=B B^T`` rather than assuming that an INT8
    decoded basis is still exactly orthogonal.  It neither re-orthogonalizes
    the input nor allocates a 160x160 deployment matrix.
    """

    directions = np.asarray(basis, dtype=np.float64)
    values = np.asarray(attenuation, dtype=np.float64)
    t = _unit(np.asarray(transformed, dtype=np.float64), field="RDCE transformed point")
    denominator = float(ax_squared_norm)
    if (
        directions.shape != (RDCE_RANK, FEATURE_DIM)
        or values.shape != (RDCE_RANK,)
        or not np.isfinite(directions).all()
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
        or np.any(values >= 1.0)
        or not math.isfinite(denominator)
        or denominator <= EPSILON * EPSILON
    ):
        raise D122RDCEGroundHeadError("D122 Jacobian low-rank inputs are invalid")
    coefficient = 1.0 - np.sqrt(1.0 - values)
    gram = directions @ directions.T
    d_gram = coefficient[:, None] * gram
    trace_a_squared = (
        float(FEATURE_DIM)
        - 2.0 * float(np.trace(d_gram))
        + float(np.trace(d_gram @ d_gram))
    )
    u = directions @ t
    at_squared_norm = (
        1.0
        - 2.0 * float(np.dot(u, coefficient * u))
        + float(np.dot(u, coefficient * (gram @ (coefficient * u))))
    )
    numerator = trace_a_squared - at_squared_norm
    multiplier = numerator / (float(FEATURE_DIM) * denominator)
    if (
        not math.isfinite(trace_a_squared)
        or not math.isfinite(at_squared_norm)
        or not math.isfinite(multiplier)
        or multiplier <= 0.0
    ):
        raise D122RDCEGroundHeadError("D122 Jacobian transport multiplier is non-positive")
    return multiplier


def d122_dense_jacobian_multiplier(
    *,
    basis: np.ndarray,
    attenuation: np.ndarray,
    raw_unit_point: np.ndarray,
) -> float:
    """Dense audit-only counterpart of the frozen low-rank multiplier.

    It deliberately constructs ``A`` from the *same supplied decoded basis*;
    callers must not use this function in predictor deployment.
    """

    x = _unit(raw_unit_point, field="D122 dense audit raw point")
    directions = np.asarray(basis, dtype=np.float64)
    values = np.asarray(attenuation, dtype=np.float64)
    if (
        directions.shape != (RDCE_RANK, FEATURE_DIM)
        or values.shape != (RDCE_RANK,)
        or not np.isfinite(directions).all()
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
        or np.any(values >= 1.0)
    ):
        raise D122RDCEGroundHeadError("D122 dense Jacobian audit inputs are invalid")
    coefficient = 1.0 - np.sqrt(1.0 - values)
    matrix = np.eye(FEATURE_DIM, dtype=np.float64) - directions.T @ (
        coefficient[:, None] * directions
    )
    ax = matrix @ x
    norm = float(np.linalg.norm(ax))
    if not math.isfinite(norm) or norm <= EPSILON:
        raise D122RDCEGroundHeadError("D122 dense Jacobian audit transform is degenerate")
    t = ax / norm
    jacobian = (np.eye(FEATURE_DIM, dtype=np.float64) - np.outer(t, t)) @ matrix / norm
    multiplier = float(np.sum(np.square(jacobian)) / FEATURE_DIM)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise D122RDCEGroundHeadError("D122 dense Jacobian multiplier is non-positive")
    return multiplier


def _validate_rdce_state(
    rdce_state: Mapping[str, Any],
    normalized_raw_support: np.ndarray,
    active_k: int,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Verify D106's frozen source-held receipt without reading an asset anew."""

    if not isinstance(rdce_state, Mapping):
        return None
    payload = rdce_state.get("payload")
    receipt = rdce_state.get("receipt")
    if not isinstance(payload, Mapping) or type(receipt) is not str or _sha(payload) != receipt:
        return None
    try:
        if (
            payload.get("scope") != "SOURCE_HELD_NON_TARGET_NO_P2_AUTHORITY"
            or int(payload.get("K")) != int(active_k)
            or payload.get("query_rows_used_for_fit") != 0
            or payload.get("query_state_updates") != 0
            or payload.get("support_root_sha256")
            != hashlib.sha256(
                np.ascontiguousarray(normalized_raw_support, dtype=np.float64).tobytes()
            ).hexdigest()
        ):
            return None
        basis = np.asarray(rdce_state.get("basis"), dtype=np.float64)
        attenuation = np.asarray(rdce_state.get("attenuation"), dtype=np.float64)
        encoded_attenuation = np.asarray(payload.get("attenuation_fp16"), dtype=np.float64)
        if (
            basis.shape != (RDCE_RANK, FEATURE_DIM)
            or attenuation.shape != (RDCE_RANK,)
            or encoded_attenuation.shape != (RDCE_RANK,)
            or not np.isfinite(basis).all()
            or not np.isfinite(attenuation).all()
            or np.any(attenuation <= 0.0)
            or np.any(attenuation >= 1.0)
            or not np.array_equal(attenuation, encoded_attenuation)
        ):
            return None
        _require_sha256(str(payload.get("asset_receipt_sha256", "")), "D106 asset receipt")
    except (D122RDCEGroundHeadError, TypeError, ValueError):
        return None
    return np.ascontiguousarray(basis, dtype=np.float64), np.ascontiguousarray(
        attenuation, dtype=np.float64
    ), receipt


@dataclass(frozen=True, slots=True)
class D122RDCEGroundHeadState:
    """Immutable per-row, support-only state for D122's M_JOINT arm."""

    classes: tuple[str, ...]
    old_class_indices: tuple[int, ...]
    anchors: np.ndarray
    rho: np.ndarray
    information_valid: np.ndarray
    fallback_to_m_da: np.ndarray
    support_transport_multiplier: np.ndarray
    ground_transport_multiplier: np.ndarray
    sigma0_amb_transported: np.ndarray
    v_g_amb_transported: np.ndarray
    v_s_amb: np.ndarray
    discrepancy_amb: np.ndarray
    global_component_valid: bool
    global_failure_reason: str
    bank_receipt_sha256: str
    config_lock_digest: str
    d112_bundle_content_root_sha256: str
    d112_reference_state_receipt_sha256: str
    rdce_state_receipt_sha256: str
    rdce_basis_sha256: str
    rdce_attenuation_sha256: str
    resource_receipt: Mapping[str, int]
    state_receipt_sha256: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        count = len(self.classes)
        expected = (count,)
        arrays = (
            self.rho,
            self.information_valid,
            self.fallback_to_m_da,
            self.support_transport_multiplier,
            self.ground_transport_multiplier,
            self.sigma0_amb_transported,
            self.v_g_amb_transported,
            self.v_s_amb,
            self.discrepancy_amb,
        )
        if (
            self.schema != SCHEMA
            or len(set(self.classes)) != count
            or len(self.old_class_indices) != EXPECTED_OLD_CLASS_COUNT
            or len(set(self.old_class_indices)) != EXPECTED_OLD_CLASS_COUNT
            or self.anchors.shape != (count, FEATURE_DIM)
            or any(array.shape != expected for array in arrays)
            or any(array.flags.writeable for array in (self.anchors, *arrays))
        ):
            raise D122RDCEGroundHeadError("D122 state shape/schema/readonly drift")


def _state_payload(state: D122RDCEGroundHeadState) -> dict[str, Any]:
    array_names = (
        "anchors",
        "rho",
        "information_valid",
        "fallback_to_m_da",
        "support_transport_multiplier",
        "ground_transport_multiplier",
        "sigma0_amb_transported",
        "v_g_amb_transported",
        "v_s_amb",
        "discrepancy_amb",
    )
    return {
        "schema": state.schema,
        "classes": list(state.classes),
        "old_class_indices": list(state.old_class_indices),
        "global_component_valid": bool(state.global_component_valid),
        "global_failure_reason": state.global_failure_reason,
        "bank_receipt_sha256": state.bank_receipt_sha256,
        "config_lock_digest": state.config_lock_digest,
        "d112_bundle_content_root_sha256": state.d112_bundle_content_root_sha256,
        "d112_reference_state_receipt_sha256": state.d112_reference_state_receipt_sha256,
        "rdce_state_receipt_sha256": state.rdce_state_receipt_sha256,
        "rdce_basis_sha256": state.rdce_basis_sha256,
        "rdce_attenuation_sha256": state.rdce_attenuation_sha256,
        "arrays": {name: _array_receipt(getattr(state, name)) for name in array_names},
        "resource_receipt": {str(key): int(value) for key, value in state.resource_receipt.items()},
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
    }


def _verify_state(state: D122RDCEGroundHeadState) -> None:
    if type(state) is not D122RDCEGroundHeadState:
        raise D122RDCEGroundHeadError("D122 scoring requires an exact state")
    if _sha(_state_payload(state)) != state.state_receipt_sha256:
        raise D122RDCEGroundHeadError("D122 state receipt verification failed")
    old = np.asarray(state.old_class_indices, dtype=np.int64)
    if (
        np.any(old < 0)
        or np.any(old >= len(state.classes))
        or not np.isfinite(state.anchors).all()
        or not np.isfinite(state.rho).all()
        or np.any(state.rho < 0.0)
        or np.any(state.rho >= 1.0)
        or any(
            not np.isfinite(array).all() or np.any(array < 0.0)
            for array in (
                state.support_transport_multiplier,
                state.ground_transport_multiplier,
                state.sigma0_amb_transported,
                state.v_g_amb_transported,
                state.v_s_amb,
                state.discrepancy_amb,
            )
        )
    ):
        raise D122RDCEGroundHeadError("D122 state numeric invariant drift")
    old_mask = np.zeros(len(state.classes), dtype=bool)
    old_mask[old] = True
    if np.any(state.information_valid & ~old_mask) or np.any(state.fallback_to_m_da & ~old_mask):
        raise D122RDCEGroundHeadError("D122 non-old activation/fallback drift")
    if np.any(state.information_valid & state.fallback_to_m_da):
        raise D122RDCEGroundHeadError("D122 active/fallback columns overlap")
    if np.any((state.rho > 0.0) != state.information_valid):
        raise D122RDCEGroundHeadError("D122 rho activation drift")
    if not state.global_component_valid and (
        np.any(state.information_valid) or not np.all(state.fallback_to_m_da[old])
    ):
        raise D122RDCEGroundHeadError("D122 global fallback must be exact M_DA")
    active = np.flatnonzero(state.information_valid)
    if len(active) and not np.allclose(
        np.linalg.norm(state.anchors[active].astype(np.float64), axis=1),
        1.0,
        atol=1.0e-5,
        rtol=0.0,
    ):
        raise D122RDCEGroundHeadError("D122 active transformed anchors lost unit norm")
    for field in (
        "bank_receipt_sha256",
        "config_lock_digest",
        "d112_bundle_content_root_sha256",
        "d112_reference_state_receipt_sha256",
    ):
        _require_sha256(getattr(state, field), field)
    if state.global_component_valid:
        _require_sha256(state.rdce_state_receipt_sha256, "rdce_state_receipt_sha256")
        _require_sha256(state.rdce_basis_sha256, "rdce_basis_sha256")
        _require_sha256(state.rdce_attenuation_sha256, "rdce_attenuation_sha256")


def _make_state(
    *,
    bank: TypedINT8ZIDSupportBank,
    old_class_indices: tuple[int, ...],
    d112_bundle_content_root_sha256: str,
    d112_reference_state_receipt_sha256: str,
    rdce_state_receipt_sha256: str,
    rdce_basis_sha256: str,
    rdce_attenuation_sha256: str,
    global_component_valid: bool,
    global_failure_reason: str,
    anchors: np.ndarray,
    rho: np.ndarray,
    information_valid: np.ndarray,
    fallback_to_m_da: np.ndarray,
    support_transport_multiplier: np.ndarray,
    ground_transport_multiplier: np.ndarray,
    sigma0_amb_transported: np.ndarray,
    v_g_amb_transported: np.ndarray,
    v_s_amb: np.ndarray,
    discrepancy_amb: np.ndarray,
) -> D122RDCEGroundHeadState:
    arrays = {
        "anchors": _readonly(anchors, np.float32),
        "rho": _readonly(rho, np.float64),
        "information_valid": _readonly(information_valid, np.bool_),
        "fallback_to_m_da": _readonly(fallback_to_m_da, np.bool_),
        "support_transport_multiplier": _readonly(support_transport_multiplier, np.float64),
        "ground_transport_multiplier": _readonly(ground_transport_multiplier, np.float64),
        "sigma0_amb_transported": _readonly(sigma0_amb_transported, np.float64),
        "v_g_amb_transported": _readonly(v_g_amb_transported, np.float64),
        "v_s_amb": _readonly(v_s_amb, np.float64),
        "discrepancy_amb": _readonly(discrepancy_amb, np.float64),
    }
    resource = {
        "persistent_numeric_bytes": int(sum(array.nbytes for array in arrays.values())),
        "enrollment_rdce_macs_for_six_ground_anchors": EXPECTED_OLD_CLASS_COUNT
        * 2
        * RDCE_RANK
        * FEATURE_DIM,
        "enrollment_jacobian_low_rank_scalar_terms": EXPECTED_OLD_CLASS_COUNT
        * RDCE_RANK
        * RDCE_RANK,
        "extra_query_macs_per_row_upper_bound": EXPECTED_OLD_CLASS_COUNT * FEATURE_DIM,
        "query_dependent_state_bytes": 0,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
    }
    state = D122RDCEGroundHeadState(
        classes=tuple(bank.classes),
        old_class_indices=old_class_indices,
        global_component_valid=bool(global_component_valid),
        global_failure_reason=str(global_failure_reason),
        bank_receipt_sha256=bank.bank_receipt_sha256,
        config_lock_digest=bank.config_lock_digest,
        d112_bundle_content_root_sha256=d112_bundle_content_root_sha256,
        d112_reference_state_receipt_sha256=d112_reference_state_receipt_sha256,
        rdce_state_receipt_sha256=rdce_state_receipt_sha256,
        rdce_basis_sha256=rdce_basis_sha256,
        rdce_attenuation_sha256=rdce_attenuation_sha256,
        resource_receipt=MappingProxyType(resource),
        state_receipt_sha256="0" * 64,
        **arrays,
    )
    object.__setattr__(state, "state_receipt_sha256", _sha(_state_payload(state)))
    _verify_state(state)
    return state


def fit_d122_rdce_ground_head_source_held_g1_state(
    bundle: D112Bundle,
    bank: TypedINT8ZIDSupportBank,
    raw_support_zid: np.ndarray,
    support_labels: Sequence[str],
    rdce_state: Mapping[str, Any],
) -> D122RDCEGroundHeadState:
    """Build D122 M_JOINT state from legal support and frozen aggregate assets.

    The function has deliberately no query, truth, role, quota, or target
    argument.  It uses the D112 G1 public fitter as the immutable aggregate
    loader/validator; D122 does not rebuild or replace any Phase1 asset.
    """

    if type(bundle) is not D112Bundle or type(bank) is not TypedINT8ZIDSupportBank:
        raise D122RDCEGroundHeadError("D122 fit requires exact D112 bundle and qKNN bank")
    # This is the exact D112 source-held G1 asset gate.  Its local head values
    # are intentionally not reused; only its sealed aggregate binding is.
    reference = fit_d112_ground_head_source_held_g1_state(bundle, bank)
    if (
        reference.bundle_component_state != G1_COMPONENT_STATE
        or reference.evaluation_scope != G1_EVALUATION_SCOPE
        or len(reference.old_class_indices) != EXPECTED_OLD_CLASS_COUNT
    ):
        raise D122RDCEGroundHeadError("D122 D112 source-held aggregate binding drift")
    old_indices = tuple(int(value) for value in reference.old_class_indices)
    count = len(bank.classes)
    normalized_raw = _normalized_raw_rows(raw_support_zid, field="raw support")
    labels = tuple(str(value) for value in support_labels)
    if (
        len(labels) != len(normalized_raw)
        or len(normalized_raw) != bank.support_row_count
        or any(label not in bank.classes for label in labels)
        or any(labels.count(class_id) != bank.active_k for class_id in bank.classes)
    ):
        raise D122RDCEGroundHeadError("D122 raw support/label/bank closure drift")

    zeros_vector = np.zeros(count, dtype=np.float64)
    zeros_anchor = np.zeros((count, FEATURE_DIM), dtype=np.float64)
    no_active = np.zeros(count, dtype=np.bool_)
    global_fallback = np.zeros(count, dtype=np.bool_)
    global_fallback[np.asarray(old_indices, dtype=np.int64)] = True
    bundle_root = str(bundle.manifest.get("content_root_sha256", ""))
    _require_sha256(bundle_root, "D112 bundle content root")

    if not reference.global_bundle_valid:
        return _make_state(
            bank=bank,
            old_class_indices=old_indices,
            d112_bundle_content_root_sha256=bundle_root,
            d112_reference_state_receipt_sha256=reference.state_receipt_sha256,
            rdce_state_receipt_sha256="",
            rdce_basis_sha256="",
            rdce_attenuation_sha256="",
            global_component_valid=False,
            global_failure_reason="D112_GLOBAL_BUNDLE_INVALID",
            anchors=zeros_anchor,
            rho=zeros_vector,
            information_valid=no_active,
            fallback_to_m_da=global_fallback,
            support_transport_multiplier=zeros_vector,
            ground_transport_multiplier=zeros_vector,
            sigma0_amb_transported=zeros_vector,
            v_g_amb_transported=zeros_vector,
            v_s_amb=zeros_vector,
            discrepancy_amb=zeros_vector,
        )

    validated_rdce = _validate_rdce_state(rdce_state, normalized_raw, bank.active_k)
    if validated_rdce is None:
        return _make_state(
            bank=bank,
            old_class_indices=old_indices,
            d112_bundle_content_root_sha256=bundle_root,
            d112_reference_state_receipt_sha256=reference.state_receipt_sha256,
            rdce_state_receipt_sha256="",
            rdce_basis_sha256="",
            rdce_attenuation_sha256="",
            global_component_valid=False,
            global_failure_reason="RDCE_STATE_RECEIPT_OR_BINDING_INVALID",
            anchors=zeros_anchor,
            rho=zeros_vector,
            information_valid=no_active,
            fallback_to_m_da=global_fallback,
            support_transport_multiplier=zeros_vector,
            ground_transport_multiplier=zeros_vector,
            sigma0_amb_transported=zeros_vector,
            v_g_amb_transported=zeros_vector,
            v_s_amb=zeros_vector,
            discrepancy_amb=zeros_vector,
        )
    basis, attenuation, rdce_receipt = validated_rdce
    # The typed bank canonicalizes its row order, so raw row position itself
    # is not a valid binding.  Rebuild through the same D106 transform and
    # qKNN canonicalizer, then bind the complete receipt.  A same-count bank
    # made from another label pairing or another transformed support surface is
    # therefore a hard error rather than an M_DA fallback.
    try:
        expected_transformed_support, _ = _d106_like_transform(
            raw_support_zid, basis, attenuation
        )
        expected_bank = build_typed_zid_support_bank(
            expected_transformed_support,
            labels,
            bank.classes,
            config=bank.config,
        )
    except Exception as error:
        raise D122RDCEGroundHeadError(
            "D122 cannot reconstruct the RDCE-bound typed support bank"
        ) from error
    if expected_bank.bank_receipt_sha256 != bank.bank_receipt_sha256:
        raise D122RDCEGroundHeadError("D122 RDCE transformed support-bank binding drift")
    anchors = np.zeros((count, FEATURE_DIM), dtype=np.float64)
    rho = np.zeros(count, dtype=np.float64)
    information_valid = np.zeros(count, dtype=np.bool_)
    fallback_to_m_da = np.zeros(count, dtype=np.bool_)
    support_multiplier = np.zeros(count, dtype=np.float64)
    ground_multiplier = np.zeros(count, dtype=np.float64)
    sigma0_transported = np.zeros(count, dtype=np.float64)
    v_ground_transported = np.zeros(count, dtype=np.float64)
    v_support = np.zeros(count, dtype=np.float64)
    discrepancy = np.zeros(count, dtype=np.float64)

    decoded_transformed = normalize_zid_rows(
        decode_zid_support_bank(bank).astype(np.float32)
    ).astype(np.float64)
    raw_ground = np.asarray(bundle.g, dtype=np.float64)
    sigma0 = np.asarray(bundle.sigma0_amb, dtype=np.float64)
    v_ground = np.asarray(bundle.v_g_amb, dtype=np.float64)
    for old_position, class_index in enumerate(old_indices):
        try:
            class_id = bank.classes[class_index]
            raw_local = normalized_raw[np.asarray(labels) == class_id]
            if raw_local.shape != (bank.active_k, FEATURE_DIM):
                raise D122RDCEGroundHeadError("D122 class-specific raw support count drift")
            raw_prototype = _unit(np.sum(raw_local, axis=0), field="raw support prototype")
            transported_prototype, support_ax_squared = _d106_like_transform(
                raw_prototype[None, :], basis, attenuation
            )
            transported_ground, ground_ax_squared = _d106_like_transform(
                _unit(raw_ground[old_position], field="D112 ground anchor")[None, :],
                basis,
                attenuation,
            )
            prototype_point = transported_prototype[0].astype(np.float64)
            anchor_point = transported_ground[0].astype(np.float64)
            local = decoded_transformed[bank.class_indices_int16 == class_index]
            if local.shape != (bank.active_k, FEATURE_DIM):
                raise D122RDCEGroundHeadError("D122 transformed support-bank count drift")
            # The frozen D122 center is T(s_c), not normalize(sum_k T(x_ck)).
            if bank.active_k == 1:
                hat_sigma = 0.0
            else:
                hat_sigma = float(
                    np.sum(np.square(local - prototype_point[None, :]))
                    / ((bank.active_k - 1) * FEATURE_DIM)
                )
            r_support = _low_rank_jacobian_multiplier(
                basis=basis,
                attenuation=attenuation,
                transformed=prototype_point,
                ax_squared_norm=float(support_ax_squared[0]),
            )
            r_ground = _low_rank_jacobian_multiplier(
                basis=basis,
                attenuation=attenuation,
                transformed=anchor_point,
                ax_squared_norm=float(ground_ax_squared[0]),
            )
            local_sigma0 = r_support * float(sigma0[old_position])
            local_v_ground = r_ground * float(v_ground[old_position])
            local_v_support = (local_sigma0 + hat_sigma) / float(bank.active_k)
            local_discrepancy = float(
                np.sum(np.square(prototype_point - anchor_point)) / FEATURE_DIM
            )
            denominator = local_v_support + local_v_ground + local_discrepancy
            local_rho = local_v_support / denominator
            values = (
                hat_sigma,
                r_support,
                r_ground,
                local_sigma0,
                local_v_ground,
                local_v_support,
                local_discrepancy,
                denominator,
                local_rho,
            )
            if (
                not all(math.isfinite(value) and value >= 0.0 for value in values[:-1])
                or not math.isfinite(local_rho)
                or denominator <= 0.0
                or not 0.0 < local_rho < 1.0
            ):
                raise D122RDCEGroundHeadError("D122 transported ground quality is invalid")
            anchors[class_index] = anchor_point
            rho[class_index] = local_rho
            information_valid[class_index] = True
            support_multiplier[class_index] = r_support
            ground_multiplier[class_index] = r_ground
            sigma0_transported[class_index] = local_sigma0
            v_ground_transported[class_index] = local_v_ground
            v_support[class_index] = local_v_support
            discrepancy[class_index] = local_discrepancy
        except (D122RDCEGroundHeadError, FloatingPointError, TypeError, ValueError):
            # Per-class geometry failure is explicitly an exact M_DA fallback.
            fallback_to_m_da[class_index] = True

    return _make_state(
        bank=bank,
        old_class_indices=old_indices,
        d112_bundle_content_root_sha256=bundle_root,
        d112_reference_state_receipt_sha256=reference.state_receipt_sha256,
        rdce_state_receipt_sha256=rdce_receipt,
        rdce_basis_sha256=hashlib.sha256(
            np.ascontiguousarray(basis, dtype=np.float64).tobytes()
        ).hexdigest(),
        rdce_attenuation_sha256=hashlib.sha256(
            np.ascontiguousarray(attenuation, dtype=np.float64).tobytes()
        ).hexdigest(),
        global_component_valid=True,
        global_failure_reason="NONE",
        anchors=anchors,
        rho=rho,
        information_valid=information_valid,
        fallback_to_m_da=fallback_to_m_da,
        support_transport_multiplier=support_multiplier,
        ground_transport_multiplier=ground_multiplier,
        sigma0_amb_transported=sigma0_transported,
        v_g_amb_transported=v_ground_transported,
        v_s_amb=v_support,
        discrepancy_amb=discrepancy,
    )


def score_d122_rdce_ground_head_source_held_g1_logits(
    state: D122RDCEGroundHeadState,
    bank: TypedINT8ZIDSupportBank,
    held_query_zid: np.ndarray,
) -> np.ndarray:
    """Score independent held queries without fitting, selection, or truth."""

    if type(bank) is not TypedINT8ZIDSupportBank:
        raise D122RDCEGroundHeadError("D122 scoring requires an exact qKNN bank")
    _verify_state(state)
    if (
        state.classes != tuple(bank.classes)
        or state.bank_receipt_sha256 != bank.bank_receipt_sha256
        or state.config_lock_digest != bank.config_lock_digest
    ):
        raise D122RDCEGroundHeadError("D122 state/support-bank binding drift")
    baseline = score_zid_student_t_logits(
        bank,
        held_query_zid,
        metric=identity_shared_psd_metric(config=bank.config),
    )
    output = np.array(baseline, dtype=np.float32, copy=True)
    active = np.flatnonzero(state.information_valid)
    if len(active):
        query = normalize_zid_rows(held_query_zid).astype(np.float64)
        dimension = int(bank.config.kernel_effective_dim)
        nu = float(bank.config.student_nu)
        for class_index in active:
            local_rho = float(state.rho[class_index])
            anchor = np.asarray(state.anchors[class_index], dtype=np.float64)
            cosine = np.clip(query @ anchor, -1.0, 1.0)
            distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
            h = float(bank.class_scales_fp16[class_index])
            anchor_kernel = (
                -dimension * math.log(h)
                - 0.5 * (nu + dimension) * np.log1p(distance / (nu * h * h))
            )
            mixed = np.logaddexp(
                math.log1p(-local_rho) + baseline[:, class_index].astype(np.float64),
                math.log(local_rho) + anchor_kernel,
            )
            if not np.isfinite(mixed).all():
                raise D122RDCEGroundHeadError("D122 ground-head logits became non-finite")
            output[:, class_index] = mixed.astype(np.float32)
    if not np.isfinite(output).all():
        raise D122RDCEGroundHeadError("D122 final logits became non-finite")
    old = np.asarray(state.old_class_indices, dtype=np.int64)
    inactive_old = old[state.fallback_to_m_da[old] | ~state.information_valid[old]]
    new_mask = np.ones(len(state.classes), dtype=bool)
    new_mask[old] = False
    if (
        (len(inactive_old) and not np.array_equal(output[:, inactive_old], baseline[:, inactive_old]))
        or not np.array_equal(output[:, new_mask], baseline[:, new_mask])
    ):
        raise D122RDCEGroundHeadError("D122 exact M_DA score boundary drift")
    return _readonly(output, np.float32)


def unique_d122_argmax(logits: np.ndarray, registry: Sequence[str]) -> tuple[str, ...]:
    """Return class labels only when each query has one bit-exact winner."""

    scores = np.asarray(logits, dtype=np.float32)
    classes = tuple(str(value) for value in registry)
    if (
        scores.ndim != 2
        or scores.shape[1] != len(classes)
        or len(set(classes)) != len(classes)
        or not np.isfinite(scores).all()
    ):
        raise D122RDCEGroundHeadError("D122 argmax score/registry drift")
    result: list[str] = []
    for row in scores:
        maximum = np.max(row)
        numeric_ties = np.flatnonzero(row == maximum)
        maximum_bits = np.asarray([maximum], dtype=np.float32).view(np.uint32)[0]
        bit_ties = np.flatnonzero(row.view(np.uint32) == maximum_bits)
        if len(numeric_ties) != 1 or len(bit_ties) != 1:
            raise D122RDCEGroundHeadError("CLASS_SCORE_TIE_UNRESOLVED")
        result.append(classes[int(bit_ties[0])])
    return tuple(result)


def predict_d122_rdce_ground_head_source_held_g1(
    state: D122RDCEGroundHeadState,
    bank: TypedINT8ZIDSupportBank,
    held_query_zid: np.ndarray,
) -> tuple[str, ...]:
    """Return truth-free, per-query D122 source-held predictions."""

    return unique_d122_argmax(
        score_d122_rdce_ground_head_source_held_g1_logits(state, bank, held_query_zid),
        state.classes,
    )


def audit_d122_rdce_ground_head_state(
    state: D122RDCEGroundHeadState,
) -> Mapping[str, Any]:
    """Return only state/resource receipts; no query labels or performance."""

    _verify_state(state)
    old = np.asarray(state.old_class_indices, dtype=np.int64)
    return MappingProxyType(
        {
            "schema": state.schema,
            "state_receipt_sha256": state.state_receipt_sha256,
            "global_component_valid": bool(state.global_component_valid),
            "global_failure_reason": state.global_failure_reason,
            "old_class_indices": [int(value) for value in old],
            "active_old_class_count": int(np.count_nonzero(state.information_valid[old])),
            "fallback_old_class_count": int(np.count_nonzero(state.fallback_to_m_da[old])),
            "anchors": _array_receipt(state.anchors[old]),
            "rho": [float(state.rho[index]) for index in old],
            "support_transport_multiplier": [
                float(state.support_transport_multiplier[index]) for index in old
            ],
            "ground_transport_multiplier": [
                float(state.ground_transport_multiplier[index]) for index in old
            ],
            "rdce_state_receipt_sha256": state.rdce_state_receipt_sha256,
            "rdce_basis_sha256": state.rdce_basis_sha256,
            "rdce_attenuation_sha256": state.rdce_attenuation_sha256,
            "d112_bundle_content_root_sha256": state.d112_bundle_content_root_sha256,
            "d112_reference_state_receipt_sha256": state.d112_reference_state_receipt_sha256,
            "resource_receipt": {
                str(key): int(value) for key, value in state.resource_receipt.items()
            },
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "target_access": False,
            "formal_p2_authority": False,
        }
    )


__all__ = [
    "D122RDCEGroundHeadError",
    "D122RDCEGroundHeadState",
    "SCHEMA",
    "audit_d122_rdce_ground_head_state",
    "d122_dense_jacobian_multiplier",
    "fit_d122_rdce_ground_head_source_held_g1_state",
    "predict_d122_rdce_ground_head_source_held_g1",
    "score_d122_rdce_ground_head_source_held_g1_logits",
    "unique_d122_argmax",
]

"""D123 leave-one-out cross-class residual-excess ground-head shrinkage.

The fit surfaces accept only a sealed D112 bundle, labelled support state and,
for the joint arm, D106's already-fitted RDCE receipt.  Query rows are admitted
only by the scorers.  The immutable Phase1 bundle registry is the sole old-class
authority; scorer-side held roles are deliberately absent from every API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_d112_seam_bundle import D112Bundle, FEATURE_DIM
from cvsrffi.stage2_d112_seam_qknn import (
    D112SEAMState,
    fit_d112_ground_head_source_held_g1_state,
    score_d112_seam_source_held_g1_logits,
)
from cvsrffi.stage2_d122_rdce_ground_head import (
    D122RDCEGroundHeadState,
    fit_d122_rdce_ground_head_source_held_g1_state,
    score_d122_rdce_ground_head_source_held_g1_logits,
    unique_d122_argmax,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    TypedINT8ZIDSupportBank,
    identity_shared_psd_metric,
    normalize_zid_rows,
    score_zid_student_t_logits,
)


IDENTITY_SCHEMA = "cvs.stage2.d123.loo_cres.identity.score_state.v1"
RDCE_SCHEMA = "cvs.stage2.d123.loo_cres.rdce.score_state.v1"
EXPECTED_OLD_CLASS_COUNT = 6
MIN_DONOR_COUNT = 3


class D123LOOCRESError(ValueError):
    """Raised when D123 cannot preserve its frozen support-only semantics."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    frozen.setflags(write=False)
    return frozen


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _payload(state: Any) -> dict[str, Any]:
    return {
        "schema": state.schema,
        "classes": list(state.classes),
        "old_class_indices": list(state.old_class_indices),
        "bank_receipt_sha256": state.bank_receipt_sha256,
        "config_lock_digest": state.config_lock_digest,
        "reference_state_receipt_sha256": state.reference_state.state_receipt_sha256,
        "rho": _array_receipt(state.rho),
        "delta": _array_receipt(state.delta),
        "donor_count": _array_receipt(state.donor_count),
        "cres_applied": _array_receipt(state.cres_applied),
    }


@dataclass(frozen=True, slots=True)
class D123LOOCRESGroundHeadState:
    classes: tuple[str, ...]
    old_class_indices: tuple[int, ...]
    reference_state: D112SEAMState
    rho: np.ndarray
    delta: np.ndarray
    donor_count: np.ndarray
    cres_applied: np.ndarray
    bank_receipt_sha256: str
    config_lock_digest: str
    state_receipt_sha256: str
    schema: str = IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        _verify_state(self, D112SEAMState, IDENTITY_SCHEMA, allow_placeholder=True)


@dataclass(frozen=True, slots=True)
class D123LOOCRESRDCEGroundHeadState:
    classes: tuple[str, ...]
    old_class_indices: tuple[int, ...]
    reference_state: D122RDCEGroundHeadState
    rho: np.ndarray
    delta: np.ndarray
    donor_count: np.ndarray
    cres_applied: np.ndarray
    bank_receipt_sha256: str
    config_lock_digest: str
    state_receipt_sha256: str
    schema: str = RDCE_SCHEMA

    def __post_init__(self) -> None:
        _verify_state(self, D122RDCEGroundHeadState, RDCE_SCHEMA, allow_placeholder=True)


def _verify_state(
    state: Any,
    reference_type: type,
    schema: str,
    *,
    allow_placeholder: bool = False,
) -> None:
    if state.schema != schema or type(state.reference_state) is not reference_type:
        raise D123LOOCRESError("D123 state schema/reference type drift")
    if (
        state.classes != tuple(state.reference_state.classes)
        or state.old_class_indices != tuple(state.reference_state.old_class_indices)
        or state.bank_receipt_sha256 != state.reference_state.bank_receipt_sha256
        or state.config_lock_digest != state.reference_state.config_lock_digest
        or len(state.old_class_indices) != EXPECTED_OLD_CLASS_COUNT
        or len(set(state.old_class_indices)) != EXPECTED_OLD_CLASS_COUNT
    ):
        raise D123LOOCRESError("D123 reference/registry binding drift")
    count = len(state.classes)
    arrays = (state.rho, state.delta, state.donor_count, state.cres_applied)
    if any(value.shape != (count,) or value.flags.writeable for value in arrays):
        raise D123LOOCRESError("D123 state array shape/mutability drift")
    if (
        state.rho.dtype != state.reference_state.rho.dtype
        or state.delta.dtype != np.float64
        or state.donor_count.dtype != np.int16
        or state.cres_applied.dtype != np.bool_
        or not np.isfinite(state.rho).all()
        or not np.isfinite(state.delta).all()
        or np.any(state.rho < 0.0)
        or np.any(state.rho >= 1.0)
        or np.any(state.delta < 0.0)
        or np.any(state.donor_count < 0)
    ):
        raise D123LOOCRESError("D123 state numeric invariant drift")
    old_mask = np.zeros(count, dtype=bool)
    old_mask[np.asarray(state.old_class_indices, dtype=np.int64)] = True
    if (
        np.any(state.rho[~old_mask] != 0.0)
        or np.any(state.delta[~old_mask] != 0.0)
        or np.any(state.donor_count[~old_mask] != 0)
        or np.any(state.cres_applied[~old_mask])
        or np.any(state.rho > np.asarray(state.reference_state.rho, dtype=state.rho.dtype))
    ):
        raise D123LOOCRESError("D123 old-only shrinkage boundary drift")
    if state.state_receipt_sha256 == "0" * 64:
        if not allow_placeholder:
            raise D123LOOCRESError("D123 placeholder state receipt is not scoreable")
    elif state.state_receipt_sha256 != _sha(_payload(state)):
        raise D123LOOCRESError("D123 state receipt drift")


def _cres_arrays(
    *,
    reference_rho: np.ndarray,
    information_valid: np.ndarray,
    old_class_indices: tuple[int, ...],
    v_s: np.ndarray,
    v_g: np.ndarray,
    discrepancy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(reference_rho)
    reference = np.asarray(reference_rho)
    if reference.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise D123LOOCRESError("D123 reference rho dtype is unsupported")
    rho = reference.copy()
    delta = np.zeros(count, dtype=np.float64)
    donor_count = np.zeros(count, dtype=np.int16)
    applied = np.zeros(count, dtype=np.bool_)
    valid = np.zeros(count, dtype=np.bool_)
    old = np.asarray(old_class_indices, dtype=np.int64)
    for index in old:
        values = (float(v_s[index]), float(v_g[index]), float(discrepancy[index]))
        valid[index] = bool(
            information_valid[index]
            and all(math.isfinite(value) and value >= 0.0 for value in values)
            and values[0] > 0.0
        )
    for index in old:
        if not valid[index]:
            continue
        donors = np.asarray([value for value in old if value != index and valid[value]], dtype=np.int64)
        donor_count[index] = len(donors)
        if len(donors) < MIN_DONOR_COUNT:
            continue
        excess = discrepancy[donors] - v_s[donors] - v_g[donors]
        local_delta = max(0.0, float(np.median(excess)))
        denominator = float(v_s[index] + v_g[index] + discrepancy[index] + local_delta)
        if not math.isfinite(local_delta) or not math.isfinite(denominator) or denominator <= 0.0:
            raise D123LOOCRESError("D123 cross-old residual excess is invalid")
        candidate = np.asarray(float(v_s[index]) / denominator, dtype=reference.dtype).item()
        bounded = np.asarray(
            min(float(reference_rho[index]), float(candidate)), dtype=reference.dtype
        ).item()
        delta[index] = local_delta
        rho[index] = bounded
        applied[index] = not np.array_equal(
            np.asarray(bounded, dtype=reference.dtype),
            np.asarray(reference_rho[index], dtype=reference.dtype),
        )
    return (
        _readonly(rho, reference.dtype),
        _readonly(delta, np.float64),
        _readonly(donor_count, np.int16),
        _readonly(applied, np.bool_),
    )


def _make_state(state_type: type, reference_state: Any, arrays: tuple[np.ndarray, ...]) -> Any:
    state = state_type(
        classes=tuple(reference_state.classes),
        old_class_indices=tuple(int(value) for value in reference_state.old_class_indices),
        reference_state=reference_state,
        rho=arrays[0],
        delta=arrays[1],
        donor_count=arrays[2],
        cres_applied=arrays[3],
        bank_receipt_sha256=reference_state.bank_receipt_sha256,
        config_lock_digest=reference_state.config_lock_digest,
        state_receipt_sha256="0" * 64,
    )
    object.__setattr__(state, "state_receipt_sha256", _sha(_payload(state)))
    _verify_state(state, type(reference_state), state.schema)
    return state


def fit_d123_loo_cres_ground_head_source_held_g1_state(
    bundle: D112Bundle,
    bank: TypedINT8ZIDSupportBank,
) -> D123LOOCRESGroundHeadState:
    """Fit identity-coordinate D123 from the sealed old registry and support."""

    reference = fit_d112_ground_head_source_held_g1_state(bundle, bank)
    count = len(bank.classes)
    v_g = np.zeros(count, dtype=np.float64)
    for position, index in enumerate(reference.old_class_indices):
        v_g[int(index)] = float(bundle.v_g_amb[position])
    arrays = _cres_arrays(
        reference_rho=reference.rho,
        information_valid=reference.information_valid,
        old_class_indices=tuple(reference.old_class_indices),
        v_s=np.asarray(reference.v_s_amb, dtype=np.float64),
        v_g=v_g,
        discrepancy=np.asarray(reference.discrepancy_amb, dtype=np.float64),
    )
    return _make_state(D123LOOCRESGroundHeadState, reference, arrays)


def fit_d123_loo_cres_rdce_ground_head_source_held_g1_state(
    bundle: D112Bundle,
    bank: TypedINT8ZIDSupportBank,
    raw_support_zid: np.ndarray,
    support_labels: Sequence[str],
    rdce_state: Mapping[str, Any],
) -> D123LOOCRESRDCEGroundHeadState:
    """Fit RDCE-coordinate D123 without a query, truth or scorer role input."""

    reference = fit_d122_rdce_ground_head_source_held_g1_state(
        bundle,
        bank,
        raw_support_zid,
        support_labels,
        rdce_state,
    )
    arrays = _cres_arrays(
        reference_rho=reference.rho,
        information_valid=reference.information_valid,
        old_class_indices=tuple(reference.old_class_indices),
        v_s=np.asarray(reference.v_s_amb, dtype=np.float64),
        v_g=np.asarray(reference.v_g_amb_transported, dtype=np.float64),
        discrepancy=np.asarray(reference.discrepancy_amb, dtype=np.float64),
    )
    return _make_state(D123LOOCRESRDCEGroundHeadState, reference, arrays)


def _score_adjusted(
    *,
    state: Any,
    bank: TypedINT8ZIDSupportBank,
    query_zid: np.ndarray,
    reference_logits: np.ndarray,
) -> np.ndarray:
    _verify_state(state, type(state.reference_state), state.schema)
    if (
        type(bank) is not TypedINT8ZIDSupportBank
        or state.classes != tuple(bank.classes)
        or state.bank_receipt_sha256 != bank.bank_receipt_sha256
        or state.config_lock_digest != bank.config_lock_digest
    ):
        raise D123LOOCRESError("D123 state/support-bank binding drift")
    output = np.array(reference_logits, dtype=np.float32, copy=True)
    active = np.flatnonzero(state.cres_applied)
    if len(active):
        support_logits = score_zid_student_t_logits(
            bank,
            query_zid,
            metric=identity_shared_psd_metric(config=bank.config),
        )
        query = normalize_zid_rows(query_zid).astype(np.float64)
        dimension = int(bank.config.kernel_effective_dim)
        nu = float(bank.config.student_nu)
        for class_index in active:
            local_rho = float(state.rho[class_index])
            anchor = np.asarray(state.reference_state.anchors[class_index], dtype=np.float64)
            cosine = np.clip(query @ anchor, -1.0, 1.0)
            distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
            h = float(bank.class_scales_fp16[class_index])
            anchor_kernel = (
                -dimension * math.log(h)
                - 0.5 * (nu + dimension) * np.log1p(distance / (nu * h * h))
            )
            mixed = np.logaddexp(
                math.log1p(-local_rho) + support_logits[:, class_index].astype(np.float64),
                math.log(local_rho) + anchor_kernel,
            )
            if not np.isfinite(mixed).all():
                raise D123LOOCRESError("D123 adjusted ground-head logits became non-finite")
            output[:, class_index] = mixed.astype(np.float32)
    old_mask = np.zeros(len(state.classes), dtype=bool)
    old_mask[np.asarray(state.old_class_indices, dtype=np.int64)] = True
    if not np.array_equal(output[:, ~old_mask], reference_logits[:, ~old_mask]):
        raise D123LOOCRESError("D123 non-old bit-exact score boundary drift")
    if not np.isfinite(output).all():
        raise D123LOOCRESError("D123 final logits became non-finite")
    return _readonly(output, np.float32)


def score_d123_loo_cres_ground_head_source_held_g1_logits(
    state: D123LOOCRESGroundHeadState,
    bank: TypedINT8ZIDSupportBank,
    held_query_zid: np.ndarray,
) -> np.ndarray:
    reference = score_d112_seam_source_held_g1_logits(
        state.reference_state, bank, held_query_zid
    )
    return _score_adjusted(
        state=state,
        bank=bank,
        query_zid=held_query_zid,
        reference_logits=reference,
    )


def score_d123_loo_cres_rdce_ground_head_source_held_g1_logits(
    state: D123LOOCRESRDCEGroundHeadState,
    bank: TypedINT8ZIDSupportBank,
    held_query_zid: np.ndarray,
) -> np.ndarray:
    reference = score_d122_rdce_ground_head_source_held_g1_logits(
        state.reference_state, bank, held_query_zid
    )
    return _score_adjusted(
        state=state,
        bank=bank,
        query_zid=held_query_zid,
        reference_logits=reference,
    )


def predict_d123_loo_cres_source_held_g1(
    state: D123LOOCRESGroundHeadState | D123LOOCRESRDCEGroundHeadState,
    bank: TypedINT8ZIDSupportBank,
    held_query_zid: np.ndarray,
) -> tuple[str, ...]:
    if type(state) is D123LOOCRESGroundHeadState:
        logits = score_d123_loo_cres_ground_head_source_held_g1_logits(
            state, bank, held_query_zid
        )
    elif type(state) is D123LOOCRESRDCEGroundHeadState:
        logits = score_d123_loo_cres_rdce_ground_head_source_held_g1_logits(
            state, bank, held_query_zid
        )
    else:
        raise D123LOOCRESError("D123 prediction requires an exact D123 state")
    if type(state) is D123LOOCRESRDCEGroundHeadState:
        return unique_d122_argmax(logits, state.classes)
    return tuple(state.classes[index] for index in np.argmax(logits, axis=1))


def audit_d123_loo_cres_state(
    state: D123LOOCRESGroundHeadState | D123LOOCRESRDCEGroundHeadState,
) -> Mapping[str, Any]:
    if type(state) is D123LOOCRESGroundHeadState:
        reference_type = D112SEAMState
    elif type(state) is D123LOOCRESRDCEGroundHeadState:
        reference_type = D122RDCEGroundHeadState
    else:
        raise D123LOOCRESError("D123 audit requires an exact D123 state")
    _verify_state(state, reference_type, state.schema)
    old = np.asarray(state.old_class_indices, dtype=np.int64)
    return MappingProxyType(
        {
            "schema": state.schema,
            "state_receipt_sha256": state.state_receipt_sha256,
            "reference_state_receipt_sha256": state.reference_state.state_receipt_sha256,
            "old_class_count": len(old),
            "cres_applied_old_count": int(np.sum(state.cres_applied[old])),
            "positive_delta_old_count": int(np.sum(state.delta[old] > 0.0)),
            "min_donor_count_old": int(np.min(state.donor_count[old])),
            "max_delta": float(np.max(state.delta[old])),
            "max_rho_ratio": float(
                np.max(
                    np.divide(
                        state.rho[old],
                        state.reference_state.rho[old],
                        out=np.zeros(len(old), dtype=np.float32),
                        where=state.reference_state.rho[old] > 0.0,
                    )
                )
            ),
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "truth_role_quota_inputs": 0,
        }
    )


__all__ = [
    "D123LOOCRESError",
    "D123LOOCRESGroundHeadState",
    "D123LOOCRESRDCEGroundHeadState",
    "IDENTITY_SCHEMA",
    "RDCE_SCHEMA",
    "audit_d123_loo_cres_state",
    "fit_d123_loo_cres_ground_head_source_held_g1_state",
    "fit_d123_loo_cres_rdce_ground_head_source_held_g1_state",
    "predict_d123_loo_cres_source_held_g1",
    "score_d123_loo_cres_ground_head_source_held_g1_logits",
    "score_d123_loo_cres_rdce_ground_head_source_held_g1_logits",
]

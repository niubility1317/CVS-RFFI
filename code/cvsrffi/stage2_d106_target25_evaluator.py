"""Truth-free single-state four-arm evaluator for D106 Target25.

This module only orchestrates the public RDCE, Student-t qKNN, and RCMR APIs.
It contains no alternative DA/head mathematics and has no truth, metric,
receiver-selection, or performance-selection input surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

import numpy as np

from cvsrffi.stage2_d106_k_conditioned_router import TARGET25_ROW_SCHEMA
from cvsrffi.stage2_d106_rcmr_2v_qknn import (
    D106RCMR2VBinding,
    build_d106_rcmr_2v_state,
    prepare_d106_rcmr_2v_scoring_context,
    score_d106_rcmr_2v_query,
)
from cvsrffi.stage2_d106_rdce_asset import D106RDCEAsset, Z_DIM
from cvsrffi.stage2_d106_rdce_runtime import (
    D106RDCESupportRows,
    fit_d106_rdce_runtime,
    prepare_d106_rdce_scoring_context,
    transform_d106_rdce_query,
    transform_d106_rdce_zid,
)
from cvsrffi.stage2_lpo_rc_qknn import (
    validate_lpo_rc_physical_id_disjointness,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)


ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
ALLOWED_K = (1, 5, 10)


class D106Target25EvaluatorError(ValueError):
    """Raised when a Target25 state or four-arm prediction fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise D106Target25EvaluatorError("canonical JSON payload is invalid") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise D106Target25EvaluatorError(f"{name} must be non-empty builtin text")
    return value


def _tokens(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise D106Target25EvaluatorError(f"{name} must be an ordered sequence")
    result = tuple(value)
    if not result or any(type(item) is not str or not item for item in result):
        raise D106Target25EvaluatorError(
            f"{name} must contain non-empty builtin strings"
        )
    if len(set(result)) != len(result):
        raise D106Target25EvaluatorError(f"{name} must contain unique values")
    return result


def _float32_rows(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise D106Target25EvaluatorError(f"{name} must be a numpy array")
    if (
        value.dtype != np.float32
        or value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] != Z_DIM
        or not np.isfinite(value).all()
    ):
        raise D106Target25EvaluatorError(
            f"{name} must be finite float32 [N,{Z_DIM}]"
        )
    return np.ascontiguousarray(value, dtype=np.float32)


def _physical_root(physical_ids: Sequence[str]) -> str:
    return _sha256(sorted(physical_ids))


def _unique_student_t_predictions(
    logits: np.ndarray, registry: tuple[str, ...]
) -> list[str]:
    values = np.asarray(logits)
    if (
        values.dtype != np.float32
        or values.ndim != 2
        or values.shape[1] != len(registry)
        or not np.isfinite(values).all()
    ):
        raise D106Target25EvaluatorError("Student-t logits layout drift")
    predictions: list[str] = []
    for row in values:
        winners = np.flatnonzero(row == np.float32(np.max(row)))
        if len(winners) != 1:
            raise D106Target25EvaluatorError(
                "Student-t cross-class maximum tie is fail-closed"
            )
        predictions.append(registry[int(winners[0])])
    return predictions


def _paired_view_receipt(
    *,
    support_physical_ids: tuple[str, ...],
    support_plus: np.ndarray,
    support_signed: np.ndarray,
    da_receipt_sha256: str,
) -> str:
    return _sha256(
        {
            "schema": "cvs.phase2.d106.target25.paired_support_views.v1",
            "support_physical_ids": list(support_physical_ids),
            "support_plus": _array_receipt(support_plus),
            "support_signed": _array_receipt(support_signed),
            "da_receipt_sha256": da_receipt_sha256,
        }
    )


def _identity_view_receipt(
    *,
    support_plus: np.ndarray,
    support_signed: np.ndarray,
    query_plus: np.ndarray,
    query_signed: np.ndarray,
) -> str:
    return _sha256(
        {
            "schema": "cvs.phase2.d106.target25.identity_views.v1",
            "support_plus": _array_receipt(support_plus),
            "support_signed": _array_receipt(support_signed),
            "query_plus": _array_receipt(query_plus),
            "query_signed": _array_receipt(query_signed),
        }
    )


def evaluate_d106_target25_state(
    *,
    row_id: str,
    receiver: str,
    scene: str,
    active_k: int,
    support_rows: D106RDCESupportRows,
    support_signed: np.ndarray,
    query_plus: np.ndarray,
    query_signed: np.ndarray,
    query_physical_ids: Sequence[str],
    registered_classes: Sequence[str],
    rdce_asset: D106RDCEAsset,
    rdce_row_authority: Any,
    rcmr_method_lock: Any,
) -> dict[str, Any]:
    """Evaluate all four frozen arms for one Target state without truth access."""

    row_token = _text(row_id, "row_id")
    receiver_token = _text(receiver, "receiver")
    scene_token = _text(scene, "scene")
    if type(active_k) is not int or active_k not in ALLOWED_K:
        raise D106Target25EvaluatorError("active_k must be exactly 1, 5, or 10")
    if type(support_rows) is not D106RDCESupportRows:
        raise D106Target25EvaluatorError("exact D106RDCESupportRows required")
    if support_rows.row_id != row_token:
        raise D106Target25EvaluatorError("support row_id binding drift")

    support_plus = _float32_rows(support_rows.support_z_id, "support_plus")
    signed_support = _float32_rows(support_signed, "support_signed")
    plus_query = _float32_rows(query_plus, "query_plus")
    signed_query = _float32_rows(query_signed, "query_signed")
    if support_plus.shape != signed_support.shape:
        raise D106Target25EvaluatorError("support plus/signed row closure drift")
    if plus_query.shape != signed_query.shape:
        raise D106Target25EvaluatorError("query plus/signed row closure drift")
    if not np.array_equal(
        support_plus, np.maximum(signed_support, np.float32(0.0))
    ) or not np.array_equal(
        plus_query, np.maximum(signed_query, np.float32(0.0))
    ):
        raise D106Target25EvaluatorError(
            "plus views must equal ReLU of the same signed pre_relu rows"
        )

    registry = _tokens(registered_classes, "registered_classes")
    bank = support_rows.qknn_bank
    if (
        tuple(bank.classes) != registry
        or bank.active_k != active_k
        or len(support_plus) != len(registry) * active_k
    ):
        raise D106Target25EvaluatorError("support registry/K binding drift")
    support_labels = tuple(str(value) for value in support_rows.support_labels.tolist())
    support_ids = tuple(
        str(value) for value in support_rows.support_physical_ids.tolist()
    )
    query_ids = _tokens(query_physical_ids, "query_physical_ids")
    if len(query_ids) != len(plus_query):
        raise D106Target25EvaluatorError("query physical-ID/order closure drift")
    validate_lpo_rc_physical_id_disjointness(support_ids, query_ids)
    if (
        _physical_root(query_ids)
        != support_rows.split_handle.query_physical_root_sha256
    ):
        raise D106Target25EvaluatorError("query physical-root binding drift")

    rdce_state = fit_d106_rdce_runtime(
        rdce_asset,
        support_rows,
        row_authority=rdce_row_authority,
    )
    rdce_context = prepare_d106_rdce_scoring_context(rdce_state)
    da_support_plus = transform_d106_rdce_zid(
        rdce_state, support_plus, context=rdce_context
    )
    da_support_signed = transform_d106_rdce_zid(
        rdce_state, signed_support, context=rdce_context
    )
    da_query_plus = transform_d106_rdce_query(
        rdce_state, plus_query, context=rdce_context
    )
    da_query_signed = transform_d106_rdce_query(
        rdce_state, signed_query, context=rdce_context
    )

    qknn_lock = bank.config
    identity_metric = identity_shared_psd_metric(config=qknn_lock)
    da_bank = build_typed_zid_support_bank(
        da_support_plus,
        support_labels,
        registry,
        config=qknn_lock,
    )
    arm_predictions: dict[str, list[str]] = {
        "M0": _unique_student_t_predictions(
            score_zid_student_t_logits(bank, plus_query, metric=identity_metric),
            registry,
        ),
        "M_DA": _unique_student_t_predictions(
            score_zid_student_t_logits(
                da_bank, da_query_plus, metric=identity_metric
            ),
            registry,
        ),
    }

    identity_da_receipt = _sha256(
        {
            "schema": "cvs.phase2.d106.target25.identity_da.v1",
            "mapping": "identity",
            "query_state_updates": 0,
        }
    )
    bindings = {
        "M_HEAD": D106RCMR2VBinding(
            capsule_id=support_rows.split_handle.capsule_id,
            split_id=support_rows.split_handle.split_id,
            validator_receipt_sha256=(
                support_rows.split_handle.validator_receipt_sha256
            ),
            support_physical_root_sha256=(
                support_rows.split_handle.support_physical_root_sha256
            ),
            row_id=row_token,
            seed=support_rows.seed,
            active_k=active_k,
            da_receipt_sha256=identity_da_receipt,
            paired_view_receipt_sha256=_paired_view_receipt(
                support_physical_ids=support_ids,
                support_plus=support_plus,
                support_signed=signed_support,
                da_receipt_sha256=identity_da_receipt,
            ),
        ),
        "M_JOINT": D106RCMR2VBinding(
            capsule_id=support_rows.split_handle.capsule_id,
            split_id=support_rows.split_handle.split_id,
            validator_receipt_sha256=(
                support_rows.split_handle.validator_receipt_sha256
            ),
            support_physical_root_sha256=(
                support_rows.split_handle.support_physical_root_sha256
            ),
            row_id=row_token,
            seed=support_rows.seed,
            active_k=active_k,
            da_receipt_sha256=rdce_state.runtime_receipt_sha256,
            paired_view_receipt_sha256=_paired_view_receipt(
                support_physical_ids=support_ids,
                support_plus=da_support_plus,
                support_signed=da_support_signed,
                da_receipt_sha256=rdce_state.runtime_receipt_sha256,
            ),
        ),
    }
    states = {
        "M_HEAD": build_d106_rcmr_2v_state(
            support_plus,
            signed_support,
            support_labels,
            support_ids,
            registry,
            binding=bindings["M_HEAD"],
            method_lock=rcmr_method_lock,
        ),
        "M_JOINT": build_d106_rcmr_2v_state(
            da_support_plus,
            da_support_signed,
            support_labels,
            support_ids,
            registry,
            binding=bindings["M_JOINT"],
            method_lock=rcmr_method_lock,
        ),
    }
    contexts = {
        arm: prepare_d106_rcmr_2v_scoring_context(state)
        for arm, state in states.items()
    }
    arm_predictions["M_HEAD"] = [
        score_d106_rcmr_2v_query(
            states["M_HEAD"],
            plus,
            signed,
            da_receipt_sha256=identity_da_receipt,
            context=contexts["M_HEAD"],
        ).predicted_class
        for plus, signed in zip(plus_query, signed_query, strict=True)
    ]
    arm_predictions["M_JOINT"] = [
        score_d106_rcmr_2v_query(
            states["M_JOINT"],
            plus,
            signed,
            da_receipt_sha256=rdce_state.runtime_receipt_sha256,
            context=contexts["M_JOINT"],
        ).predicted_class
        for plus, signed in zip(da_query_plus, da_query_signed, strict=True)
    ]
    if set(arm_predictions) != set(ARMS) or any(
        len(value) != len(query_ids) for value in arm_predictions.values()
    ):
        raise D106Target25EvaluatorError("four-arm prediction closure drift")

    method_lock_sha256 = getattr(rcmr_method_lock, "document_sha256", None)
    if type(method_lock_sha256) is not str or len(method_lock_sha256) != 64:
        raise D106Target25EvaluatorError("RCMR method-lock receipt unavailable")
    row: dict[str, Any] = {
        "schema": TARGET25_ROW_SCHEMA,
        "row_id": row_token,
        "receiver": receiver_token,
        "scene": scene_token,
        "K": active_k,
        "registered_classes": list(registry),
        "query_physical_ids": list(query_ids),
        "arm_predictions": arm_predictions,
        "shared_component_receipts": {
            "M_DA_M_JOINT_rdce_state_sha256": rdce_state.runtime_receipt_sha256,
            "M0_M_HEAD_identity_view_sha256": _identity_view_receipt(
                support_plus=support_plus,
                support_signed=signed_support,
                query_plus=plus_query,
                query_signed=signed_query,
            ),
            "M0_M_DA_student_t_lock_sha256": qknn_lock.lock_digest,
            "M_HEAD_M_JOINT_rcmr_method_lock_sha256": method_lock_sha256,
            "M_HEAD_state_sha256": states["M_HEAD"].state_receipt_sha256,
            "M_JOINT_state_sha256": states["M_JOINT"].state_receipt_sha256,
        },
        "query_truth_access": False,
        "query_role_access": False,
        "query_selection": False,
        "query_state_updates": 0,
    }
    row["prediction_receipt_sha256"] = _sha256(row)
    return row


__all__ = [
    "ALLOWED_K",
    "ARMS",
    "D106Target25EvaluatorError",
    "evaluate_d106_target25_state",
]

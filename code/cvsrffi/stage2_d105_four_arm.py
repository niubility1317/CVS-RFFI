"""Typed four-arm integration for the frozen D105 Stage2 experiment.

The integration owns no new estimator.  It fits one shared D105-CBRC state,
then applies the same locked Student-t qKNN and LPO-RC code to the base and
adapted representations:

``M0``      base representation + base head
``M_DA``    D105 representation + base head
``M_HEAD``  base representation + LPO-RC head
``M_JOINT`` D105 representation + LPO-RC head
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .rxid_metabias4_bundle import RXIDMetaBias4Bundle
from .stage2_d105_cbrc import (
    D105CBRCBundleHandle,
    D105CBRCError,
    D105CBRCState,
    audit_d105_cbrc_resources,
    fit_d105_cbrc_state,
    transform_d105_cbrc,
)
from .stage2_lpo_rc_qknn import (
    LPORCQKNNError,
    TypedLPORCQKNNState,
    TypedValidatedOnceP2SplitHandle,
    audit_lpo_rc_resource,
    build_lpo_rc_qknn_state,
    score_lpo_rc_qknn_logits,
)
from .stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    TypedINT8ZIDSupportBank,
    TypedSharedPSDMetric,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    normalize_zid_rows,
    score_zid_student_t_logits,
)


SCHEMA = "cvs.phase2.d105.four_arm_state.v1"
RESULT_SCHEMA = "cvs.phase2.d105.four_arm_logits.v1"
ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")


class D105FourArmError(ValueError):
    """Raised when the frozen four-arm binding or no-query-update rule drifts."""


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


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _physical_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or any(not value for value in result) or len(set(result)) != len(result):
        raise D105FourArmError(f"{name} must contain unique non-empty physical IDs")
    return result


def _base_zid(pre_relu: np.ndarray) -> np.ndarray:
    value = np.asarray(pre_relu)
    if (
        value.dtype != np.float32
        or value.ndim != 2
        or value.shape[1] != 160
        or len(value) < 1
        or not np.isfinite(value).all()
    ):
        raise D105FourArmError("base pre-ReLU z_id contract drift")
    return normalize_zid_rows(np.maximum(value, np.float32(0.0)))


def _state_receipt_payload(
    *,
    da_state: D105CBRCState,
    base_bank: TypedINT8ZIDSupportBank,
    da_bank: TypedINT8ZIDSupportBank,
    metric: TypedSharedPSDMetric,
    base_head: TypedLPORCQKNNState,
    da_head: TypedLPORCQKNNState,
    split_handle: TypedValidatedOnceP2SplitHandle,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "arms": list(ARMS),
        "active_k": da_state.active_k,
        "stage": da_state.stage,
        "da_state_receipt_sha256": da_state.state_receipt_sha256,
        "base_bank_receipt_sha256": base_bank.bank_receipt_sha256,
        "da_bank_receipt_sha256": da_bank.bank_receipt_sha256,
        "metric_receipt_sha256": metric.metric_receipt_sha256,
        "base_head_receipt_sha256": base_head.receipt_sha256,
        "da_head_receipt_sha256": da_head.receipt_sha256,
        "config_lock_digest": base_bank.config_lock_digest,
        "split_handle_digest": split_handle.handle_digest,
        "same_da_state_for_da_and_joint": True,
        "same_head_code_config_for_head_and_joint": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_truth_surface": False,
        "role_quota_or_global_assignment": False,
    }


@dataclass(frozen=True, slots=True)
class D105FourArmState:
    da_state: D105CBRCState
    base_bank: TypedINT8ZIDSupportBank
    da_bank: TypedINT8ZIDSupportBank
    metric: TypedSharedPSDMetric
    base_head: TypedLPORCQKNNState
    da_head: TypedLPORCQKNNState
    split_handle: TypedValidatedOnceP2SplitHandle
    receipt_sha256: str
    schema: str = SCHEMA


@dataclass(frozen=True, slots=True)
class D105FourArmLogits:
    m0: np.ndarray
    m_da: np.ndarray
    m_head: np.ndarray
    m_joint: np.ndarray
    query_physical_root_sha256: str
    state_receipt_sha256: str
    schema: str = RESULT_SCHEMA

    def __post_init__(self) -> None:
        rows: int | None = None
        columns: int | None = None
        for name in ("m0", "m_da", "m_head", "m_joint"):
            value = np.asarray(getattr(self, name))
            if (
                value.dtype != np.float32
                or value.ndim != 2
                or not np.isfinite(value).all()
            ):
                raise D105FourArmError(f"{name} logits contract drift")
            if rows is None:
                rows, columns = value.shape
            elif value.shape != (rows, columns):
                raise D105FourArmError("four-arm logit shapes differ")
            frozen = np.ascontiguousarray(value, dtype=np.float32).copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

    @property
    def by_arm(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(
            {
                "M0": self.m0,
                "M_DA": self.m_da,
                "M_HEAD": self.m_head,
                "M_JOINT": self.m_joint,
            }
        )


def _verify_state(state: D105FourArmState) -> None:
    if type(state) is not D105FourArmState or state.schema != SCHEMA:
        raise D105FourArmError("exact typed four-arm state required")
    payload = _state_receipt_payload(
        da_state=state.da_state,
        base_bank=state.base_bank,
        da_bank=state.da_bank,
        metric=state.metric,
        base_head=state.base_head,
        da_head=state.da_head,
        split_handle=state.split_handle,
    )
    if _sha256(payload) != state.receipt_sha256:
        raise D105FourArmError("four-arm state receipt drift")
    if (
        state.base_bank.active_k != state.da_bank.active_k
        or state.base_bank.active_k != state.da_state.active_k
        or state.base_bank.classes != state.da_bank.classes
        or state.base_head.classes != state.da_head.classes
        or state.base_head.classes != state.base_bank.classes
        or state.metric.config_lock_digest != state.base_bank.config_lock_digest
        or state.da_bank.config_lock_digest != state.base_bank.config_lock_digest
    ):
        raise D105FourArmError("four-arm component binding drift")


def build_d105_four_arm_state(
    bundle: RXIDMetaBias4Bundle,
    bundle_handle: D105CBRCBundleHandle,
    support_pre_relu: np.ndarray,
    support_zdom: np.ndarray,
    support_labels: Sequence[str],
    support_physical_ids: Sequence[str],
    registered_classes: Sequence[str],
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    *,
    config: Phase1ZIDStudentTLock,
    split_handle: TypedValidatedOnceP2SplitHandle,
    active_k: int,
    stage: str,
    support_receipt_sha256: str,
) -> D105FourArmState:
    """Fit the two representations and two heads from one support row only."""

    if type(config) is not Phase1ZIDStudentTLock or config.active_k != active_k:
        raise D105FourArmError("four-arm Phase1 lock/active K drift")
    identifiers = _physical_ids(support_physical_ids, "support physical IDs")
    da_state = fit_d105_cbrc_state(
        bundle,
        bundle_handle,
        support_pre_relu,
        support_zdom,
        support_labels,
        identifiers,
        registered_classes,
        old_classes,
        new_classes,
        active_k=active_k,
        stage=stage,
        support_receipt_sha256=support_receipt_sha256,
    )
    base_support = _base_zid(support_pre_relu)
    da_support = transform_d105_cbrc(da_state, support_pre_relu)
    base_bank = build_typed_zid_support_bank(
        base_support, support_labels, registered_classes, config=config
    )
    da_bank = build_typed_zid_support_bank(
        da_support, support_labels, registered_classes, config=config
    )
    metric = identity_shared_psd_metric(config=config)
    base_head = build_lpo_rc_qknn_state(
        base_bank,
        base_support,
        support_labels,
        registered_classes,
        metric=metric,
        support_physical_ids=identifiers,
        split_handle=split_handle,
    )
    da_head = build_lpo_rc_qknn_state(
        da_bank,
        da_support,
        support_labels,
        registered_classes,
        metric=metric,
        support_physical_ids=identifiers,
        split_handle=split_handle,
    )
    payload = _state_receipt_payload(
        da_state=da_state,
        base_bank=base_bank,
        da_bank=da_bank,
        metric=metric,
        base_head=base_head,
        da_head=da_head,
        split_handle=split_handle,
    )
    state = D105FourArmState(
        da_state=da_state,
        base_bank=base_bank,
        da_bank=da_bank,
        metric=metric,
        base_head=base_head,
        da_head=da_head,
        split_handle=split_handle,
        receipt_sha256=_sha256(payload),
    )
    _verify_state(state)
    return state


def score_d105_four_arm_logits(
    state: D105FourArmState,
    query_pre_relu: np.ndarray,
    *,
    query_physical_ids: Sequence[str],
    chunk_size: int | None = None,
) -> D105FourArmLogits:
    """Score all four arms after one full-root check, optionally in chunks."""

    _verify_state(state)
    query_rows = np.asarray(query_pre_relu)
    identifiers = _physical_ids(query_physical_ids, "query physical IDs")
    if len(identifiers) != len(query_rows):
        raise D105FourArmError("query rows and physical IDs must align")
    query_root = _sha256(sorted(identifiers))
    if query_root != state.split_handle.query_physical_root_sha256:
        raise D105FourArmError("validated split handle query physical root mismatch")
    if chunk_size is None:
        chunk_size = len(query_rows)
    if type(chunk_size) is not int or chunk_size < 1:
        raise D105FourArmError("chunk_size must be a positive integer or None")
    receipt_before = state.receipt_sha256
    chunks: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    for start in range(0, len(query_rows), chunk_size):
        local = np.ascontiguousarray(
            query_rows[start : start + chunk_size], dtype=query_rows.dtype
        )
        base_query = _base_zid(local)
        da_query = transform_d105_cbrc(state.da_state, local)
        chunks["M0"].append(
            score_zid_student_t_logits(
                state.base_bank, base_query, metric=state.metric
            )
        )
        chunks["M_DA"].append(
            score_zid_student_t_logits(
                state.da_bank, da_query, metric=state.metric
            )
        )
        chunks["M_HEAD"].append(
            score_lpo_rc_qknn_logits(
                state.base_head,
                base_query,
                bank=state.base_bank,
                metric=state.metric,
                split_handle=state.split_handle,
            )
        )
        chunks["M_JOINT"].append(
            score_lpo_rc_qknn_logits(
                state.da_head,
                da_query,
                bank=state.da_bank,
                metric=state.metric,
                split_handle=state.split_handle,
            )
        )
    m0 = np.concatenate(chunks["M0"], axis=0)
    m_da = np.concatenate(chunks["M_DA"], axis=0)
    m_head = np.concatenate(chunks["M_HEAD"], axis=0)
    m_joint = np.concatenate(chunks["M_JOINT"], axis=0)
    _verify_state(state)
    if state.receipt_sha256 != receipt_before:
        raise D105FourArmError("query scoring changed the four-arm state")
    if state.da_state.active_k == 1:
        if not np.array_equal(m_head, m0) or not np.array_equal(m_joint, m_da):
            raise D105FourArmError("K1 HEAD identity closure failed")
    return D105FourArmLogits(
        m0=m0,
        m_da=m_da,
        m_head=m_head,
        m_joint=m_joint,
        query_physical_root_sha256=query_root,
        state_receipt_sha256=state.receipt_sha256,
    )


def audit_d105_four_arm_resources(state: D105FourArmState) -> dict[str, Any]:
    """Expose component receipts without aggregating unlike resource units."""

    _verify_state(state)
    return {
        "schema": "cvs.phase2.d105.four_arm_resource_audit.v1",
        "da": audit_d105_cbrc_resources(state.da_state),
        "base_head": audit_lpo_rc_resource(
            state.base_head,
            state.base_bank,
            state.metric,
            state.split_handle,
        ),
        "da_head": audit_lpo_rc_resource(
            state.da_head,
            state.da_bank,
            state.metric,
            state.split_handle,
        ),
        "query_state_updates": 0,
        "query_rows_used_for_fit": 0,
    }


__all__ = [
    "ARMS",
    "D105FourArmError",
    "D105FourArmLogits",
    "D105FourArmState",
    "audit_d105_four_arm_resources",
    "build_d105_four_arm_state",
    "score_d105_four_arm_logits",
]

"""Frozen four-arm source-held G1 composition for D114 HBPD-qKNN."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from cvsrffi.stage2_d112_seam_qknn import (
    D112SEAMState,
    score_d112_seam_source_held_g1_logits,
)
from cvsrffi.stage2_d114_hbpd_qknn import (
    D114State,
    score_d114_hbpd_logits,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    TypedINT8ZIDSupportBank,
    identity_shared_psd_metric,
    normalize_zid_rows,
    score_zid_student_t_logits,
)


ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")


class D114G1Error(ValueError):
    """Raised when the frozen D114 four-arm composition drifts."""


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float32)
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    frozen.setflags(write=False)
    return frozen


def _verify(
    hbpd: D114State, head: D112SEAMState, bank: TypedINT8ZIDSupportBank
) -> None:
    if (
        type(hbpd) is not D114State
        or type(head) is not D112SEAMState
        or type(bank) is not TypedINT8ZIDSupportBank
        or hbpd.classes != tuple(bank.classes)
        or head.classes != tuple(bank.classes)
        or hbpd.bank_receipt_sha256 != bank.bank_receipt_sha256
        or head.bank_receipt_sha256 != bank.bank_receipt_sha256
        or hbpd.config_lock_digest != bank.config_lock_digest
        or head.config_lock_digest != bank.config_lock_digest
        or float(bank.config.kernel_volume_gamma) != 1.0
    ):
        raise D114G1Error("D114 G1 state/bank binding drift")


def score_d114_g1_arms(
    hbpd: D114State,
    head: D112SEAMState,
    bank: TypedINT8ZIDSupportBank,
    held_query_zid: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Score four frozen arms; truth never enters this API."""

    _verify(hbpd, head, bank)
    m0 = score_zid_student_t_logits(
        bank,
        held_query_zid,
        metric=identity_shared_psd_metric(config=bank.config),
    )
    m_da = score_d114_hbpd_logits(hbpd, bank, held_query_zid)
    m_head = score_d112_seam_source_held_g1_logits(head, bank, held_query_zid)
    joint = np.asarray(m_da, dtype=np.float64).copy()
    active = np.flatnonzero(head.information_valid)
    if len(active):
        query = normalize_zid_rows(held_query_zid).astype(np.float64)
        dimension = bank.config.kernel_effective_dim
        nu = float(bank.config.student_nu)
        for class_index in active:
            rho = float(head.rho[class_index])
            h = float(hbpd.predictive_bandwidth[class_index])
            anchor = np.asarray(head.anchors[class_index], dtype=np.float64)
            distance = np.maximum(
                2.0 * (1.0 - np.clip(query @ anchor, -1.0, 1.0)), 0.0
            )
            anchor_kernel = (
                -dimension * math.log(h)
                - 0.5
                * (nu + dimension)
                * np.log1p(distance / (nu * h * h))
            )
            joint[:, class_index] = np.logaddexp(
                math.log1p(-rho) + joint[:, class_index],
                math.log(rho) + anchor_kernel,
            )
    if not np.isfinite(joint).all():
        raise D114G1Error("D114 joint logits became non-finite")
    values = {
        "M0": _readonly(m0),
        "M_DA": _readonly(m_da),
        "M_HEAD": _readonly(m_head),
        "M_JOINT": _readonly(joint),
    }
    if tuple(values) != ARMS:
        raise D114G1Error("D114 four-arm closure drift")
    return MappingProxyType(values)


def audit_d114_g1_states(
    hbpd: D114State, head: D112SEAMState, bank: TypedINT8ZIDSupportBank
) -> Mapping[str, Any]:
    _verify(hbpd, head, bank)
    old = set(head.old_class_indices)
    return MappingProxyType(
        {
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "truth_role_quota_inputs": 0,
            "head_active_count": int(np.count_nonzero(head.information_valid)),
            "head_new_active_count": int(
                sum(bool(head.information_valid[index]) for index in range(len(bank.classes)) if index not in old)
            ),
            "hbpd_state_numeric_bytes": int(
                hbpd.resource_receipt["persistent_numeric_bytes"]
            ),
            "head_state_numeric_bytes": int(
                head.resource_receipt["persistent_numeric_bytes"]
            ),
            "query_dependent_state_bytes": 0,
        }
    )


__all__ = ["ARMS", "D114G1Error", "audit_d114_g1_states", "score_d114_g1_arms"]

#!/usr/bin/env python3
"""Run D123's frozen four-arm source-held G1 on D122's sealed lifecycle.

The lifecycle implementation is reused unchanged.  This file replaces only
the two head arms with D123 LOO-CRES and changes the immutable output identity.
Prediction still completes before the independent truth-side score command.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_ROOT.parent
for value in (SCRIPT_ROOT, CODE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_d122_g1_sourceheld_standalone as base  # noqa: E402
from cvsrffi.stage2_d123_loo_cres import (  # noqa: E402
    audit_d123_loo_cres_state,
    fit_d123_loo_cres_ground_head_source_held_g1_state,
    fit_d123_loo_cres_rdce_ground_head_source_held_g1_state,
    score_d123_loo_cres_ground_head_source_held_g1_logits,
    score_d123_loo_cres_rdce_ground_head_source_held_g1_logits,
)


CANDIDATE_ID = "D123_LOO_CRES_GROUND_HEAD"
PREDICTION_SCHEMA = "cvs.d123.loo_cres_ground_head.sourceheld.predictions.v1"
SCORE_SCHEMA = "cvs.d123.loo_cres_ground_head.sourceheld.scores.v1"
ARMS = base.ARMS


def _build_four_arm_predictions(
    *,
    bundle: Any,
    rdce_asset: Any,
    support_signed: np.ndarray,
    labels: tuple[str, ...],
    query_signed: np.ndarray,
    registry: tuple[str, ...],
    k_shot: int,
    package_sha256: str,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Build D123's frozen 2x2 row without a truth or held-role input."""

    support_plus = np.ascontiguousarray(np.maximum(support_signed, np.float32(0.0)))
    query_plus = np.ascontiguousarray(np.maximum(query_signed, np.float32(0.0)))
    lock = base._lock(k_shot, package_sha256)

    identity_bank = base.build_typed_zid_support_bank(
        support_plus, labels, registry, config=lock
    )
    m0_logits = base._baseline_logits(identity_bank, query_plus)
    head_state = fit_d123_loo_cres_ground_head_source_held_g1_state(
        bundle, identity_bank
    )
    head_logits = score_d123_loo_cres_ground_head_source_held_g1_logits(
        head_state, identity_bank, query_plus
    )

    da_state = base.fit_rdce_sourceheld_state(rdce_asset, support_plus, labels, k_shot)
    da_support = base.apply_rdce_state(da_state, support_plus)
    da_query = base.apply_rdce_state(da_state, query_plus)
    da_bank = base.build_typed_zid_support_bank(
        da_support, labels, registry, config=lock
    )
    da_logits = base._baseline_logits(da_bank, da_query)
    joint_state = fit_d123_loo_cres_rdce_ground_head_source_held_g1_state(
        bundle, da_bank, support_plus, labels, da_state
    )
    joint_logits = score_d123_loo_cres_rdce_ground_head_source_held_g1_logits(
        joint_state, da_bank, da_query
    )

    old_indices = np.asarray(joint_state.old_class_indices, dtype=np.int64)
    non_old = np.ones(len(registry), dtype=bool)
    non_old[old_indices] = False
    if (
        not np.array_equal(head_logits[:, non_old], m0_logits[:, non_old])
        or not np.array_equal(joint_logits[:, non_old], da_logits[:, non_old])
    ):
        raise base.D122G1Error("D123 non-old logit boundary is not bit-exact")

    head_audit = base._jsonable(audit_d123_loo_cres_state(head_state))
    joint_audit = base._jsonable(audit_d123_loo_cres_state(joint_state))
    joint_reference_audit = base._jsonable(
        base.audit_d122_rdce_ground_head_state(joint_state.reference_state)
    )
    base._assert_query_zero({"M_HEAD": head_audit, "M_JOINT": joint_audit})
    if (
        joint_reference_audit.get("global_component_valid") is not True
        or joint_reference_audit.get("global_failure_reason") != "NONE"
        or joint_reference_audit.get("rdce_state_receipt_sha256")
        != str(da_state["receipt"])
    ):
        raise base.D122G1Error("D123 RDCE reference binding is fail-closed")

    logits = {
        "M0": m0_logits,
        "M_DA": da_logits,
        "M_HEAD": head_logits,
        "M_JOINT": joint_logits,
    }
    predictions = {
        arm: list(base.unique_d122_argmax(value, registry)) for arm, value in logits.items()
    }
    if set(predictions) != set(ARMS) or any(
        len(values) != len(query_plus) for values in predictions.values()
    ):
        raise base.D122G1Error("D123 four-arm prediction closure drift")
    return predictions, {
        "student_t_lock_sha256": lock.lock_digest,
        "M_DA_M_JOINT_rdce_state_sha256": str(da_state["receipt"]),
        "M_HEAD_state_audit": head_audit,
        "M_JOINT_state_audit": joint_audit,
        "M_JOINT_D122_reference_audit": joint_reference_audit,
        "arm_logits": {arm: base._array_receipt(value) for arm, value in logits.items()},
        "non_old_logit_boundary_bit_exact": True,
    }


def _install_overlay() -> None:
    base.CANDIDATE_ID = CANDIDATE_ID
    base.PREDICTION_SCHEMA = PREDICTION_SCHEMA
    base.SCORE_SCHEMA = SCORE_SCHEMA
    base._build_four_arm_predictions = _build_four_arm_predictions


def main(argv: Sequence[str] | None = None) -> int:
    _install_overlay()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

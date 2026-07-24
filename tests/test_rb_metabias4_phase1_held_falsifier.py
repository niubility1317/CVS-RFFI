from __future__ import annotations

import pytest

from cvsrffi.phase1_rb_metabias4_bundle import RBMetaBias4Config
from cvsrffi.rb_metabias4_phase1_held_falsifier import (
    RBMetaBias4HeldError,
    run_rb_metabias4_phase1_held_falsifier,
)
from test_phase1_rb_metabias4_bundle import (
    SHA_A,
    SHA_B,
    SHA_C,
    synthetic_tap_archive,
)


def _run(arrays):
    return run_rb_metabias4_phase1_held_falsifier(
        arrays,
        checkpoint_sha256=SHA_A,
        runtime_sha256=SHA_B,
        method_lock_sha256=SHA_C,
        config=RBMetaBias4Config(),
    )


def test_held_falsifier_closes_receiver_k_loco_tx_permutation_and_determinism() -> None:
    arrays = synthetic_tap_archive()
    first = _run(arrays)
    second = _run(arrays)
    assert first["receipt_sha256"] == second["receipt_sha256"]
    assert first["receiver_held_complete_k1_k5_k10"] is True
    assert len(first["receiver_held_rows"]) == 9
    assert first["class_loco_complete"] is True
    assert len(first["class_loco_rows"]) == 9
    assert isinstance(first["class_loco_gate_passed"], bool)
    assert isinstance(first["all_gates_passed"], bool)
    assert first["all_support_query_physical_disjoint"] is True
    assert first["all_query_rows_used_for_fit_zero"] is True
    assert first["target_access"] is False
    assert first["formal_target_query_access"] is False
    assert first["phase2_old_new_claimed"] is False
    assert "not Phase2 old/new" in first["anonymous_class_partition_semantics"]
    assert first["tx_leakage_receipt"]["raw_zdom_used_for_bank_matching"] is False
    assert (
        first["tx_leakage_receipt"]["maximum_balanced_accuracy"]
        >= first["tx_leakage_receipt"]["mean_balanced_accuracy"]
    )
    assert first["tx_leakage_receipt"]["passed"] == (
        first["tx_leakage_receipt"]["maximum_balanced_accuracy"] <= 0.25
    )
    assert (
        first["label_permutation_receipt"]["consistent_permutation_equivalent"]
        is True
    )
    assert set(first["performance_summary_by_k"]) == {"1", "5", "10"}
    for summary in first["performance_summary_by_k"].values():
        assert "anonymous_group_a_net_correct" in summary
        assert "anonymous_group_b_net_correct" in summary
        assert "old_net_correct" not in summary
        assert "new_net_correct" not in summary
    assert first["claim_boundary"] == "PHASE1_SOURCE_ONLY_NOT_TARGET_PERFORMANCE"
    assert first["status"] in {
        "PHASE1_HELD_FALSIFIER_PASS",
        "PHASE1_HELD_FALSIFIER_REJECT",
    }


def test_held_falsifier_rejects_incomplete_k10_receiver_cell() -> None:
    arrays = synthetic_tap_archive(per_cell=5)
    with pytest.raises(RBMetaBias4HeldError, match="K10"):
        _run(arrays)

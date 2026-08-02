from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi.stage2_d112_seam_bundle import (
    FEATURE_DIM,
    build_d112_source_held_g1_bundle,
)
from cvsrffi.stage2_d112_seam_qknn import (
    fit_d112_ground_head_source_held_g1_state,
)
from cvsrffi.stage2_d114_hbpd_g1 import (
    ARMS,
    D114G1Error,
    audit_d114_g1_states,
    score_d114_g1_arms,
)
from cvsrffi.stage2_d114_hbpd_qknn import build_d114_bundle, fit_d114_state
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
)
from scripts.run_d114_hbpd_g1_sourceheld import _add_factorial_interaction


OLD = tuple(f"old-{index}" for index in range(6))
CLASSES = OLD + ("new-0",)


def _lock(k: int, salt: str = "3") -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        k, 3.0, 12, 1.0, 0.2, 2.0, 0.5, 2.0, 1.0, salt * 64, "4" * 64
    )


def _support(k: int) -> tuple[np.ndarray, list[str]]:
    rows, labels = [], []
    for index, name in enumerate(CLASSES):
        base = np.zeros(FEATURE_DIM, dtype=np.float64)
        base[10 + index] = 1.0
        for shot in range(k):
            value = base.copy()
            value[40 + shot] += 0.02 + 0.012 * (shot - (k - 1) / 2.0)
            value /= np.linalg.norm(value)
            rows.append(value)
            labels.append(name)
    return np.asarray(rows, dtype=np.float32), labels


def _bank(k: int, salt: str = "3"):
    support, labels = _support(k)
    return build_typed_zid_support_bank(
        support, labels, CLASSES, config=_lock(k, salt)
    )


def _d112_bundle():
    ground = np.zeros((6, FEATURE_DIM), dtype=np.float64)
    for index in range(6):
        ground[index, 10 + index] = 1.0
    q0 = np.sum(ground, axis=0)
    q0 /= np.linalg.norm(q0)
    basis = np.zeros((3, FEATURE_DIM), dtype=np.float64)
    basis[np.arange(3), np.arange(3)] = 1.0
    return build_d112_source_held_g1_bundle(
        class_registry=OLD,
        g=ground,
        q0=q0,
        U=basis,
        sigma0_r=np.linspace(0.0020, 0.0025, 6),
        sigma0_amb=np.linspace(0.0020, 0.0025, 6),
        v_g_r=np.linspace(0.0010, 0.0015, 6),
        v_g_amb=np.linspace(0.0010, 0.0015, 6),
        tau_h_r=0.004,
        checkpoint_sha256="1" * 64,
        source_aggregate_sha256="2" * 64,
        phase1_seal_sha256="5" * 64,
        source_held_split_sha256="6" * 64,
        global_bundle_valid=True,
    )


def _d114_bundle():
    return build_d114_bundle(
        class_registry=OLD,
        sigma0_old=np.linspace(0.0004, 0.0024, 6),
        checkpoint_sha256="1" * 64,
        source_aggregate_sha256="2" * 64,
        allowed_config_lock_digests=tuple(_lock(k).lock_digest for k in (1, 5, 10)),
    )


@pytest.mark.parametrize("k", [1, 5, 10])
def test_four_arms_are_finite_and_factorially_identifiable(k: int) -> None:
    bank = _bank(k)
    hbpd = fit_d114_state(_d114_bundle(), bank)
    head = fit_d112_ground_head_source_held_g1_state(_d112_bundle(), bank)
    query, _labels = _support(k)
    arms = score_d114_g1_arms(hbpd, head, bank, query)
    assert tuple(arms) == ARMS
    assert all(value.shape == (len(query), len(CLASSES)) for value in arms.values())
    assert all(np.isfinite(value).all() and not value.flags.writeable for value in arms.values())
    assert np.max(np.abs(arms["M_DA"] - arms["M0"])) > 0.0
    assert np.max(np.abs(arms["M_HEAD"][:, :6] - arms["M0"][:, :6])) > 0.0
    assert np.max(np.abs(arms["M_JOINT"][:, :6] - arms["M_DA"][:, :6])) > 0.0
    assert np.array_equal(arms["M_HEAD"][:, 6], arms["M0"][:, 6])
    assert np.array_equal(arms["M_JOINT"][:, 6], arms["M_DA"][:, 6])
    audit = audit_d114_g1_states(hbpd, head, bank)
    assert audit["query_rows_used_for_fit"] == 0
    assert audit["query_state_updates"] == 0
    assert audit["head_new_active_count"] == 0


def test_fit_surfaces_have_no_query_truth_role_or_quota() -> None:
    assert set(inspect.signature(fit_d114_state).parameters) == {"bundle", "bank"}
    forbidden = {"truth", "role", "quota", "query_labels"}
    assert not forbidden & set(inspect.signature(score_d114_g1_arms).parameters)


def test_foreign_bank_is_rejected_before_scoring() -> None:
    bank = _bank(1)
    hbpd = fit_d114_state(_d114_bundle(), bank)
    head = fit_d112_ground_head_source_held_g1_state(_d112_bundle(), bank)
    foreign = _bank(1, "a")
    query, _labels = _support(1)
    with pytest.raises(D114G1Error, match="binding drift"):
        score_d114_g1_arms(hbpd, head, foreign, query)


def test_score_adds_exact_four_arm_factorial_interaction() -> None:
    metrics = {
        "M0": {name: 0.50 for name in ("old_balanced_accuracy", "seen_new_accuracy", "H_old_new", "old_floor")},
        "M_DA": {name: 0.55 for name in ("old_balanced_accuracy", "seen_new_accuracy", "H_old_new", "old_floor")},
        "M_HEAD": {name: 0.60 for name in ("old_balanced_accuracy", "seen_new_accuracy", "H_old_new", "old_floor")},
        "M_JOINT": {name: 0.68 for name in ("old_balanced_accuracy", "seen_new_accuracy", "H_old_new", "old_floor")},
    }
    row = {"arm_metrics": metrics, "same_row_effects": {}}
    value = {
        "performance_rows": [row for _ in range(63)],
        "negative_tail_row_counts": {},
        "score_set_receipt_sha256": "0" * 64,
    }
    actual = _add_factorial_interaction(value)
    for result_row in actual["performance_rows"]:
        assert all(
            np.isclose(delta, 0.03)
            for delta in result_row["same_row_effects"]["FACTORIAL_INTERACTION"].values()
        )
    assert actual["negative_tail_row_counts"]["FACTORIAL_INTERACTION"] == {
        "old_balanced_accuracy": 0,
        "seen_new_accuracy": 0,
        "H_old_new": 0,
        "old_floor": 0,
    }

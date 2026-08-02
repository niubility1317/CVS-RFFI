from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi.stage2_d114_hbpd_qknn import (
    D114HBPDError,
    build_d114_bundle,
    fit_d114_state,
    score_d114_hbpd_logits,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)


P = 160
OLD = tuple(f"old-{index}" for index in range(6))
CLASSES = OLD + ("new-0", "new-1")


def _lock(k: int, *, foreign: bool = False) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        k,
        3.0,
        12,
        1.0,
        0.2,
        2.0,
        0.5,
        2.0,
        1.0,
        ("a" if foreign else "3") * 64,
        ("b" if foreign else "4") * 64,
    )


def _bundle(order: tuple[int, ...] | None = None):
    selected = tuple(range(6)) if order is None else order
    sigma = np.asarray([0.0004, 0.0007, 0.0010, 0.0013, 0.0018, 0.0024])
    return build_d114_bundle(
        class_registry=tuple(OLD[index] for index in selected),
        sigma0_old=sigma[np.asarray(selected)],
        checkpoint_sha256="1" * 64,
        source_aggregate_sha256="2" * 64,
        allowed_config_lock_digests=tuple(_lock(k).lock_digest for k in (1, 5, 10)),
    )


def _support(k: int):
    rows = []
    labels = []
    for index, name in enumerate(CLASSES):
        base = np.zeros(P, dtype=np.float64)
        base[10 + index] = 1.0
        for shot in range(k):
            value = base.copy()
            if k > 1:
                value[40 + shot] += (shot - (k - 1) / 2.0) * 0.01
                value /= np.linalg.norm(value)
            rows.append(value)
            labels.append(name)
    return np.asarray(rows, dtype=np.float32), labels


def _bank(k: int, *, foreign: bool = False):
    support, labels = _support(k)
    return build_typed_zid_support_bank(
        support, labels, CLASSES, config=_lock(k, foreign=foreign)
    )


def test_k1_is_nonidentity_and_new_classes_share_the_pooled_prior():
    bank = _bank(1)
    state = fit_d114_state(_bundle(), bank)
    query, _labels = _support(1)
    m0 = score_zid_student_t_logits(
        bank, query, metric=identity_shared_psd_metric(config=bank.config)
    )
    hbpd = score_d114_hbpd_logits(state, bank, query)
    new = [bank.classes.index("new-0"), bank.classes.index("new-1")]
    assert np.max(np.abs(hbpd - m0)) > 0.0
    assert state.predictive_bandwidth[new[0]] == state.predictive_bandwidth[new[1]]
    assert len(np.unique(state.predictive_bandwidth)) > 2


def test_k1_k5_k10_are_finite_and_fit_has_no_query_argument():
    assert tuple(inspect.signature(fit_d114_state).parameters) == ("bundle", "bank")
    for k in (1, 5, 10):
        bank = _bank(k)
        state = fit_d114_state(_bundle(), bank)
        query, _labels = _support(k)
        output = score_d114_hbpd_logits(state, bank, query[:9])
        assert output.shape == (len(query[:9]), len(CLASSES))
        assert np.isfinite(output).all()
        assert state.resource_receipt["query_dependent_state_bytes"] == 0


def test_old_registry_permutation_preserves_all_class_scores():
    bank = _bank(5)
    query, _labels = _support(5)
    direct = fit_d114_state(_bundle(), bank)
    order = (5, 3, 1, 4, 2, 0)
    permuted = fit_d114_state(_bundle(order), bank)
    assert np.array_equal(
        score_d114_hbpd_logits(direct, bank, query[:11]),
        score_d114_hbpd_logits(permuted, bank, query[:11]),
    )


def test_foreign_phase1_lock_is_rejected_before_fit():
    with pytest.raises(D114HBPDError, match="config lineage mismatch"):
        fit_d114_state(_bundle(), _bank(1, foreign=True))

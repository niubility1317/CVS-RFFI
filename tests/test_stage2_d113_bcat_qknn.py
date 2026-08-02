from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi.stage2_d113_bcat_qknn import (
    D113BCATError,
    bcat_inverse,
    build_d113_bundle,
    fit_d113_state,
    score_d113_da_logits,
    score_d113_joint_logits,
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


def _lock(k: int) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        k, 3.0, 12, 1.0, 0.2, 2.0, 0.5, 2.0, 1.0, "3" * 64, "4" * 64
    )


def _ground() -> np.ndarray:
    value = np.zeros((6, P), dtype=np.float32)
    value[np.arange(6), 10 + np.arange(6)] = 1.0
    return value


def _bundle(order: tuple[int, ...] | None = None):
    selected = tuple(range(6)) if order is None else order
    return build_d113_bundle(
        class_registry=tuple(OLD[index] for index in selected),
        ground=_ground()[np.asarray(selected)],
        sigma0=np.asarray([0.002 + index * 1.0e-5 for index in selected]),
        v_ground=np.asarray([0.001 + index * 1.0e-5 for index in selected]),
        quantization_mse=np.asarray([1.0e-6] * 6),
        tau_b2=0.004,
        checkpoint_sha256="1" * 64,
        source_aggregate_sha256="2" * 64,
        allowed_config_lock_digests=tuple(_lock(k).lock_digest for k in (1, 5, 10)),
    )


def _support(k: int):
    shift = np.zeros(P, dtype=np.float64)
    shift[:3] = [0.08, -0.04, 0.02]
    rows = []
    labels = []
    for index, name in enumerate(CLASSES):
        latent = np.zeros(P, dtype=np.float64)
        latent[10 + index] = 1.0
        observed = latent + shift
        observed /= np.linalg.norm(observed)
        for shot in range(k):
            local = observed.copy()
            if k > 1:
                local[40 + shot] += (shot - (k - 1) / 2.0) * 1.0e-3
                local /= np.linalg.norm(local)
            rows.append(local)
            labels.append(name)
    return np.asarray(rows, dtype=np.float32), labels, shift


def _bank(k: int):
    support, labels, _ = _support(k)
    return build_typed_zid_support_bank(support, labels, CLASSES, config=_lock(k))


def test_exact_inverse_recovers_effective_additive_model():
    rng = np.random.default_rng(9)
    latent = rng.normal(size=(12, P))
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)
    b = np.zeros(P)
    b[:3] = [0.12, -0.08, 0.04]
    observed = latent + b
    observed /= np.linalg.norm(observed, axis=1, keepdims=True)
    recovered = bcat_inverse(observed, b)
    assert np.allclose(recovered, latent, atol=1.0e-10, rtol=0.0)


def test_support_only_fit_and_four_arm_column_contract():
    bank = _bank(1)
    bundle = _bundle()
    state = fit_d113_state(bundle, bank)
    query, _labels, _shift = _support(1)
    query = query[:16]
    metric = identity_shared_psd_metric(config=bank.config)
    m0 = score_zid_student_t_logits(bank, query, metric=metric)
    da = score_d113_da_logits(state, bank, query)
    joint = score_d113_joint_logits(state, bundle, bank, query)
    new = [bank.classes.index("new-0"), bank.classes.index("new-1")]
    assert np.linalg.norm(state.b) > 0.0
    assert np.linalg.norm(state.b) < 0.5
    assert np.max(np.abs(da - m0)) > 0.0
    assert np.all(state.rho[np.asarray(state.old_class_indices)] > 0.0)
    assert np.array_equal(joint[:, new], da[:, new])
    assert np.max(np.abs(joint[:, np.asarray(state.old_class_indices)] - da[:, np.asarray(state.old_class_indices)])) > 0.0


def test_k1_k5_k10_are_finite_and_query_is_not_a_fit_argument():
    assert tuple(inspect.signature(fit_d113_state).parameters) == ("bundle", "bank")
    for k in (1, 5, 10):
        bank = _bank(k)
        state = fit_d113_state(_bundle(), bank)
        query, _labels, _shift = _support(k)
        selected = query[:9]
        output = score_d113_joint_logits(state, _bundle(), bank, selected)
        assert output.shape == (len(selected), len(CLASSES))
        assert np.isfinite(output).all()
        assert state.resource_receipt["query_dependent_state_bytes"] == 0


def test_old_registry_permutation_keeps_common_shift_and_scores():
    bank = _bank(1)
    order = (5, 3, 1, 4, 2, 0)
    direct = fit_d113_state(_bundle(), bank)
    permuted_bundle = _bundle(order)
    permuted = fit_d113_state(permuted_bundle, bank)
    query, _labels, _shift = _support(1)
    assert np.allclose(direct.b, permuted.b, atol=1.0e-12, rtol=0.0)
    assert np.array_equal(
        score_d113_joint_logits(direct, _bundle(), bank, query[:8]),
        score_d113_joint_logits(permuted, permuted_bundle, bank, query[:8]),
    )


def test_joint_score_rejects_a_same_registry_different_bundle():
    bank = _bank(1)
    bundle = _bundle()
    state = fit_d113_state(bundle, bank)
    altered = build_d113_bundle(
        class_registry=OLD,
        ground=_ground(),
        sigma0=np.full(6, 0.003),
        v_ground=np.full(6, 0.001),
        quantization_mse=np.full(6, 1.0e-6),
        tau_b2=0.004,
        checkpoint_sha256="1" * 64,
        source_aggregate_sha256="2" * 64,
        allowed_config_lock_digests=tuple(_lock(k).lock_digest for k in (1, 5, 10)),
    )
    query, _labels, _shift = _support(1)
    with pytest.raises(D113BCATError, match="bundle/registry drift"):
        score_d113_joint_logits(state, altered, bank, query[:4])


def test_fit_rejects_a_support_bank_from_another_phase1_lock():
    support, labels, _shift = _support(1)
    foreign_lock = Phase1ZIDStudentTLock(
        1, 3.0, 12, 1.0, 0.2, 2.0, 0.5, 2.0, 1.0, "a" * 64, "b" * 64
    )
    foreign_bank = build_typed_zid_support_bank(
        support, labels, CLASSES, config=foreign_lock
    )
    with pytest.raises(D113BCATError, match="config lineage mismatch"):
        fit_d113_state(_bundle(), foreign_bank)

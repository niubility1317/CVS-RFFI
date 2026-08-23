from __future__ import annotations

import numpy as np

from cvsrffi.stage2_m24_features import physical_if256
from cvsrffi.stage2_m25_anchored_residual import B3, fit_m25_anchored_residual
from cvsrffi.stage2_m28_local_flip_risk import (
    C1,
    C2,
    apply_local_flip_policy,
    fit_local_flip_risk_model,
    fit_m28_local_flip_risk,
)
from test_stage2_m26_td_src256 import _fixture


def _risk_support() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(82801)
    classes = ("old-0", "old-1", "new-0")
    centres = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.1],
        ],
        dtype=np.float32,
    )
    shift = np.linspace(-0.12, 0.12, 6, dtype=np.float32)
    rows = []
    labels = []
    for name, centre in zip(classes, centres, strict=True):
        for _ in range(5):
            rows.append(centre + shift + 0.015 * rng.normal(size=6))
            labels.append(name)
    support = np.asarray(rows, dtype=np.float32)
    base_scores = support @ centres.T
    return support, np.asarray(labels), base_scores.astype(np.float32), classes


def test_local_policy_emits_only_exact_b0_or_b3_rows_and_distinguishes_c1_c2() -> None:
    b0 = np.asarray(
        [[2.0, 1.0, 0.0], [0.8, 0.7, 0.1], [0.8, 0.7, 0.1], [0.8, 0.7, 0.1]],
        dtype=np.float32,
    )
    b3 = np.asarray(
        [[2.1, 1.0, 0.0], [0.7, 0.9, 0.1], [0.7, 0.1, 0.9], [0.7, 0.9, 0.1]],
        dtype=np.float32,
    )
    representation = np.asarray(
        [[3.0, 1.0, 0.0], [0.1, 0.9, 0.2], [0.1, 0.9, 0.8], [0.1, 0.9, 0.2]],
        dtype=np.float32,
    )
    conformal = np.asarray(
        [[0.8, 0.2, 0.1], [0.1, 0.8, 0.2], [0.1, 0.7, 0.8], [0.7, 0.2, 0.1]],
        dtype=np.float32,
    )
    posterior_mean = np.asarray(
        [[0.7, 0.7], [0.7, 0.7], [0.7, 0.7], [0.7, 0.7]], dtype=np.float32
    )
    posterior_lower = np.asarray(
        [[0.4, 0.4], [0.4, 0.4], [0.4, 0.4], [0.4, 0.4]], dtype=np.float32
    )
    rank_events = np.full((4, 2), 8, dtype=np.int64)
    class_stability = np.full((4, 3), 0.8, dtype=np.float32)
    radial = np.full(4, 0.8, dtype=np.float32)

    c1, audit1 = apply_local_flip_policy(
        b0,
        b3,
        representation,
        conformal,
        radial,
        posterior_mean,
        posterior_lower,
        rank_events,
        class_stability,
        arm=C1,
        k_shot=5,
    )
    c2, audit2 = apply_local_flip_policy(
        b0,
        b3,
        representation,
        conformal,
        radial,
        posterior_mean,
        posterior_lower,
        rank_events,
        class_stability,
        arm=C2,
        k_shot=5,
    )
    np.testing.assert_array_equal(c1[0], b3[0])
    np.testing.assert_array_equal(c1[1], b3[1])
    np.testing.assert_array_equal(c1[2], b0[2])
    np.testing.assert_array_equal(c1[3], b0[3])
    np.testing.assert_array_equal(c2[2], b3[2])
    assert all(
        np.array_equal(row, b0[index]) or np.array_equal(row, b3[index])
        for index, row in enumerate(c2)
    )
    assert audit1["accepted_rank1_flip_count"] == 1
    assert audit1["accepted_rank2_flip_count"] == 0
    assert audit2["accepted_rank2_flip_count"] == 1


def test_local_risk_fit_is_support_only_order_invariant_and_target_centered() -> None:
    rows, labels, base_scores, classes = _risk_support()
    model, audit = fit_local_flip_risk_model(
        rows,
        labels,
        base_support_scores=base_scores,
        classes=classes,
        old_class_count=2,
        k_shot=5,
    )
    assert audit["support_only"] is True
    assert audit["query_rows_used"] == 0
    assert audit["target_shift_source"] == "CLASS_BALANCED_OLD_SUPPORT"
    np.testing.assert_allclose(
        np.mean(model.centered_old_class_centres, axis=0), 0.0, atol=1.0e-6
    )

    permutation = np.random.default_rng(82802).permutation(len(rows))
    repeated, repeated_audit = fit_local_flip_risk_model(
        rows[permutation],
        labels[permutation],
        base_support_scores=base_scores[permutation],
        classes=classes,
        old_class_count=2,
        k_shot=5,
    )
    np.testing.assert_allclose(repeated.shared_target_center, model.shared_target_center)
    np.testing.assert_allclose(repeated.class_prototypes, model.class_prototypes)
    np.testing.assert_allclose(repeated.pair_posterior_mean, model.pair_posterior_mean)
    assert repeated_audit["state_digest"] == audit["state_digest"]


def test_m28_wraps_unmodified_b3_and_query_scoring_is_batch_independent() -> None:
    blocks, labels, classes, old, _anchor, base = _fixture(5)
    direct, _ = fit_m25_anchored_residual(
        arm=B3,
        base_state=base,
        support_blocks=blocks,
        support_labels=labels,
        classes=classes,
        k_shot=5,
        old_class_count=len(old),
        domain_digest="synthetic-m28",
    )
    state, audit = fit_m28_local_flip_risk(
        arm=C2,
        base_state=base,
        support_blocks=blocks,
        support_labels=labels,
        classes=classes,
        k_shot=5,
        old_class_count=len(old),
        domain_digest="synthetic-m28",
    )
    np.testing.assert_array_equal(state.b3_state.score(blocks), direct.score(blocks))
    assert audit["performance_branch"] == B3
    joint = state.score(blocks[:3])
    separate = np.concatenate([state.score(row[None, :]) for row in blocks[:3]], axis=0)
    np.testing.assert_array_equal(np.argmax(joint, axis=1), np.argmax(separate, axis=1))
    np.testing.assert_allclose(joint, separate, atol=3.0e-6, rtol=3.0e-6)


def test_k1_m28_is_exact_b0_fallback() -> None:
    blocks, labels, classes, old, _anchor, base = _fixture(1)
    state, audit = fit_m28_local_flip_risk(
        arm=C1,
        base_state=base,
        support_blocks=blocks,
        support_labels=labels,
        classes=classes,
        k_shot=1,
        old_class_count=len(old),
        domain_digest="synthetic-m28-k1",
    )
    selected, query_audit = state.score_with_audit(blocks)
    np.testing.assert_array_equal(selected, base.score(physical_if256(blocks)))
    assert audit["risk_fit"]["fallback_policy"] == "K_LT_5_EXACT_B0"
    assert query_audit["fallback_reason"] == "K_LT_5_EXACT_B0"

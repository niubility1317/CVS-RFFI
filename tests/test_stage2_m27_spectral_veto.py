from __future__ import annotations

import numpy as np

from cvsrffi.stage2_m24_features import physical_if256
from cvsrffi.stage2_m27_spectral_veto import (
    V1,
    apply_consensus_veto,
    fit_m27_spectral_veto,
    fit_target_centered_competition,
)
from cvsrffi.stage2_m25_anchored_residual import B3, fit_m25_anchored_residual
from test_stage2_m26_td_src256 import _fixture


def _support_fixture() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(82701)
    classes = ("old-0", "old-1", "new-0")
    shared_shift = np.linspace(-0.15, 0.15, 6)
    centres = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.2, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.2, 0.0, 0.0],
        ]
    )
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for name, centre in zip(classes, centres, strict=True):
        for _ in range(5):
            rows.append(centre + shared_shift + 0.01 * rng.normal(size=6))
            labels.append(name)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels), classes


def test_consensus_veto_emits_only_exact_b0_or_exact_b3_rows() -> None:
    b0 = np.asarray(
        [[2.0, 1.0, 0.0], [0.1, 0.2, 0.0], [0.0, 0.2, 0.3]],
        dtype=np.float32,
    )
    b3 = np.asarray(
        [[2.1, 1.0, 0.0], [0.1, 0.0, 0.3], [0.0, 0.3, 0.2]],
        dtype=np.float32,
    )
    representation = np.asarray(
        [[4.0, 0.0, 0.0], [0.0, 0.1, 0.9], [0.0, 0.2, 0.8]],
        dtype=np.float32,
    )
    selected, audit = apply_consensus_veto(
        b0,
        b3,
        representation,
        reliability_accepted=True,
        margin_threshold=0.25,
    )
    np.testing.assert_array_equal(selected[0], b3[0])
    np.testing.assert_array_equal(selected[1], b3[1])
    np.testing.assert_array_equal(selected[2], b0[2])
    assert all(
        np.array_equal(selected[index], b0[index])
        or np.array_equal(selected[index], b3[index])
        for index in range(len(selected))
    )
    assert audit["selected_b3_count"] == 2
    assert audit["vetoed_b3_flip_count"] == 1


def test_unreliable_representation_is_exact_b0_fallback() -> None:
    b0 = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b3 = b0[:, ::-1].copy()
    representation = b3.copy()
    selected, audit = apply_consensus_veto(
        b0,
        b3,
        representation,
        reliability_accepted=False,
        margin_threshold=0.0,
    )
    np.testing.assert_array_equal(selected, b0)
    assert audit["fallback_reason"] == "SUPPORT_REPRESENTATION_UNRELIABLE"


def test_target_model_is_support_only_class_balanced_and_order_invariant() -> None:
    rows, labels, classes = _support_fixture()
    model, audit = fit_target_centered_competition(
        rows,
        labels,
        classes=classes,
        old_class_count=2,
        k_shot=5,
    )
    assert audit["support_only"] is True
    assert audit["query_rows_used"] == 0
    assert audit["target_shift_source"] == "CLASS_BALANCED_OLD_SUPPORT"
    np.testing.assert_allclose(
        np.mean(model.centered_old_class_centres, axis=0),
        0.0,
        atol=1.0e-6,
    )

    permutation = np.random.default_rng(82702).permutation(len(rows))
    repeated, repeated_audit = fit_target_centered_competition(
        rows[permutation],
        labels[permutation],
        classes=classes,
        old_class_count=2,
        k_shot=5,
    )
    np.testing.assert_allclose(repeated.shared_target_center, model.shared_target_center)
    np.testing.assert_allclose(repeated.class_prototypes, model.class_prototypes)
    assert repeated_audit["state_digest"] == audit["state_digest"]

    contaminated = rows.copy()
    contaminated[0] += 1000.0
    robust, _robust_audit = fit_target_centered_competition(
        contaminated,
        labels,
        classes=classes,
        old_class_count=2,
        k_shot=5,
    )
    np.testing.assert_allclose(
        robust.shared_target_center,
        model.shared_target_center,
        atol=0.03,
    )


def test_target_model_score_is_batch_and_label_permutation_invariant() -> None:
    rows, labels, classes = _support_fixture()
    model, _audit = fit_target_centered_competition(
        rows,
        labels,
        classes=classes,
        old_class_count=2,
        k_shot=5,
    )
    query = rows[[1, 7, 12]] + 0.003
    joint = model.score(query)
    separate = np.concatenate([model.score(item[None, :]) for item in query], axis=0)
    np.testing.assert_array_equal(joint, separate)

    permuted_classes = (classes[2], classes[0], classes[1])
    permuted, _audit = fit_target_centered_competition(
        rows,
        labels,
        classes=permuted_classes,
        old_class_count=2,
        k_shot=5,
        old_classes=(classes[0], classes[1]),
    )
    index = [permuted_classes.index(name) for name in classes]
    np.testing.assert_allclose(permuted.score(query)[:, index], joint, atol=1.0e-6)


def test_m27_wraps_the_unmodified_b3_fit_and_scores() -> None:
    blocks, labels, classes, old, _anchor, base = _fixture(5)
    direct, _direct_audit = fit_m25_anchored_residual(
        arm=B3,
        base_state=base,
        support_blocks=blocks,
        support_labels=labels,
        classes=classes,
        k_shot=5,
        old_class_count=len(old),
        domain_digest="synthetic-domain",
    )
    wrapped, audit = fit_m27_spectral_veto(
        arm=V1,
        base_state=base,
        support_blocks=blocks,
        support_labels=labels,
        classes=classes,
        k_shot=5,
        old_class_count=len(old),
        domain_digest="synthetic-domain",
    )
    np.testing.assert_array_equal(wrapped.b3_state.score(blocks), direct.score(blocks))
    assert wrapped.b3_state.config_hash == direct.config_hash
    assert audit["performance_branch"] == B3


def test_k1_m27_is_exact_b0_fallback() -> None:
    blocks, labels, classes, old, _anchor, base = _fixture(1)
    wrapped, audit = fit_m27_spectral_veto(
        arm=V1,
        base_state=base,
        support_blocks=blocks,
        support_labels=labels,
        classes=classes,
        k_shot=1,
        old_class_count=len(old),
        domain_digest="synthetic-domain-k1",
    )

    selected, query_audit = wrapped.score_with_audit(blocks)
    np.testing.assert_array_equal(selected, base.score(physical_if256(blocks)))
    assert audit["representation_fit"]["reliability_accepted"] is False
    assert audit["representation_fit"]["fallback_policy"] == "K1_EXACT_B0"
    assert query_audit["fallback_reason"] == "SUPPORT_REPRESENTATION_UNRELIABLE"

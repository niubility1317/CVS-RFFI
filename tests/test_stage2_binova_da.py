from __future__ import annotations

import numpy as np
import pytest
import torch

from cvsrffi.stage2_binova_da import (
    NOVA_DA_Config,
    NOVA_DA_Module,
    affine_explained_ratio,
    apply_nova_da,
    build_pseudo_role_rotations,
    fit_nova_da,
    support_crossfit_masks,
)
from cvsrffi.stage2_binova_features import BiNOVAFeatures, BiNOVAQuery, BiNOVASupport


def _features(class_count: int = 6, shots: int = 10) -> BiNOVAFeatures:
    rows = class_count * shots
    rng = np.random.default_rng(19)
    labels = np.repeat(np.arange(class_count), shots)
    centers = rng.normal(size=(class_count, 160)).astype(np.float32)
    identity = centers[labels] + 0.04 * rng.normal(size=(rows, 160)).astype(np.float32)
    return BiNOVAFeatures(
        identity160=identity,
        late_time160=identity + 0.03 * np.tanh(identity),
        domain160=rng.normal(size=(rows, 160)).astype(np.float32),
        fft96=rng.normal(size=(rows, 96)).astype(np.float32),
        physical6=rng.normal(size=(rows, 6)).astype(np.float32),
        physical_ids=tuple(f"p{index}" for index in range(rows)),
    )


def _support() -> BiNOVASupport:
    features = _features()
    return BiNOVASupport(
        features=features,
        labels=np.repeat(np.arange(6), 10),
        ranks=np.tile(np.arange(10), 6),
        context={
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": "cap-test",
            "split_id": "split-test",
        },
    )


def test_zero_initialized_stage_a_is_exact_identity() -> None:
    support = _support()
    module = NOVA_DA_Module()
    adapted = apply_nova_da(module, support.features)
    np.testing.assert_array_equal(adapted, support.features.identity160)


def test_role_rotations_are_four_base_plus_two_pseudo_new_and_cover_roles() -> None:
    rotations = build_pseudo_role_rotations(tuple(range(6)))
    assert len(rotations) == 3
    assert all(len(base) == 4 and len(new) == 2 for base, new in rotations)
    assert {item for _, pseudo_new in rotations for item in pseudo_new} == set(range(6))


def test_rank_crossfit_is_exactly_eight_train_two_held_per_class() -> None:
    support = _support()
    train, held = support_crossfit_masks(support.labels, support.ranks)
    for class_id in range(6):
        selected = support.labels == class_id
        assert int(np.sum(train & selected)) == 8
        assert int(np.sum(held & selected)) == 2
    assert not np.any(train & held)


def test_affine_explained_ratio_distinguishes_affine_and_nonlinear_residuals() -> None:
    torch.manual_seed(7)
    rows = torch.randn(80, 12)
    affine = rows * torch.randn(12) * 0.1 + torch.randn(12) * 0.1
    nonlinear = torch.sin(rows * 3.0)
    assert float(affine_explained_ratio(rows, affine)) > 0.99
    assert float(affine_explained_ratio(rows, nonlinear)) < 0.80


def test_stage_a_fit_accepts_support_only_and_returns_finite_non_affine_state() -> None:
    support = _support()
    state = fit_nova_da(
        support,
        NOVA_DA_Config(steps=2, learning_rate=2.0e-3, seed=5),
        device="cpu",
    )
    assert state.audit["query_rows_used"] == 0
    assert state.audit["role_rotation_count"] == 3
    assert state.audit["gate_held_labels_used_for_gradient"] is False
    assert state.audit["optimizer_d92_fit_per_class"] == 6
    assert state.audit["optimizer_score_per_class"] == 2
    assert 0.0 <= state.audit["non_affine_fraction"] <= 1.0
    assert np.isfinite(apply_nova_da(state.module, support.features)).all()


def test_stage_a_rejects_query_object() -> None:
    support = _support()
    query = BiNOVAQuery(features=support.features, context=support.context)
    with pytest.raises((TypeError, ValueError), match="support"):
        fit_nova_da(query, NOVA_DA_Config(steps=1), device="cpu")

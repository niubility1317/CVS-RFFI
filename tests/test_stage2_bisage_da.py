from __future__ import annotations

import inspect

import numpy as np
import torch

from cvsrffi.stage2_binova_features import BiNOVAFeatures, BiNOVASupport
from cvsrffi.stage2_bisage_da import (
    SAGEDConfig,
    SAGEDModule,
    apply_sage_d,
    build_role_rotations,
    coordinate_median_consensus,
    evaluate_sage_d_crossfit,
    fit_sage_d,
    stage_a_gate,
    support_crossfit_masks,
)


def _support(shots: int) -> BiNOVASupport:
    rng = np.random.default_rng(713102)
    class_count = 6
    labels = np.repeat(np.arange(class_count), shots)
    rows = len(labels)
    centers = rng.normal(size=(class_count, 160)).astype(np.float32)
    identity = centers[labels] + 0.05 * rng.normal(size=(rows, 160)).astype(np.float32)
    features = BiNOVAFeatures(
        identity160=identity,
        late_time160=identity + 0.02 * np.tanh(identity),
        domain160=rng.normal(size=(rows, 160)).astype(np.float32),
        fft96=rng.normal(size=(rows, 96)).astype(np.float32),
        physical6=rng.normal(size=(rows, 6)).astype(np.float32),
        physical_ids=tuple(f"old-{index}" for index in range(rows)),
    )
    return BiNOVASupport(
        features=features,
        labels=labels,
        ranks=np.tile(np.arange(shots), class_count),
        context={
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": "cap-stage-a",
            "split_id": "split-stage-a",
        },
    )


def test_role_rotations_cover_every_old_class_as_pseudo_new() -> None:
    rotations = build_role_rotations(range(6))
    assert len(rotations) == 3
    assert all(len(item.pseudo_base) == 4 for item in rotations)
    assert all(len(item.pseudo_new) == 2 for item in rotations)
    assert {value for item in rotations for value in item.pseudo_new} == set(range(6))


def test_stage_a_forward_has_no_class_id_or_label_input() -> None:
    parameters = inspect.signature(SAGEDModule.forward).parameters
    assert "class_ids" not in parameters
    assert "labels" not in parameters
    assert "targets" not in parameters


def test_zero_initialized_stage_a_is_exact_identity() -> None:
    support = _support(2)
    module = SAGEDModule()
    output = apply_sage_d(module, support.features, np.zeros(166, dtype=np.float64))
    np.testing.assert_array_equal(output, support.features.identity160)


def test_coordinate_median_consensus_keeps_majority_and_zeros_tie() -> None:
    gradients = [
        torch.tensor([3.0, 2.0, 0.0]),
        torch.tensor([2.0, -2.0, 0.0]),
        torch.tensor([-1.0, 1.0, 0.0]),
    ]
    result = coordinate_median_consensus(gradients, normalize=False)
    torch.testing.assert_close(result, torch.tensor([2.0, 1.0, 0.0]))


def test_crossfit_uses_ranked_fit_and_held_samples() -> None:
    support = _support(5)
    fit, held = support_crossfit_masks(support.labels, support.ranks)
    for class_id in range(6):
        selected = support.labels == class_id
        assert int(np.sum(fit & selected)) == 4
        assert int(np.sum(held & selected)) == 1
    assert not np.any(fit & held)


def test_k1_stage_a_returns_s0_fallback_without_training() -> None:
    support = _support(1)
    state = fit_sage_d(support, SAGEDConfig(steps=1), device="cpu")
    assert state.audit["selected_mode"] == "S0"
    assert state.audit["k1_fallback"] is True
    assert state.audit["query_rows_used"] == 0
    np.testing.assert_array_equal(apply_sage_d(state, support.features), support.features.identity160)


def test_k5_stage_a_executes_one_consensus_update() -> None:
    support = _support(5)
    state = fit_sage_d(
        support,
        SAGEDConfig(
            steps=1,
            learning_rate=1.0e-3,
            late_rank=4,
            identity_rank=4,
            context_dim=4,
            covariance_rank=3,
        ),
        device="cpu",
    )
    assert state.audit["selected_mode"] == "S1_CANDIDATE"
    assert state.audit["role_rotation_count"] == 3
    assert state.audit["gradient_consensus"] == "normalized_coordinate_median"
    assert state.audit["crossfit_fit_per_class"] == 4
    assert state.audit["crossfit_held_per_class"] == 1
    assert np.isfinite(apply_sage_d(state, support.features)).all()
    metrics = evaluate_sage_d_crossfit(state, support)
    assert metrics["query_rows_used"] == 0
    assert np.isfinite(list(metrics.values())).all()
    assert "delta_lcb_h_pseudo" in metrics
    assert "prediction_change_count" in metrics


def test_stage_a_gate_requires_all_registered_mechanism_thresholds() -> None:
    passing = {
        "prediction_change_count": 3,
        "delta_lcb_h_pseudo": 0.002,
        "nonaffine_energy": 0.2,
        "pseudo_old_accuracy": 0.8,
        "baseline_pseudo_old_accuracy": 0.8,
        "pseudo_old_floor": 0.7,
        "baseline_pseudo_old_floor": 0.7,
        "pseudo_forgetting": 0.0,
        "baseline_pseudo_forgetting": 0.0,
    }
    assert stage_a_gate(passing)["stage_a_gate_passed"] is True
    for field, failed_value in (
        ("prediction_change_count", 0),
        ("delta_lcb_h_pseudo", 0.001),
        ("nonaffine_energy", 0.09),
        ("pseudo_old_accuracy", 0.79),
        ("pseudo_old_floor", 0.69),
        ("pseudo_forgetting", 0.01),
    ):
        failed = dict(passing)
        failed[field] = failed_value
        result = stage_a_gate(failed)
        assert result["stage_a_gate_passed"] is False
        assert result["status"] == "STOPPED_SCIENTIFIC_GATE"

from __future__ import annotations

from dataclasses import replace
import pytest
import torch

from cvsrffi.slow_fast_adapter import SlowFastAdapterState, SlowFastCandidate
from cvsrffi.slow_fast_selection import (
    SupportTrustPolicy,
    _movement_statistics,
    _stratified_crossfit_splits,
    choose_crossfit_lambda,
    evaluate_frozen_support_state,
    select_support_only_state,
    support_state_diagnostics,
)


def _common_state() -> SlowFastAdapterState:
    direction = torch.tensor([1.0, -1.0])
    direction = direction / torch.linalg.vector_norm(direction)
    basis = torch.zeros(2, 4)
    basis[:, 0] = direction
    return SlowFastAdapterState(
        candidate=SlowFastCandidate.COMMON_SHIFT_R4,
        slow_u=basis,
        common_coeff=torch.zeros(4),
    )


def test_k1_never_uses_a_mathematical_view_as_a_second_shot() -> None:
    prototypes = torch.eye(2)
    features = prototypes.clone()
    labels = torch.tensor([0, 1])

    selected, audit = select_support_only_state(
        features,
        labels,
        prototypes,
        _common_state(),
        k_shot=1,
        logit_scale=8.0,
        trust_radius=0.15,
    )

    assert selected.rho == 0.0
    assert audit["selected_lambda"] == 0.0
    assert audit["reason"] == "K1_NO_INDEPENDENT_LOO_FALLBACK_DA0"
    assert audit["gradient_updates"] == 0


def test_leave_one_out_selects_common_shift_when_it_improves_every_class() -> None:
    prototypes = torch.eye(2)
    shift = torch.tensor([0.9, -0.9])
    features = torch.stack(
        [
            prototypes[0] + shift,
            prototypes[0] + shift + torch.tensor([0.0, 0.02]),
            prototypes[1] + shift,
            prototypes[1] + shift + torch.tensor([0.02, 0.0]),
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])

    selected, audit = select_support_only_state(
        features,
        labels,
        prototypes,
        _common_state(),
        k_shot=2,
        logit_scale=8.0,
        trust_radius=0.5,
    )

    assert audit["selected_lambda"] > 0.0
    assert audit["selected_macro_accuracy"] >= audit["baseline_macro_accuracy"]
    assert audit["selected_floor_accuracy"] >= audit["baseline_floor_accuracy"]
    assert selected.common_coeff.abs().sum().item() > 0.0


def test_support_gate_falls_back_when_baseline_is_already_strictly_better() -> None:
    prototypes = torch.eye(2)
    features = torch.stack([prototypes[0], prototypes[0], prototypes[1], prototypes[1]])
    labels = torch.tensor([0, 0, 1, 1])

    selected, audit = select_support_only_state(
        features,
        labels,
        prototypes,
        _common_state(),
        k_shot=2,
        logit_scale=8.0,
        trust_radius=0.15,
    )

    assert selected.rho == 0.0
    assert audit["selected_lambda"] == 0.0


def test_k10_crossfit_is_five_five_and_class_balanced_for_every_fold() -> None:
    labels = torch.arange(3).repeat_interleave(10)

    splits = _stratified_crossfit_splits(labels, k_shot=10, seed=17, repeats=3)

    assert len(splits) == 6
    for train, validation in splits:
        assert not set(train.tolist()) & set(validation.tolist())
        assert torch.bincount(labels[train], minlength=3).tolist() == [5, 5, 5]
        assert torch.bincount(labels[validation], minlength=3).tolist() == [5, 5, 5]


def test_k5_crossfit_keeps_every_fold_class_balanced() -> None:
    labels = torch.arange(2).repeat_interleave(5)

    splits = _stratified_crossfit_splits(labels, k_shot=5, seed=23, repeats=1)

    assert len(splits) == 2
    assert [torch.bincount(labels[train], minlength=2).tolist() for train, _ in splits] == [
        [3, 3],
        [2, 2],
    ]
    assert [
        torch.bincount(labels[validation], minlength=2).tolist()
        for _, validation in splits
    ] == [[2, 2], [3, 3]]


def test_crossfit_removes_complement_duplicates_and_keeps_physical_ids_disjoint() -> None:
    # K=2只有一个无序二分，无法产生8个唯一repeat；K=4每类有3个无序二分，
    # 两类组合后可产生9个唯一repeat，足以检验去重逻辑。
    labels = torch.arange(2).repeat_interleave(4)
    physical_ids = tuple(f"class{class_id}-sample{sample_id}" for class_id in range(2) for sample_id in range(4))

    splits = _stratified_crossfit_splits(
        labels,
        k_shot=4,
        seed=7,
        repeats=8,
        physical_ids=physical_ids,
    )

    assert len(splits) == 16
    canonical_partitions: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for train, validation in splits[::2]:
        halves = sorted((tuple(sorted(train.tolist())), tuple(sorted(validation.tolist()))))
        canonical_partitions.add((halves[0], halves[1]))
    assert len(canonical_partitions) == 8
    for train, validation in splits:
        train_ids = {physical_ids[index] for index in train.tolist()}
        validation_ids = {physical_ids[index] for index in validation.tolist()}
        assert train_ids.isdisjoint(validation_ids)


def test_crossfit_rejects_a_physical_id_repeated_across_rows() -> None:
    labels = torch.arange(2).repeat_interleave(4)
    physical_ids = ("duplicate", "duplicate", "a", "b", "c", "d", "e", "f")

    with pytest.raises(ValueError, match="physical IDs must be unique"):
        _stratified_crossfit_splits(
            labels,
            k_shot=4,
            seed=7,
            repeats=1,
            physical_ids=physical_ids,
        )


def test_support_normalized_strength_records_quantile_relative_and_fold_stability() -> None:
    prototypes = torch.eye(2)
    features = torch.tensor([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]])
    labels = torch.tensor([0, 0, 1, 1])
    fitted = replace(
        _common_state(), common_coeff=torch.tensor([0.8, 0.0, 0.0, 0.0])
    )
    policy = SupportTrustPolicy(
        q90_move=0.10,
        hard_move=0.30,
        q90_relative_move=0.8,
        minimum_positive_folds=5,
    )

    diagnostics = support_state_diagnostics(
        features,
        labels,
        prototypes,
        fitted,
        nominal_lambda=0.5,
        policy=policy,
        fold_risk_gains=(0.06, 0.05, 0.04, 0.03, 0.02, -0.001),
    )

    assert 0.0 < diagnostics["effective_lambda"] < 0.5
    assert diagnostics["q50_feature_move"] <= diagnostics["q90_feature_move"]
    assert diagnostics["q90_feature_move"] <= 0.10 + 1.0e-6
    assert diagnostics["max_feature_move"] <= 0.30
    assert diagnostics["q90_relative_move"] >= 0.0
    assert diagnostics["positive_fold_count"] == 5
    assert diagnostics["fold_gain_std"] > 0.0
    assert diagnostics["fold_gain_lcb90"] > 0.0


def test_policy_selector_records_nominal_effective_geometry_and_unique_fold_gains() -> None:
    prototypes = torch.eye(2)
    shift = torch.tensor([0.45, -0.45])
    features = torch.cat(
        (
            prototypes[0].repeat(10, 1) + shift,
            prototypes[1].repeat(10, 1) + shift,
        ),
        dim=0,
    )
    labels = torch.arange(2).repeat_interleave(10)
    policy = SupportTrustPolicy(
        q90_move=0.20,
        hard_move=0.60,
        q90_relative_move=10.0,
        minimum_positive_folds=1,
    )

    selected, audit = select_support_only_state(
        features,
        labels,
        prototypes,
        _common_state(),
        k_shot=10,
        logit_scale=8.0,
        trust_radius=0.6,
        repeats=3,
        physical_ids=tuple(f"support-{index}" for index in range(20)),
        trust_policy=policy,
    )

    assert audit["selection_protocol"] == "repeated_stratified_2fold"
    assert audit["selected_effective_lambda"] <= audit["selected_lambda"]
    assert selected.rho == pytest.approx(audit["selected_effective_lambda"])
    assert audit["crossfit_normalizer_scope"] == "fold_train_only"
    assert len(audit["crossfit_fold_strength_normalizers"]) == audit["crossfit_fit_count"]
    assert all(
        0.0 < value <= 1.0 for value in audit["crossfit_fold_strength_normalizers"]
    )
    for row in audit["lambda_trace"]:
        assert row["effective_lambda"] <= row["lambda"]
        assert len(row["fold_risk_gains"]) == audit["crossfit_fit_count"]
        assert {
            "q50_feature_move",
            "q90_feature_move",
            "q90_relative_move",
            "positive_fold_count",
            "fold_gain_std",
            "fold_gain_lcb90",
        } <= set(row)


def test_repeat_stability_averages_complementary_fold_directions_before_gating() -> None:
    prototypes = torch.eye(2)
    features = torch.tensor([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]])
    labels = torch.tensor([0, 0, 1, 1])
    fitted = replace(
        _common_state(), common_coeff=torch.tensor([0.8, 0.0, 0.0, 0.0])
    )
    policy = SupportTrustPolicy(
        q90_move=0.10,
        hard_move=0.30,
        q90_relative_move=10.0,
        minimum_positive_folds=5,
        require_fold_lcb=False,
    )

    diagnostics = support_state_diagnostics(
        features,
        labels,
        prototypes,
        fitted,
        nominal_lambda=0.5,
        policy=policy,
        fold_risk_gains=(0.1, -1.0, 0.1, 0.1, 0.1, 0.1),
    )

    assert diagnostics["positive_fold_count"] == 5
    assert diagnostics["repeat_risk_gains"] == pytest.approx([-0.45, 0.1, 0.1])
    assert diagnostics["positive_repeat_count"] == 2
    assert diagnostics["required_positive_repeats"] == 3
    assert diagnostics["fold_stability_pass"] is False


def test_directional_margin_trust_protects_correct_rows_and_repairs_wrong_rows() -> None:
    prototypes = torch.eye(2)
    labels = torch.tensor([0, 1])
    baseline = torch.tensor([[1.0, 0.0], [0.8, 0.6]])
    adapted = torch.tensor([[0.8, 0.6], [0.9, 0.4358899]])

    diagnostics = _movement_statistics(adapted, baseline, labels, prototypes)

    assert diagnostics["correct_margin_violation_count"] == 1
    assert diagnostics["error_margin_nonimprovement_count"] == 1
    assert diagnostics["minimum_correct_margin_ratio"] == pytest.approx(0.5)
    assert diagnostics["directional_margin_pass"] is False


def test_frozen_support_state_diagnostic_is_explicitly_in_sample_and_support_only() -> None:
    prototypes = torch.eye(2)
    shift = torch.tensor([0.45, -0.45])
    features = torch.cat(
        (prototypes[0].repeat(2, 1) + shift, prototypes[1].repeat(2, 1) + shift),
        dim=0,
    )
    labels = torch.tensor([0, 0, 1, 1])
    fitted = replace(
        _common_state(), common_coeff=torch.tensor([0.6, 0.0, 0.0, 0.0])
    )

    diagnostics = evaluate_frozen_support_state(
        features, labels, prototypes, fitted, logit_scale=8.0
    )

    assert diagnostics["diagnostic_role"] == "full_support_diagnostic_only"
    assert diagnostics["query_opened"] is False
    assert diagnostics["q90_feature_move"] > 0.0
    assert isinstance(diagnostics["risk_gain"], float)


def test_gate_chooses_lowest_risk_not_largest_passing_lambda() -> None:
    trace = [
        {"lambda": 0.0, "risk": 1.0, "eligible": True},
        {"lambda": 0.125, "risk": 0.8, "eligible": True},
        {"lambda": 0.5, "risk": 0.9, "eligible": True},
        {"lambda": 1.0, "risk": 0.7, "eligible": False},
    ]

    selected = choose_crossfit_lambda(trace)

    assert selected == 0.125


def test_gate_records_all_lambda_risks_and_attempted_work_even_on_fallback() -> None:
    prototypes = torch.eye(2)
    features = torch.stack([prototypes[0], prototypes[0], prototypes[1], prototypes[1]])
    labels = torch.tensor([0, 0, 1, 1])

    _selected, audit = select_support_only_state(
        features,
        labels,
        prototypes,
        _common_state(),
        k_shot=2,
        logit_scale=3.5,
        trust_radius=0.15,
        lambda_grid=(0.0, 0.125, 0.25, 0.5, 0.75, 1.0),
        crossfit_seed=31,
        repeats=2,
    )

    assert audit["selected_lambda"] == 0.0
    assert audit["crossfit_fit_count"] == 4
    assert audit["loo_fit_count"] == 0
    assert audit["selection_protocol"] == "repeated_stratified_2fold"
    assert audit["crossfit_updates"] == 0
    assert audit["deployment_candidate_updates"] == 0
    assert audit["attempted_gradient_updates"] == 0
    assert audit["committed_gradient_updates"] == 0
    assert audit["support_logit_scale"] == 3.5
    assert audit["trust_radius"] == 0.15
    assert [row["lambda"] for row in audit["lambda_trace"]] == [
        0.0,
        0.125,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    required = {
        "macro_accuracy",
        "floor_accuracy",
        "macro_ce",
        "class_cvar_ce",
        "mean_margin",
        "min_class_margin",
        "mean_feature_move",
        "max_feature_move",
        "prediction_flip_count",
        "positive_flip_count",
        "negative_flip_count",
        "risk",
        "rejection_reasons",
    }
    assert all(required <= set(row) for row in audit["lambda_trace"])

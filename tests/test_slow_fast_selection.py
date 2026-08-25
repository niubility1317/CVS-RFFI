from __future__ import annotations

import torch

from cvsrffi.slow_fast_adapter import SlowFastAdapterState, SlowFastCandidate
from cvsrffi.slow_fast_selection import (
    _stratified_crossfit_splits,
    choose_crossfit_lambda,
    select_support_only_state,
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

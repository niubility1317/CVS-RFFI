from __future__ import annotations

import torch

from cvsrffi.slow_fast_adapter import SlowFastAdapterState, SlowFastCandidate
from cvsrffi.slow_fast_selection import select_support_only_state


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
    )

    assert selected.rho == 0.0
    assert audit["selected_lambda"] == 0.0

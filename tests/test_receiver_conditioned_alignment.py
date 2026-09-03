from __future__ import annotations

import pytest
import torch

from cvsrffi.receiver_conditioned_alignment import (
    GroupTailRiskEMA,
    ReceiverConditionedPrototypeBank,
    excitation_descriptors,
    receiver_conditioned_alignment_loss,
)


def test_excitation_descriptors_return_five_finite_physical_summaries() -> None:
    descriptors = excitation_descriptors(torch.randn(6, 2, 128))

    assert descriptors.shape == (6, 5)
    assert torch.isfinite(descriptors).all()


def test_receiver_alignment_never_pairs_different_transmitters() -> None:
    bank = ReceiverConditionedPrototypeBank(feature_dim=2, min_count=1)
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1]])
    tx = torch.tensor([0, 0, 1, 1])
    receiver = torch.tensor([1, 3, 1, 3])
    excitation_bin = torch.zeros(4, dtype=torch.long)
    bank.update(features, tx=tx, receiver=receiver, excitation_bin=excitation_bin)

    result = bank.alignment_loss()

    assert result["pair_count"] == 2
    assert float(result["loss"]) < 0.02


def test_group_tail_risk_uses_only_eligible_worst_groups_and_caps_weights() -> None:
    tracker = GroupTailRiskEMA(alpha=0.5, momentum=0.0, min_group_size=2, max_weight=2.0)
    group_ids = torch.tensor([0, 0, 1, 1, 2])
    losses = torch.tensor([0.1, 0.1, 0.8, 1.0, 5.0])

    result = tracker.update(group_ids=group_ids, losses=losses)

    assert result["eligible_group_count"] == 2
    assert result["selected_group_ids"].tolist() == [1]
    assert float(result["loss"]) == pytest.approx(0.9)
    assert float(result["max_effective_weight"]) <= 2.0


def test_receiver_conditioned_batch_alignment_is_differentiable_and_tx_local() -> None:
    features = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]], requires_grad=True
    )
    result = receiver_conditioned_alignment_loss(
        features,
        tx=torch.tensor([0, 0, 1, 1]),
        receiver=torch.tensor([1, 3, 1, 3]),
        excitation_bin=torch.tensor([0, 0, 0, 0]),
    )
    result["loss"].backward()

    assert result["pair_count"] == 2
    assert float(features.grad.abs().sum()) > 0.0

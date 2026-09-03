from __future__ import annotations

import pytest
import torch

from cvsrffi.receiver_style_bank import OnlineReceiverStyleBank, ReceiverStyleBank, extract_receiver_statistics


def test_receiver_style_bank_rejects_non_source_roles() -> None:
    with pytest.raises(ValueError, match="source-only"):
        ReceiverStyleBank.fit(
            torch.randn(3, 9), receiver_ids=torch.tensor([1, 3, 4]), role="target", rank=2
        )


def test_receiver_style_bank_round_trips_and_replays_virtual_receiver_sampling() -> None:
    stats = torch.tensor(
        [
            [0.0, 0.2, 0.1, 0.0, 0.2, 0.0, 0.1, 0.0, 0.2],
            [0.2, 0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 0.1, 0.0],
            [-0.1, 0.1, 0.2, 0.0, 0.1, -0.1, 0.2, 0.0, 0.1],
        ]
    )
    bank = ReceiverStyleBank.fit(
        stats, receiver_ids=torch.tensor([1, 3, 4]), role="source", rank=2
    )

    first = bank.sample(batch_size=5, seed=19, extension=(0.8, 1.2))
    restored = ReceiverStyleBank.from_state_dict(bank.state_dict())
    second = restored.sample(batch_size=5, seed=19, extension=(0.8, 1.2))

    assert torch.allclose(first, second)
    assert first.shape == (5, 9)
    assert torch.isfinite(first).all()
    assert restored.receiver_ids.tolist() == [1, 3, 4]


def test_receiver_statistics_are_finite_and_receiver_transform_changes_iq() -> None:
    x = torch.randn(4, 2, 128)
    stats = extract_receiver_statistics(x)
    bank = ReceiverStyleBank.fit(
        stats, receiver_ids=torch.tensor([1, 3, 4, 6]), role="source", rank=2
    )
    style = bank.sample(batch_size=4, seed=3)
    transformed = bank.apply(x, style)

    assert stats.shape == (4, 9)
    assert torch.isfinite(transformed).all()
    assert bool((transformed - x).abs().sum(dim=(1, 2)).gt(0.0).all())


def test_online_receiver_style_bank_becomes_ready_from_source_receivers_only() -> None:
    runtime = OnlineReceiverStyleBank(rank=2, min_receivers=2)
    iq = torch.randn(6, 2, 32)
    receivers = torch.tensor([1, 1, 1, 3, 3, 3])

    runtime.update(iq, receiver_ids=receivers, role="source")
    changed = runtime.apply_sampled(iq, seed=11)

    assert runtime.ready is True
    assert changed.shape == iq.shape
    with pytest.raises(ValueError, match="source-only"):
        runtime.update(iq, receiver_ids=receivers, role="target")

import pytest
import torch

from experiments.phase1_dura_orthogonal.source_receiver_balance import (
    receiver_weights, receiver_weighted_guard,
)
from experiments.phase1_dura_orthogonal.train_source import source_loss


def test_inverse_frequency_gives_each_source_receiver_equal_total_weight():
    receivers = [1, 1, 3, 4, 6, 8]
    weights, counts = receiver_weights(receivers)
    assert counts == {'1': 2, '3': 1, '4': 1, '6': 1, '8': 1}
    for receiver in set(receivers):
        assert sum(w for r, w in zip(receivers, weights) if r == receiver) == pytest.approx(1.2)
    assert sum(weights) == pytest.approx(len(receivers))


def test_all_one_weights_match_original_dura_guard_and_gradient():
    torch.manual_seed(2)
    clean = torch.randn(5, 2, 256)
    anchor = torch.randn_like(clean)
    output = torch.randn_like(clean, requires_grad=True)
    weighted = receiver_weighted_guard(output, anchor, clean, torch.ones(5))
    original = source_loss(output, anchor, clean, objective='dura_guard')
    torch.testing.assert_close(weighted, original)
    g1 = torch.autograd.grad(weighted, output, retain_graph=True)[0]
    g2 = torch.autograd.grad(original, output)[0]
    torch.testing.assert_close(g1, g2)


def test_missing_or_unknown_source_receiver_is_rejected():
    with pytest.raises(ValueError):
        receiver_weights([1, 3, 4, 6])
    with pytest.raises(ValueError):
        receiver_weights([1, 3, 4, 6, 8, 9])

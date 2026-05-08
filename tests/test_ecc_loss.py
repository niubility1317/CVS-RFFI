import torch

from train import compute_ecc_loss, ecc_tau_for_epoch


def test_ecc_tau_ramps_to_final_value():
    assert ecc_tau_for_epoch(1, 0.65, 0.95, 60, 1) == 0.65
    assert ecc_tau_for_epoch(61, 0.65, 0.95, 60, 1) == 0.95


def test_ecc_loss_penalizes_only_probs_above_tau():
    logits = torch.tensor([
        [4.0, 0.0],
        [0.2, 0.1],
    ])
    loss, max_prob = compute_ecc_loss(logits, tau=0.70)

    expected = torch.relu(torch.softmax(logits, dim=1).max(dim=1).values - 0.70).pow(2).mean()
    assert torch.allclose(loss, expected)
    assert max_prob > 0.70


def test_ecc_loss_can_be_sample_gated():
    logits = torch.tensor([
        [4.0, 0.0],
        [4.0, 0.0],
    ])
    gate = torch.tensor([1.0, 0.0])
    loss, _ = compute_ecc_loss(logits, tau=0.70, gate=gate)

    penalties = torch.relu(torch.softmax(logits, dim=1).max(dim=1).values - 0.70).pow(2)
    assert torch.allclose(loss, penalties[:1].mean())

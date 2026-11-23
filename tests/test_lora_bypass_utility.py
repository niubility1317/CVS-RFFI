import torch

from experiments.pamf_lora_main.bypass_utility import actions, signed_gain


def test_policy_keeps_existing_packets_and_requires_both_gates():
    eligible = torch.tensor([[False, False], [True, True], [True, True], [True, False], [True, True]])
    disturbed = torch.tensor([True, False, True, True, True])
    gains = torch.tensor([[.3, .2], [.3, .2], [.01, -.3], [.2, .5], [.1, .2]])
    assert actions(eligible, disturbed, gains).tolist() == [0, 0, 0, 1, 2]


def test_signed_gain_distinguishes_repair_from_damage_at_equal_dose():
    base = torch.ones(2, 3)
    proposals = torch.stack((base * .5, base * 1.5), dim=1)
    score = signed_gain(base, proposals)
    assert score.shape == (2, 2)
    assert (score[:, 0] > 0).all() and (score[:, 1] < 0).all()
    assert torch.equal(score[:, 0], -score[:, 1])

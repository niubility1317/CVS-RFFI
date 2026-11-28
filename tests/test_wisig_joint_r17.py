import torch
from torch import nn

from experiments.pamf_lora_main.dura_real_gain_wisig_joint import configure_student, seeded_gain


def test_joint_fit_updates_only_r17_priors():
    student = nn.Module()
    student.frozen = nn.Linear(2, 2)
    student.prior = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    configure_student(student)
    assert all(not p.requires_grad for p in student.frozen.parameters())
    assert all(p.requires_grad for p in student.prior.parameters())


def test_gain_initialization_does_not_depend_on_prior_random_operations():
    first = seeded_gain('cpu')
    torch.randn(71)
    second = seeded_gain('cpu')
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

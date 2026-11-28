from torch import nn

from experiments.pamf_lora_main.dura_real_gain_wisig_unroll_joint import configure_prior


def test_only_original_r17_priors_are_trainable():
    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = nn.Linear(2, 2)
            self.prior = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

    student = configure_prior(Student())
    assert all(p.requires_grad for p in student.prior.parameters())
    assert not any(p.requires_grad for p in student.frozen.parameters())

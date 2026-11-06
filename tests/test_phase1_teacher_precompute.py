import torch

from experiments.phase1_dura_orthogonal.benchmark_teacher_precompute import targets
from experiments.phase1_dura_orthogonal.masked_prior_constraint import visible_pair


def test_precomputed_teacher_inputs_match_training_path():
    class Teacher(torch.nn.Module):
        def forward(self, observed, mask, scale):
            return observed / scale + mask[:, None].to(observed.dtype)

    torch.manual_seed(71)
    clean = torch.randn(70, 2, 256)
    starts = torch.randint(1, 224, (70,))
    _, _, observed, scale, mask = visible_pair(clean, clean, starts)
    expected = Teacher()(observed, mask, scale)
    assert torch.equal(targets(Teacher(), clean, starts, 32), expected)
    assert torch.equal(targets(Teacher(), clean, starts, 256), expected)

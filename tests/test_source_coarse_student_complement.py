import torch

from experiments.phase1_dura_orthogonal.source_coarse_student_complement import pair_metrics


def test_oracle_mixture_and_endpoint_constraints():
    first = torch.tensor([[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]])
    second = torch.ones_like(first)
    clean = torch.tensor([[[.25, .25], [.25, .25]], [[2., 2.], [2., 2.]], [[-1., -1.], [-1., -1.]]])
    values = pair_metrics(first, second, clean)
    torch.testing.assert_close(values[:, 3], torch.tensor([.25, 1., 0.]))
    assert torch.all(values[:, 2] <= torch.minimum(values[:, 0], values[:, 1])+1e-6)
    assert values[0, 2] == 0

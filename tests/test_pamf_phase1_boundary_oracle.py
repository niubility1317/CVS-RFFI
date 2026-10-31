import torch

from experiments.pamf_phase1_flow.boundary_source_oracle import projected_targeted_steps


def test_projection_respects_each_sample_budget_and_moves_toward_target():
    x = torch.tensor([[[-0.10, 0.0]], [[-0.20, 0.0]]])
    scale = torch.tensor([1.0, 0.5])
    targets = torch.tensor([0, 0])
    def classify(z):
        value = z[:, 0, 0]
        return torch.stack((value, -value), dim=1)
    y = projected_targeted_steps(classify, x, targets, scale, epsilon=0.5, steps=8)
    delta = (y - x).flatten(1).norm(dim=1)
    assert torch.all(delta <= 0.5 * scale + 1e-6)
    assert torch.all(y[:, 0, 0] > x[:, 0, 0])
    assert torch.equal(classify(y).argmax(1), targets)
    assert not y.requires_grad


def test_zero_gradient_does_not_create_nonfinite_waveform():
    x = torch.ones(2, 2, 4)
    def classify(z):
        value = z.flatten(1).sum(1) * 0
        return torch.stack((value, value), dim=1)
    y = projected_targeted_steps(classify, x, torch.zeros(2, dtype=torch.long), torch.ones(2), epsilon=0.1, steps=2)
    assert torch.equal(y, x)
    assert torch.isfinite(y).all()

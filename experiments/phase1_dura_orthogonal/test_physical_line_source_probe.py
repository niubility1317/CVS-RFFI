import torch
from experiments.phase1_dura_orthogonal.physical_line_source_probe import bounded_step, metrics


def test_bounded_step_real_complex_and_degenerate():
    d = torch.tensor([[1+1j, 2-1j], [0j, 0j], [1+0j, 0j], [1+0j, 0j]])
    r = torch.stack((.4*d[0], torch.ones(2, dtype=d.dtype), torch.tensor([2+0j, 0j]), torch.tensor([-1+0j, 0j])))
    torch.testing.assert_close(bounded_step(r, d), torch.tensor([.4, 0., 1., 0.]))


def test_physical_step_uses_observed_residual():
    anchor = torch.zeros(2, 2, 256)
    candidate = torch.zeros_like(anchor)
    candidate[:, 0] = 1.
    received = torch.zeros_like(anchor)
    received[0, 0] = .25
    received[1, 0] = .75
    clean = torch.zeros_like(anchor)
    clean[:, 0] = .5
    out = metrics(received, anchor, candidate, clean, lambda x: x)
    torch.testing.assert_close(out[:, 4], torch.tensor([.25, .75]))
    torch.testing.assert_close(out[:, 5], torch.tensor([.5, .5]))
    assert torch.all(out[:, 3] <= out[:, 2])

import torch

from experiments.phase1_dura_orthogonal.conditional_phase_dura import ConditionalPhaseDura
from experiments.phase1_dura_orthogonal.conditional_phase_source_probe import analyze_batch


def test_zero_initialized_branches_reproduce_anchor_without_corrections():
    torch.manual_seed(1)
    x, anchor, innovation, clean = (torch.randn(4, 2, 256) for _ in range(4))
    model = ConditionalPhaseDura(width=4).eval()
    values = analyze_batch(model, x, anchor, innovation, torch.full((4,), .5), clean)
    assert values.shape == (4, 7)
    torch.testing.assert_close(values[:, 0], torch.zeros(4))
    torch.testing.assert_close(values[:, 1], torch.full((4,), .5))
    torch.testing.assert_close(values[:, 2:5], torch.zeros(4, 3))
    torch.testing.assert_close(values[:, 5], values[:, 6])


def test_nonzero_phase_correction_has_expected_mixture_and_finite_metrics():
    torch.manual_seed(2)
    x, anchor, innovation, clean = (torch.randn(4, 2, 256) for _ in range(4))
    model = ConditionalPhaseDura(width=4).eval()
    with torch.no_grad():
        model.summary[-1].bias[0] = .5
    values = analyze_batch(model, x, anchor, innovation, torch.full((4,), .5), clean)
    assert (values[:, 0] > 0).all() and (values[:, 2] > 0).all()
    assert torch.equal(values[:, 3], torch.zeros(4))
    assert torch.isfinite(values).all()

import math
import torch

from experiments.phase1_dura_orthogonal.conditional_phase_dura import ConditionalPhaseDura
from experiments.phase1_dura_orthogonal.conditional_phase_oracle_probe import analyze_batch, rotate


def test_oracle_recovers_known_angle_and_zero_head_does_not_correct():
    torch.manual_seed(3)
    anchor = torch.randn(4, 2, 256)
    clean = rotate(anchor, torch.full((4,), .3))
    model = ConditionalPhaseDura(width=4).eval()
    values = analyze_batch(model, anchor, anchor, torch.zeros_like(anchor), torch.full((4,), .5), clean)
    torch.testing.assert_close(values[:, 0], torch.full((4,), .3))
    torch.testing.assert_close(values[:, 1], torch.zeros(4), atol=0, rtol=0)
    torch.testing.assert_close(values[:, 4], torch.zeros(4), atol=1e-12, rtol=0)
    torch.testing.assert_close(values[:, 3], values[:, 5])


def test_oracle_phase_bound_and_error():
    torch.manual_seed(4)
    anchor = torch.randn(4, 2, 256)
    clean = rotate(anchor, torch.full((4,), 2.2))
    model = ConditionalPhaseDura(width=4).eval()
    values = analyze_batch(model, anchor, anchor, torch.zeros_like(anchor), torch.full((4,), .5), clean)
    torch.testing.assert_close(values[:, 0], torch.full((4,), math.pi/2))
    assert (values[:, 4] < values[:, 3]).all()

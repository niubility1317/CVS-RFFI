import torch

from experiments.phase1_dura_orthogonal.source_quartic_equal_dim import features


def test_equal_feature_sizes_and_phase_shift_invariance():
    generator = torch.Generator().manual_seed(342)
    x = torch.randn(3, 2, 256, generator=generator)
    z = torch.complex(x[:, 0], x[:, 1])
    changed = z.roll(11, 1)*torch.polar(torch.ones(3), torch.tensor([.3, -.5, 1.2]))[:, None]
    y = torch.stack((changed.real, changed.imag), 1)
    power, closure = features(x)
    other_power, other_closure = features(y)
    assert power.shape == closure.shape == (3, 256)
    torch.testing.assert_close(power, other_power, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(closure, other_closure, rtol=3e-4, atol=3e-5)

import torch
from experiments.phase1_dura_orthogonal.complex_gain_source_probe import gain_metrics


def test_complex_oracle_removes_gain_and_phase_but_phase_only_keeps_amplitude_error():
    clean = torch.tensor([[[1., 2., -1.], [2., -1., 1.]]])
    z = torch.complex(clean[:, 0], clean[:, 1]) * (2j)
    prediction = torch.stack((z.real, z.imag), dim=1)
    unaligned, phase, complex_gain = gain_metrics(prediction, clean).unbind(1)
    torch.testing.assert_close(complex_gain, torch.zeros_like(complex_gain), atol=1e-7, rtol=0)
    torch.testing.assert_close(phase, torch.ones_like(phase), atol=1e-7, rtol=0)
    assert (unaligned > phase).all()


def test_zero_estimate_is_finite_and_not_falsely_improved():
    clean = torch.ones(2, 2, 4)
    result = gain_metrics(torch.zeros_like(clean), clean)
    torch.testing.assert_close(result, torch.ones(2, 3))

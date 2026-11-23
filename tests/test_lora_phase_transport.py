import torch

from experiments.pamf_lora_main.natural_detail import projection_risk


def test_projected_weak_detail_risk_detects_gain_loss_and_phase_rotation():
    spectrum = torch.zeros(2, 6, 256, dtype=torch.complex64)
    spectrum[..., 1] = 1. + 2j
    spectrum[..., 14] = -2. + .5j
    expected = torch.fft.ifft(spectrum, norm='ortho')
    assert projection_risk(expected, expected) == 0
    assert projection_risk(.3 * expected, expected) > .3
    assert projection_risk(1j * expected, expected) > .9
    assert projection_risk(-expected, expected) > 1.4
    predicted = (torch.zeros_like(expected)).requires_grad_()
    loss = projection_risk(predicted, expected)
    loss.backward()
    assert predicted.grad.norm() > 0 and torch.isfinite(predicted.grad).all()


def test_projection_ignores_samples_without_reference_energy():
    empty = torch.zeros(2, 6, 256, dtype=torch.complex64)
    signal = torch.ones_like(empty)
    assert projection_risk(signal, empty) == 0

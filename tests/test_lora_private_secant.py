import torch

from experiments.pamf_lora_main.natural_detail import (
    private_difference,
    private_gain_error,
    weak_difference,
)


def fixture_wave():
    frequency = torch.arange(256, dtype=torch.float32) * (2 * torch.pi / 256)
    chirp = torch.arange(6, dtype=torch.float32)
    amplitude = torch.arange(1, 4, dtype=torch.float32)
    tone = torch.polar(torch.ones_like(frequency), frequency)
    shape = (chirp - chirp.mean()).view(1, -1, 1) * (tone**5 + 0.35j * tone**19)
    common = (tone**3 + tone**70).view(1, 1, -1)
    return amplitude.view(-1, 1, 1) * (shape + common)


def test_private_source_difference_excludes_shared_and_outside_bins():
    source = fixture_wave()
    difference = private_difference(source)
    torch.testing.assert_close(difference.mean(1), torch.zeros_like(difference[:, 0]), atol=3e-7, rtol=0)
    spectrum = torch.fft.fft(difference, norm='ortho')
    assert spectrum[..., 5].abs().mean() > 1.
    assert spectrum[..., 19].abs().mean() > 1.
    assert spectrum[..., [0, 3, 70]].abs().max() < 1e-5
    weak = weak_difference(source)
    torch.testing.assert_close(difference, weak - weak.mean(1, keepdim=True), atol=4e-7, rtol=0)


def test_private_gain_error_identifies_attenuation_and_quadrature():
    expected = private_difference(fixture_wave())
    shifted = (0.7 + 0.2j) * expected + (0.6 - 0.2j)
    error = private_gain_error(shifted, expected)
    torch.testing.assert_close(error, torch.full_like(error, -0.3 + 0.2j), atol=2e-6, rtol=0)
    exact = private_gain_error(expected, expected)
    torch.testing.assert_close(exact, torch.zeros_like(exact), atol=1e-7, rtol=0)


def test_private_gain_error_is_finite_and_has_gradient_on_response_only():
    expected = private_difference(fixture_wave()).detach()
    expected[0] = 0
    prediction = (0.6 * expected).clone().requires_grad_()
    error = private_gain_error(prediction, expected)
    assert error[0].abs().max().item() == 0.
    error.abs().square().mean().backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0].abs().max().item() == 0.
    assert prediction.grad[1:].abs().max().item() > 0.

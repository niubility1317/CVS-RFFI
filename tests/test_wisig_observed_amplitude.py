import torch

from experiments.pamf_repl_view_shuffle.observable_action import (
    ObservableAction, amplitude_evidence, objective, preserve_spectral_amplitude,
)


def test_confident_magnitude_blending_keeps_proposed_phase():
    size = 128
    received = torch.zeros(2, size, dtype=torch.complex64)
    receive_spectrum = torch.full_like(received, 2j)
    received[:] = torch.fft.ifft(receive_spectrum, norm='ortho')
    proposed_spectrum = torch.full_like(received, -1.)
    proposal = torch.fft.ifft(proposed_spectrum, norm='ortho')
    spectrum, trust = amplitude_evidence(received, torch.tensor([[0.], [10.]]))
    torch.testing.assert_close(trust[0], torch.full((size,), .75))
    assert torch.equal(trust[1], torch.zeros(size))
    result = preserve_spectral_amplitude(proposal, spectrum, trust)
    output = torch.fft.fft(result, norm='ortho')
    torch.testing.assert_close(output[0], proposed_spectrum[0] * 1.75)
    torch.testing.assert_close(output[1], proposed_spectrum[1])


def test_zero_spectrum_uses_available_phase_and_has_finite_gradients():
    received = torch.ones(1, 256, dtype=torch.complex64)
    proposal = torch.zeros_like(received, requires_grad=True)
    spectrum, trust = amplitude_evidence(received, torch.zeros(1, 1))
    output = preserve_spectral_amplitude(proposal, spectrum, trust)
    torch.testing.assert_close(output, .75 * received)
    output.abs().sum().backward()
    assert torch.isfinite(proposal.grad).all()


def test_end_to_end_amplitude_protected_fit_has_both_stage_gradients():
    torch.manual_seed(11)
    model = ObservableAction(True, protect_amplitude=True)
    source = torch.randn(8, 256, dtype=torch.complex64, generator=torch.Generator().manual_seed(7))
    loss, parts = objective(model, source, torch.Generator().manual_seed(12))
    assert torch.isfinite(parts).all()
    loss.backward()
    for step in (model.coarse, model.fine):
        gradient = step.out.weight.grad
        assert torch.isfinite(gradient).all() and gradient.norm() > 0

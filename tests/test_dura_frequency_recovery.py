import torch

from experiments.phase1_dura_orthogonal.dura_frequency_recovery import DuraFrequencyRecovery
from experiments.phase1_dura_orthogonal.train_frequency_source import fit


def test_zero_spectral_correction_preserves_dura_and_ignores_innovation():
    torch.manual_seed(7)
    model = DuraFrequencyRecovery().eval()
    received = torch.randn(5, 2, 256)
    anchor = torch.randn_like(received)
    variance = torch.linspace(.001, .1, 5)
    actual = model(received, anchor, torch.randn_like(received), variance)
    torch.testing.assert_close(actual, anchor, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual, model(received, anchor, torch.zeros_like(received), variance))


def test_fitted_frequency_residual_improves_disjoint_synthetic_iq():
    torch.manual_seed(8)
    n = 512
    received = torch.randn(n, 2, 256)
    anchor = torch.randn_like(received)
    variance = torch.linspace(.001, .1, n)
    clean = 1.2*anchor + .08*received
    train = (received[:400], anchor[:400], torch.zeros_like(anchor[:400]), variance[:400], clean[:400])
    model, counts = fit(train)
    output = model(received[400:], anchor[400:], torch.zeros_like(anchor[400:]), variance[400:])
    original_error = (anchor[400:]-clean[400:]).square().mean()
    corrected_error = (output-clean[400:]).square().mean()
    assert all(counts) and sum(counts) == 400
    assert corrected_error < .01*original_error

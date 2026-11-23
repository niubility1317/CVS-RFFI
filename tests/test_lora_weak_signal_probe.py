import torch

from experiments.pamf_lora_main.weak_signal_probe import (ProjectedDenoise, probe_response,
                                                         reconstruction, source_probe)


def test_probe_is_a_single_added_wave_across_both_frames():
    torch.manual_seed(141)
    first = torch.randn(3, 6, 256, dtype=torch.complex64)
    angles = torch.randn(3, 6, 256)
    rotation = torch.polar(torch.ones_like(angles), angles)
    chart = torch.stack((torch.ones_like(rotation), rotation), dim=1)
    clean = first[:, None] / chart
    noise = .1*torch.randn_like(clean)
    delta = source_probe(clean, chart, torch.arange(3))
    torch.testing.assert_close(delta[:, 0], delta[:, 1]*rotation)
    for sign in (-1., 1.):
        altered_input = (clean+noise) + sign*delta
        altered_signal = clean + sign*delta
        torch.testing.assert_close(altered_input-altered_signal, noise)


def test_small_signal_probe_measures_gain_loss_without_a_physical_role_read():
    class Shrinker(torch.nn.Module):
        def forward(self, y, q):
            return .3*y

    torch.manual_seed(78)
    y = torch.randn(2, 2, 6, 256, dtype=torch.complex64)
    q = torch.full((2, 2, 6, 1), .8)
    signal = .8*y
    delta = source_probe(signal, torch.ones_like(y), torch.arange(len(y)))
    model = ProjectedDenoise()
    error, gain = probe_response(model, y, q, delta)
    assert torch.equal(error, torch.zeros_like(error))
    assert torch.equal(gain, torch.ones_like(gain))
    _, shrunk = probe_response(Shrinker(), y, q, delta)
    torch.testing.assert_close(shrunk, torch.full((2,), .7))
    value, _ = reconstruction(model, y, q, signal)
    value.backward()
    assert all(h.out.weight.grad.norm() > 0 for h in (model.spectral, model.temporal))

import torch

from experiments.phase1_dura_orthogonal.source_quartic_closure import closure_features


def test_closure_is_invariant_to_global_phase_and_circular_shift():
    generator = torch.Generator().manual_seed(213)
    x = torch.randn(4, 2, 256, generator=generator)
    z = torch.complex(x[:, 0], x[:, 1])
    rotated = torch.polar(torch.ones(4), torch.tensor([.1, .7, -1.2, 2.]))[:, None]*z.roll(17, dims=1)
    y = torch.stack((rotated.real, rotated.imag), dim=1)
    torch.testing.assert_close(closure_features(x), closure_features(y), rtol=3e-4, atol=3e-5)


def test_closure_observes_phase_coupling_beyond_power():
    generator = torch.Generator().manual_seed(214)
    z = torch.complex(torch.randn(3, 256, generator=generator), torch.randn(3, 256, generator=generator))
    spectrum = torch.fft.fft(z)
    phases = torch.rand(3, 256, generator=generator)*6.283185307179586
    scrambled = torch.fft.ifft(spectrum.abs()*torch.polar(torch.ones_like(phases), phases))
    torch.testing.assert_close(torch.fft.fft(scrambled).abs(), spectrum.abs(), rtol=1e-5, atol=1e-5)
    x = torch.stack((z.real, z.imag), dim=1)
    y = torch.stack((scrambled.real, scrambled.imag), dim=1)
    assert (closure_features(x)-closure_features(y)).abs().mean() > 0.01

import torch

from experiments.pamf_lora_main.chirp_band_power import (
    band_power, estimators, frequency_groups, observation_uncertainty, packet_basis, spectrum,
)


def test_even_odd_cross_power_is_near_minus_alias():
    source = torch.randn(4, 6, 256, dtype=torch.complex128)
    basis = packet_basis(source)
    x = basis.to(source.dtype) @ source
    freq = torch.cat(frequency_groups(256, source.device))
    phase = torch.exp(-2j*torch.pi*freq.double()/256)
    even = torch.fft.fft(x[..., ::2], norm='ortho')[..., freq % 128] * (2**.5)
    odd = torch.fft.fft(x[..., 1::2], norm='ortho')[..., freq % 128] * phase * (2**.5)
    full = spectrum(source, basis)
    torch.testing.assert_close((even * odd.conj()).real,
                               full[..., freq].abs().square() - full[..., (freq + 128) % 256].abs().square(),
                               rtol=1e-12, atol=1e-12)


def test_packet_moment_propagates_proper_noise_and_alias_variance():
    source = torch.zeros(2, 6, 256, dtype=torch.complex64)
    basis = packet_basis(source)
    groups = frequency_groups(256, source.device)
    variance = torch.full((2, 6), .4)
    signal_power = torch.full((2, 2), .02)
    noise, normal = observation_uncertainty(variance, basis, groups, signal_power, parity=False)
    _, alias = observation_uncertainty(variance, basis, groups, signal_power, parity=True)
    torch.testing.assert_close(noise, torch.full_like(noise, .4))
    for i, count in enumerate((1, 5)):
        for j, bin_count in enumerate((16, 48)):
            torch.testing.assert_close(alias[:, i, j] - normal[:, i, j], torch.full((2,), .4**2/(count*bin_count)))


def test_shrunk_estimator_has_exact_clean_band_energy_in_absence_of_noise_and_alias():
    source = torch.zeros(3, 6, 256, dtype=torch.complex64)
    source[..., 3] = torch.arange(6).float()
    source[..., 10] = torch.arange(6).flip(0).float()
    y = torch.fft.ifft(source, norm='ortho')
    prior = torch.full((2, 2), .001)
    values, basis, groups, power, _ = estimators(y, prior, torch.ones_like(prior))
    torch.testing.assert_close(power, band_power(basis.to(source.dtype) @ source, groups))
    for best in ('median_shrunk', 'parity_shrunk'):
        torch.testing.assert_close(values[best], power)

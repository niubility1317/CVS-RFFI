import torch

from experiments.pamf_lora_main.private_dose import (
    action_parts, band_energy, match_band_energy, component_swaps,
)


def example():
    gen = torch.Generator().manual_seed(761)
    shape = (3, 6, 256)
    first = torch.randn(shape, generator=gen, dtype=torch.complex64)
    second = torch.randn(shape, generator=gen, dtype=torch.complex64) * 0.4
    rms = torch.linspace(.5, 2., 18).reshape(3, 6, 1)
    frequency = (256 * torch.fft.fftfreq(256)).round().abs()
    limits = (0, 8, 32, 64, 128)
    bands = torch.stack([(frequency > a) & (frequency <= b)
                         for a, b in zip(limits, limits[1:])]).float()
    return first, second, rms, bands


def test_components_are_orthogonal_under_physical_iq_energy():
    first, _, rms, bands = example()
    shared, private = action_parts(first, rms)
    torch.testing.assert_close(shared + private, first)
    torch.testing.assert_close((rms.square() * private).sum(1), torch.zeros_like(first[:, 0]),
                               atol=3e-6, rtol=0)
    torch.testing.assert_close(band_energy(first, rms, bands),
                               band_energy(shared, rms, bands) + band_energy(private, rms, bands))


def test_match_preserves_each_packet_band_dose_and_uses_donor_direction():
    first, second, rms, bands = example()
    matched = match_band_energy(first, second, rms, bands)
    torch.testing.assert_close(matched[..., 0], first[..., 0])
    torch.testing.assert_close(band_energy(matched, rms, bands), band_energy(first, rms, bands))
    for band in bands.bool():
        value, proposal = matched[..., band], second[..., band]
        cross = (value * proposal.conj()).sum((1, 2))
        assert torch.all(cross.real > 0)
        torch.testing.assert_close(cross.imag, torch.zeros_like(cross.real), atol=1e-5, rtol=0)


def test_zero_direction_falls_back_and_default_equal_direction_preserves_input():
    first, second, rms, bands = example()
    second[1, :, bands[1].bool()] = 0
    result = match_band_energy(first, second, rms, bands)
    torch.testing.assert_close(result[1, :, bands[1].bool()], first[1, :, bands[1].bool()])
    torch.testing.assert_close(match_band_energy(first, first, rms, bands), first)


def test_private_direction_swap_keeps_shared_and_private_doses():
    first, second, rms, bands = example()
    swapped, reverse, private_only, shared_only = component_swaps(first, second, rms, bands)
    private_parts = action_parts(private_only, rms)
    shared_parts = action_parts(shared_only, rms)
    a, b = action_parts(first, rms)
    torch.testing.assert_close(private_parts[0], a)
    torch.testing.assert_close(shared_parts[1], b)
    torch.testing.assert_close(band_energy(swapped, rms, bands), band_energy(first, rms, bands))
    torch.testing.assert_close(band_energy(reverse, rms, bands), band_energy(second, rms, bands))


def test_mixed_batch_telemetry_crosses_numpy_as_native_values(monkeypatch):
    from experiments.pamf_lora_main.private_dose_source import summarize_telemetry

    def reject_array_conversion(_):
        raise TypeError("Incompatible torch NumPy array type")

    monkeypatch.setattr(torch.Tensor, "numpy", reject_array_conversion)
    values = torch.ones((2, 2, 4, 4))
    values[:, :, 1] *= 2.
    result = summarize_telemetry([(torch.empty((0, 2, 4, 4)), 0.), (values, 1e-8)])
    assert result["active"] == 2
    assert result["private"]["9_to_32"]["ratio_median"] == 2.
    assert result["max_matched_band_rms_error"] == 1e-8

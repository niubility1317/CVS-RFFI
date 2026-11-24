import torch

from experiments.pamf_lora_main.joint_mode_covariance import (
    chirp_basis, decode, encode, match_rank_direction, moments, noise_covariance, noise_edit,
    paired_bins, without_cross_covariance,
)


def test_decode_leaves_unselected_bins_unchanged_as_zero_correction():
    spectrum = torch.randn(3, 6, 256, dtype=torch.complex128)
    bins, basis = paired_bins(spectrum.device), chirp_basis(spectrum)
    result = decode(encode(spectrum, basis, bins), basis, bins, spectrum)
    expected = torch.zeros_like(spectrum)
    expected[..., bins] = spectrum[..., bins]
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)


def test_ablation_keeps_marginals_and_uses_same_observation_covariance():
    rng = torch.Generator().manual_seed(10)
    clean = torch.randn(124, 8, 96, generator=rng, dtype=torch.float64)
    clean[..., 16:] += .5 * clean[..., :16].repeat(1, 1, 5)
    mean, covariance = moments(clean)
    bins, basis = paired_bins(clean.device), chirp_basis(clean)
    prior = without_cross_covariance(covariance, bins)
    torch.testing.assert_close(prior[:, :16, :16], covariance[:, :16, :16])
    torch.testing.assert_close(prior[:, 16:, 16:], covariance[:, 16:, 16:])
    assert not prior[:, :16, 16:].any()
    assert covariance[:, :16, 16:].abs().max() > .1
    noise = noise_covariance(torch.linspace(.2, 1.2, 6, dtype=clean.dtype).view(1, 6), basis, bins)
    assert noise[..., :16, 16:].abs().max() > .01
    y = mean + clean[:1]
    actual = noise_edit(y, mean, covariance, noise)
    solved = torch.linalg.solve(covariance + noise, (y - mean).unsqueeze(-1))
    expected = -(noise @ solved).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, noise_edit(y, mean, prior, noise))


def test_equal_physical_dose_uses_original_per_chirp_scales():
    rank = torch.randn(2, 6, 256, dtype=torch.complex128)
    correction = torch.randn_like(rank)
    rms = torch.linspace(.5, 3., 6, dtype=torch.float64).view(1, 6, 1)
    actual = match_rank_direction(rank, correction, rms)
    f = torch.fft.fftfreq(256).abs()
    for band in ((f > 0) & (f <= 1 / 32), (f > 1 / 32) & (f <= 1 / 8)):
        assert not torch.equal(actual[..., band], rank[..., band])
        torch.testing.assert_close(
            torch.linalg.vector_norm(actual[..., band] * rms, dim=(-2, -1)),
            torch.linalg.vector_norm(rank[..., band] * rms, dim=(-2, -1)))
    unselected = (f == 0) | (f > 1 / 8)
    assert torch.equal(actual[..., unselected], rank[..., unselected])

"""Estimate weak LoRa band energy from one received chirp packet."""

import torch

from .private_transport import contrast_basis
from .weak_signal_probe import observed_noise_rms


BANDS = ('1_to_8', '9_to_32')
COMPONENTS = ('common', 'private')


def frequency_groups(length, device):
    magnitude = torch.arange(length, device=device)
    magnitude = torch.minimum(magnitude, length - magnitude)
    return tuple(((magnitude >= low) & (magnitude <= high)).nonzero().flatten()
                 for low, high in ((1, 8), (9, 32)))


def packet_basis(observed):
    n = observed.shape[1]
    contrast = contrast_basis(n, dtype=observed.real.dtype, device=observed.device)
    common = torch.full_like(contrast[:, :1], n ** -.5)
    return torch.cat((common, contrast), dim=-1).T


def spectrum(observed, basis):
    return torch.fft.fft(basis.to(observed.dtype) @ observed, norm='ortho')


def band_power(coefficients, groups):
    modes = (coefficients[:, :1], coefficients[:, 1:])
    return torch.stack([torch.stack([v[..., band].abs().square().mean((-2, -1)) for band in groups], dim=-1)
                        for v in modes], dim=1)


def covariances(variance, basis):
    projected = (basis * variance[:, None, :]) @ basis.T
    return projected[:, :1, :1], projected[:, 1:, 1:]


def observation_uncertainty(variance, basis, groups, prior_mean, *, parity):
    """Approximate signal-power noise using a proper additive Gaussian model."""
    pairs = []
    for i, covariance in enumerate(covariances(variance, basis)):
        modes = covariance.shape[-1]
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        squared = covariance.square().sum((-2, -1))
        bins = trace.new_tensor([len(band) for band in groups])
        # Under isotropic power, E[xᴴΣx] = mean signal power × tr(Σ).
        noise_variance = ((2 if parity else 1) * squared[:, None]
                          + 2 * trace[:, None] * prior_mean[i]) / (modes**2 * bins)
        pairs.append((trace / modes, noise_variance))
    return tuple(torch.stack(values, dim=1) for values in zip(*pairs, strict=True))


def estimators(observed, prior_mean, prior_variance):
    basis = packet_basis(observed)
    groups = frequency_groups(observed.shape[-1], observed.device)
    transformed = spectrum(observed, basis)
    alias = tuple((band + observed.shape[-1] // 2) % observed.shape[-1] for band in groups)
    observed_power = band_power(transformed, groups)
    alias_power = band_power(transformed, alias)
    noise = observed_noise_rms(observed).square().squeeze(-1)
    mean_floor, floor_variance = observation_uncertainty(noise, basis, groups, prior_mean, parity=False)
    _, alias_variance = observation_uncertainty(noise, basis, groups, prior_mean, parity=True)
    raw_floor = observed_power - mean_floor[..., None]
    cross = observed_power - alias_power

    def shrink(measured, variance):
        gain = prior_variance / (prior_variance + variance).clamp_min(1e-12)
        return (prior_mean + gain*(measured - prior_mean)).clamp_min(0.)

    return (dict(prior=prior_mean.expand_as(observed_power), median_raw=raw_floor, parity_raw=cross,
                 median_shrunk=shrink(raw_floor, floor_variance), parity_shrunk=shrink(cross, alias_variance)),
            basis, groups, observed_power, alias_power)

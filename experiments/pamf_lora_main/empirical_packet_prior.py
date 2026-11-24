"""Estimate LoRa detail from a received-spectrum likelihood over unlabeled source IQ."""

import torch

from .joint_mode_covariance import match_rank_direction
from .weak_signal_probe import observed_noise_rms


def detail_bins(device):
    return torch.cat((torch.arange(1, 33, device=device), torch.arange(224, 256, device=device)))


def observation_bins(bins):
    return (bins <= 8) | (bins >= 248)


def split(spectrum):
    common = spectrum.mean(1, keepdim=True)
    return common, spectrum - common


def bank_summary(bank):
    _, private = split(bank)
    return private.mean(0, keepdim=True), private.var(0, unbiased=False, keepdim=True)


def weights(measurement, bank, variance):
    """Evaluate a proper complex Gaussian likelihood using common-mode variance."""
    errors = (measurement[:, None, :] - bank[None, :, :]).abs().square().sum(-1)
    return torch.softmax(-errors / variance.clamp_min(1e-7), dim=-1)


def posterior(near, variance, bank, private_mean, private_power, keys):
    observed_common, observed_private = split(near)
    source_common, source_private = split(bank)
    n = near.shape[1]
    common_noise = variance.sum(1, keepdim=True) / n**2
    weight = weights(observed_common[:, 0, keys], source_common[:, 0, keys], common_noise)
    coefficients = weight.to(bank.dtype)
    common = (coefficients @ source_common[:, 0]).unsqueeze(1)
    conditional = (coefficients @ source_private.flatten(1)).reshape_as(observed_private)

    # diag[(I - 11^T/n) diag(variance) (I - 11^T/n)].
    contrast_noise = (1 - 2/n) * variance + common_noise
    private_gain = private_power / (private_power + contrast_noise.unsqueeze(-1)).clamp_min(1e-7)

    def estimate(center):
        estimate_private = center + private_gain * (observed_private - center)
        return common + split(estimate_private)[1]

    return estimate(private_mean), estimate(conditional), weight


def candidate_edits(y, rot, quality, rms, model, prior):
    bank, mean, power = prior
    bins = detail_bins(y.device)
    spectrum = torch.fft.fft(y, norm='ortho')
    near = spectrum[..., bins]
    variance = observed_noise_rms(y).square().squeeze(-1)
    ordinary, linked, weight = posterior(near, variance, bank, mean, power, observation_bins(bins))
    proposals = []
    for estimate in (ordinary, linked):
        edit = torch.zeros_like(spectrum)
        edit[..., bins] = .5 * (estimate - near)
        proposals.append(edit)
    parent = torch.lerp(y, model(y, quality, rot)[-1], .5) - y
    parent_spectrum = torch.fft.fft(parent, norm='ortho')
    matched = [match_rank_direction(parent_spectrum, edit, rms) for edit in proposals]
    waves = (parent, *(torch.fft.ifft(v, norm='ortho') for v in (*proposals, *matched)))
    return waves, weight

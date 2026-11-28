"""Receiver-only novelty relative to lawful source clean packets."""

import torch

from experiments.pamf_lora_main.w24_native_dura import capture_scale


def capture_features(raw):
    signal = raw / capture_scale(raw)
    blocks = signal.unflatten(-1, (8, -1))
    spectrum = torch.fft.fft(blocks, norm='ortho').abs().square()
    frequencies = torch.fft.fftfreq(blocks.shape[-1], device=raw.device).abs()
    energy = spectrum.sum(-1).clamp_min(1e-8)
    edge = spectrum[..., frequencies >= .35].sum(-1) / energy
    flatness = (spectrum / (energy[..., None] / blocks.shape[-1])).clamp_min(1e-7).log().mean(-1)
    crest = blocks.abs().pow(4).mean(-1) / blocks.abs().square().mean(-1).square().clamp_min(1e-8)
    repeated = (blocks[:, 1:] * blocks[:, :-1].conj()).sum(-1).abs()
    repeated = repeated / (blocks[:, 1:].norm(dim=-1) * blocks[:, :-1].norm(dim=-1)).clamp_min(1e-8)
    features = [edge.mean(-1), edge.std(-1), flatness.mean(-1), flatness.std(-1),
                crest.clamp_min(1e-8).log().mean(-1), repeated.mean(-1), repeated.std(-1)]
    for lag in (1, 2, 4):
        features.append((signal[:, lag:] * signal[:, :-lag].conj()).mean(-1).abs())
    return torch.stack(features, dim=-1)


def novelty(features, center, precision):
    deviation = features - center
    return ((deviation @ precision) * deviation).sum(-1).clamp_min(0.)


def fit_clean_geometry(clean_fit, clean_calibration):
    center = clean_fit.mean(0)
    deviation = clean_fit - center
    covariance = deviation.T @ deviation / (len(deviation) - 1)
    regularized = .9 * covariance + .1 * covariance.diag().clamp_min(1e-6).diag()
    precision = torch.linalg.inv(regularized)
    cutoff = torch.quantile(novelty(clean_calibration, center, precision), .995).clamp_min(1.)
    if not (torch.isfinite(precision).all() and torch.isfinite(cutoff)):
        raise ValueError('Expected finite clean observation geometry')
    return dict(center=center.tolist(), precision=precision.tolist(), cutoff=float(cutoff))


def clean_geometry_weight(raw, calibration):
    features = capture_features(raw)
    center = features.new_tensor(calibration['center'])
    precision = features.new_tensor(calibration['precision'])
    score = novelty(features, center, precision) / calibration['cutoff']
    amount = (score - 1.).clamp(0., 1.)
    return amount.square() * (3. - 2. * amount)

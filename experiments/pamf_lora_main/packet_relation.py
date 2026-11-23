"""Preserve measured LoRa packet relations across independent corruption draws."""

import torch
from torch.nn import functional as F

from .channel_blind_source import waveform_risk
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .lapped_posterior import LappedPosterior


def fingerprint(wave):
    """Measure complex repeat means and mirror-sideband closure in two bands."""
    rms = wave.abs().square().mean(dim=(-2,-1), keepdim=True).clamp_min(1e-8).sqrt()
    spectrum = torch.fft.fft(wave / rms, norm='ortho')
    carrier = spectrum[..., :1]
    phase = carrier.conj() / carrier.abs().clamp_min(1e-6)
    positive = spectrum[..., 1:33] * phase
    negative = spectrum[..., -32:].flip(-1) * phase
    even_p, odd_p = positive[..., ::2, :].mean(-2), positive[..., 1::2, :].mean(-2)
    even_n, odd_n = negative[..., ::2, :].mean(-2), negative[..., 1::2, :].mean(-2)
    mirror = .5 * (even_p * odd_n + odd_p * even_n)
    mean_p, mean_n = positive.mean(-2), negative.mean(-2)
    features = []
    for section in (slice(0,8), slice(8,32)):
        coherent = torch.cat((mean_p[..., section], mean_n[..., section]), dim=-1)
        features.extend((torch.view_as_real(coherent).flatten(-2),
                         torch.view_as_real(mirror[..., section]).flatten(-2)))
    return tuple(features)


def squared_distances(first, second):
    return (first.unsqueeze(-2) - second.unsqueeze(-3)).square().mean(-1)


def relation_cost(reference, estimates):
    # estimates has axes (stage, packet, independent draw, chirp, sample).
    clean_features = fingerprint(reference)
    output_features = fingerprint(estimates)
    diagonal = torch.eye(reference.shape[0], device=reference.device, dtype=torch.bool)
    values = []
    for clean, output in zip(clean_features, output_features, strict=True):
        power = clean.var(dim=0, unbiased=False)
        scale = (power + .05 * power.mean() + 1e-5).sqrt()
        target = squared_distances(clean / scale, clean / scale).log1p()
        observed = squared_distances(output[:, :, 0] / scale, output[:, :, 1] / scale).log1p()
        difference = F.smooth_l1_loss(observed, target.expand_as(observed), beta=.20, reduction='none')
        diagonal_risk = difference[:, diagonal].mean(-1)
        pair_risk = difference[:, ~diagonal].sum(-1) / max(1, diagonal.numel() - reference.shape[0])
        values.append(.30 * diagonal_risk + .70 * pair_risk)
    return torch.stack(values).mean(0)


class PacketRelation(LappedPosterior):
    def __init__(self, preserve_relations):
        super().__init__(local=False)
        self.preserve_relations = preserve_relations


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    clean = complex_iq(iq)
    source, source_phase, _, source_rms = coordinates(tracker, iq)
    source_frame = source_phase * source_rms
    draws = torch.stack((recipe.perturb(clean,rng), recipe.perturb(clean,rng)), dim=1)
    if with_identity:
        draws[-2:] = clean[-2:,None]
    observed, phase, quality, level = coordinates(tracker, real_iq(draws.flatten(0,1)))
    frame = phase * level
    target = clean[:,None].expand_as(draws).flatten(0,1) / frame
    coarse, fine, prior, variance = model.restore(observed, quality)
    error = (model.transformed(target) - prior).abs().square()
    likelihood = (error / variance + variance.log())[...,model.noncarrier].mean()
    restored = torch.stack((coarse, fine)) * frame
    physical = restored.unflatten(1, draws.shape[:2]) / source_frame[:,None]
    consistency = (physical[1,:,0] - physical[1,:,1]).abs().square().mean()
    geometry = relation_cost(source, physical)
    values = torch.stack((waveform_risk(coarse,target), waveform_risk(fine,target),
                          consistency, likelihood, *geometry.unbind()))
    first, second, paired = recipe.loss_weights(step)
    loss = first * values[0] + second * values[1] + paired * consistency + .08 * likelihood
    if model.preserve_relations:
        ramp = min(1., step / 8000.)
        loss = loss + (.25 + .55 * ramp) * (first * geometry[0] + second * geometry[1])
    return loss, values

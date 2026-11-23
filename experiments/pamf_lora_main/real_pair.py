"""Protect signed complex differences of two complete measured source packets."""

import torch

from .channel_blind_source import waveform_risk
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .pilot_joint import apply_echo, echo_taps
from .rf_noise_transport import colored_observation
from .tail_risk import stratified_upper_quarter
from .weak_signal_probe import observed_noise_rms


BANDS = ((1, 8), (9, 32), (33, 64), (65, 128))


def paired_view(clean, rng):
    count = len(clean)
    if count % 2:
        clean = torch.cat((clean, clean[-1:]))
    packets = clean.unflatten(0, (-1, 2))
    template = packets[:, 0]
    shared_echo = tuple(tap[:, None] for tap in echo_taps(template, rng))
    distorted = apply_echo(packets, shared_echo)
    unit = torch.ones_like(template)
    shared_noise = colored_observation(unit, rng) - unit
    scale = packets.abs().square().mean(dim=(1, 3), keepdim=True).clamp_min(1e-8).sqrt()
    return (distorted + scale * shared_noise[:, None]).flatten(0, 1)[:count]


def signed_difference(sources):
    grouped = sources.unflatten(0, (-1, 2))
    return grouped[:, 1] - grouped[:, 0]


def band_risks(received_difference, clean_difference, severity):
    """Measure per-band complex error with a clean reference energy floor."""
    ref = torch.fft.fft(clean_difference, norm='ortho')
    error = torch.fft.fft(received_difference - clean_difference, norm='ortho')
    frequency = torch.fft.fftfreq(ref.shape[-1], device=ref.device).abs() * ref.shape[-1]
    values = []
    for low, high in BANDS:
        keep = (frequency >= low) & (frequency <= high)
        power = ref[..., keep].abs().square().mean(-1)
        risk = (error[..., keep].abs().square().mean(-1) / (power + .002)).log1p()
        values.append((risk.mean(dim=(1, 2)) * severity).sum() / severity.sum())
    return torch.stack(values)


def objective(model, tracker, iq, rng, *, with_identity, recipe, step, pair_weight, tail_weight):
    clean = complex_iq(iq)
    waves = torch.stack((paired_view(clean, rng), paired_view(clean, rng)), dim=1)
    if with_identity:
        waves[-2:] = clean[-2:, None]
    _, clean_rotation, _, clean_rms = coordinates(tracker, iq)
    clean_basis = clean_rotation * clean_rms
    observed, rotation, q, rms = coordinates(tracker, real_iq(waves.flatten(0, 1)))
    basis = rotation * rms
    target = clean[:, None].expand_as(waves).flatten(0, 1) / basis
    coarse, fine = recipe.restore(model, observed, q, rotation)
    stages = (coarse, fine)
    risks = torch.stack([waveform_risk(out, target, per_chirp=True) for out in stages])
    means = risks.flatten(1).mean(1)
    restored_iq = [(out * basis).unflatten(0, waves.shape[:2]) for out in stages]
    agreement = (restored_iq[1][:, 0] - restored_iq[1][:, 1]).abs().square().mean()
    slot = slice(None, -2) if with_identity else slice(None)
    perturbed = risks.unflatten(1, waves.shape[:2])[:, slot]
    with torch.no_grad():
        noise = observed_noise_rms(observed).squeeze(-1).unflatten(0, waves.shape[:2])[slot]
    tails = stratified_upper_quarter(perturbed, noise)
    excess = tails - perturbed.flatten(1).mean(1)
    coarse_weight, fine_weight, consistency_weight = recipe.loss_weights(step)
    loss = (coarse_weight * (means[0] + tail_weight * excess[0])
            + fine_weight * (means[1] + tail_weight * excess[1])
            + consistency_weight * agreement)

    # The source used to construct each within-receiver pair has its own clean phase
    # and scale. The relation compares both predictions in those same source frames.
    count = len(noise) - len(noise) % 2
    desired = signed_difference((clean / clean_basis)[slot][:count])[:, None]
    family_noise = noise[:count].unflatten(0, (-1, 2)).mean(dim=(1, 2, 3))
    severity = 1 + 2 * family_noise / (.15 + family_noise)
    relation = torch.stack([
        band_risks(signed_difference((out / clean_basis[:, None])[slot][:count]), desired, severity)
        for out in restored_iq
    ])
    if pair_weight:
        loss = loss + pair_weight * (coarse_weight * relation[0].mean() + fine_weight * relation[1].mean())
    metrics = torch.cat((means, agreement[None], tails, relation.flatten()))
    return loss, metrics

"""Measure bias and noise transfer with antithetic pairs of generic LoRa corruption."""

import torch

from .channel_blind_source import waveform_risk
from .crossfit_pilot import apply_transfer, generic_taps
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .natural_detail import detail_bands, transfer_risk, weak_difference


def antithetic_inputs(signal, change, perturb, rng, *, with_identity, transform=None):
    """Hold generic response and noise scale fixed across both source signals."""
    taps = generic_taps(signal, rng)
    has_taps = torch.rand((len(signal), 1), device=signal.device, generator=rng) < .5
    taps = taps * has_taps
    base, shifted = apply_transfer(signal, taps), apply_transfer(signal + change, taps)
    if transform is not None:
        base, shifted = transform(base, shifted)
    noise = perturb(base, rng) - base
    if with_identity:
        base[-2:] = signal[-2:]
        shifted[-2:] = signal[-2:] + change[-2:]
        noise[-2:] = 0
    return torch.stack((base + noise, base - noise, shifted + noise, shifted - noise), dim=1)


def pair_terms(values):
    """Split response to opposite corruptions into average and half difference."""
    first, opposite = values[:, ::2], values[:, 1::2]
    return .5*(first + opposite), .5*(first - opposite)


def component_risk(error, signal):
    """Measure common and private spectral errors in physical source coordinates."""
    delta, reference = [torch.fft.fft(x, norm='ortho') for x in (error, signal)]
    public, public_reference = (v.mean(-2, keepdim=True) for v in (delta, reference))
    component_pairs = ((public, public_reference), (delta - public, reference - public_reference))
    bands = detail_bands(reference)
    losses = []
    for residual, target in component_pairs:
        for band in bands:
            power = target[..., band].abs().square().mean((-2, -1))
            distortion = residual[..., band].abs().square().mean((-2, -1))
            losses.append((distortion / (power + .002)).log1p().mean())
    return torch.stack(losses).mean()


def objective(model, tracker, iq, rng, *, with_identity, recipe, step, split_risk,
              build_inputs=antithetic_inputs):
    clean = complex_iq(iq)
    framed, rotation, _, rms = coordinates(tracker, iq)
    source_basis = rotation * rms
    delta = weak_difference(framed)
    change = delta * source_basis
    wave = build_inputs(clean, change, recipe.perturb, rng, with_identity=with_identity)
    observed, rotation, quality, rms = coordinates(tracker, real_iq(wave.flatten(0, 1)))
    basis = rotation * rms
    target = torch.stack((clean, clean, clean + change, clean + change), dim=1)
    coarse, refined = recipe.restore(model, observed, quality, rotation)
    stage_target = target.flatten(0, 1) / basis
    coarse_risk = waveform_risk(coarse, stage_target)
    full_risk = waveform_risk(refined, stage_target)
    recovered = (refined * basis).unflatten(0, wave.shape[:2])
    half = torch.lerp(wave, recovered, .5) / source_basis[:, None]
    adapted_target = target / source_basis[:, None]
    mean, difference = pair_terms(half)
    reference = adapted_target[:, ::2]
    point = component_risk(half - adapted_target, adapted_target)
    bias = component_risk(mean - reference, reference)
    response = component_risk(difference, reference)
    transfer = transfer_risk(mean[:, 1] - mean[:, 0], delta)
    focus = bias + response if split_risk else point
    first, second, _ = recipe.loss_weights(step)
    loss = first*coarse_risk + second*full_risk + .30*focus + .35*transfer
    return loss, torch.stack((coarse_risk, full_risk, point, bias, response, transfer))

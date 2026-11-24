"""Measure bias and noise transfer with antithetic pairs of generic LoRa corruption."""

import torch

from .channel_blind_source import waveform_risk
from .crossfit_pilot import apply_transfer, generic_taps
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .lag_profile import lag_profile_risk
from .natural_detail import detail_bands, transfer_risk, weak_difference
from .packet_relation import packet_relation_risk


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


def private_direction_risk(even, odd, reference):
    """Measure half-strength bias and noise along each clean private chirp spectrum."""
    private = []
    for value in (even - reference, odd, reference):
        centered = value - value.mean(-2, keepdim=True)
        private.append(torch.fft.fft(centered, norm='ortho'))
    bias, noise, target = private
    risks = []
    for band in detail_bands(target):
        template = target[..., band]
        energy = template.abs().square().mean(-1)
        denominator = energy + 1e-4
        axis = template.conj()
        bias_gain = (bias[..., band] * axis).mean(-1) / denominator
        noise_gain = (noise[..., band] * axis).mean(-1) / denominator
        risks.append(torch.stack((bias_gain.abs().square().log1p().mean(),
                                  noise_gain.abs().square().log1p().mean())))
    return torch.stack(risks, dim=1)


def objective(model, tracker, iq, rng, *, with_identity, recipe, step, split_risk,
              build_inputs=antithetic_inputs, private_risk_weight=0., relation_weight=0., lag_weight=0.,
              observation_frame=coordinates, carrier_risk=None, mix=torch.lerp):
    clean = complex_iq(iq)
    framed, rotation, _, rms = observation_frame(tracker, iq)
    source_basis = rotation * rms
    delta = weak_difference(framed)
    change = delta * source_basis
    wave = build_inputs(clean, change, recipe.perturb, rng, with_identity=with_identity)
    observed, rotation, quality, rms = observation_frame(tracker, real_iq(wave.flatten(0, 1)))
    basis = rotation * rms
    target = torch.stack((clean, clean, clean + change, clean + change), dim=1)
    stages = recipe.restore(model, observed, quality, rotation)
    coarse, refined = recipe.transform_stages(observed, stages, rms)
    stage_target = target.flatten(0, 1) / basis
    coarse_risk = waveform_risk(coarse, stage_target)
    full_risk = waveform_risk(refined, stage_target)
    recovered = (refined * basis).unflatten(0, wave.shape[:2])
    physical_half = mix(wave, recovered, .5)
    half = physical_half / source_basis[:, None]
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
    values = torch.stack((coarse_risk, full_risk, point, bias, response, transfer))
    if private_risk_weight:
        private = private_direction_risk(mean[:, 0], difference[:, 0], reference[:, 0])
        loss = loss + private_risk_weight * private.sum(0).mean()
        values = torch.cat((values, private.flatten()))
    if relation_weight:
        relations = packet_relation_risk(half[:, :2], reference[:, 0])
        loss = loss + relation_weight * relations.mean()
        values = torch.cat((values, relations))
    if lag_weight:
        lag = lag_profile_risk(half[:, :2], reference[:, 0])
        loss = loss + lag_weight * lag.mean()
        values = torch.cat((values, lag))
    if carrier_risk is not None:
        extra, diagnostics = carrier_risk(coarse, refined, stage_target, half, adapted_target,
                                          source_frame=framed,
                                          received_half=physical_half / basis.unflatten(0, wave.shape[:2]))
        loss = loss + extra
        values = torch.cat((values, diagnostics))
    return loss, values

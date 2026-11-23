"""Learn to retain measured weak differences between unlabeled source packets."""

import torch

from .channel_blind_source import waveform_risk
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .pilot_joint import apply_echo, echo_taps


def detail_bands(value):
    # At the registered 256 kHz and 256 points, these are bins 1–8 and 9–32.
    f = torch.fft.fftfreq(value.shape[-1], device=value.device).abs()
    return (f > 0) & (f <= 1/32), (f > 1/32) & (f <= 1/8)


def weak_difference(framed):
    near, middle = detail_bands(framed)
    delta = .5 * (framed.roll(1, dims=0) - framed)
    spectrum = torch.fft.fft(delta, norm='ortho')
    return torch.fft.ifft(spectrum * (near | middle), norm='ortho')


def transfer_risk(prediction, expected):
    error = torch.fft.fft(prediction - expected, norm='ortho').abs().square()
    energy = torch.fft.fft(expected, norm='ortho').abs().square()
    # Normalize by actual trace energy per band; a floor limits gradients when
    # two source packets contain nearly identical sidebands.
    return torch.stack([
        (error[..., band].mean(-1) / (.002 + energy[..., band].mean(-1))).log1p().mean()
        for band in detail_bands(expected)
    ]).mean()


def paired_inputs(clean, perturb, rng, change, *, with_identity):
    a,b = perturb(clean, rng),perturb(clean, rng)
    if with_identity:
        a[-2:] = b[-2:] = clean[-2:]
    # Both members of the first pair have exactly the same additive noise.
    return torch.stack((a, b, a + change), dim=1)


def paired_echo_inputs(clean, perturb, rng, change, *, with_identity):
    first_taps = echo_taps(clean, rng)
    first = perturb(apply_echo(clean, first_taps), rng)
    second = perturb(apply_echo(clean, echo_taps(clean, rng)), rng)
    changed = first + apply_echo(change, first_taps)
    if with_identity:
        first[-2:] = second[-2:] = clean[-2:]
        changed[-2:] = clean[-2:] + change[-2:]
    return torch.stack((first, second, changed), dim=1)


def natural_objective(model, tracker, iq, rng, *, with_identity, recipe, step, preserve,
                      make_pair=paired_inputs):
    clean = complex_iq(iq)
    framed, rotation, _, rms = coordinates(tracker, iq)
    clean_basis = rotation * rms
    delta = weak_difference(framed)
    change = delta * clean_basis
    wave = make_pair(clean, recipe.perturb, rng, change, with_identity=with_identity)
    observed, rotation, quality, rms = coordinates(tracker, real_iq(wave.flatten(0, 1)))
    basis = rotation * rms
    targets = torch.stack((clean, clean, clean + change), dim=1).flatten(0, 1)
    target = targets / basis
    coarse, refined = recipe.restore(model, observed, quality, rotation)
    refined_iq = (refined * basis).unflatten(0, wave.shape[:2])
    consistent = (refined_iq[:, 0] - refined_iq[:, 1]).abs().square().mean()
    transport = (refined_iq[:, 2] - refined_iq[:, 0]) / clean_basis
    measured = transfer_risk(transport, delta)
    values = torch.stack((waveform_risk(coarse,target), waveform_risk(refined,target),
                          consistent, measured))
    first,second,paired = recipe.loss_weights(step)
    loss = first*values[0] + second*values[1] + paired*values[2] + preserve*values[3]
    return loss,values

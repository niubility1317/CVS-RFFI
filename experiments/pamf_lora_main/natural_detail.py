"""Learn to retain measured weak differences between unlabeled source packets."""

import torch

from .channel_blind_source import waveform_risk
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .pilot_joint import apply_echo, echo_taps
from .tail_risk import stratified_upper_quarter
from .weak_signal_probe import observed_noise_rms


def detail_bands(value):
    # At the registered 256 kHz and 256 points, these are bins 1–8 and 9–32.
    f = torch.fft.fftfreq(value.shape[-1], device=value.device).abs()
    return (f > 0) & (f <= 1/32), (f > 1/32) & (f <= 1/8)


def weak_difference(framed):
    near, middle = detail_bands(framed)
    delta = .5 * (framed.roll(1, dims=0) - framed)
    spectrum = torch.fft.fft(delta, norm='ortho')
    return torch.fft.ifft(spectrum * (near | middle), norm='ortho')


def private_difference(framed):
    delta = weak_difference(framed)
    return delta - delta.mean(1, keepdim=True)


def private_gain_error(prediction, expected):
    """Complex gain error along the clean zero-sum chirp difference in each band."""
    expected = expected - expected.mean(1, keepdim=True)
    error = prediction - expected
    error = error - error.mean(1, keepdim=True)
    ref = torch.fft.fft(expected, norm='ortho')
    delta = torch.fft.fft(error, norm='ortho')
    gains = []
    for band in detail_bands(expected):
        reference = ref[..., band]
        power = reference.abs().square().sum((1, 2))
        cross = (delta[..., band] * reference.conj()).sum((1, 2))
        gains.append(cross / power.clamp_min(1e-7))
    return torch.stack(gains, dim=-1)


def transfer_risk(prediction, expected):
    error = torch.fft.fft(prediction - expected, norm='ortho').abs().square()
    energy = torch.fft.fft(expected, norm='ortho').abs().square()
    # Normalize by actual trace energy per band; a floor limits gradients when
    # two source packets contain nearly identical sidebands.
    return torch.stack([
        (error[..., band].mean(-1) / (.002 + energy[..., band].mean(-1))).log1p().mean()
        for band in detail_bands(expected)
    ]).mean()


def projection_risk(prediction, expected):
    """Penalize gain loss and quadrature leakage of measured weak differences."""
    restored = torch.fft.fft(prediction, norm='ortho')
    reference = torch.fft.fft(expected, norm='ortho')
    cross = (restored - reference) * reference.conj()
    losses = []
    for band in detail_bands(expected):
        energy = reference[..., band].abs().square().mean(-1)
        deviation = cross[..., band].mean(-1) / (energy + 1e-4)
        losses.append(deviation.abs().square().log1p().mean())
    return torch.stack(losses).mean()


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
                      make_pair=paired_inputs, projection_weight=None, tail_weight=None,
                      difference=weak_difference, half_private_weight=None):
    clean = complex_iq(iq)
    framed, rotation, _, rms = coordinates(tracker, iq)
    clean_basis = rotation * rms
    delta = difference(framed)
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
    per_chirp = tail_weight is not None
    stage_risks = torch.stack(tuple(waveform_risk(stage, target, per_chirp=per_chirp)
                                    for stage in (coarse, refined)))
    means = stage_risks.flatten(1).mean(1) if per_chirp else stage_risks
    values = torch.cat((means, torch.stack((consistent, measured))))
    first,second,paired = recipe.loss_weights(step)
    loss = first*values[0] + second*values[1] + paired*values[2] + preserve*values[3]
    if per_chirp:
        slots = slice(None, -2) if with_identity else slice(None)
        perturbed = stage_risks.unflatten(1, wave.shape[:2])[:, slots]
        with torch.no_grad():
            noise = observed_noise_rms(observed).squeeze(-1).unflatten(0, wave.shape[:2])[slots]
        tails = stratified_upper_quarter(perturbed, noise)
        excess = tails - perturbed.flatten(1).mean(dim=1)
        loss = loss + tail_weight * (first * excess[0] + second * excess[1])
        values = torch.cat((values, tails))
    if projection_weight is not None:
        projected = projection_risk(transport, delta)
        loss = loss + projection_weight * projected
        values = torch.cat((values, projected[None]))
    if half_private_weight is not None:
        # Match the half-strength deployment readout in physical IQ.
        readout = torch.lerp(wave, refined_iq, .5)
        half_difference = (readout[:, 2] - readout[:, 0]) / clean_basis
        error = private_gain_error(half_difference, delta)
        loss = loss + half_private_weight * error.abs().square().log1p().mean()
        values = torch.cat((values, error.real.mean(0), error.imag.abs().mean(0),
                            error.abs().square().log1p().mean(0)))
    return loss,values

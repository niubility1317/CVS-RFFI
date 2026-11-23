"""Constrain learned LoRa edits using a measured shared-spectrum interval."""

from functools import partial
from math import log

import torch

from .channel_blind_source import waveform_risk
from .data import SOURCE
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .lapped_posterior import LappedPosterior
from .natural_detail import paired_inputs, transfer_risk, weak_difference
from .reference_cancel import read_relative_action
from .variance_transport import physical_norm


def edit_interval(edit, axis, amplitude, floor, confidence):
    """Clip shared parallel and quadrature edits independently."""
    mean = edit.mean(1, keepdim=True)
    components = mean * axis.conj()
    slack = (amplitude - floor).clamp_min(0.)
    radial = components.real.clamp(-slack, .20 * slack)
    angular_limit = .65 * (amplitude.square() - floor.square()).clamp_min(0.).sqrt()
    tangent = components.imag.clamp(-angular_limit, angular_limit)
    correction = confidence * (torch.complex(radial, tangent) * axis - mean)
    return edit + correction


def spectral_budget(original, preferred):
    before = original.abs().square().sum(1, keepdim=True)
    after = preferred.abs().square().sum(1, keepdim=True)
    ratio = (after / before.clamp_min(1e-14)).clamp(1e-14, 1.).sqrt()
    return original * ratio


class CoherentInterval(LappedPosterior):
    def __init__(self, directional):
        super().__init__(local=False)
        self.directional = directional
        bins = torch.arange(256)
        distance = torch.minimum(bins, 256 - bins)
        self.register_buffer('weak', (distance > 0) & (distance <= 32), persistent=False)
        self.register_buffer('outer_fft', distance > 64, persistent=False)
        self.register_buffer('power_limit', torch.ones(1, 1, 256))
        self.register_buffer('noise_floor', torch.tensor(.02))

    def spectral_evidence(self, observed, quality):
        spectrum = torch.fft.fft(observed, norm='ortho')

        def group(x, q):
            weights = .25 + .75 * q
            return (x * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True)

        odd = group(spectrum[:, ::2], quality[:, ::2])
        even = group(spectrum[:, 1::2], quality[:, 1::2])
        raw = spectrum.mean(1, keepdim=True)
        amp = raw.abs()
        axis = torch.where(amp > 1e-7, raw / amp.clamp_min(1e-7), 1.)
        outer = raw[..., self.outer_fft].abs().square().median(-1, keepdim=True).values / log(2.)
        cross_power = (odd * even.conj()).real.clamp_min(0.)
        residual = .5 * (odd - even).abs().square()
        useful = (cross_power - 1.5 * outer).clamp_min(0.)
        fraction = useful / (.001 + cross_power + residual + 1.5 * outer)
        plausible = (self.power_limit / (amp.square() + 1e-8)).clamp_max(1.)
        confidence = fraction * plausible * self.weak
        floor = (.95 * useful.sqrt()).minimum(amp)
        per_chirp = spectrum[..., self.outer_fft].abs().square().median(-1).values / log(2.)
        received_noise = per_chirp.median(1).values.sqrt()[:, None, None]
        return (axis, amp, floor, confidence), received_noise

    @staticmethod
    def project(time_edit, *, evidence, directional):
        spec = torch.fft.fft(time_edit, norm='ortho')
        corrected = edit_interval(spec, *evidence)
        if not directional:
            corrected = spectral_budget(spec, corrected)
        return torch.fft.ifft(corrected, norm='ortho')

    def stages(self, y, q, rms, *, constraint='trained', use_action=True, use_noise=True):
        action, trust, _ = read_relative_action(y, q)
        stable_action = .5 * (action + torch.cat((action[..., :1], action[..., :-1]), dim=-1))
        evidence, received_noise = self.spectral_evidence(y, q)
        directional = self.directional if constraint == 'trained' else constraint == 'directional'
        projection = None
        if constraint != 'off':
            projection = partial(self.project, evidence=evidence, directional=directional)
        noise = None if use_noise else self.noise_floor.reshape(1, 1, 1, 1)
        coarse, fine, prior, variance = self.restore(
            y, q, reference_action=stable_action if use_action else None, noise_override=noise,
            project_edit=projection,
        )
        limited = []
        raw_norm = physical_norm(y, rms)
        cap = .07 + .13 * ((received_noise - .06) / .24).clamp(0., 1.)
        for wave in (coarse, fine):
            delta = wave - y
            rms_delta = physical_norm(delta, rms)
            dose = (cap * raw_norm / rms_delta.clamp_min(1e-7)).clamp_max(1.)
            limited.append(y + delta * dose)
        return *limited, prior, variance, evidence[-1], trust


@torch.no_grad()
def initialize(model, tracker, fit, calibration):
    balanced = fit.unflatten(0, (len(SOURCE), -1))[:, :1000].flatten(0, 1)
    power = []
    for x in balanced.split(128):
        z = coordinates(tracker, x)[0]
        power.append(torch.fft.fft(z.mean(1), norm='ortho').abs().square())
    limit = .005 + 2. * torch.quantile(torch.cat(power), .99, dim=0)
    model.power_limit.copy_(limit)
    outer = []
    for x in calibration.split(128):
        z = coordinates(tracker, x)[0]
        outer.append(model.estimate_power(model.transformed(z)).flatten())
    model.noise_floor.copy_(torch.cat(outer).median())


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    source = complex_iq(iq)
    clean, clean_rotation, _, clean_rms = coordinates(tracker, iq)
    clean_frame = clean_rotation * clean_rms
    weak = weak_difference(clean)
    delta = weak * clean_frame
    wave = paired_inputs(source, recipe.perturb, rng, delta, with_identity=with_identity)
    y, angle, q, rms = coordinates(tracker, real_iq(wave.flatten(0, 1)))
    frame = angle * rms
    target = torch.stack((source, source, source + delta), dim=1).flatten(0, 1) / frame
    coarse, fine, prior, variance, confidence, _ = model.stages(y, q, rms)
    error = (model.transformed(target) - prior).abs().square()
    prior_risk = (error / variance + variance.log())[..., model.noncarrier].mean()
    physical = (fine * frame).unflatten(0, wave.shape[:2])
    paired_risk = ((physical[:, 0] - physical[:, 1]) / clean_frame).abs().square().mean()
    change = (physical[:, 2] - physical[:, 0]) / clean_frame
    source_shared = weak.mean(1, keepdim=True)
    got_shared = change.mean(1, keepdim=True)
    shared_error = transfer_risk(got_shared, source_shared)
    private_error = transfer_risk(change - got_shared, weak - source_shared)
    values = torch.stack((waveform_risk(coarse, target), waveform_risk(fine, target),
                          paired_risk, shared_error, private_error, prior_risk,
                          confidence[..., model.weak].mean()))
    first, second, agreement = recipe.loss_weights(step)
    ramp = min(1., step / 8000.)
    loss = (first * values[0] + second * values[1] + agreement * paired_risk
            + ramp * (.26 * shared_error + .09 * private_error) + .08 * prior_risk)
    return loss, values

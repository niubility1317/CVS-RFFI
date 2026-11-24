"""Learn the direction of a physically matched LoRa complex correction."""

import torch
from torch import nn

from .channel_blind_source import waveform_risk
from .common_anchor import head, iq_field
from .cross_frequency import CrossFrequencyPrior
from .dura_repair import complex_iq, real_iq
from .joint_spectrum import limit, split
from .joint_unroll_source import coordinates
from .natural_detail import paired_inputs, transfer_risk, weak_difference
from .received_update import ReceivedUpdate
from .split_shrink import spectrum, wave


def inner(first, second, rms):
    return ((first * second.conj()).real * rms.square()).mean((1, 2), keepdim=True)


def match_direction(anchor, proposal, rms):
    anchor_power = inner(anchor, anchor, rms)
    radial = inner(proposal, anchor, rms) / anchor_power.clamp_min(1e-12)
    tangent = proposal - radial * anchor
    tangent_power = inner(tangent, tangent, rms).clamp_min(1e-12)
    tangent = tangent * (2.25 * anchor_power / tangent_power).clamp(1e-12, 1.).sqrt()
    candidate = anchor + tangent
    candidate_power = inner(candidate, candidate, rms).clamp_min(1e-12)
    matched = candidate * (anchor_power / candidate_power).clamp_min(1e-12).sqrt()
    return torch.where(anchor_power > 1e-12, matched, anchor)


class AngularTransport(nn.Module):
    def __init__(self, angular):
        super().__init__()
        self.angular = angular
        self.prior = CrossFrequencyPrior()
        self.coarse = head(15, 2)
        self.fine = head(21, 2)
        frequency = torch.fft.fftshift(torch.fft.fftfreq(256))
        self.register_buffer('frequency', frequency, persistent=False)

    @staticmethod
    def read(net, fields):
        data = torch.stack(fields, dim=-2).flatten(0, 1)
        mixed = net.training and data.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            logits = net(data)
        return logits.float().unflatten(0, fields[0].shape[:2])

    def observe(self, y, quality, rms, *, use_action=True, use_noise=True):
        pooled = self.prior.spectrum(y)
        measured = self.prior.noise(pooled)
        noise = measured if use_noise else torch.full_like(measured, .05)
        action, reference, mismatch = ReceivedUpdate.estimate(y, noise.mean(-1)[:, None, None])
        if not use_action:
            action = torch.ones_like(action)
        anchor = .5 * limit(self.prior.transport(pooled, noise), y, rms, y.real.new_tensor(.30))
        raw = spectrum(y)
        axes = spectrum(anchor)
        reference_spectrum = spectrum(reference)
        closure = spectrum(action.conj() * (y - action * y))
        same_shape = axes.real
        broadcast = (quality, (10. * noise[:, 0, None, None]).log1p(),
                     (10. * noise[:, 1, None, None]).log1p(), (4. * mismatch).log1p(),
                     self.frequency)
        fields = (*iq_field(raw, axes, reference_spectrum, closure), raw.abs().log1p(), axes.abs().log1p(),
                  *(x.expand_as(same_shape) for x in broadcast))
        scale = (axes.abs().square() + .0625 * raw.abs().square()).sqrt()
        uncertainty = .015 + 2. * noise.mean(-1)[:, None, None] + mismatch
        return anchor, axes, scale, fields, action, uncertainty, measured, mismatch

    def stages(self, y, rms, observed, *, angular=None):
        angular = self.angular if angular is None else angular
        anchor, axes, scale, fields, action, _, _, _ = observed

        def readout(logits):
            real, imag = logits.tanh().unbind(-2)
            proposal = torch.complex(real, imag) * scale
            if not angular:
                unit = axes / axes.abs().clamp_min(1e-8)
                proposal = (proposal * unit.conj()).real * unit
            proposal = proposal * (self.frequency != 0.)
            return match_direction(anchor, wave(proposal), rms)

        first_logits = self.read(self.coarse, fields)
        first = readout(first_logits)
        innovation = action.conj() * (y - action * (y + first))
        fine_fields = (*fields, *iq_field(spectrum(first), spectrum(y + first), spectrum(innovation)))
        final_logits = first_logits + self.read(self.fine, fine_fields)
        return first, readout(final_logits)


@torch.no_grad()
def initialize(model, tracker, training, _calibration):
    model.prior.calibrate(coordinates(tracker, batch)[0] for batch in training.split(96))


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    clean, clean_phase, _, clean_rms = coordinates(tracker, iq)
    clean_frame = clean_phase * clean_rms
    source = complex_iq(iq)
    difference = weak_difference(clean)
    weak = difference * clean_frame
    views = paired_inputs(source, recipe.perturb, rng, weak, with_identity=with_identity)
    y, phase, quality, rms = coordinates(tracker, real_iq(views.flatten(0, 1)))
    frame = phase * rms
    target = torch.stack((source, source, source + weak), dim=1).flatten(0, 1) / frame
    observed = model.observe(y, quality, rms)
    coarse, fine = model.stages(y, rms, observed)
    _, _, _, _, action, uncertainty, _, _ = observed
    normalizer = inner(y, y, rms).clamp_min(1e-8)
    coarse_error, fine_error = (inner(error, error, rms) / normalizer
                                for error in (y + coarse - target, y + fine - target))
    closure = ((y - action * (y + fine)).abs().square() / uncertainty).mean((1, 2))
    observed_risk = (closure - 1.).clamp_min(0.).square().mean()
    restored = ((y + fine) * frame).unflatten(0, views.shape[:2])
    stability = ((restored[:, 0] - restored[:, 1]) / clean_frame).abs().square().mean()
    projected = split((restored[:, 2] - restored[:, 0]) / clean_frame)
    expected = split(difference)
    shared = transfer_risk(projected[:, :1], expected[:, :1])
    unique = transfer_risk(projected[:, 1:], expected[:, 1:])
    spectral = waveform_risk(y + fine, target)
    values = torch.stack((coarse_error.log1p().mean(), fine_error.log1p().mean(), spectral,
                          stability, shared, unique, observed_risk))
    first, final, paired = recipe.loss_weights(step)
    return (first * values[0] + final * values[1] + .35 * spectral + paired * stability
            + .20 * shared + .35 * unique + .12 * observed_risk), values

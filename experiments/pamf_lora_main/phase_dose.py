"""Recover LoRa time-domain phase by allocating edits inside the current packet."""

from math import log

import torch
from torch import nn

from .channel_blind_source import waveform_risk
from .common_anchor import head
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .local_phase import project, rotation, weak_phase_difference, window_mean
from .natural_detail import transfer_risk
from .phase_observation_diagnostic import repeat_angle
from .received_update import ReceivedUpdate


class PhaseDose(nn.Module):
    def __init__(self, local):
        super().__init__()
        self.local = local
        self.coarse = head(12, 1)
        self.fine = head(16, 1)
        self.register_buffer('time', torch.linspace(-1., 1., 256), persistent=False)
        freq = torch.fft.fftfreq(256).abs()
        self.register_buffer('outer', freq > .25, persistent=False)
        self.register_buffer('weak_bands', torch.stack(((freq > 0) & (freq <= 1/32),
                                                      (freq > 1/32) & (freq <= 1/8))), persistent=False)

    def read(self, net, fields):
        data = torch.stack(fields, dim=-2).flatten(0, 1)
        mixed = self.training and data.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            output = net(data)
        return output.float().reshape_as(fields[0])

    def distribute(self, amount, weight, local):
        if local:
            return amount
        return (amount * weight).sum((1, 2), keepdim=True) / weight.sum((1, 2), keepdim=True).clamp_min(1e-8)

    def phases(self, y, quality, rms, *, local=None, use_action=True, use_noise=True):
        local = self.local if local is None else local
        amplitude = y.abs()
        raw_power = (amplitude * rms).square()
        shared_rms = rms.square().mean(1, keepdim=True).sqrt()
        shared = y * (rms / shared_rms)
        direction = repeat_angle(shared, quality)
        outer = torch.fft.fft(y, norm='ortho')[..., self.outer].abs().square().median(-1, keepdim=True).values / log(2.)
        noise = outer if use_noise else torch.full_like(outer, .10)
        action, repeat, mismatch = ReceivedUpdate.estimate(y, noise)
        if not use_action:
            action = torch.ones_like(action)
        expected = action * repeat
        closure = rotation((y * expected.conj()).angle())
        coarse_fields = (y.real, y.imag, amplitude.log1p(), quality.expand_as(amplitude), direction / torch.pi,
                         window_mean(direction.abs()) / torch.pi, noise.log1p().expand_as(amplitude),
                         closure.real, closure.imag, (action - 1.).real, action.imag,
                         mismatch.log1p().expand_as(amplitude))
        coarse_logit = self.read(self.coarse, coarse_fields)
        window = torch.ones_like(self.time)
        support = project(direction, raw_power, window, self.time, angle_cap=.65)
        weight = raw_power * support.square()
        coarse_dose = self.distribute((coarse_logit + .5).sigmoid(), weight, local)
        first = project(support * coarse_dose, raw_power, window, self.time, angle_cap=.65)
        observed_innovation = y - action * (y * rotation(first))
        fine_fields = (*coarse_fields, first, observed_innovation.real, observed_innovation.imag,
                       coarse_dose.expand_as(amplitude))
        fine_logit = coarse_logit + self.read(self.fine, fine_fields)
        dose = self.distribute((fine_logit + .5).sigmoid(), weight, local)
        final = project(support * dose, raw_power, window, self.time, angle_cap=.65)
        evidence = torch.stack((outer.mean((1, 2)), mismatch.mean((1, 2)),
                                (action - 1.).abs().mean((1, 2)), dose.expand_as(amplitude).std(
                                    dim=(1, 2), unbiased=False)), -1)
        return first, final, (action, noise, mismatch, evidence)


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    ground = complex_iq(iq)
    clean, source_rotation, _, source_rms = coordinates(tracker, iq)
    weak = weak_phase_difference(ground, clean, model.weak_bands)
    source = torch.stack((ground, ground + weak), dim=1)
    replay = torch.Generator(device=ground.device)
    replay.set_state(rng.get_state())
    views = torch.stack((recipe.perturb(source[:, 0], rng), recipe.perturb(source[:, 1], replay)), dim=1)
    if with_identity:
        views[-2:] = source[-2:]
    y, phasor, quality, rms = coordinates(tracker, real_iq(views.flatten(0, 1)))
    frame = phasor * rms
    target = source.flatten(0, 1) / frame
    first, final, (action, noise, mismatch, _) = model.phases(y, quality, rms)
    restored = y * rotation(final)
    target_phase = (target * y.conj()).angle()
    signal_support = ((target.abs().square() / (.05 + target.abs().square())) *
                      (y.abs().square() / (.05 + y.abs().square()))).sqrt()
    phase_first = ((1. - (target_phase - first).cos()) * signal_support).mean()
    phase_final = ((1. - (target_phase - final).cos()) * signal_support).mean()
    reconstruction = waveform_risk(restored, target)
    observed = (((y - action * restored).abs().square() - (y - action * y).abs().square()) /
                (.15 + 2 * noise + mismatch)).mean()
    pair = (restored * frame).reshape_as(views)
    source_frame = source_rotation * source_rms
    weak_loss = transfer_risk((pair[:, 1] - pair[:, 0]) / source_frame, weak / source_frame)
    a, b, _ = recipe.loss_weights(step)
    loss = a * phase_first + b * phase_final + .16 * reconstruction + .035 * observed + .18 * weak_loss
    return loss, torch.stack((phase_first, phase_final, reconstruction, observed, weak_loss))

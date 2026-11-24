"""Build a LoRa phase direction from samples within the same received chirp."""

from math import log

import torch
from torch.nn import functional as F

from .common_anchor import head
from .local_phase import project, rotation, window_mean
from .phase_dose import PhaseDose
from .phase_observation_diagnostic import repeat_angle
from .received_update import ReceivedUpdate


def neighbor_angle(y):
    """Predict center phase from two disjoint symmetric sets, without wraparound."""
    center = y[..., 4:-4]
    near = .5 * (y[..., 3:-5] + y[..., 5:-3])
    farther = (y[..., 2:-6] + y[..., 6:-2] + .5 * (y[..., :-8] + y[..., 8:])) / 3.
    unit_near = near / near.abs().clamp_min(1e-7)
    unit_farther = farther / farther.abs().clamp_min(1e-7)
    agreement = (unit_near * unit_farther.conj()).real.clamp(0., 1.)
    support = agreement * torch.minimum(near.abs(), farther.abs()).clamp_max(1.)
    direction = ((unit_near + unit_farther) * center.conj()).angle()
    return F.pad(direction * support, (4, 4)), F.pad(support, (4, 4))


def remove_parallel(phase, reference, power):
    cross = (power * phase * reference).sum(-1, keepdim=True)
    energy = (power * reference.square()).sum(-1, keepdim=True)
    return phase - (cross / energy.clamp_min(1e-10)) * reference


class NeighborPhaseDose(PhaseDose):
    def __init__(self, use_neighbor):
        super().__init__(local=True)
        self.use_neighbor = use_neighbor
        self.coarse = head(15, 2)
        self.fine = head(19, 2)
        self.register_buffer('valid_neighbors', F.pad(torch.ones(248), (4, 4)), persistent=False)

    def read_pair(self, net, fields):
        data = torch.stack(fields, dim=-2).flatten(0, 1)
        mixed = self.training and data.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            result = net(data)
        return result.float().unflatten(0, fields[0].shape[:2]).unbind(dim=-2)

    def phases(self, y, quality, rms, *, use_neighbor=None, use_action=True, use_noise=True):
        enabled = self.use_neighbor if use_neighbor is None else use_neighbor
        amplitude = y.abs()
        raw_power = (amplitude * rms).square()
        shared_rms = rms.square().mean(1, keepdim=True).sqrt()
        across = repeat_angle(y * (rms / shared_rms), quality)
        adjacent, agreement = neighbor_angle(y)
        outer = torch.fft.fft(y, norm='ortho')[..., self.outer].abs().square().median(-1, keepdim=True).values / log(2.)
        noise = outer if use_noise else torch.full_like(outer, .10)
        action, repeat, mismatch = ReceivedUpdate.estimate(y, noise)
        if not use_action:
            action = torch.ones_like(action)
        closure = rotation((y * (action * repeat).conj()).angle())
        all_points = torch.ones_like(self.time)

        def bounded(phase, window):
            return project(phase, raw_power, window, self.time, angle_cap=.65)

        repeat_direction = bounded(across, all_points)
        repeat_inside = bounded(repeat_direction, self.valid_neighbors)
        adjacent_direction = bounded(adjacent, self.valid_neighbors)
        fields = (y.real, y.imag, amplitude.log1p(), quality.expand_as(amplitude), across / torch.pi,
                  window_mean(across.abs()) / torch.pi, noise.log1p().expand_as(amplitude),
                  closure.real, closure.imag, (action - 1.).real, action.imag,
                  mismatch.log1p().expand_as(amplitude), adjacent / torch.pi, agreement,
                  window_mean(adjacent.abs()) / torch.pi)

        def update(repeat_logit, neighbor_logit):
            dose = (repeat_logit + .5).sigmoid()
            repeated = repeat_direction * dose
            additional = torch.zeros_like(repeated)
            if enabled:
                candidate = bounded(adjacent_direction * neighbor_logit.tanh(), self.valid_neighbors)
                additional = remove_parallel(candidate, repeat_inside, raw_power)
            return bounded(repeated + additional, all_points), dose, additional

        a, b = self.read_pair(self.coarse, fields)
        first, dose, _ = update(a, b)
        innovation = y - action * (y * rotation(first))
        c, d = self.read_pair(self.fine, (*fields, first, innovation.real, innovation.imag, dose))
        final, dose, additional = update(a + c, b + d)
        neighbor_rms = ((raw_power * additional.square()).sum((1, 2)) /
                        raw_power.sum((1, 2)).clamp_min(1e-10)).sqrt()
        evidence = torch.stack((outer.mean((1, 2)), mismatch.mean((1, 2)),
                                (action - 1.).abs().mean((1, 2)), agreement.mean((1, 2)), neighbor_rms), -1)
        return first, final, (action, noise, mismatch, evidence)

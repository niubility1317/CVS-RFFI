"""A constant complex phase correction estimated from a split WiSig STF."""

import torch
from torch import nn

from experiments.pamf_r65_four_route.stage_polyphase import StageDistilled
from experiments.pamf_restoration.models import complex_iq, pair
from .receiver_pilot import BEGIN, PERIOD, estimate, phase
from .receiver_timing import NOMINAL, preamble_template


QUANTILE = .999
MIN_CORRELATION = .50
MIN_AGREEMENT = .80


class ShortFieldPhase(nn.Module):
    def __init__(self):
        super().__init__()
        index = torch.arange(3 * PERIOD) + BEGIN
        delays = torch.arange(PERIOD)
        template = preamble_template((index[None] - NOMINAL - delays[:, None].float()) % PERIOD - PERIOD)
        self.register_buffer('template', template / template.norm(dim=-1, keepdim=True),
                             persistent=False)

    def measure(self, received):
        # CFO is removed for the matched filter only, at the original time gauge.
        observed = estimate(received)
        parts = observed.aligned[:, BEGIN:BEGIN + 6 * PERIOD].unflatten(-1, (2, 3 * PERIOD))
        z = parts @ self.template.mH
        norms = parts.norm(dim=-1).clamp_min(1e-8)
        t = (z / norms[..., None]).sum(dim=1).abs().argmax(dim=-1)
        c = z.gather(-1, t[:, None, None].expand(-1, 2, 1)).squeeze(-1)
        fit = (c.abs() / norms).amin(dim=-1)
        agreement = (c[:, 0] * c[:, 1].conj()).real / c.abs().prod(dim=1).clamp_min(1e-8)
        return c.sum(dim=1).angle(), fit, agreement, t

    def rotation(self, received, center, limit):
        angle, fit, agreement, lag = self.measure(received)
        error = phase(angle - center).angle()
        trustworthy = (fit >= MIN_CORRELATION) & (agreement >= MIN_AGREEMENT)
        active = trustworthy & (error.abs() > limit)
        return phase(-error), active, dict(phase_delta_rad=error, phase_fit=fit,
                                            phase_agreement=agreement, phase_lag=lag.float(),
                                            phase_active=active.float())


class ScratchUnroll(nn.Module):
    """The two-stage architecture is constructed empty. No parent weights are loaded."""
    def __init__(self, *, canonical, center, limit):
        super().__init__()
        self.solver = StageDistilled()
        self.gauge = ShortFieldPhase()
        self.canonical = canonical
        self.center = center
        self.limit = limit

    @property
    def coarse(self):
        return self.solver.frozen.coarse.temporal

    @property
    def fine(self):
        return self.solver.prior[-1]

    def forward(self, received):
        if self.canonical:
            rotation, active, _ = self.gauge.rotation(received, self.center, self.limit)
            received = torch.where(active[:, None], received * rotation[:, None], received)
        stages, variance, _, base = self.solver.stage_outputs(pair(received))
        return complex_iq(base), complex_iq(stages[-1]), dict(map_variance=variance[:, 0])

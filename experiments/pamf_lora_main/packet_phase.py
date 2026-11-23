"""Share quadratic phase across a packet while keeping independent linear slopes."""

import torch
from torch import nn

from .dura_repair import PacketRestorer


class PacketPhaseTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = PacketRestorer('consensus')
        time = torch.linspace(-1., 1., 256)
        self.register_buffer('time', time, persistent=False)
        self.register_buffer('quadratic', time.square() - time.square().mean(), persistent=False)
        self.register_buffer('grid', torch.linspace(-.6, .6, 9), persistent=False)

    def curvature(self, wave, quality):
        """Profile out independent complex amplitudes, then require split agreement."""
        templates = torch.polar(torch.ones_like(self.grid[:, None] * self.quadratic),
                                -self.grid[:, None] * self.quadratic)
        power = wave.abs().square().mean(-1).clamp_min(1e-10)
        coherent = (wave @ templates.T / wave.shape[-1]).abs().square() / power[..., None]
        weight = quality.square().unsqueeze(-1)

        def group_fit(scores, weights):
            values = (scores * weights).sum(1)
            best = values.argmax(-1, keepdim=True)
            left = values.gather(-1, (best - 1).clamp_min(0))
            center = values.gather(-1, best)
            right = values.gather(-1, (best + 1).clamp_max(len(self.grid) - 1))
            second = left - 2 * center + right
            eligible = (best > 0) & (best < len(self.grid) - 1) & (second < -1e-10)
            offset = .5 * (left - right) / second.clamp_max(-1e-10)
            fitted = self.grid[best] + (self.grid[1] - self.grid[0]) * torch.where(
                eligible, offset.clamp(-.5, .5), 0.)
            return torch.where(weights.sum(1) > 0, fitted, 0.)

        a = group_fit(coherent[:, ::2], weight[:, ::2])
        b = group_fit(coherent[:, 1::2], weight[:, 1::2])
        return torch.where(a * b > 0, a.sign() * torch.minimum(a.abs(), b.abs()), 0.)

    def fine_slope(self, wave, coherence):
        """Refine the local concentrated-likelihood peak inside one bounded interval."""
        delta = torch.zeros_like(wave[..., :1].real)
        current = wave
        for _ in range(2):
            tone = current.mean(-1, keepdim=True)
            first = (-1j * self.time * current).mean(-1, keepdim=True)
            second = (-self.time.square() * current).mean(-1, keepdim=True)
            gradient = (tone.conj() * first).real
            curvature = first.abs().square() + (tone.conj() * second).real
            proposal = (delta - gradient / curvature.clamp_max(-1e-10)).clamp(-.4, .4)
            angle = -proposal * self.time
            test = wave * torch.polar(torch.ones_like(angle), angle)
            improves = test.mean(-1, keepdim=True).abs() >= tone.abs()
            accept = improves & (curvature < -1e-10) & (coherence[..., None] >= .2)
            delta = torch.where(accept, proposal, delta)
            current = torch.where(accept, test, current)
        return delta

    @torch.no_grad()
    def refine(self, frame, curvature=None):
        wave, orientation, coherence, _, gate = frame
        if curvature is not None:
            phase = curvature[:, None] * self.quadratic
            rotation = torch.polar(torch.ones_like(phase), phase)
            wave, orientation = wave * rotation.conj(), orientation * rotation
        delta = self.fine_slope(wave, coherence)
        angle = delta * self.time
        linear_rotation = torch.polar(torch.ones_like(angle), angle)
        wave, orientation = wave * linear_rotation.conj(), orientation * linear_rotation
        dc = wave.mean(-1, keepdim=True)
        phase = torch.where(dc.abs() > 1e-8, dc / dc.abs().clamp_min(1e-8), torch.ones_like(dc))
        wave, orientation = wave * phase.conj(), orientation * phase
        quality = (dc.abs().square().squeeze(-1) / wave.abs().square().mean(-1).clamp_min(1e-8)).clamp(0., 1.)
        return wave, orientation, quality, quality.median(1).values, gate

    @torch.no_grad()
    def track(self, received):
        initial = self.base.track(received)
        fitted = self.curvature(initial[0], initial[2])
        return self.refine(initial, fitted)


def frame_inputs(frame):
    wave, orientation, q, _, _ = frame
    rms = wave.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    return wave / rms, orientation, q.unsqueeze(-1), rms

"""Refine the common complex spectrum without editing chirp contrasts."""

import torch
from torch import nn

from .w24_packet_unet import CENTER, PacketUNet


def common_band_risk(output, target, observed, rms):
    """Separate near/far common residuals at the physical receive scale."""
    error, disturbance = (
        torch.fft.fft(((z - target) * rms).mean(-2), norm='ortho').abs().square()
        for z in (output, observed)
    )
    freq = (torch.fft.fftfreq(CENTER, device=output.device) * CENTER).round().abs()
    near = freq <= 32
    floor = .001 * rms.square().mean((-2, -1))
    return torch.stack([(error[..., band].mean(-1) /
                         (disturbance[..., band].mean(-1) + floor).clamp_min(1e-12)).log1p().mean()
                        for band in (near, ~near)])


class CommonRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = PacketUNet(across_chirps=True).requires_grad_(False)
        self.head = nn.Sequential(
            nn.Conv1d(10, 48, 7, padding=3, padding_mode='circular'), nn.GroupNorm(8, 48), nn.SiLU(),
            nn.Conv1d(48, 48, 9, padding=8, dilation=2, padding_mode='circular'),
            nn.GroupNorm(8, 48), nn.SiLU(),
            nn.Conv1d(48, 48, 7, padding=3, padding_mode='circular'), nn.GroupNorm(8, 48), nn.SiLU(),
            nn.Conv1d(48, 2, 5, padding=2, padding_mode='circular'),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        frequency = torch.fft.fftfreq(CENTER)
        self.register_buffer('frequency', 2. * frequency, persistent=False)
        self.register_buffer('noise_band', frequency.abs() >= 1 / 3, persistent=False)

    def forward(self, y, quality, rms, edges):
        time, full = self.base(y, quality, rms, edges)
        # Normalize by packet RMS so the edit remains a common physical
        # correction across chirps with unequal observed RMS.
        packet_rms = rms.square().mean(-2, keepdim=True).sqrt().clamp_min(1e-8)
        scale = rms / packet_rms
        receive, preliminary = (
            torch.fft.fft(view * scale, norm='ortho') for view in (y, torch.lerp(y, full, .5))
        )
        coherent = receive.mean(-2)
        current = preliminary.mean(-2)
        quality_weight = .05 + quality.square()
        weighted = (quality_weight * receive).sum(-2) / quality_weight.sum(-2)
        dispersion = (receive - coherent[:, None]).abs().square().mean(-2)
        variance = (receive[..., self.noise_band].abs().square().flatten(1).median(-1).values
                    / .69314718056).clamp_min(0.)
        features = torch.stack((
            coherent.real, coherent.imag, current.real, current.imag, weighted.real, weighted.imag,
            dispersion.log1p(), quality.mean(-2).expand_as(coherent.real),
            self.frequency.expand_as(coherent.real), variance[:, None].log1p().expand_as(coherent.real),
        ), dim=1)
        edit = self.head(features).float()
        delta = torch.fft.ifft(torch.complex(*edit.unbind(1)), norm='ortho')
        budget = delta.real.new_tensor(.22)
        strength = variance[:, None] / (.008 + variance[:, None])
        bound = budget / (budget.square() + delta.abs().square().mean(-1, keepdim=True)).sqrt()
        change = (strength * bound * delta).unsqueeze(-2) / scale.clamp_min(1e-8)
        return time, full + 2. * change

    def propose(self, y, quality, rms, edges):
        return self(y, quality, rms, edges),

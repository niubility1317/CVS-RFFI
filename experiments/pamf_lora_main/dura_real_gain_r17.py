"""Fuse a real-gain prior with the original R17 and its physical residual."""

import torch
from torch import nn


class R17Gain(nn.Module):
    def __init__(self, width):
        super().__init__()
        if width not in (32, 1024):
            raise ValueError('Expected registered WiSig or LoRa group width')
        self.width = width
        self.net = nn.Sequential(
            nn.Conv1d(8, 32, 7, padding=3, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(32, 32, 9, padding=12, dilation=3, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(32, 2, 5, padding=2, padding_mode='circular'),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('edit_band', torch.fft.fftfreq(width).abs() > 0, persistent=False)

    def forward(self, received, prior, r17, action):
        if (received.ndim != 2 or received.shape != prior.shape or received.shape != r17.shape
                or received.shape[-1] != 8 * self.width):
            raise ValueError('Matching full physical packets required')
        backprojection = action.adjoint(received - action(prior))
        y, p, r, d = (torch.fft.fft(z.unflatten(-1, (8, self.width)), norm='ortho')
                      for z in (received, prior, r17, backprojection))
        scale = y.abs().square().mean((-1, -2), keepdim=True).clamp_min(1e-6)
        local = tuple(z.abs().square() / scale for z in (y, p, r, d))
        shared = tuple(z.mean(1, keepdim=True).expand_as(z) for z in local)
        features = torch.stack((*local, *shared), dim=-2).log1p()
        gains = self.net(features.flatten(0, 1)).unflatten(0, features.shape[:2]).tanh()
        delta = .25 * self.edit_band * (gains[..., 0, :] * d + gains[..., 1, :] * (r - p))
        return prior + torch.fft.ifft(delta, norm='ortho').flatten(-2)

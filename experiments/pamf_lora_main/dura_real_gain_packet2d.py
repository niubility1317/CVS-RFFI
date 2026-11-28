"""Packet-context real spectral gains for the R17 physical directions."""

import torch
from torch import nn
from torch.nn import functional as F


class PacketR17Gain(nn.Module):
    def __init__(self, width):
        super().__init__()
        if width not in (32, 1024):
            raise ValueError('Expected registered WiSig or LoRa group width')
        self.width = width
        self.net = nn.Sequential(
            nn.Conv2d(10, 64, (3, 7), padding=(1, 3)), nn.SiLU(),
            nn.Conv2d(64, 64, (3, 5), padding=(1, 2)), nn.SiLU(),
            nn.Conv2d(64, 64, (3, 5), padding=(1, 2)), nn.SiLU(),
            nn.Conv2d(64, 3, (3, 5), padding=(1, 2)),
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
        magnitude = torch.stack((*local, *shared), dim=1).log1p()
        cross = p[:, 1:] * p[:, :-1].conj()
        coherence = cross / (p[:, 1:].abs() * p[:, :-1].abs()).clamp_min(1e-6)
        coherence = F.pad(coherence, (0, 0, 1, 0))
        features = torch.cat((magnitude, coherence.real[:, None], coherence.imag[:, None]), dim=1)
        gains = self.net(features).tanh()
        delta = self.edit_band * (.1 * gains[:, 0] * p + .25 * gains[:, 1] * d
                                  + .25 * gains[:, 2] * (r - p))
        return prior + torch.fft.ifft(delta, norm='ortho').flatten(-2)

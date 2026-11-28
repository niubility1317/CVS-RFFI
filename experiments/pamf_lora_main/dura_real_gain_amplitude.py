"""Real-valued spectral correction of the prior and two DURA directions."""

import torch
from torch import nn

from experiments.pamf_lora_main.dura_real_gain_r17 import R17Gain


class R17AmplitudeGain(R17Gain):
    def __init__(self, width):
        super().__init__(width)
        self.net[-1] = nn.Conv1d(32, 3, 5, padding=2, padding_mode='circular')
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

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
        delta = self.edit_band * (.1 * gains[..., 0, :] * p + .25 * gains[..., 1, :] * d
                                  + .25 * gains[..., 2, :] * (r - p))
        return prior + torch.fft.ifft(delta, norm='ortho').flatten(-2)

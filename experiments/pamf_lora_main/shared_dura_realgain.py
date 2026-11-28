"""One DURA residual and real-frequency-gain rule for both registered packet shapes."""
import torch
from torch import nn


class SharedDuraRealGain(nn.Module):
    """Correct frozen R17 with two real gains, using only the current packet."""

    def __init__(self, width):
        super().__init__()
        if width not in (32, 1024):
            raise ValueError('Expected eight groups of 32 or 1024 samples')
        self.width = width
        self.net = nn.Sequential(
            nn.Conv1d(8, 32, 7, padding=3, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(32, 32, 9, dilation=3, padding=12, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(32, 2, 5, padding=2, padding_mode='circular'),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('edit_band', torch.fft.fftfreq(width).abs() > 0, persistent=False)

    def forward(self, received, prior, action):
        if (received.ndim != 2 or received.shape != prior.shape or
                received.shape[-1] != 8 * self.width or not received.is_complex()):
            raise ValueError('Expected matching complex packets of the registered length')
        backprojected = action.adjoint(received - action(prior))
        y, p, residual = (torch.fft.fft(z.unflatten(-1, (8, self.width)), norm='ortho')
                          for z in (received, prior, backprojected))
        scale = y.abs().square().mean((-1, -2), keepdim=True).clamp_min(1e-6)
        local = tuple(z.abs().square() / scale for z in (y, p, residual))
        shared = tuple(z.mean(1, keepdim=True).expand_as(z) for z in local)
        features = torch.stack((*local, *shared), dim=-2).log1p()
        cross = p * y.conj() / (scale * ((1 + local[0]) * (1 + local[1])).sqrt())
        features = torch.cat((features, torch.stack((cross.real, cross.imag), dim=-2)), dim=-2)
        gain = self.net(features.flatten(0, 1)).unflatten(0, features.shape[:2]).tanh()
        correction = self.edit_band * (gain[..., 0, :] * (y - p) + .25 * gain[..., 1, :] * residual)
        return prior + torch.fft.ifft(correction, norm='ortho').flatten(-2)


class SharedTwoPriorGain(nn.Module):
    """Use real gains to combine real-gain and R17 estimates with physical residual."""

    def __init__(self, width):
        super().__init__()
        if width not in (32, 1024):
            raise ValueError('Expected eight groups of 32 or 1024 samples')
        self.width = width
        self.net = nn.Sequential(
            nn.Conv1d(8, 32, 7, padding=3, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(32, 32, 9, dilation=3, padding=12, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(32, 2, 5, padding=2, padding_mode='circular'),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('edit_band', torch.fft.fftfreq(width).abs() > 0, persistent=False)

    def forward(self, received, real_gain, r17, action):
        if (received.ndim != 2 or received.shape != real_gain.shape or received.shape != r17.shape or
                received.shape[-1] != 8 * self.width or not received.is_complex()):
            raise ValueError('Expected three matching complex packets of the registered length')
        backprojected = action.adjoint(received - action(real_gain))
        y, a, b, residual = (torch.fft.fft(z.unflatten(-1, (8, self.width)), norm='ortho')
                             for z in (received, real_gain, r17, backprojected))
        scale = y.abs().square().mean((-1, -2), keepdim=True).clamp_min(1e-6)
        local = tuple(z.abs().square() / scale for z in (y, a, b, residual))
        shared = tuple(z.mean(1, keepdim=True).expand_as(z) for z in local)
        features = torch.stack((*local, *shared), dim=-2).log1p()
        gain = self.net(features.flatten(0, 1)).unflatten(0, features.shape[:2]).tanh()
        correction = self.edit_band * (gain[..., 0, :] * (b - a) + .25 * gain[..., 1, :] * residual)
        return real_gain + torch.fft.ifft(correction, norm='ortho').flatten(-2)


class SharedConvexGain(SharedTwoPriorGain):
    """Use a bounded real frequency weight to interpolate two frozen estimates."""

    def __init__(self, width):
        super().__init__(width)
        self.net[-1] = nn.Conv1d(32, 1, 5, padding=2, padding_mode='circular')
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def initial_reference(self, real_gain, r17):
        difference = torch.fft.fft((r17-real_gain).unflatten(-1, (8, self.width)), norm='ortho')
        return real_gain + torch.fft.ifft(.5*self.edit_band*difference, norm='ortho').flatten(-2)

    def forward(self, received, real_gain, r17, action):
        if (received.ndim != 2 or received.shape != real_gain.shape or received.shape != r17.shape or
                received.shape[-1] != 8*self.width or not received.is_complex()):
            raise ValueError('Expected three matching complex packets of the registered length')
        backprojected = action.adjoint(received - action(real_gain))
        y, a, b, residual = (torch.fft.fft(z.unflatten(-1, (8, self.width)), norm='ortho')
                             for z in (received, real_gain, r17, backprojected))
        scale = y.abs().square().mean((-1, -2), keepdim=True).clamp_min(1e-6)
        local = tuple(z.abs().square()/scale for z in (y, a, b, residual))
        shared = tuple(z.mean(1, keepdim=True).expand_as(z) for z in local)
        features = torch.stack((*local, *shared), dim=-2).log1p()
        weight = self.net(features.flatten(0, 1)).unflatten(0, features.shape[:2]).squeeze(-2).sigmoid()
        return real_gain + torch.fft.ifft(self.edit_band*weight*(b-a), norm='ortho').flatten(-2)

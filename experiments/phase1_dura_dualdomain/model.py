"""Phase-equivariant time/frequency correction anchored to frozen DURA."""
import torch
from torch import nn
from torch.nn import functional as F


class DualDomainRecovery(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_entry = nn.Conv1d(5, 48, 7, padding=3, padding_mode='reflect')
        self.freq_entry = nn.Conv1d(5, 48, 7, padding=3, padding_mode='circular')
        self.time_blocks = nn.ModuleList([
            nn.Conv1d(48, 48, 5, padding=d*2, dilation=d, padding_mode='reflect')
            for d in (1, 2, 4)])
        self.freq_blocks = nn.ModuleList([
            nn.Conv1d(48, 48, 5, padding=d*2, dilation=d, padding_mode='circular')
            for d in (1, 2, 4)])
        self.gate_output = nn.Conv1d(48, 4, 1)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    @property
    def entry(self):
        return self.time_entry

    def _correction(self, z, a, noise, entry, blocks):
        cross = z * a.conj()
        features = torch.stack((z.abs(), a.abs(), cross.real, cross.imag), dim=1)
        h = F.silu(entry(torch.cat((features, noise.expand(-1, 1, 256)), dim=1)))
        for block in blocks:
            h = h + 0.2 * F.silu(block(h))
        w = self.gate_output(h).tanh()
        return torch.complex(w[:, 0], w[:, 1]) * a + torch.complex(w[:, 2], w[:, 3]) * z

    def forward(self, received, initial, log_variance):
        if (received.ndim != 3 or received.shape[1:] != (2, 256)
                or initial.shape != received.shape or log_variance.shape != (received.shape[0],)):
            raise ValueError('Matching IQ windows and one blind log variance are required')
        if not all(torch.isfinite(t).all() for t in (received, initial, log_variance)):
            raise FloatingPointError('Finite received evidence is required')
        power = received.square().mean((1, 2), keepdim=True).clamp_min(1e-8)
        scale = power.sqrt()
        z = torch.complex(received[:, 0], received[:, 1]) / scale[:, 0]
        a = torch.complex(initial[:, 0], initial[:, 1]) / scale[:, 0]
        noise = ((log_variance - power[:, 0, 0].log()).clamp(-16, 8)/8)[:, None, None]
        time = self._correction(z, a, noise, self.time_entry, self.time_blocks)
        zf, af = torch.fft.fft(z, norm='ortho'), torch.fft.fft(a, norm='ortho')
        freq = self._correction(zf, af, noise, self.freq_entry, self.freq_blocks)
        correction = 0.1 * (time + torch.fft.ifft(freq, norm='ortho'))
        return initial + torch.stack((correction.real, correction.imag), dim=1) * scale

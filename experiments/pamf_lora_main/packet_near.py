"""Phase-aware cross-chirp spectral context for LoRa DURA."""

import torch
from torch import nn

from experiments.pamf_restoration.models import zero_output

from .complex_vector import fft
from .denoise_components import bands
from .near_spectrum import NearSpectrum


def phase_transfer(z):
    # Rows are reference chirps and columns are companions transported to them.
    cross = z @ z.conj().transpose(-1, -2)
    energy = z.abs().square().sum(-1).clamp_min(1e-8).sqrt()
    confidence = (cross.abs()/(energy[..., :, None]*energy[..., None, :])).clamp(0., 1.)
    magnitude = cross.abs()
    unit = torch.where(magnitude > 1e-8, cross/magnitude.clamp_min(1e-8), 1.)
    return unit, confidence


class PacketContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(8, 24), nn.SiLU(), nn.LayerNorm(24))
        self.key_query = nn.Linear(24, 32)
        self.value = nn.Linear(24, 2)
        self.out = zero_output(nn.Linear(6, 2))

    def forward(self, observed, current, quality, *, packet):
        a, b = bands(observed.shape[-1], observed.device)
        select = a | b
        y, z = (fft(x)[..., select] for x in (observed, current))
        count = z.shape[1]
        time = torch.arange(count, device=z.device, dtype=z.real.dtype)
        position = (time/(max(count-1, 1))*2.-1.)[None, :, None].expand_as(z.real)
        freq = 8.*torch.fft.fftfreq(current.shape[-1], device=z.device)[select]
        scope = freq[None, None].expand_as(z.real)
        q = quality.expand_as(z.real)
        feat = torch.stack((y.real, y.imag, z.real, z.imag, (y-z).abs().log1p(), q,
                            position, scope), dim=-1).transpose(1, 2).contiguous()
        token = self.token(feat)
        keys, queries = self.key_query(token).chunk(2, dim=-1)
        logits = (queries @ keys.transpose(-1, -2)) * .25
        align, reliability = phase_transfer(z)
        distance = (time[:, None]-time[None]).abs()
        log_prior = (reliability+.05).log() + .25*(quality[..., 0]+.05).log()[:, None, :] - .13*distance
        logits = logits+log_prior[:, None]
        if not packet:
            logits = logits.masked_fill(~torch.eye(count, dtype=torch.bool, device=z.device), -torch.inf)
        values = self.value(token)
        values = z.transpose(1, 2) + .1*torch.complex(values[..., 0], values[..., 1])
        transported = align[:, None] * values[:, :, None, :]
        context = (logits.softmax(-1)*transported).sum(-1).transpose(1, 2)
        residual = context-z
        fields = torch.stack((context.real, context.imag, z.real, z.imag, residual.real, residual.imag), dim=-1)
        proposal = self.out(fields)
        result = torch.zeros_like(observed)
        result[..., select] = torch.complex(proposal[..., 0], proposal[..., 1])
        return result


class PacketNear(NearSpectrum):
    """Add spectral attention over the same local proposal in both arms.

    The inherited proposal keeps its complementary chirp mean. The packet
    switch controls off-diagonal attention only.
    """

    def __init__(self, *, packet):
        super().__init__()
        self.packet = packet
        self.context = PacketContext()

    def forward(self, observed, baseline, quality, frozen_update):
        current = baseline+torch.fft.ifft(frozen_update, norm='ortho')
        local = self.proposal(observed, current, quality)
        cross = self.context(observed, current, quality, packet=self.packet)
        return self.reorient(frozen_update, .3*local+.3*cross)

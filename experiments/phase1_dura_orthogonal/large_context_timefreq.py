"""Joint time-frequency attention for source-only DURA refinement."""
import torch
from torch import nn

from experiments.phase1_dura_orthogonal.large_context_stable import StableLargeContextDura
from experiments.phase1_dura_orthogonal.large_context_dura import SCALE
from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate

FREQUENCY_TOKENS = 64
TIME_FREQUENCY_KIND = 'complex_fft_joint_attention_v1'


class TimeFrequencyContextDura(StableLargeContextDura):
    """Exchange DURA context between time and frequency representations."""
    def __init__(self, width=384, layers=8):
        super().__init__(width=width, layers=layers)
        self.frequency_local = nn.Sequential(
            nn.Conv1d(7, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.GELU())
        self.frequency_down = nn.Conv1d(64, width, 9, stride=4, padding=4)
        self.frequency_position = nn.Parameter(torch.zeros(1, FREQUENCY_TOKENS, width))

    def forward(self, received, anchor, innovation, variance, *, validate=True):
        if (received.ndim != 3 or received.shape[1:] != (2, 256) or anchor.shape != received.shape
                or innovation.shape != received.shape or variance.shape != (len(received),)):
            raise ValueError('Matching IQ, DURA context and variance required')
        if validate and (not all(torch.isfinite(v).all() for v in (received, anchor, innovation, variance))
                         or (variance <= 0).any()):
            raise ValueError('Finite IQ and positive variance required')
        power = anchor.square().mean((1, 2), keepdim=True).clamp_min(1e-12)
        scale = power.sqrt()
        signals = (received/scale, anchor/scale, innovation/scale)
        noise = torch.log1p(variance[:, None, None].clamp_min(0.)/power).expand(-1, 1, 256)
        local = self.local(torch.cat((*signals, noise), 1))
        spectra = []
        for x in signals:
            spectrum = torch.fft.fft(torch.complex(x[:, 0], x[:, 1]), dim=-1, norm='ortho')
            spectra.append(torch.stack((spectrum.real, spectrum.imag), 1))
        freq = self.frequency_local(torch.cat((*spectra, noise), 1))
        time_tokens = self.down(local).transpose(1, 2) + self.position
        freq_tokens = self.frequency_down(freq).transpose(1, 2) + self.frequency_position
        context = self.global_context(torch.cat((time_tokens, freq_tokens), 1))
        context = context[:, :time_tokens.shape[1]].transpose(1, 2).repeat_interleave(4, dim=-1)
        delta = self.out(self.decode(torch.cat((local, context), 1)))
        delta = delta * torch.rsqrt(1 + delta.square().sum(1, keepdim=True)) * scale * SCALE
        return anchor + noise_gate(anchor, variance) * delta

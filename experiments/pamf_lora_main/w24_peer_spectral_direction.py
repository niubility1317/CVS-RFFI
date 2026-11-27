"""Native spectral correction with a matched within-chirp and peer readout."""

import torch
from torch import nn

from .w24_packet_dura import CHIRP, CHIRPS, PACKET, chirp_reference, receive_coordinates


class SpectralContext(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv1d(width, width, 5, dilation=dilation, padding=2*dilation,
                      groups=width, padding_mode='circular'),
            nn.Conv1d(width, width, 1), nn.GroupNorm(8, width), nn.SiLU(),
        )

    def forward(self, x):
        return x + self.context(x)


class PeerSpectralDirection(nn.Module):
    width = 48

    def __init__(self):
        super().__init__()
        width = self.width
        self.embed = nn.Sequential(nn.Conv1d(8, width, 1), nn.SiLU(),
                                   *(SpectralContext(width, d) for d in (1, 4, 16, 64)))
        self.local = nn.Linear(width, 2)
        self.peer_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.peer = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 2))
        for output in (self.local, self.peer[-1]):
            nn.init.normal_(output.weight, std=.002)
            nn.init.zeros_(output.bias)
        self.register_buffer('chirp', chirp_reference(), persistent=False)
        self.register_buffer('self_mask', torch.eye(CHIRPS, dtype=torch.bool), persistent=False)
        frequency = torch.fft.fftfreq(CHIRP)
        self.register_buffer('frequency', 2 * frequency, persistent=False)
        self.register_buffer('noise_band', frequency.abs() >= .35, persistent=False)

    def local_coefficients(self, tokens):
        return self.local(tokens)

    def peer_context(self, tokens):
        b, chirps, tones, width = tokens.shape
        sequence = self.peer_norm(tokens).transpose(1, 2).reshape(b*tones, chirps, width)
        # PyTorch 2.1 converts the boolean mask before autocast projects Q/K/V.
        # Keep this masked kernel in FP32 so its mask matches those projections.
        with torch.autocast('cuda', enabled=False):
            sequence = sequence.float()
            context = self.attn(sequence, sequence, sequence, attn_mask=self.self_mask,
                                need_weights=False)[0]
        return context.reshape(b, tones, chirps, width).transpose(1, 2)

    def forward(self, observed, base):
        if observed.ndim != 2 or observed.shape[-1] != PACKET or not observed.is_complex() or base.shape != observed.shape:
            raise ValueError('Expected matching native complex LoRa observations and edits')
        received, _, frame, quality = receive_coordinates(observed, self.chirp)
        aligned_base = base.unflatten(-1, (CHIRPS, CHIRP)) / frame
        y, e = torch.fft.fft(received, norm='ortho'), torch.fft.fft(aligned_base, norm='ortho')
        power = y.abs().square()
        noise = (power[..., self.noise_band].median(-1, keepdim=True).values / .69314718056
                 ).clamp(1e-6, 4.).log() / 6.
        values = (y.real, y.imag, e.real, e.imag, power.log1p(),
                  noise.expand_as(power), quality.expand_as(power), self.frequency.expand_as(power))
        mixed = self.training and y.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            embedded = self.embed(torch.stack(values, dim=-2).flatten(0, 1))
            tokens = embedded.unflatten(0, (-1, CHIRPS)).transpose(-1, -2)
            local = self.local_coefficients(tokens)
            peer = self.peer(torch.cat((tokens, self.peer_context(tokens)), dim=-1))
        dose = aligned_base.abs().square().mean((-2, -1), keepdim=True).sqrt()

        def restore(coefficients):
            correction = torch.view_as_complex(coefficients.float().contiguous())
            return base + (dose * frame * torch.fft.ifft(correction, norm='ortho')).flatten(-2)

        return restore(local), restore(local + peer)

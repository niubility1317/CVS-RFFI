"""A single-pass, position-dependent inverse filter for a complete IQ packet."""

import torch
from torch import nn
from torch.nn import functional as F


def repeat_context(iq):
    """Use four observed 20-point periods as context, in the original IQ gauge."""
    x = torch.complex(iq[:, 0, 80:160], iq[:, 1, 80:160]).reshape(-1, 4, 20)
    left, right = x[:, :-1], x[:, 1:]
    cross = (left.conj() * right).sum((1, 2))
    magnitude = left.abs().square().sum((1, 2)) * right.abs().square().sum((1, 2))
    coherence = cross / magnitude.clamp_min(1e-20).sqrt()
    nonzero = torch.where(cross.abs() > 0, cross, torch.ones_like(cross))
    phase = torch.angle(nonzero)
    steps = torch.arange(4, device=iq.device, dtype=iq.dtype)
    aligned = (x * torch.exp(-1j * phase[:, None, None] * steps[None, :, None])).mean(1)
    return torch.cat((aligned.real, aligned.imag, coherence.real[:, None], coherence.imag[:, None]), dim=1)


def apply_inverse(iq, taps, innovation):
    """Add a real-linear complex filter correction and a learned IQ residual."""
    radius = taps.shape[2] // 2
    neighborhood = F.pad(iq, (radius, radius), mode='reflect').unfold(-1, 2*radius+1, 1).transpose(2, 3)
    real, imag = neighborhood.unbind(1)
    main_real, main_imag, image_real, image_imag = taps.unbind(1)
    real_delta = ((main_real + image_real) * real + (image_imag - main_imag) * imag).sum(1)
    imag_delta = ((main_imag + image_imag) * real + (main_real - image_real) * imag).sum(1)
    return iq + torch.stack((real_delta, imag_delta), dim=1) + innovation


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.update = nn.Sequential(
            nn.GroupNorm(8, width),
            nn.Conv1d(width, width, 5, padding=2*dilation, dilation=dilation, groups=width),
            nn.Conv1d(width, 2*width, 1),
            nn.GELU(),
            nn.Conv1d(2*width, width, 1),
        )

    def forward(self, value):
        return value + self.update(value)


class PacketInverseFilter(nn.Module):
    """Predict distinct complex filter coefficients across packet regions."""

    def __init__(self):
        super().__init__()
        width, taps = 96, 17
        self.tap_count = taps
        time = torch.arange(256, dtype=torch.float32)
        front = ((88-time)/16).clamp(0, 1)
        tail = ((time-152)/16).clamp(0, 1)
        self.register_buffer('region_weights', torch.stack((front, 1-front-tail, tail))[None])
        self.register_buffer('position', ((time-127.5)/127.5)[None, None])
        self.stem = nn.Conv1d(4, width, 5, padding=2)
        self.reference = nn.Sequential(nn.Linear(42, width), nn.GELU(), nn.Linear(width, width))
        self.context = nn.Sequential(*(TemporalBlock(width, dilation) for dilation in (1, 2, 4, 8, 16, 32)))
        # Heads overlap at the two boundaries to keep the filter transitions continuous.
        self.heads = nn.ModuleList([
            nn.Sequential(*(TemporalBlock(width, dilation) for dilation in dilations),
                          nn.Conv1d(width, 4*taps+2, 1))
            for dilations in ((1, 4), (1,), (1, 4, 8))
        ])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, iq):
        if iq.ndim != 3 or iq.shape[1:] != (2, 256) or iq.dtype != torch.float32:
            raise ValueError('Expected FP32 received IQ [N,2,256]')
        power = iq.square().sum(1, keepdim=True).log1p()
        local = self.stem(torch.cat((iq, power, self.position.expand(len(iq), -1, -1)), dim=1))
        global_reference = self.reference(repeat_context(iq))[:, :, None]
        hidden = self.context(local + global_reference)
        prediction = sum(head(hidden)*self.region_weights[:, i:i+1] for i, head in enumerate(self.heads))
        taps = prediction[:, :-2].reshape(len(iq), 4, self.tap_count, 256)
        return apply_inverse(iq, taps, prediction[:, -2:])


def scratch_packet_filter(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PacketInverseFilter()

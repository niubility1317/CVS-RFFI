"""Complex multi-resolution spectral mask for source-only IQ restoration."""
from __future__ import annotations

import torch
from torch import nn


class ComplexMultiResolutionMask(nn.Module):
    """Filter a frozen-restorer waveform with bounded, sample-conditioned masks.

    Inputs are [batch, 2, 256] real-valued IQ tensors. Received alignment uses
    only received IQ and the restorer output. No labels or paired clean
    waveforms are inputs to this model.
    """

    def __init__(self, channels: int = 32):
        super().__init__()
        self.mask_net = nn.Sequential(
            nn.Conv2d(5, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.mask_net[-1].weight)
        nn.init.zeros_(self.mask_net[-1].bias)
        self.register_buffer('window32', torch.hann_window(32), persistent=False)
        self.register_buffer('window64', torch.hann_window(64), persistent=False)

    def _filter(self, anchor: torch.Tensor, received: torch.Tensor, n_fft: int):
        window = getattr(self, f'window{n_fft}').to(dtype=anchor.real.dtype)
        hop = n_fft // 4
        z = torch.stft(anchor, n_fft=n_fft, hop_length=hop, window=window,
                       center=True, onesided=False, return_complex=True)
        x = torch.stft(received, n_fft=n_fft, hop_length=hop, window=window,
                       center=True, onesided=False, return_complex=True)
        norm = float(n_fft) ** 0.5
        feature = torch.stack((z.real / norm, z.imag / norm, x.real / norm,
                               x.imag / norm, (x-z).abs() / norm), dim=1)
        params = self.mask_net(feature)
        amplitude = torch.exp(0.35 * torch.tanh(params[:, 0]))
        angle = 0.35 * torch.tanh(params[:, 1])
        masked = z * torch.complex(amplitude * torch.cos(angle),
                                   amplitude * torch.sin(angle))
        return torch.istft(masked, n_fft=n_fft, hop_length=hop, window=window,
                           center=True, onesided=False, length=anchor.shape[-1],
                           return_complex=True)

    def forward(self, anchor_iq: torch.Tensor, received_aligned_iq: torch.Tensor,
                gate: torch.Tensor) -> torch.Tensor:
        if anchor_iq.ndim != 3 or anchor_iq.shape[1:] != (2, 256):
            raise ValueError('anchor_iq must have shape [batch, 2, 256]')
        if received_aligned_iq.shape != anchor_iq.shape or gate.shape != (len(anchor_iq),):
            raise ValueError('received IQ and per-record gate must match anchor')
        rms = anchor_iq.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        a = anchor_iq / rms
        r = received_aligned_iq / rms
        az = torch.complex(a[:, 0], a[:, 1])
        rz = torch.complex(r[:, 0], r[:, 1])
        restored = (self._filter(az, rz, 32) + self._filter(az, rz, 64)) * 0.5
        restored_iq = torch.stack((restored.real, restored.imag), dim=1) * rms
        return anchor_iq + gate[:, None, None].to(anchor_iq.dtype) * (restored_iq - anchor_iq)


def aligned_observation(anchor_iq: torch.Tensor, received_iq: torch.Tensor):
    """Align received IQ to anchor and compute the predeclared blind gate."""
    if anchor_iq.shape != received_iq.shape or anchor_iq.ndim != 3 or anchor_iq.shape[1:] != (2, 256):
        raise ValueError('matching [batch, 2, 256] IQ tensors required')

    def align(wave_iq, reference_iq):
        padded = torch.nn.functional.pad(wave_iq, (64, 64), mode='reflect')
        full = torch.complex(padded[:, 0], padded[:, 1]).unfold(-1, 256, 1)
        part = full[:, :, 64:-64]
        reference = torch.complex(reference_iq[:, 0, 64:-64], reference_iq[:, 1, 64:-64])[:, None, :]
        gain = (part.conj()*reference).sum(-1) / part.abs().square().sum(-1).clamp_min(1e-20)
        error = (gain[:, :, None]*part-reference).abs().square().mean(-1)
        choice = error.argmin(1)
        row = torch.arange(len(wave_iq), device=wave_iq.device)
        output = gain[row, choice, None] * full[row, choice]
        return torch.stack((output.real, output.imag), dim=1), error[row, choice]

    aligned, _ = align(received_iq, anchor_iq)
    _, residual = align(anchor_iq, received_iq)
    power = received_iq[:, :, 64:-64].square().sum(1).mean(1).clamp_min(1e-20)
    fraction = (residual/power).clamp(1e-8, 1.)
    proxy = 10.*torch.log10((1.-fraction).clamp_min(1e-8)/fraction)
    ratio = ((received_iq-anchor_iq).square().sum((1,2)) / received_iq.square().sum((1,2)).clamp_min(1e-12)).sqrt()
    return aligned, (ratio > .2) & (proxy < 10.)

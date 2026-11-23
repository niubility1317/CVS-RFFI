"""LoRa short-time Wiener restoration using only current-packet IQ."""

from math import log

import torch
from torch import nn


FFT = 32
HOP = FFT // 4
GROUP = 4
SUPPORT = 2


class LocalWiener(nn.Module):
    """Estimate separate common and chirp-specific local complex covariances."""

    def __init__(self):
        super().__init__()
        self.register_buffer('window', torch.hann_window(FFT))
        self.register_buffer('outer', torch.fft.fftshift(torch.fft.fftfreq(FFT)).abs() >= .375)

    def analyze(self, y):
        raw = torch.stft(y.flatten(0, 1), FFT, HOP, window=self.window,
                         normalized=True, onesided=False, return_complex=True)
        # Batch, chirp, frame, frequency group, frequency within group.
        spectrum = torch.fft.fftshift(raw, dim=-2).transpose(-2, -1)
        spectrum = spectrum.reshape(*y.shape[:2], -1, FFT // GROUP, GROUP)
        # Difference across repeats removes the truly shared component.
        # The 1-1/C factor corrects the variance of each demeaned observation.
        distinct = spectrum - spectrum.mean(1, keepdim=True)
        outside = distinct.flatten(-2)[..., self.outer]
        noise = outside.abs().square().flatten(1).median(dim=1).values / (log(2) * (1 - 1/y.shape[1]))
        power = spectrum.abs().square().mean(dim=(1, 2, 3, 4))
        return spectrum, noise, noise / power.clamp_min(1e-8)

    def synthesize(self, spectrum, y):
        flat = spectrum.flatten(-2).flatten(0, 1).transpose(-2, -1)
        restored = torch.istft(torch.fft.ifftshift(flat, dim=-2), FFT, HOP, window=self.window,
                               normalized=True, onesided=False, length=y.shape[-1], return_complex=True)
        return restored.reshape_as(y)

    @staticmethod
    def covariance(spectrum):
        cross = spectrum.unsqueeze(-1) * spectrum.conj().unsqueeze(-2)
        time = cross.mean(dim=1)
        frames = torch.arange(time.shape[1], device=time.device)
        support = frames[:, None] + torch.arange(-SUPPORT, SUPPORT + 1, device=time.device)
        return time[:, support.clamp(0, len(frames) - 1)].mean(dim=2)

    @staticmethod
    def filter(spectrum, covariance, noise, *, correlated):
        effective = noise[:, None, None, None]
        if correlated:
            level, axes = torch.linalg.eigh(covariance)
            rotated = (axes.mH[:, None] @ spectrum.unsqueeze(-1)).squeeze(-1)
        else:
            level = covariance.diagonal(dim1=-2, dim2=-1).real
            rotated = spectrum
        # Estimate positive signal power by subtracting the measured noise floor.
        gain = (level - effective).clamp_min(0) / level.clamp_min(1e-8)
        output = rotated * gain[:, None]
        return (axes[:, None] @ output.unsqueeze(-1)).squeeze(-1) if correlated else output

    def restore(self, y, spectrum, noise, *, correlated):
        common = spectrum.mean(dim=1, keepdim=True)
        distinct = spectrum - common
        n = y.shape[1]
        common = self.filter(common, self.covariance(common), noise / n, correlated=correlated)
        distinct = self.filter(distinct, self.covariance(distinct), noise * (1-1/n), correlated=correlated)
        delta = self.synthesize(common + distinct, y) - y
        # Keep the tracked complex carrier unchanged in each chirp.
        return delta - delta.mean(dim=-1, keepdim=True)

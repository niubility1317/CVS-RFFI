"""Observation whitening from native repeated preambles and source clean dispersion."""

import torch
from torch.nn import functional as F

from experiments.pamf_lora_main.repeated_edit import PREAMBLE, repeat_rotation
from experiments.pamf_lora_main.w24_native_dura import capture_scale


def repeat_dispersion(received):
    start, count, period = PREAMBLE[received.shape[-1]]
    segment = received[:, start:start + count * period]
    normalized = (segment / capture_scale(segment)).unflatten(-1, (count, period))
    aligned = normalized * repeat_rotation(normalized).conj().unsqueeze(-1)
    spectrum = torch.fft.fft(aligned, norm='ortho')
    return (spectrum-spectrum.mean(1,keepdim=True)).abs().square().sum(1) / (count-1)


def preamble_weights(received, clean_floor):
    dispersion = repeat_dispersion(received)
    excess = (dispersion-clean_floor).clamp_min(0.)
    excess = .5 * excess + .25 * (excess.roll(1, -1) + excess.roll(-1, -1))
    center = excess.mean(-1, keepdim=True)
    stabilizer = .25 * center + 1e-4
    weights = ((center + stabilizer) / (excess + stabilizer)).sqrt().clamp(.5, 2.)
    shifted = torch.fft.fftshift(weights, dim=-1)[:, None]
    resized = F.interpolate(shifted, size=received.shape[-1], mode='linear', align_corners=False)[:, 0]
    weights = torch.fft.ifftshift(resized, dim=-1)
    return weights / weights.square().mean(-1,keepdim=True).sqrt()


class WhitenedAction:
    """Compose a physical action with a self-adjoint positive FFT multiplier."""

    def __init__(self, action, weights):
        self.action, self.weights = action, weights

    def whiten(self, value):
        return torch.fft.ifft(self.weights * torch.fft.fft(value, norm='ortho'), norm='ortho')

    def __call__(self, value):
        return self.whiten(self.action(value))

    def adjoint(self, value):
        return self.action.adjoint(self.whiten(value))

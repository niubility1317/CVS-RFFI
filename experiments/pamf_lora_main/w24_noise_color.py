"""Match W24 source noise energy while changing its native-band spectrum."""

import torch

from .w24_white_corruption import center_power


def shape_noise(white, half_bandwidth, center_frequency):
    """Smooth continuous passband on the contiguous capture clock."""
    f = torch.fft.fftfreq(white.shape[-1], device=white.device, dtype=white.real.dtype)
    separation = (f - center_frequency).abs() / half_bandwidth
    transfer = (1 + separation.pow(8)).rsqrt()
    return torch.fft.ifft(torch.fft.fft(white, norm='ortho') * transfer, norm='ortho')


def match_noise_energy(aperture_noise, target_power):
    ratio = target_power / center_power(aperture_noise).clamp_min(1e-12)
    return aperture_noise * ratio.sqrt()


def noise_views(raw, clean, aperture, generic, rng, *, colored):
    """Both arms retain the same mixed cases and draw the same source-only streams."""
    native = raw.unflatten(-1, (8, 1024))
    observations = []
    for _ in range(2):
        anchor = aperture(generic(native, rng).flatten(-2))
        packet = (len(raw), 1)
        white = torch.randn(raw.shape, device=raw.device, dtype=raw.dtype, generator=rng)
        choices = torch.rand((len(raw), 3), dtype=raw.real.dtype, device=raw.device, generator=rng)
        width, center, mixed = (draw.reshape(packet) for draw in choices.unbind(-1))
        noise = shape_noise(white, .045 + .205 * width, .14 * (center - .5)) if colored else white
        noise = match_noise_energy(aperture(noise), center_power(anchor - clean))
        observations.append(torch.where(mixed[..., None] < .35, anchor, clean + noise))
    return observations

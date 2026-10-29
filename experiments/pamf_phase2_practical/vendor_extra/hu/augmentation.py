from __future__ import annotations

import math

import torch


def _uniform(low: float, high: float, *, generator: torch.Generator | None, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return low + (high - low) * torch.rand((), generator=generator, device=device, dtype=dtype)


def augment_iq(
    iq: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    sample_rate_hz: float = 1_000_000.0,
    max_delay_samples: int = 10,
) -> torch.Tensor:
    """Apply the paper's multipath, Doppler and noise families with fixed defaults."""
    if iq.ndim != 3 or tuple(iq.shape[1:]) != (2, 256):
        raise ValueError("iq must have shape [batch, 2, 256]")
    if sample_rate_hz <= 0 or max_delay_samples <= 0:
        raise ValueError("sample_rate_hz and max_delay_samples must be positive")
    complex_iq = torch.complex(iq[:, 0], iq[:, 1])
    batch, length = complex_iq.shape
    output = torch.empty_like(complex_iq)
    sample_index = torch.arange(length, device=iq.device, dtype=iq.dtype)
    for index in range(batch):
        path_count = int(torch.randint(1, 6, (), generator=generator, device=iq.device))
        maximum_delay = int(torch.randint(5, max_delay_samples + 1, (), generator=generator, device=iq.device))
        multipath = torch.zeros_like(complex_iq[index])
        for _ in range(path_count):
            delay = int(torch.randint(0, maximum_delay + 1, (), generator=generator, device=iq.device))
            attenuation = _uniform(0.5, 1.0, generator=generator, device=iq.device, dtype=iq.dtype)
            phase = _uniform(0.0, 2.0 * math.pi, generator=generator, device=iq.device, dtype=iq.dtype)
            multipath = multipath + attenuation * torch.exp(1j * phase) * torch.roll(complex_iq[index], delay)
        doppler_hz = _uniform(-15.0, 15.0, generator=generator, device=iq.device, dtype=iq.dtype)
        doppler = torch.exp(1j * (2.0 * math.pi * doppler_hz * sample_index / sample_rate_hz))
        signal = multipath * doppler
        snr_db = _uniform(15.0, 30.0, generator=generator, device=iq.device, dtype=iq.dtype)
        signal_power = signal.abs().square().mean().clamp_min(torch.finfo(iq.dtype).eps)
        noise_scale = torch.sqrt(signal_power / torch.pow(torch.tensor(10.0, device=iq.device, dtype=iq.dtype), snr_db / 10.0) / 2.0)
        noise = noise_scale * (
            torch.randn(length, generator=generator, device=iq.device, dtype=iq.dtype)
            + 1j * torch.randn(length, generator=generator, device=iq.device, dtype=iq.dtype)
        )
        output[index] = signal + noise
    return torch.stack((output.real, output.imag), dim=1)

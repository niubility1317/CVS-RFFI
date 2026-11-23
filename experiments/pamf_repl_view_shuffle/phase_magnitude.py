"""Split WiSig Fourier magnitude and phase in the same received coordinates."""

import torch

from .receiver_pilot import phase, power


def magnitude_with_phase(magnitude_source, phase_source):
    magnitude = magnitude_source.abs()
    phase_magnitude = phase_source.abs()
    shaped = magnitude * (phase_source / phase_magnitude.clamp_min(1e-8))
    return torch.where(phase_magnitude > 1e-8, shaped, magnitude_source)


def match_power(y, x):
    fit_power = power(x).sum(dim=-1, keepdim=True)
    recv_power = power(y).sum(dim=-1, keepdim=True)
    return torch.where(fit_power > 1e-12, x * (recv_power / fit_power.clamp_min(1e-12)).sqrt(), y)


def weighted_spectrum(y, adjusted, noise_power):
    # Keep the original magnitude and move its phase only where both spectra
    # have received evidence above the effective STF disturbance.
    evidence = torch.minimum(power(y), power(adjusted))
    confidence = evidence / (evidence + noise_power[:, None] + 1e-8)
    rotation = (adjusted * y.conj()).angle()
    return y * phase(confidence * rotation)


def candidates(received, fitted, noise_power):
    y = torch.fft.fft(received, norm='ortho')
    x = torch.fft.fft(fitted, norm='ortho')
    adjusted = match_power(y, x)
    parts = dict(fit_rms=adjusted, fit_phase=magnitude_with_phase(y, x),
                 fit_magnitude=magnitude_with_phase(x, y),
                 weighted_phase=weighted_spectrum(y, adjusted, noise_power))
    return {name: torch.fft.ifft(value, norm='ortho') for name, value in parts.items()}


def weighted_phase(received, fitted, noise_power):
    y = torch.fft.fft(received, norm='ortho')
    x = torch.fft.fft(fitted, norm='ortho')
    return torch.fft.ifft(weighted_spectrum(y, match_power(y, x), noise_power), norm='ortho')


def paired_reference_diagnostic(received, clean):
    y = torch.fft.fft(received, norm='ortho')
    x = torch.fft.fft(clean, norm='ortho')
    return dict(reference_phase=torch.fft.ifft(magnitude_with_phase(y, x), norm='ortho'),
                reference_magnitude=torch.fft.ifft(magnitude_with_phase(x, y), norm='ortho'),
                paired_clean=clean)

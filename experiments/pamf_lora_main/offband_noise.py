"""Predict complex LoRa noise from observed dechirped witness frequencies."""

import torch


def native_spectrum(spectrum, rotation):
    tracked = torch.fft.ifft(spectrum, norm='ortho')
    return torch.fft.fft(tracked * rotation, norm='ortho')


def covariance_action(spectrum, rotation, density):
    native = native_spectrum(spectrum, rotation)
    colored = torch.fft.ifft(native * density, norm='ortho')
    return torch.fft.fft(colored * rotation.conj(), norm='ortho')


def received_profile(spectrum, rotation, witness, *, share):
    """Estimate a regularized RF noise spectrum from the witness projection."""
    power = native_spectrum(spectrum * witness, rotation).abs().square()
    smooth = sum(power.roll(k, dims=-1) for k in range(-4, 5)) / 9
    mean = smooth.mean(-1, keepdim=True)
    normalized = torch.where(mean > 1e-12, smooth / mean.clamp_min(1e-12), 1.)
    coverage = witness.float().mean()
    # Approximate a swept chirp projector by its diagonal fraction and a
    # diffuse spectral pedestal, then bound the resulting covariance.
    profile = ((normalized - (1 - coverage)) / coverage).clamp(.20, 5.)
    profile = profile.lerp(profile.median(dim=1, keepdim=True).values, share)
    return profile / profile.mean(-1, keepdim=True)


def conditional_noise(spectrum, rotation, density, witness, *, ridge=.05, steps=16):
    """Solve the witness-restricted complex covariance system using FFTs."""
    def applied(value):
        return covariance_action(value, rotation, density) * witness + ridge * value

    residual = spectrum * witness
    direction = residual
    solution = torch.zeros_like(residual)
    power = residual.abs().square().sum(-1, keepdim=True)
    cutoff = power * 1e-12 + 1e-20
    for _ in range(steps):
        mapped = applied(direction)
        product = (direction.conj() * mapped).real.sum(-1, keepdim=True)
        step = torch.where(power > cutoff, power / product.clamp_min(1e-20), 0.)
        solution = solution + step * direction
        residual = residual - step * mapped
        new_power = residual.abs().square().sum(-1, keepdim=True)
        factor = torch.where(power > cutoff, new_power / power.clamp_min(1e-20), 0.)
        direction = residual + factor * direction
        power = new_power
    return covariance_action(solution, rotation, density) * ~witness

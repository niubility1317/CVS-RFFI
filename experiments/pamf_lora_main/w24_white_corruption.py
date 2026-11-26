"""Match W24 source perturbation energy while varying its structure."""

import torch


def center_power(wave):
    return wave[..., :384].abs().square().mean(dim=(-2, -1), keepdim=True)


def training_observations(raw, clean, aperture, generic, rng, *, white):
    native = raw.unflatten(-1, (8, 1024))
    observed = []
    for _ in range(2):
        original = aperture(generic(native, rng).flatten(-2))
        # Both arms advance the same generator through the same two draws.
        innovation = torch.randn(raw.shape, device=raw.device, dtype=raw.dtype, generator=rng)
        if white:
            unit_noise = aperture(innovation)
            ratio = center_power(original - clean) / center_power(unit_noise).clamp_min(1e-12)
            original = clean + ratio.sqrt() * unit_noise
        observed.append(original)
    return tuple(observed)

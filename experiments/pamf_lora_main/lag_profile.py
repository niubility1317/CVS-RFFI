"""Measure within-chirp amplitude and phase variation in normalized receiver IQ."""

import torch


LAGS = (2, 8, 24)
WINDOW = 64


def lag_profile(wave):
    """Center local complex lag products and remove the global phase of each lag."""
    windows = wave.unflatten(-1, (-1, WINDOW))
    products = torch.stack([(windows[..., lag:] * windows[..., :-lag].conj()).mean(-1)
                            for lag in LAGS], dim=-2)
    carrier = products.mean(-1, keepdim=True)
    # Products with negligible coherent support contribute smoothly less.
    phase = carrier.conj() / (carrier.abs().square() + 1e-6).sqrt()
    return (products - carrier) * phase


def lag_profile_risk(prediction, reference):
    """Preserve shared and chirp-specific nonstationary lag profiles separately."""
    clean = lag_profile(reference)
    error = lag_profile(prediction) - clean[:, None]
    common_ref = clean.mean(-3, keepdim=True)
    common_error = error.mean(-3, keepdim=True)
    components = ((common_error, common_ref), (error - common_error, clean - common_ref))
    results = []
    for discrepancy, target in components:
        power = target.abs().square().mean(dim=(-3, -1))
        distortion = discrepancy.abs().square().mean(dim=(-3, -1))
        results.append((distortion / (power[:, None] + .002)).log1p().mean())
    return torch.stack(results)

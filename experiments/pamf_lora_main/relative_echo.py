"""Cross-validated short relative channel action on six LoRa chirps."""

import torch
import torch.nn.functional as F


SHORT = (0, -2, -1, 1, 2)
SCALAR = (0,)
BLOCK_SAMPLES = 16
GUARD = max(map(abs, SHORT))


def shift(value, offset):
    """Translate on the measured interval, with no periodic extension."""
    if offset > 0:
        return F.pad(value[..., :-offset], (offset, 0))
    if offset < 0:
        return F.pad(value[..., -offset:], (0, -offset))
    return value


def columns(value, phase, offsets):
    raw = phase * value
    return torch.stack([phase.conj() * shift(raw, offset) for offset in offsets], -1)


def action(value, phase, coefficients, offsets, *, adjoint=False):
    if adjoint:
        coefficients = coefficients.conj()
        offsets = tuple(-offset for offset in offsets)
    edit = (columns(value, phase, offsets) * coefficients.unsqueeze(-2)).sum(-1)
    return value + edit


def fit_fold(design, residual, samples, ridge):
    support = design[..., samples, :]
    target = residual[..., samples]
    gram = support.mH @ support / len(samples)
    rhs = (support.mH @ target.unsqueeze(-1)) / len(samples)
    eye = torch.eye(design.shape[-1], dtype=design.dtype, device=design.device)
    coefficient = torch.linalg.solve(gram + ridge.unsqueeze(-1) * eye, rhs).squeeze(-1)
    return coefficient - coefficient.mean(1, keepdim=True)


def fold_samples(length, device):
    positions = torch.arange(length, device=device)
    within_block = positions.remainder(BLOCK_SAMPLES)
    interior = (within_block >= GUARD) & (within_block < BLOCK_SAMPLES - GUARD)
    positions = positions[interior]
    interleaved = positions.div(BLOCK_SAMPLES, rounding_mode="floor").remainder(2)
    return tuple(positions[interleaved == i] for i in range(2))


def fit_relative(y, phase, sigma, offsets):
    """Estimate a relative transfer and score it off the fitting indices."""
    reference = torch.stack((y[:, 1::2].mean(1), y[:, ::2].mean(1)), 1).repeat(1, 3, 1)
    design = columns(reference, phase, offsets)
    residual = y - reference
    masks = fold_samples(y.shape[-1], y.device)
    ridge = .05 + .25 * sigma
    estimates = [fit_fold(design, residual, samples, ridge) for samples in masks]
    held_errors = [
        (residual[..., samples] - (design[..., samples, :] * fit.unsqueeze(-2)).sum(-1))
        .abs().square().mean(-1)
        for samples, fit in zip(masks, reversed(estimates), strict=True)
    ]
    held = .5 * (held_errors[0] + held_errors[1])
    baseline = .5 * sum(residual[..., samples].abs().square().mean(-1) for samples in masks)
    return .5 * (estimates[0] + estimates[1]), held, baseline


def relative_models(y, phase, sigma):
    """Apply the extra taps only where they beat the scalar held-out fit."""
    scalar, scalar_error, reference_error = fit_relative(y, phase, sigma, SCALAR)
    extended, extended_error, _ = fit_relative(y, phase, sigma, SHORT)
    variance = sigma.squeeze(-1)
    baseline = reference_error.mean(1, keepdim=True)
    one_tap = scalar_error.mean(1, keepdim=True)
    multi_tap = extended_error.mean(1, keepdim=True)
    scalar_trust = ((baseline - one_tap) / (.02 + baseline)).clamp(0., 1.)
    echo_margin = (one_tap - multi_tap - .05 * variance).clamp_min(0.)
    extra_trust = (echo_margin / (.01 + one_tap)).clamp_max(1.)
    single = F.pad(scalar * scalar_trust[..., None], (0, len(SHORT) - 1))
    full = single + extra_trust[..., None] * (extended - single)
    return single, full, (reference_error, scalar_error, extended_error, extra_trust, echo_margin)


def correction(y, phase, coefficients):
    """One damped adjoint step from the received samples, leaving the carrier fixed."""
    residual = y - action(y, phase, coefficients, SHORT)
    update = action(residual, phase, coefficients, SHORT, adjoint=True)
    update = update / (1. + coefficients.abs().sum(-1, keepdim=True)).square()
    # All fold fits disregard the boundary. Protect it and the carrier on readout.
    interior = update[..., GUARD:-GUARD]
    return F.pad(interior - interior.mean(-1, keepdim=True), (GUARD, GUARD))

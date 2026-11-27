"""Isolated source counterfactuals at fixed total center correction energy."""

import torch

from .w24_noise_components import aligned_spectrum_parts


def components(value, phase):
    _, common, private, near, frame = aligned_spectrum_parts(value, phase)
    return torch.stack((common * near, common * ~near, private * near, private * ~near), dim=1), frame


def counterfactual_edits(estimated, exact, phase):
    """Replace four component norms or directions while holding overall edit power fixed."""
    # A non-unit phase would alter the power of the fixed-norm interventions.
    if (not torch.isfinite(phase).all() or (phase.abs() <= 1e-12).any()):
        raise ValueError('Expected finite nonzero measured phase')
    estimate, frame = components(estimated, phase)
    reference, _ = components(exact, phase)
    est_norm, ref_norm = (torch.linalg.vector_norm(value, dim=(-2, -1), keepdim=True)
                          for value in (estimate, reference))
    est_total, ref_total = (torch.linalg.vector_norm(value, dim=(-2, -1), keepdim=True).unsqueeze(1)
                            for value in (estimated, exact))
    est_unit = estimate / est_norm.clamp_min(1e-12)
    ref_unit = reference / ref_norm.clamp_min(1e-12)
    active = (est_total > 1e-12) & (ref_total > 1e-12)
    ref_dose = ref_norm * (est_total / ref_total.clamp_min(1e-12))

    # With no learned direction in an active reference component, keep the whole
    # learned edit for the allocation-only row. Do not lend the oracle direction.
    est_floor = (est_total * 1e-6).clamp_min(1e-12)
    ref_floor = (ref_total * 1e-6).clamp_min(1e-12)
    defined = ~((est_norm <= est_floor) & (ref_dose > est_floor)).any(1, keepdim=True)
    partitioned = (
        torch.where(active & defined, est_unit * ref_dose, estimate),
        torch.where(active & (ref_norm > ref_floor), ref_unit * est_norm, estimate),
    )
    both = exact * (est_total / ref_total.clamp_min(1e-12)).flatten()[:, None, None]
    outputs = (*(torch.fft.ifft(value.sum(1), norm='ortho') * frame for value in partitioned), both)
    active_rows = active.flatten()
    fixed_dose = tuple(torch.where(active_rows[:, None, None], output, estimated) for output in outputs)
    for output in fixed_dose:
        torch.testing.assert_close(torch.linalg.vector_norm(output, dim=(-2, -1)), est_total.flatten(),
                                   rtol=2e-5, atol=1e-6)
    return (*fixed_dose,
            torch.where((ref_total.flatten() > 1e-12)[:, None, None], exact, estimated))

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import torch


def finite_difference_linearity_error(
    evaluate: Callable[[float], torch.Tensor],
    *,
    deltas: Iterable[float],
) -> dict[float, float]:
    """Compare D_delta with D_delta/2 around the same deterministic base point."""

    base = torch.as_tensor(evaluate(0.0)).detach().float()
    errors: dict[float, float] = {}
    for raw_delta in deltas:
        delta = float(raw_delta)
        if delta <= 0.0:
            raise ValueError("calibration deltas must be positive")
        full = (torch.as_tensor(evaluate(delta)).detach().float() - base) / delta
        half = (torch.as_tensor(evaluate(delta / 2.0)).detach().float() - base) / (delta / 2.0)
        error = (full - half).norm() / half.norm().clamp_min(1e-12)
        errors[delta] = float(error.item())
    return errors


def select_largest_stable_delta(errors: Mapping[float, float], *, max_error: float) -> float:
    if not 0.0 < float(max_error) < 1.0:
        raise ValueError("max_error must be in (0,1)")
    stable = [float(delta) for delta, error in errors.items() if float(delta) > 0.0 and float(error) <= float(max_error)]
    if not stable:
        raise ValueError("no calibrated delta satisfies the linearity bound")
    return max(stable)

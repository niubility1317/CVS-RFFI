"""Bounded, observation-budget auxiliary loss with no fitted state."""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch.nn import functional as F


def _constant(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real constant")
    value = float(value)
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _observation(value, shape, device, name):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.shape != shape or value.device != device or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be a finite real floating tensor of shape {list(shape)} on the prediction device")
    return value.detach().to(dtype=torch.float64)


def budget_loss(pred, target, signal, budget, *, coefficient=.1, eps=1e-8,
                normalization_count=None, role="U_s"):
    """Loss on source-only ``pred[N,14]``; all observation inputs are detached.

    D=signal+budget+eps, w=signal/D, a=relu((MSE-budget)/D).
    Return coefficient*sum(w*a/(1+a))/FULL count. FULL defaults to N; an
    explicitly supplied integer count must be at least N. No active-only
    denominator is used. Each unnormalized packet contribution is bounded by
    coefficient*w. MSE <= budget gives exact zero loss and zero derivative.

    Prediction retains its graph. Target, signal and budget may be FP32/FP64
    observations on that device and are detached before FP64 calculation.
    Source L_s/U_s are the only accepted roles. Caller establishes actual source
    provenance; no labels, W, model, optimizer, teacher argument or RNG is used.
    The budget is an IID-noise-motivated tolerance, not a proven cache noise floor.
    """
    if role not in ("L_s", "U_s"):
        raise ValueError("budget_loss accepts only source roles L_s and U_s")
    coefficient = _constant(coefficient, "coefficient")
    eps = _constant(eps, "eps", positive=True)
    if (not isinstance(pred, torch.Tensor) or pred.layout != torch.strided
            or pred.ndim != 2 or pred.shape[0] < 1 or pred.shape[1] != 14
            or not pred.is_floating_point() or not bool(torch.isfinite(pred).all())):
        raise ValueError("pred must be a nonempty finite real dense floating tensor [N,14]")
    count = len(pred)
    full_count = count if normalization_count is None else normalization_count
    if isinstance(full_count, bool) or not isinstance(full_count, Integral) or full_count < count:
        raise ValueError("normalization_count must be an integer at least N")
    full_count = int(full_count)
    target = _observation(target, pred.shape, pred.device, "target")
    signal = _observation(signal, (count,), pred.device, "signal")
    budget = _observation(budget, (count,), pred.device, "budget")
    if bool((signal < 0).any()) or bool((budget < 0).any()):
        raise ValueError("signal and budget must be nonnegative observations")
    prediction = pred.to(dtype=torch.float64)
    mse = (prediction - target).square().mean(dim=1)
    denominator = signal + budget + eps
    weight = signal / denominator
    excess = F.relu((mse - budget) / denominator)
    bounded = excess / (1 + excess)
    per_packet = coefficient * weight * bounded
    weighted_loss = per_packet.sum() / full_count
    if not all(bool(torch.isfinite(value).all()) for value in (mse, denominator, weight, excess, bounded, per_packet, weighted_loss)):
        raise ValueError("budget loss arithmetic must remain finite")
    with torch.no_grad():
        active = (excess > 0) & (weight > 0) & (coefficient > 0)
        summary = {
            "role": role, "input_count": count, "normalization_count": full_count,
            "coefficient": coefficient, "eps": eps, "active_count": int(active.sum()),
            "fulfilled_count": int((mse <= budget).sum()),
            "above_budget_count": int((mse > budget).sum()), "zero_signal_count": int((signal == 0).sum()),
            "MSE_mean": float(mse.mean()), "signal_mean": float(signal.mean()),
            "budget_mean": float(budget.mean()), "weight_mean": float(weight.mean()),
            "unweighted_mean": float((weight * bounded).mean()),
            "weighted_loss": float(weighted_loss),
            "maximum_packet_contribution": float(per_packet.max()),
            "maximum_packet_bound": float((coefficient * weight).max()),
            "denominator_rule": "full packet count; no active-only normalization",
            "observations_detached": True, "prediction_graph_retained": True,
            "global_RNG_consumed": False, "new_parameters": False,
            "scope": "source-only observation tolerance motivated by IID noise; upstream equalization and noise independence unknown; no TX, DG or RX-removal claim",
        }
    return {"weighted_loss": weighted_loss, "summary": summary, "tensors": {
        "MSE": mse, "D": denominator, "w": weight, "a": excess,
        "bounded_excess": bounded, "per_packet_loss": per_packet,
    }}

"""Clean-only, conditional cosine-gradient Fishr (project adaptation).

Variance uses n, coordinate reduction is mean, and a two-domain pair uses 1/4.
EMA is independently zero-debiased per (RX, day, actual scene), unlike the
original author's compensation. No virtual, style or target feature is valid.
"""
from __future__ import annotations

import math
import statistics
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def cosine_sample_gradients(z: Tensor, weight: Tensor, labels: Tensor,
                            mode: str = "direction", scale: float = 30.0,
                            eps: float = 1e-4) -> Tensor:
    """Differentiable [B,6,160] individual CE gradients, with all paths intact.

    The tangent formula is invalid in F.normalize's clamped-norm region. Such
    inputs are rejected explicitly instead of applying an incorrect derivative.
    """
    if z.ndim != 2 or z.shape[1] != 160 or weight.shape != (6, 160):
        raise ValueError("Fishr requires z[B,160] and cosine weight[6,160]")
    if labels.shape != (len(z),) or labels.dtype != torch.long:
        raise ValueError("Fishr requires an int64 source label for every sample")
    if mode not in ("direction", "raw") or scale != 30.0 or eps <= 0:
        raise ValueError("Expected raw/direction, fixed scale=30 and positive epsilon")
    if not torch.isfinite(z).all() or not torch.isfinite(weight).all():
        raise ValueError("Nonfinite cosine input")
    zn = z.norm(dim=1, keepdim=True)
    wn = weight.norm(dim=1, keepdim=True)
    if (zn <= eps).any() or (wn <= eps).any():
        raise ValueError("Cosine norm at/below epsilon; tangent Fishr undefined")
    u, v = z / zn, weight / wn
    cosine = u @ v.T
    delta = (scale * cosine).softmax(dim=1) - F.one_hot(labels, 6).to(z.dtype)
    tangent = u[:, None, :] - cosine[:, :, None] * v[None, :, :]
    gradient = scale * delta[:, :, None] * tangent
    return gradient if mode == "direction" else gradient / wn[None, :, :]


def sample_variance(gradient: Tensor) -> Tensor:
    if gradient.ndim != 3 or gradient.shape[0] < 2:
        raise ValueError("At least two actual samples required for variance")
    return (gradient - gradient.mean(dim=0, keepdim=True)).square().mean(dim=0)


def pair_penalty(left: Tensor, right: Tensor) -> Tensor:
    return (left - right).square().mean() / 4.0


class ConditionalFishr(nn.Module):
    """Each call contains the two real RX of one same-day clean B48 block.

    Malformed/missing support returns a graph-connected zero with an explicit
    reason and never changes any bucket. Evaluation cannot mutate state.
    """
    def __init__(self, mode: str = "direction", beta: float = 0.9,
                 require_b48: bool = True):
        super().__init__()
        if mode not in ("direction", "raw") or not 0 <= beta < 1:
            raise ValueError("Invalid Fishr mode/beta")
        self.mode, self.beta, self.require_b48 = mode, float(beta), require_b48
        self.buckets: dict[tuple[int, int, str], dict] = {}

    def get_extra_state(self):
        return {"version": 1, "mode": self.mode, "beta": self.beta,
                "require_b48": self.require_b48,
                "buckets": [{"key": list(key), "m": value["m"].detach().cpu().clone(),
                             "count": value["count"], "update_step": value["update_step"]}
                            for key, value in sorted(self.buckets.items())]}

    def set_extra_state(self, state):
        if (state["version"] != 1 or state["mode"] != self.mode
                or state["beta"] != self.beta or state["require_b48"] != self.require_b48):
            raise ValueError("Incompatible Fishr checkpoint definition")
        restored = {}
        for row in state["buckets"]:
            key = tuple(row["key"])
            m = row["m"].detach().clone()
            if m.shape != (6, 160) or row["count"] < 1 or not torch.isfinite(m).all():
                raise ValueError("Invalid raw EMA checkpoint")
            restored[key] = {"m": m, "count": int(row["count"]),
                             "update_step": int(row["update_step"])}
        self.buckets = restored

    def forward(self, z: Tensor, weight: Tensor, labels: Tensor, rx: Tensor,
                day: Tensor, step: int, scene: str = "clean", update: bool = True):
        zero = z.sum() * 0.0 + weight.sum() * 0.0
        meta = {"mode": self.mode, "scene": scene, "scene_weight": 1.0,
                "updated": False, "reason": None, "bucket_counts": [], "buckets": []}
        def missing(reason):
            meta["reason"] = reason
            return zero, meta
        if scene != "clean":
            raise ValueError("This no-satellite experiment permits only real clean Fishr")
        if rx.shape != labels.shape or day.shape != labels.shape:
            raise ValueError("RX/day metadata shape mismatch")
        receivers = torch.unique(rx).tolist()
        days = torch.unique(day).tolist()
        if len(receivers) != 2 or len(days) != 1:
            return missing("requires_exactly_two_RX_same_day")
        if self.require_b48 and len(z) != 48:
            return missing("requires_B48")
        for receiver in receivers:
            counts = torch.bincount(labels[rx == receiver], minlength=6)
            if len(counts) != 6 or (counts < 2).any() or not (counts == counts[0]).all():
                return missing("requires_all_six_TX_equal_actual_support")
            if self.require_b48 and not (counts == 4).all():
                return missing("requires_four_packets_per_TX_RX")
        gradients = cosine_sample_gradients(z, weight, labels, self.mode)
        moments, pending = [], []
        mutate = bool(update and self.training)
        for receiver in receivers:
            key = (int(receiver), int(days[0]), scene)
            current = sample_variance(gradients[rx == receiver])
            old = self.buckets.get(key)
            count = (old["count"] if old else 0) + 1
            previous = old["m"].to(current).detach() if old else torch.zeros_like(current)
            m = self.beta * previous + (1 - self.beta) * current
            coefficient = (1 - self.beta) / (1 - self.beta ** count)
            moments.append(m / (1 - self.beta ** count))
            pending.append((key, {"m": m.detach().clone(), "count": count,
                                  "update_step": int(step)}))
            meta["bucket_counts"].append(count)
            meta["buckets"].append({"key": list(key), "count": count,
                                    "prior_update_step": old["update_step"] if old else None,
                                    "age_steps": int(step) - old["update_step"] if old else None,
                                    "current_gradient_coefficient": coefficient})
        loss = pair_penalty(*moments)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Fishr penalty; EMA not updated")
        if mutate:
            self.buckets.update(pending)
        meta.update(updated=mutate, penalty=float(loss.detach()),
                    variance_gap=float((moments[0] - moments[1]).abs().mean().detach()),
                    weight_row_norms=weight.detach().norm(dim=1).cpu().tolist())
        return loss, meta


def loss_gradients(loss: Tensor, parameters: Iterable[Tensor]):
    parameters = tuple(p for p in parameters if p.requires_grad)
    return torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)


def gradient_norm(gradients) -> float:
    squares = [g.detach().double().square().sum() for g in gradients if g is not None]
    return math.sqrt(sum(float(x) for x in squares))


class GradientRatioCalibrator:
    """E20 source-L audit; median of per-audit norm ratios, fixed thereafter.

    Audit losses must originate in the same native CE and clean relation graphs.
    No optimizer step or EMA update is performed here. The 3/5/10% ratio is an
    ACTIVE auxiliary-step ratio, not an epoch-integrated contribution.
    """
    def __init__(self, target_ratio: float = .05, min_bucket_updates: int = 8):
        if target_ratio not in (.03, .05, .10) or min_bucket_updates < 2:
            raise ValueError("Expected preregistered 3%, 5% or 10% and mature buckets")
        self.target_ratio = target_ratio
        self.min_bucket_updates = int(min_bucket_updates)
        self.observations = []
        self.lambda_max = None

    def observe(self, ce_loss, fishr_loss, parameters, bucket_counts, epoch=20):
        if self.lambda_max is not None:
            raise RuntimeError("Fishr lambda is frozen; adaptive reweighting prohibited")
        if epoch != 20:
            raise ValueError("Gradient calibration is restricted to E20 source L audit")
        if len(bucket_counts) != 2 or min(bucket_counts) < self.min_bucket_updates:
            return {"accepted": False, "reason": "immature_buckets"}
        parameters = tuple(parameters)
        cg = loss_gradients(ce_loss, parameters)
        fg = loss_gradients(fishr_loss, parameters)
        cn, fn = gradient_norm(cg), gradient_norm(fg)
        if not math.isfinite(cn + fn) or min(cn, fn) <= 1e-12:
            return {"accepted": False, "reason": "nonfinite_or_zero_gradient"}
        dot = sum(float((a.detach().double() * b.detach().double()).sum())
                  for a, b in zip(cg, fg) if a is not None and b is not None)
        row = {"accepted": True, "ce_gradient_norm": cn, "fishr_gradient_norm": fn,
               "cosine_with_ce": dot / (cn * fn), "bucket_counts": list(bucket_counts),
               "norm_ratio": cn / fn, "epoch": 20}
        self.observations.append(row)
        return row.copy()

    def freeze(self):
        if self.lambda_max is None:
            if not self.observations:
                raise RuntimeError("No mature nonzero E20 source-L gradient calibration")
            self.lambda_max = self.target_ratio * statistics.median(
                row["norm_ratio"] for row in self.observations)
        return self.lambda_max

    def weight(self, epoch):
        if epoch <= 20:
            return 0.0
        if self.lambda_max is None:
            raise RuntimeError("E20 calibration must be frozen before enabling Fishr")
        return self.lambda_max * min(1.0, (epoch - 20) / 20.0)

    def state_dict(self):
        return {"target_ratio": self.target_ratio, "min_bucket_updates": self.min_bucket_updates,
                "observations": [x.copy() for x in self.observations], "lambda_max": self.lambda_max}

    def load_state_dict(self, state):
        if (state["target_ratio"] != self.target_ratio or
                state["min_bucket_updates"] != self.min_bucket_updates):
            raise ValueError("Incompatible fixed calibration checkpoint")
        self.observations = [x.copy() for x in state["observations"]]
        self.lambda_max = state["lambda_max"]

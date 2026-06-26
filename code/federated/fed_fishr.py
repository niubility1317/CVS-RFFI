from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F


def _as_2d_float(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).view(int(x.size(0)), -1)


def _safe_num_classes(labels: torch.Tensor, logits: torch.Tensor, num_classes: int | None) -> int:
    if num_classes is not None and int(num_classes) > 0:
        return int(num_classes)
    if logits.dim() == 2 and int(logits.size(1)) > 0:
        return int(logits.size(1))
    labels = labels.detach().view(-1).long()
    return int(labels.max().item()) + 1 if int(labels.numel()) > 0 else 1


def _rademacher_projection(in_dim: int, out_dim: int, *, seed: int, device, dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(in_dim) * 1009 + int(out_dim) * 9173)
    signs = torch.randint(0, 2, (int(in_dim), int(out_dim)), generator=generator, dtype=torch.int8)
    proj = signs.to(dtype=torch.float32).mul_(2.0).sub_(1.0)
    proj = proj / math.sqrt(float(max(1, out_dim)))
    return proj.to(device=device, dtype=dtype)


def fed_fishr_gradient_vectors(
    logits: torch.Tensor,
    labels: torch.Tensor,
    features: Optional[torch.Tensor] = None,
    *,
    scope: str = "classifier_head",
    sketch_dim: int = 0,
    seed: int = 0,
) -> torch.Tensor:
    """Return per-sample gradient proxies used by federated Fishr statistics."""
    if logits.dim() != 2:
        raise ValueError("FedFishr requires 2D logits [batch, num_classes].")
    labels = labels.view(-1).long()
    n = min(int(logits.size(0)), int(labels.numel()))
    if n <= 0:
        return logits.new_zeros((0, int(logits.size(1))))
    logits = logits[:n]
    labels = labels[:n]
    prob = F.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(labels.clamp(0, int(logits.size(1)) - 1), num_classes=int(logits.size(1))).to(prob.dtype)
    err = torch.nan_to_num(prob - one_hot, nan=0.0, posinf=0.0, neginf=0.0)
    scope = str(scope or "classifier_head").lower().strip()
    if scope in {"logit", "logits", "classifier_logits"}:
        grad = err
    elif scope in {"classifier_head", "head", "linear_head"}:
        if not torch.is_tensor(features):
            raise ValueError("FedFishr classifier_head scope requires a feature tensor.")
        feat = _as_2d_float(features[:n]).to(device=err.device, dtype=err.dtype)
        grad = (err.unsqueeze(2) * feat.unsqueeze(1)).reshape(n, -1)
    else:
        raise ValueError("FedFishr gradient scope must be one of: classifier_head, logit")
    sketch_dim = int(sketch_dim or 0)
    if sketch_dim > 0 and int(grad.size(1)) > sketch_dim:
        proj = _rademacher_projection(int(grad.size(1)), sketch_dim, seed=int(seed), device=grad.device, dtype=grad.dtype)
        grad = grad @ proj
    return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


def _finalize_stats(stats: Mapping[str, Any], *, min_count: int = 1) -> Dict[str, torch.Tensor]:
    if not stats:
        return {}
    if torch.is_tensor(stats.get("sum")) and torch.is_tensor(stats.get("sq_sum")):
        sum_by_class = stats["sum"].detach().cpu().float()
        sq_sum_by_class = stats["sq_sum"].detach().cpu().float()
        count = stats["count"].detach().cpu().long()
        denom = count.clamp_min(1).view(-1, 1).to(sum_by_class.dtype)
        mean = sum_by_class / denom
        var = (sq_sum_by_class / denom - mean * mean).clamp_min(0.0)
    elif torch.is_tensor(stats.get("var")):
        var = stats["var"].detach().cpu().float()
        count = stats.get("count", torch.zeros(var.size(0), dtype=torch.long)).detach().cpu().long()
        sum_by_class = torch.zeros_like(var)
        sq_sum_by_class = torch.zeros_like(var)
    else:
        return {}
    active = count >= int(min_count)
    var = torch.where(active.view(-1, 1), var, torch.zeros_like(var))
    return {
        "sum": sum_by_class,
        "sq_sum": sq_sum_by_class,
        "count": count,
        "var": var,
        "active_mask": active,
        "payload_bytes": torch.tensor(int(_tensor_payload_bytes(sum_by_class, sq_sum_by_class, count)), dtype=torch.long),
    }


def _tensor_payload_bytes(*tensors: torch.Tensor) -> int:
    total = 0
    for tensor in tensors:
        if torch.is_tensor(tensor):
            total += int(tensor.numel() * tensor.element_size())
    return total


def _stats_payload_bytes(stats: Mapping[str, Any]) -> int:
    if not stats:
        return 0
    payload = stats.get("payload_bytes")
    if torch.is_tensor(payload):
        return int(payload.item())
    if payload is not None:
        try:
            return int(payload)
        except (TypeError, ValueError):
            pass
    return _tensor_payload_bytes(
        stats.get("sum") if torch.is_tensor(stats.get("sum")) else torch.tensor([]),
        stats.get("sq_sum") if torch.is_tensor(stats.get("sq_sum")) else torch.tensor([]),
        stats.get("count") if torch.is_tensor(stats.get("count")) else torch.tensor([]),
    )


def build_fed_fishr_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    features: Optional[torch.Tensor] = None,
    *,
    num_classes: int | None = None,
    scope: str = "classifier_head",
    min_count: int = 2,
    max_samples_per_class: int = 0,
    sketch_dim: int = 0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Build compact per-class gradient-variance statistics for one client batch."""
    num_classes = _safe_num_classes(labels, logits, num_classes)
    grad = fed_fishr_gradient_vectors(
        logits,
        labels,
        features,
        scope=scope,
        sketch_dim=int(sketch_dim or 0),
        seed=int(seed),
    )
    labels = labels.view(-1).long()[: int(grad.size(0))]
    stat_dim = int(grad.size(1)) if grad.dim() == 2 else 0
    sum_by_class = torch.zeros((num_classes, stat_dim), dtype=torch.float32, device=grad.device)
    sq_sum_by_class = torch.zeros_like(sum_by_class)
    count = torch.zeros(num_classes, dtype=torch.long, device=grad.device)
    max_samples = int(max_samples_per_class or 0)
    for cls in range(num_classes):
        mask = labels == int(cls)
        if not bool(mask.any()):
            continue
        idx = torch.nonzero(mask, as_tuple=False).view(-1)
        if max_samples > 0:
            idx = idx[:max_samples]
        if int(idx.numel()) <= 0:
            continue
        g = grad[idx].float()
        sum_by_class[cls] = g.sum(dim=0)
        sq_sum_by_class[cls] = (g * g).sum(dim=0)
        count[cls] = int(idx.numel())
    stats = _finalize_stats(
        {
            "sum": sum_by_class,
            "sq_sum": sq_sum_by_class,
            "count": count,
        },
        min_count=int(min_count),
    )
    stats.update(
        {
            "scope": str(scope or "classifier_head"),
            "stat_dim": int(stat_dim),
            "sketch_dim": int(sketch_dim or 0),
        }
    )
    stats["payload_bytes"] = int(_tensor_payload_bytes(stats["sum"], stats["sq_sum"], stats["count"]))
    return stats


def merge_fed_fishr_stats(
    stats_items: Iterable[Optional[Mapping[str, Any]]],
    *,
    min_count: int = 2,
) -> Optional[Dict[str, Any]]:
    """Merge multiple batch-level FedFishr statistics for one client."""
    accum_sum = None
    accum_sq_sum = None
    accum_count = None
    scope = None
    sketch_dim = 0
    for stats in stats_items:
        finalized = _finalize_stats(stats or {}, min_count=1)
        if not finalized:
            continue
        if accum_sum is None:
            accum_sum = finalized["sum"].clone()
            accum_sq_sum = finalized["sq_sum"].clone()
            accum_count = finalized["count"].clone()
        else:
            if tuple(accum_sum.shape) != tuple(finalized["sum"].shape):
                raise ValueError(f"FedFishr stat shape mismatch: {tuple(accum_sum.shape)} vs {tuple(finalized['sum'].shape)}")
            accum_sum += finalized["sum"]
            accum_sq_sum += finalized["sq_sum"]
            accum_count += finalized["count"]
        scope = scope or stats.get("scope") if isinstance(stats, Mapping) else scope
        sketch_dim = int((stats or {}).get("sketch_dim", sketch_dim) or sketch_dim)
    if accum_sum is None:
        return None
    out = _finalize_stats(
        {
            "sum": accum_sum,
            "sq_sum": accum_sq_sum,
            "count": accum_count,
        },
        min_count=int(min_count),
    )
    out.update({"scope": str(scope or "classifier_head"), "stat_dim": int(out["sum"].size(1)), "sketch_dim": int(sketch_dim)})
    out["payload_bytes"] = int(_tensor_payload_bytes(out["sum"], out["sq_sum"], out["count"]))
    return out


def _previous_target_tensor(previous_target_var: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(previous_target_var):
        return previous_target_var.detach().cpu().float()
    if isinstance(previous_target_var, Mapping) and torch.is_tensor(previous_target_var.get("target_var")):
        return previous_target_var["target_var"].detach().cpu().float()
    return None


def merge_fed_fishr_client_stats(
    client_stats: Mapping[str, Optional[Mapping[str, Any]]],
    *,
    min_clients: int = 2,
    min_count: int = 2,
    momentum: float = 0.0,
    previous_target_var: Any = None,
) -> Dict[str, Any]:
    """Build the server class-conditional Fishr target and per-client mismatch."""
    finalized: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()
    payload_bytes = 0
    for cid, stats in (client_stats or {}).items():
        item = _finalize_stats(stats or {}, min_count=int(min_count))
        if not item:
            continue
        finalized[str(cid)] = item
        payload_bytes += _stats_payload_bytes(stats or item)
    if not finalized:
        return {
            "enabled": True,
            "active": False,
            "inactive_reason": "no_client_stats",
            "client_count": 0,
            "active_classes": 0,
            "payload_bytes": int(payload_bytes),
            "client_mismatch": {},
        }
    first = next(iter(finalized.values()))
    num_classes, stat_dim = int(first["var"].size(0)), int(first["var"].size(1))
    vars_by_client = []
    active_by_client = []
    client_ids = []
    for cid, stats in finalized.items():
        if tuple(stats["var"].shape) != (num_classes, stat_dim):
            raise ValueError(f"FedFishr client stat shape mismatch for {cid}: {tuple(stats['var'].shape)}")
        vars_by_client.append(stats["var"])
        active_by_client.append(stats["active_mask"])
        client_ids.append(cid)
    V = torch.stack(vars_by_client, dim=0)
    active = torch.stack(active_by_client, dim=0).bool()
    active_client_count = active.sum(dim=0)
    target_mask = active_client_count >= int(min_clients)
    active_classes = int(target_mask.sum().item())
    if active_classes <= 0:
        return {
            "enabled": True,
            "active": False,
            "inactive_reason": "insufficient_global_clients_per_class",
            "client_count": int(len(finalized)),
            "active_classes": 0,
            "payload_bytes": int(payload_bytes),
            "active_client_count_by_class": active_client_count,
            "target_mask": target_mask,
            "client_mismatch": {},
        }
    target_var = torch.zeros((num_classes, stat_dim), dtype=torch.float32)
    for cls in range(num_classes):
        if not bool(target_mask[cls]):
            continue
        target_var[cls] = V[active[:, cls], cls, :].mean(dim=0)
    prev = _previous_target_tensor(previous_target_var)
    mom = min(0.999, max(0.0, float(momentum or 0.0)))
    if prev is not None and tuple(prev.shape) == tuple(target_var.shape) and mom > 0.0:
        target_var = torch.where(target_mask.view(-1, 1), mom * prev + (1.0 - mom) * target_var, prev)
    client_mismatch: Dict[str, float] = OrderedDict()
    mismatch_values = []
    for idx, cid in enumerate(client_ids):
        mask = active[idx] & target_mask
        if not bool(mask.any()):
            continue
        diff = V[idx, mask, :] - target_var[mask, :]
        value = float((diff * diff).mean().item())
        if math.isfinite(value):
            client_mismatch[cid] = value
            mismatch_values.append(value)
    mismatch_values_sorted = sorted(mismatch_values)
    p90_idx = int(math.ceil(0.9 * len(mismatch_values_sorted))) - 1 if mismatch_values_sorted else 0
    summary = {
        "enabled": True,
        "active": bool(client_mismatch),
        "inactive_reason": "" if client_mismatch else "no_active_client_mismatch",
        "client_count": int(len(finalized)),
        "active_classes": int(active_classes),
        "payload_bytes": int(payload_bytes),
        "stat_dim": int(stat_dim),
        "target_var": target_var,
        "target_mask": target_mask,
        "active_client_count_by_class": active_client_count,
        "client_mismatch": client_mismatch,
        "mismatch_mean": float(sum(mismatch_values) / max(1, len(mismatch_values))) if mismatch_values else float("nan"),
        "mismatch_min": float(min(mismatch_values)) if mismatch_values else float("nan"),
        "mismatch_max": float(max(mismatch_values)) if mismatch_values else float("nan"),
        "mismatch_p90": float(mismatch_values_sorted[max(0, min(p90_idx, len(mismatch_values_sorted) - 1))]) if mismatch_values_sorted else float("nan"),
        "target_var_mean": float(target_var[target_mask].mean().item()) if active_classes > 0 else float("nan"),
    }
    return summary


def _normalize_weights(raw: Mapping[str, float]) -> OrderedDict[str, float]:
    clean = OrderedDict((str(cid), max(0.0, float(w))) for cid, w in raw.items())
    total = sum(clean.values())
    if total <= 0.0:
        n = max(1, len(clean))
        return OrderedDict((cid, 1.0 / n) for cid in clean)
    return OrderedDict((cid, w / total) for cid, w in clean.items())


def _bounded_simplex(raw: Mapping[str, float], *, floor: float, cap: float) -> OrderedDict[str, float]:
    ids = list(raw.keys())
    if not ids:
        return OrderedDict()
    n = len(ids)
    floor = max(0.0, min(float(floor or 0.0), 1.0 / n))
    cap = max(1.0 / n, min(1.0, float(cap or 1.0)))
    active = set(ids)
    fixed: Dict[str, float] = {}
    remaining = 1.0
    for _ in range(n + 1):
        if not active:
            break
        total = sum(max(0.0, float(raw[cid])) for cid in active)
        proposed = {
            cid: (remaining / len(active) if total <= 0.0 else max(0.0, float(raw[cid])) / total * remaining)
            for cid in active
        }
        changed = False
        for cid, value in list(proposed.items()):
            if value < floor:
                fixed[cid] = floor
                remaining -= floor
                active.remove(cid)
                changed = True
            elif value > cap:
                fixed[cid] = cap
                remaining -= cap
                active.remove(cid)
                changed = True
        if not changed:
            fixed.update(proposed)
            active.clear()
            break
    if active:
        per = max(0.0, remaining) / float(max(1, len(active)))
        for cid in active:
            fixed[cid] = per
    return _normalize_weights(OrderedDict((cid, fixed.get(cid, 0.0)) for cid in ids))


def fed_fishr_reweight(
    base_weights: Mapping[str, float],
    client_mismatch: Mapping[str, float],
    *,
    alpha: float = 0.0,
    floor: float = 0.0,
    cap: float = 1.0,
) -> tuple[OrderedDict[str, float], Dict[str, Any]]:
    """Downweight clients whose class-conditional gradient variance is far from the server target."""
    base = _normalize_weights(base_weights)
    alpha = max(0.0, float(alpha or 0.0))
    finite_mismatch = {
        cid: max(0.0, float(client_mismatch.get(cid, 0.0)))
        for cid in base
        if math.isfinite(float(client_mismatch.get(cid, 0.0)))
    }
    if alpha <= 0.0 or not finite_mismatch:
        return base, {
            "active": False,
            "inactive_reason": "zero_alpha_or_no_mismatch",
            "weights": dict(base),
            "base_weights": dict(base),
            "max_delta": 0.0,
        }
    scale = sum(finite_mismatch.values()) / max(1, len(finite_mismatch))
    scale = max(scale, 1e-12)
    raw = OrderedDict()
    for cid, weight in base.items():
        mismatch = max(0.0, float(client_mismatch.get(cid, 0.0) or 0.0))
        raw[cid] = float(weight) * math.exp(-alpha * (mismatch / scale))
    weights = _bounded_simplex(raw, floor=float(floor or 0.0), cap=float(cap or 1.0))
    deltas = [abs(float(weights[cid]) - float(base[cid])) for cid in base]
    summary = {
        "active": True,
        "alpha": float(alpha),
        "floor": float(floor or 0.0),
        "cap": float(cap or 1.0),
        "weights": dict(weights),
        "base_weights": dict(base),
        "weight_min": float(min(weights.values())) if weights else float("nan"),
        "weight_max": float(max(weights.values())) if weights else float("nan"),
        "max_delta": float(max(deltas)) if deltas else 0.0,
    }
    return weights, summary


def fed_fishr_target_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    features: Optional[torch.Tensor],
    *,
    target_var: torch.Tensor,
    target_mask: torch.Tensor,
    scope: str = "classifier_head",
    min_count: int = 2,
    max_samples_per_class: int = 0,
    sketch_dim: int = 0,
    seed: int = 0,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Differentiable local loss against a server FedFishr variance target."""
    zero = logits.new_tensor(0.0)
    if not torch.is_tensor(target_var) or not torch.is_tensor(target_mask):
        return zero, {"active_classes": 0, "mean_dist": float("nan"), "skip_rate": 1.0}
    num_classes = int(target_var.size(0))
    grad = fed_fishr_gradient_vectors(
        logits,
        labels,
        features,
        scope=scope,
        sketch_dim=int(sketch_dim or 0),
        seed=int(seed),
    )
    labels = labels.view(-1).long()[: int(grad.size(0))]
    max_samples = int(max_samples_per_class or 0)
    losses = []
    dists = []
    active_classes = 0
    target = target_var.to(device=grad.device, dtype=grad.dtype)
    mask_target = target_mask.to(device=grad.device).bool().view(-1)
    for cls in range(num_classes):
        if cls >= int(mask_target.numel()) or not bool(mask_target[cls]):
            continue
        idx = torch.nonzero(labels == int(cls), as_tuple=False).view(-1)
        if max_samples > 0:
            idx = idx[:max_samples]
        if int(idx.numel()) < int(min_count):
            continue
        g = grad[idx].float()
        var = g.var(dim=0, unbiased=False)
        dist = (var - target[cls]) ** 2
        losses.append(dist.mean())
        dists.append(float(dist.detach().mean().item()))
        active_classes += 1
    if not losses:
        return zero, {"active_classes": 0, "mean_dist": float("nan"), "skip_rate": 1.0}
    loss = torch.stack(losses).mean()
    skip_rate = 1.0 - float(active_classes) / float(max(1, int(mask_target.sum().item())))
    return loss, {
        "active_classes": int(active_classes),
        "mean_dist": float(sum(dists) / max(1, len(dists))) if dists else float("nan"),
        "skip_rate": float(max(0.0, min(1.0, skip_rate))),
    }


class FedFishrBank:
    def __init__(self, *, min_clients: int = 2, min_count: int = 2, momentum: float = 0.0):
        self.min_clients = int(min_clients)
        self.min_count = int(min_count)
        self.momentum = float(momentum or 0.0)
        self.target_var: Optional[torch.Tensor] = None
        self.target_mask: Optional[torch.Tensor] = None
        self.summary: Dict[str, Any] = {"enabled": True, "active": False, "inactive_reason": "not_updated"}

    def update(self, client_stats: Mapping[str, Optional[Mapping[str, Any]]]) -> Dict[str, Any]:
        summary = merge_fed_fishr_client_stats(
            client_stats,
            min_clients=self.min_clients,
            min_count=self.min_count,
            momentum=self.momentum,
            previous_target_var=self.target_var,
        )
        if torch.is_tensor(summary.get("target_var")) and bool(summary.get("active", False)):
            self.target_var = summary["target_var"].detach().cpu().float()
            self.target_mask = summary.get("target_mask").detach().cpu().bool() if torch.is_tensor(summary.get("target_mask")) else None
        self.summary = summary
        return summary

    def tensors(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.target_var, self.target_mask

    def log_summary(self) -> Dict[str, Any]:
        return fed_fishr_log_summary(self.summary)


def fed_fishr_log_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in (summary or {}).items():
        if key in {"target_var"}:
            continue
        if torch.is_tensor(value):
            if value.numel() <= 32:
                out[key] = value.detach().cpu().tolist()
            else:
                out[f"{key}_shape"] = list(value.shape)
            continue
        if isinstance(value, Mapping):
            out[key] = {str(k): float(v) if isinstance(v, (float, int)) else v for k, v in value.items()}
        else:
            out[key] = value
    return out

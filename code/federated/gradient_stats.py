from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Dict, Mapping

import torch


def _flatten_gradients(
    gradients: Mapping[str, torch.Tensor],
    keys: list[str],
    refs: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    values = []
    for key in keys:
        grad = gradients.get(key)
        if torch.is_tensor(grad):
            values.append(grad.detach().cpu().float().reshape(-1))
        else:
            values.append(torch.zeros_like(refs[key].detach().cpu().float()).reshape(-1))
    if not values:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(values, dim=0)


def _unflatten_like(flat: torch.Tensor, ref: Mapping[str, torch.Tensor], keys: list[str]) -> OrderedDict[str, torch.Tensor]:
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    offset = 0
    for key in keys:
        value = ref[key]
        numel = int(value.numel())
        out[key] = flat[offset : offset + numel].view_as(value.detach().cpu()).to(dtype=value.detach().cpu().dtype)
        offset += numel
    return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    denom = a.norm().clamp_min(1e-12) * b.norm().clamp_min(1e-12)
    return float(torch.dot(a.float(), b.float()).div(denom).item())


def _pairwise_cosine(values: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    ids = list(values.keys())
    if len(ids) < 2:
        return {"pairs": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    cosines = []
    for i, cid_a in enumerate(ids):
        for cid_b in ids[i + 1 :]:
            cosines.append(_cosine(values[cid_a], values[cid_b]))
    finite = [v for v in cosines if math.isfinite(v)]
    if not finite:
        return {"pairs": int(len(cosines)), "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "pairs": int(len(cosines)),
        "mean": float(sum(finite) / len(finite)),
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def _aggregate_flat(flat_by_client: Mapping[str, torch.Tensor], weights: Mapping[str, float]) -> torch.Tensor:
    if not flat_by_client:
        raise ValueError("flat_by_client must not be empty")
    first = next(iter(flat_by_client.values()))
    acc = torch.zeros_like(first.detach().cpu().float())
    for cid, flat in flat_by_client.items():
        acc.add_(flat.detach().cpu().float(), alpha=float(weights.get(cid, 0.0)))
    return acc


def _gradient_key_refs(client_gradients: Mapping[str, Mapping[str, torch.Tensor]]) -> OrderedDict[str, torch.Tensor]:
    if not client_gradients:
        raise ValueError("client_gradients must not be empty")
    refs: OrderedDict[str, torch.Tensor] = OrderedDict()
    for gradients in client_gradients.values():
        for key, grad in gradients.items():
            if torch.is_tensor(grad) and key not in refs:
                refs[str(key)] = grad.detach().cpu()
    if not refs:
        raise ValueError("client_gradients do not contain tensor gradients")
    ordered = OrderedDict((key, refs[key]) for key in sorted(refs))
    for key, ref in ordered.items():
        ref_shape = tuple(ref.shape)
        for cid, gradients in client_gradients.items():
            grad = gradients.get(key)
            if grad is None:
                continue
            if not torch.is_tensor(grad):
                continue
            if tuple(grad.shape) != ref_shape:
                raise ValueError(f"Gradient shape mismatch for {key} from client {cid}")
    return ordered


def _missing_gradient_entries(client_gradients: Mapping[str, Mapping[str, torch.Tensor]], keys: list[str]) -> int:
    missing = 0
    for gradients in client_gradients.values():
        for key in keys:
            if not torch.is_tensor(gradients.get(key)):
                missing += 1
    return int(missing)


def _cosine_clip(flat_by_client: Mapping[str, torch.Tensor], weights: Mapping[str, float]) -> tuple[Dict[str, torch.Tensor], int]:
    reference = _aggregate_flat(flat_by_client, weights)
    denom = torch.dot(reference, reference).clamp_min(1e-12)
    corrected: Dict[str, torch.Tensor] = {}
    resolved = 0
    for cid, flat in flat_by_client.items():
        dot = torch.dot(flat.float(), reference.float())
        if float(dot.item()) < 0.0:
            corrected[cid] = flat.float() - dot / denom * reference.float()
            resolved += 1
        else:
            corrected[cid] = flat.float().clone()
    return corrected, resolved


def _pcgrad(flat_by_client: Mapping[str, torch.Tensor]) -> tuple[Dict[str, torch.Tensor], int]:
    ids = list(flat_by_client.keys())
    corrected = {cid: flat.detach().cpu().float().clone() for cid, flat in flat_by_client.items()}
    resolved = 0
    for cid in ids:
        grad = corrected[cid]
        for other_id in ids:
            if cid == other_id:
                continue
            other = corrected[other_id].detach()
            dot = torch.dot(grad, other)
            if float(dot.item()) < 0.0:
                grad = grad - dot / torch.dot(other, other).clamp_min(1e-12) * other
                resolved += 1
        corrected[cid] = grad
    return corrected, resolved


def conflict_aware_aggregate_gradients(
    client_gradients: Mapping[str, Mapping[str, torch.Tensor]],
    weights: Mapping[str, float],
    *,
    mode: str = "none",
) -> tuple[OrderedDict[str, torch.Tensor], Dict[str, Any]]:
    """Aggregate client gradients and optionally remove negative cross-client components."""
    key_refs = _gradient_key_refs(client_gradients)
    keys = list(key_refs.keys())
    missing_entries = _missing_gradient_entries(client_gradients, keys)
    flat_by_client = {str(cid): _flatten_gradients(grads, keys, key_refs) for cid, grads in client_gradients.items()}
    before = _pairwise_cosine(flat_by_client)
    conflicts_detected = int(sum(1 for v in [before.get("min", float("nan"))] if math.isfinite(v) and v < 0.0))

    mode_norm = str(mode or "none").lower()
    if mode_norm in {"", "none", "off"}:
        corrected = flat_by_client
        conflicts_resolved = 0
        mode_norm = "none"
    elif mode_norm == "cosine_clip":
        corrected, conflicts_resolved = _cosine_clip(flat_by_client, weights)
    elif mode_norm == "pcgrad":
        corrected, conflicts_resolved = _pcgrad(flat_by_client)
    else:
        raise ValueError("--fl_conflict_agg must be one of: none, cosine_clip, pcgrad")

    agg_flat = _aggregate_flat(corrected, weights)
    aggregated = _unflatten_like(agg_flat, key_refs, keys)
    after = _pairwise_cosine(corrected)
    raw_agg = _aggregate_flat(flat_by_client, weights)
    metrics = {
        "conflict_mode": mode_norm,
        "clients": int(len(flat_by_client)),
        "gradient_keys": int(len(keys)),
        "missing_gradient_entries": int(missing_entries),
        "conflicts_detected": int(conflicts_detected),
        "conflicts_resolved": int(conflicts_resolved),
        "grad_cos_pairs": int(before.get("pairs", 0)),
        "grad_cos_mean_before": float(before.get("mean", float("nan"))),
        "grad_cos_min_before": float(before.get("min", float("nan"))),
        "grad_cos_mean_after": float(after.get("mean", float("nan"))),
        "grad_cos_min_after": float(after.get("min", float("nan"))),
        "grad_norm_before": float(raw_agg.norm().item()),
        "grad_norm_after": float(agg_flat.norm().item()),
    }
    return aggregated, metrics

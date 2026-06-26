from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unpack_batch(batch):
    x = batch[0]
    y = batch[1]
    extra = batch[2:] if isinstance(batch, (tuple, list)) and len(batch) > 2 else ()
    return x, y, extra


def extract_domain_from_extra(extra, device) -> Optional[torch.Tensor]:
    if extra is None or len(extra) == 0:
        return None
    d0 = extra[0]
    if torch.is_tensor(d0):
        return d0.to(device, non_blocking=True).view(-1)
    try:
        return torch.as_tensor(d0, device=device).view(-1)
    except Exception:
        return None


def parse_csv_indices(s: str):
    s = str(s).strip()
    if s == "":
        return None
    out = []
    for item in s.split(","):
        item = item.strip()
        if item == "":
            continue
        try:
            out.append(int(item))
        except Exception:
            out.append(item)
    return out if len(out) > 0 else None


def parse_float_csv(s: str, default: Optional[List[float]] = None) -> List[float]:
    ss = str(s).strip()
    if ss == "":
        return list(default or [])
    out = []
    for item in ss.split(","):
        item = item.strip()
        if item == "":
            continue
        out.append(float(item))
    return out if len(out) > 0 else list(default or [])


def sample_strength_from_tiers(batch_size: int, tiers: List[float], device, dtype=torch.float32) -> torch.Tensor:
    if tiers is None or len(tiers) == 0:
        return torch.rand((batch_size,), device=device, dtype=dtype)
    vals = torch.as_tensor(tiers, device=device, dtype=dtype).clamp(0.0, 1.0)
    idx = torch.randint(low=0, high=int(vals.numel()), size=(batch_size,), device=device)
    return vals[idx]


def safe_l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    n = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True).clamp_min(float(eps))
    return x / n


def safe_cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return (safe_l2_normalize(a, dim=dim, eps=eps) * safe_l2_normalize(b, dim=dim, eps=eps)).sum(dim=dim).clamp(-1.0, 1.0)


def safe_batch_var(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size(dim) <= 1:
        return torch.zeros_like(x.mean(dim=dim))
    return x.var(dim=dim, unbiased=False).clamp_min(float(eps))


def safe_batch_std(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
    return safe_batch_var(x, dim=dim, eps=eps).sqrt()


def safe_iq_tensor(x: torch.Tensor, clamp: float = 8.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=float(clamp), neginf=-float(clamp)).clamp(-float(clamp), float(clamp))


def batch_domain_stats(d: Optional[torch.Tensor], y: torch.Tensor, num_domains: int) -> Dict[str, Any]:
    valid = None
    if d is not None:
        valid = d.view(-1).long() >= 0
    if d is None or valid is None or not bool(valid.any()):
        return {"valid": valid, "num_valid": 0, "num_domains": 0, "domain_frac": 0.0, "has_cross_pairs": False}
    d_valid = d.view(-1).long()[valid]
    y_valid = y.view(-1).long()[valid]
    uniq_d = torch.unique(d_valid)
    has_cross_pairs = False
    for cls in torch.unique(y_valid):
        if torch.unique(d_valid[y_valid == cls]).numel() >= 2:
            has_cross_pairs = True
            break
    return {
        "valid": valid,
        "num_valid": int(d_valid.numel()),
        "num_domains": int(uniq_d.numel()),
        "domain_frac": float(uniq_d.numel()) / float(max(1, num_domains)),
        "has_cross_pairs": bool(has_cross_pairs),
    }


def make_torch_generator(device, seed: int):
    try:
        gen = torch.Generator(device=device)
    except Exception:
        gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def get_nested_tensor(out: Dict[str, Any], top_key: str, nested_group: str, nested_key: str) -> torch.Tensor:
    v = out.get(top_key, None)
    if torch.is_tensor(v):
        return v
    aux = out.get(nested_group, {})
    v = aux.get(nested_key, None) if isinstance(aux, dict) else None
    if not torch.is_tensor(v):
        raise KeyError(f"Cannot find tensor {top_key} / {nested_group}.{nested_key}")
    return v


def remap_domain_tensor(d: Optional[torch.Tensor], domain_label_map: Dict[int, int], device) -> Optional[torch.Tensor]:
    """Map raw WiSig domain labels to compact train-domain labels.

    Returns -1 for raw labels not present in training domains. Those samples are
    ignored by domain accuracy in evaluation and should not enter domain CE.
    """
    if d is None:
        return None
    out = torch.full_like(d.view(-1).long(), fill_value=-1, device=device)
    for raw, mapped in domain_label_map.items():
        out[d.view(-1).long() == int(raw)] = int(mapped)
    return out


def unwrap_wisig_dataset(dataset):
    """Return the underlying Wi-Sig dataset-like object from Subset/Concat wrappers."""

    cur = dataset
    visited = set()
    while True:
        oid = id(cur)
        if oid in visited:
            break
        visited.add(oid)
        if hasattr(cur, "index") and hasattr(cur, "_domain_lut"):
            return cur
        if hasattr(cur, "dataset"):
            cur = cur.dataset
            continue
        break
    return dataset


def build_domain_label_map(dataset) -> Dict[int, int]:
    """Build raw-domain to compact train-domain mapping from the train split."""

    obj = unwrap_wisig_dataset(dataset)
    if not (hasattr(obj, "index") and hasattr(obj, "_domain_lut")):
        return {}
    raw_labels = sorted({int(obj._domain_lut[(it.rx_i, it.day_i)]) for it in obj.index})
    return {raw: idx for idx, raw in enumerate(raw_labels)}

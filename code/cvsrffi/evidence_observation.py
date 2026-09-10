"""Partial observations: a missing coordinate is never a measured zero."""
from dataclasses import dataclass
from typing import Mapping, Tuple
import math

import torch


@dataclass(frozen=True)
class EvidenceObservation:
    z: torch.Tensor
    observed: torch.Tensor
    block_sizes: Tuple[int, ...]


def normalize_observed(z, observed, block_sizes):
    """Fixed block units, never re-normalize after marginalizing dimensions.

    sqrt(block width) is a data-independent constant. Thus masking one coordinate
    cannot change any retained coordinate, including when missing data are NaN.
    """
    if z.ndim != 2 or observed.shape != z.shape or observed.dtype != torch.bool:
        raise ValueError("z and bool observed must have identical [B,D] shape")
    if sum(block_sizes) != z.shape[1] or any(d <= 0 for d in block_sizes):
        raise ValueError("invalid block schema")
    if not torch.isfinite(z[observed]).all():
        raise ValueError("nonfinite observed evidence")
    parts = []
    start = 0
    for width in block_sizes:
        part = torch.where(observed[:, start:start+width], z[:, start:start+width], 0.)
        parts.append(part / math.sqrt(width))
        start += width
    return torch.cat(parts, -1)


def extract_evidence(aux: Mapping, layout="joint", observed=None):
    if layout == "joint":
        parts = [aux["feat_joint"]]
    elif layout == "blocks":
        parts = [aux[key] for key in ("t_emb", "f_emb", "pa_local")]
    else:
        raise ValueError("layout must be joint or blocks")
    z = torch.cat(parts, -1).float()
    mask = torch.ones_like(z, dtype=torch.bool) if observed is None else observed
    sizes = tuple(p.shape[-1] for p in parts)
    return EvidenceObservation(normalize_observed(z, mask, sizes), mask, sizes)


def physical_support_ids(ids):
    ids = tuple(str(item) for item in ids)
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError("support requires unique nonempty physical sample IDs; views are not shots")
    return ids

from enum import IntEnum

import torch


class TailRegion(IntEnum):
    CORE = 0
    SOFT_TAIL = 1
    EXTREME_TAIL = 2
    OUTSIDE = 3


def partition_tail_regions(
    distances_deg: torch.Tensor,
    *,
    r_core_deg: float,
    r_accept_deg: float,
    r_tail_deg: float,
) -> torch.Tensor:
    d = distances_deg.float()
    regions = torch.full_like(d, int(TailRegion.OUTSIDE), dtype=torch.long)
    regions[d <= float(r_tail_deg)] = int(TailRegion.EXTREME_TAIL)
    regions[d <= float(r_accept_deg)] = int(TailRegion.SOFT_TAIL)
    regions[d <= float(r_core_deg)] = int(TailRegion.CORE)
    return regions


def region_to_ce_weights(
    regions: torch.Tensor,
    *,
    core_weight: float = 1.0,
    soft_tail_weight: float = 0.25,
    extreme_tail_weight: float = 0.05,
    outside_weight: float = 0.0,
) -> torch.Tensor:
    r = regions.long()
    w = torch.full(r.shape, float(outside_weight), dtype=torch.float32, device=r.device)
    w[r == int(TailRegion.EXTREME_TAIL)] = float(extreme_tail_weight)
    w[r == int(TailRegion.SOFT_TAIL)] = float(soft_tail_weight)
    w[r == int(TailRegion.CORE)] = float(core_weight)
    return w


def tail_cvar_loss(distances_deg: torch.Tensor, r_target_deg: float, top_frac: float = 0.05) -> torch.Tensor:
    overflow = torch.relu(distances_deg.float() - float(r_target_deg)).pow(2)
    if overflow.numel() == 0:
        return distances_deg.float().sum() * 0.0
    k = max(1, int(round(float(top_frac) * overflow.numel())))
    return torch.topk(overflow.view(-1), k=min(k, overflow.numel())).values.mean()


def overflow_cap_loss(distances_deg: torch.Tensor, r_accept_deg: float, region: torch.Tensor | None = None) -> torch.Tensor:
    overflow = torch.relu(distances_deg.float() - float(r_accept_deg)).pow(2)
    if region is not None:
        w = torch.ones_like(overflow)
        w[region.long() == int(TailRegion.EXTREME_TAIL)] = 0.10
        w[region.long() == int(TailRegion.OUTSIDE)] = 0.0
        overflow = overflow * w
    return overflow.mean() if overflow.numel() else distances_deg.float().sum() * 0.0


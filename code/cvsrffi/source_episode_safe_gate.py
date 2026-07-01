from dataclasses import dataclass

import torch

from cvsrffi.component_geometry import angular_distance_deg


@dataclass
class SourceEpisodePartition:
    known_query_mask: torch.Tensor
    uncertain_query_mask: torch.Tensor
    overflow_query_mask: torch.Tensor
    debug: dict


def source_episode_safe_partition(z: torch.Tensor, labels: torch.Tensor, bank, radius_eps_deg: float = 0.25) -> SourceEpisodePartition:
    known = torch.zeros(z.size(0), dtype=torch.bool, device=z.device)
    uncertain = torch.zeros(z.size(0), dtype=torch.bool, device=z.device)
    overflow = torch.zeros(z.size(0), dtype=torch.bool, device=z.device)
    for i in range(z.size(0)):
        cls = int(labels.view(-1)[i].item())
        try:
            comp = bank.nearest_own_component(z[i], cls)
        except KeyError:
            overflow[i] = True
            continue
        d = float(angular_distance_deg(z[i], comp.mu).item())
        density = bank.knn_density(z[i], comp)
        density_fail = density is not None and comp.density_tail_min is not None and density < comp.density_tail_min
        eps = float(radius_eps_deg)
        if d <= float(comp.r_core_deg) + eps and not density_fail:
            known[i] = True
        elif d <= float(comp.r_accept_deg) + eps and not density_fail:
            uncertain[i] = True
        else:
            overflow[i] = True
    return SourceEpisodePartition(
        known_query_mask=known,
        uncertain_query_mask=uncertain,
        overflow_query_mask=overflow,
        debug={
            "source_ep_known_query_frac": float(known.float().mean().item()) if z.numel() else 0.0,
            "source_ep_uncertain_query_frac": float(uncertain.float().mean().item()) if z.numel() else 0.0,
            "source_ep_overflow_query_frac": float(overflow.float().mean().item()) if z.numel() else 0.0,
        },
    )

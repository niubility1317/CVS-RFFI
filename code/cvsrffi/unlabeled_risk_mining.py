from dataclasses import dataclass

import torch

from cvsrffi.component_geometry import angular_distance_deg


@dataclass
class UnlabeledRiskConfig:
    pseudo_known_maxprob_min: float = 0.90
    pseudo_known_requires_density: bool = True
    risk_maxprob_min: float = 0.70
    risk_geo_margin_min_deg: float = 2.0


@dataclass
class UnlabeledMiningResult:
    pseudo_known_mask: torch.Tensor
    risk_mask: torch.Tensor
    ignore_mask: torch.Tensor
    pseudo_labels: torch.Tensor
    risk_scores: torch.Tensor
    debug: dict


def mine_unlabeled_risk(z: torch.Tensor, logits: torch.Tensor, bank, cfg: UnlabeledRiskConfig) -> UnlabeledMiningResult:
    prob = logits.float().softmax(dim=-1)
    max_prob, pred = prob.max(dim=-1)
    pseudo = torch.zeros(z.size(0), dtype=torch.bool, device=z.device)
    risk = torch.zeros(z.size(0), dtype=torch.bool, device=z.device)
    scores = torch.zeros(z.size(0), dtype=torch.float32, device=z.device)
    for i in range(z.size(0)):
        cls = int(pred[i].item())
        try:
            own = bank.nearest_own_component(z[i], cls)
            other = bank.nearest_other_component(z[i], cls)
        except KeyError:
            continue
        d_own = float(angular_distance_deg(z[i], own.mu).item())
        d_other = float(angular_distance_deg(z[i], other.mu).item())
        geo_margin = d_other - d_own
        density = bank.knn_density(z[i], own)
        density_pass = True
        if cfg.pseudo_known_requires_density and density is not None and own.density_tail_min is not None:
            density_pass = density >= own.density_tail_min
        in_core = d_own <= float(own.r_core_deg)
        in_accept = d_own <= float(own.r_accept_deg)
        if float(max_prob[i].item()) >= float(cfg.pseudo_known_maxprob_min) and in_core and density_pass and geo_margin >= float(cfg.risk_geo_margin_min_deg):
            pseudo[i] = True
        elif float(max_prob[i].item()) >= float(cfg.risk_maxprob_min) and ((not in_core) or (not in_accept) or (not density_pass) or geo_margin < float(cfg.risk_geo_margin_min_deg)):
            risk[i] = True
            scores[i] = float(max_prob[i].item()) + max(0.0, d_own - float(own.r_core_deg)) / 180.0
    ignore = ~(pseudo | risk)
    return UnlabeledMiningResult(
        pseudo_known_mask=pseudo,
        risk_mask=risk,
        ignore_mask=ignore,
        pseudo_labels=pred,
        risk_scores=scores,
        debug={
            "unl_pseudo_core_count": int(pseudo.sum().item()),
            "unl_risk_count": int(risk.sum().item()),
            "unl_ignore_count": int(ignore.sum().item()),
        },
    )


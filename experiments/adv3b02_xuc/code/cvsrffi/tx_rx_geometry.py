from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(x, dim=dim, eps=1e-6)


def _zero(ref: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(ref) and ref.requires_grad:
        return torch.nan_to_num(ref.float(), nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
    return torch.zeros((), device=ref.device if torch.is_tensor(ref) else None, dtype=torch.float32)


def pair_masks(y_tx: torch.Tensor, d: torch.Tensor) -> Dict[str, torch.Tensor]:
    y = y_tx.view(-1).long()
    dom = d.view(-1).long()
    if y.numel() != dom.numel():
        raise ValueError("y_tx and d must have the same length")
    eye = torch.eye(y.numel(), device=y.device, dtype=torch.bool)
    valid = (y >= 0).view(-1, 1) & (dom >= 0).view(-1, 1)
    valid_pair = valid & valid.t() & (~eye)
    same_tx = y.view(-1, 1).eq(y.view(1, -1))
    same_domain = dom.view(-1, 1).eq(dom.view(1, -1))
    return {
        "same_tx_cross_domain": same_tx & (~same_domain) & valid_pair,
        "same_domain_cross_tx": same_domain & (~same_tx) & valid_pair,
        "different_tx": (~same_tx) & valid_pair,
        "different_domain": (~same_domain) & valid_pair,
        "valid_pair": valid_pair,
    }


def masked_supcon_loss(
    z: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    temperature: float = 0.12,
) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError("z must have shape [N, D]")
    n = z.size(0)
    if positive_mask.shape != (n, n):
        raise ValueError("positive_mask must have shape [N, N]")
    if valid_mask is None:
        valid_mask = torch.ones((n, n), device=z.device, dtype=torch.bool)
        valid_mask.fill_diagonal_(False)
    elif valid_mask.shape != (n, n):
        raise ValueError("valid_mask must have shape [N, N]")
    if n <= 1:
        return _zero(z)
    feat = _norm(z, dim=1)
    logits = (feat @ feat.t()) / max(1e-4, float(temperature))
    logits = logits - logits.detach().max(dim=1, keepdim=True).values
    pos = positive_mask.to(device=z.device).bool() & valid_mask.to(device=z.device).bool()
    denom = valid_mask.to(device=z.device).bool()
    has_pos = pos.sum(dim=1) > 0
    if not bool(has_pos.any()):
        return _zero(z)
    exp_logits = torch.exp(logits) * denom.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    pos_log_prob = (log_prob * pos.float()).sum(dim=1) / pos.float().sum(dim=1).clamp_min(1.0)
    return -pos_log_prob[has_pos].mean()


def _cell_centers(z: torch.Tensor, y_tx: torch.Tensor, d: torch.Tensor) -> Dict[tuple[int, int], torch.Tensor]:
    centers: Dict[tuple[int, int], torch.Tensor] = {}
    y = y_tx.view(-1).long().to(z.device)
    dom = d.view(-1).long().to(z.device)
    for tx in torch.unique(y[y >= 0]):
        for rx in torch.unique(dom[(y == tx) & (dom >= 0)]):
            m = (y == tx) & (dom == rx)
            if bool(m.any()):
                centers[(int(tx.item()), int(rx.item()))] = _norm(z[m].mean(dim=0, keepdim=True), dim=1).squeeze(0)
    return centers


def tx_rx_rectangle_identity_loss(z_tx: torch.Tensor, y_tx: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, int]:
    centers = _cell_centers(z_tx, y_tx, d)
    tx_values = sorted({k[0] for k in centers})
    d_values = sorted({k[1] for k in centers})
    losses = []
    for tx1, tx2 in combinations(tx_values, 2):
        for d1, d2 in combinations(d_values, 2):
            keys = [(tx1, d1), (tx1, d2), (tx2, d1), (tx2, d2)]
            if not all(k in centers for k in keys):
                continue
            diff1 = centers[(tx1, d1)] - centers[(tx2, d1)]
            diff2 = centers[(tx1, d2)] - centers[(tx2, d2)]
            losses.append(1.0 - F.cosine_similarity(diff1.view(1, -1), diff2.view(1, -1), dim=1).squeeze(0))
    if not losses:
        return _zero(z_tx), 0
    return torch.stack(losses).mean(), len(losses)


def tx_rx_rectangle_receiver_loss(z_rx: torch.Tensor, y_tx: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, int]:
    centers = _cell_centers(z_rx, y_tx, d)
    tx_values = sorted({k[0] for k in centers})
    d_values = sorted({k[1] for k in centers})
    losses = []
    for tx1, tx2 in combinations(tx_values, 2):
        for d1, d2 in combinations(d_values, 2):
            keys = [(tx1, d1), (tx1, d2), (tx2, d1), (tx2, d2)]
            if not all(k in centers for k in keys):
                continue
            diff1 = centers[(tx1, d1)] - centers[(tx1, d2)]
            diff2 = centers[(tx2, d1)] - centers[(tx2, d2)]
            losses.append(1.0 - F.cosine_similarity(diff1.view(1, -1), diff2.view(1, -1), dim=1).squeeze(0))
    if not losses:
        return _zero(z_rx), 0
    return torch.stack(losses).mean(), len(losses)


def tx_rx_anova_metrics(z: torch.Tensor, y_tx: torch.Tensor, d: torch.Tensor) -> Dict[str, float]:
    if z.ndim != 2:
        raise ValueError("z must have shape [N, D]")
    if z.size(0) == 0:
        return {"var_total": 0.0, "var_tx_ratio": float("nan"), "var_rx_ratio": float("nan")}
    feat = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
    total = feat.var(dim=0, unbiased=False).sum().clamp_min(1e-12)
    y = y_tx.view(-1).long().to(feat.device)
    dom = d.view(-1).long().to(feat.device)

    def between_var(labels: torch.Tensor) -> torch.Tensor:
        centers = []
        weights = []
        for val in torch.unique(labels[labels >= 0]):
            m = labels == val
            if bool(m.any()):
                centers.append(feat[m].mean(dim=0))
                weights.append(float(m.float().mean().item()))
        if not centers:
            return feat.new_tensor(0.0)
        C = torch.stack(centers, dim=0)
        W = torch.tensor(weights, device=feat.device, dtype=feat.dtype).view(-1, 1)
        mean = feat.mean(dim=0, keepdim=True)
        return (W * (C - mean).pow(2)).sum()

    tx_var = between_var(y)
    rx_var = between_var(dom)
    inter = (total - tx_var - rx_var).clamp_min(0.0)
    return {
        "var_total": float(total.detach().item()),
        "var_tx_ratio": float((tx_var / total).detach().item()),
        "var_rx_ratio": float((rx_var / total).detach().item()),
        "var_interaction_ratio": float((inter / total).detach().item()),
    }


def domain_shift_losses(
    tx_domain_bank,
    tx_bank,
    domain_bank=None,
    shift_predictor=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    shifts = tx_domain_bank.compute_domain_shifts(tx_bank)
    delta = shifts["delta"]
    mask = shifts["mask"]
    if not bool(mask.any()):
        return _zero(delta), {"shift_cons": float("nan"), "interaction": float("nan"), "shift_pred": float("nan")}

    cons_losses = []
    for domain in range(delta.size(1)):
        active = mask[:, domain]
        if int(active.sum().item()) <= 1:
            continue
        D = _norm(delta[active, domain], dim=1)
        sim = D @ D.t()
        iu = torch.triu_indices(sim.size(0), sim.size(1), offset=1, device=sim.device)
        cons_losses.append((1.0 - sim[iu[0], iu[1]]).mean())
    shift_cons = torch.stack(cons_losses).mean() if cons_losses else _zero(delta)
    interaction = shifts["interaction"][mask].pow(2).sum(dim=1).mean()
    shift_pred = _zero(delta)
    if domain_bank is not None and shift_predictor is not None:
        domain_proto = domain_bank.get().to(delta.device)
        pred = shift_predictor(domain_proto)
        target = shifts["domain_shift"].detach()
        active_domains = shifts["domain_counts"] > 0
        if bool(active_domains.any()):
            shift_pred = (1.0 - F.cosine_similarity(pred[active_domains], target[active_domains], dim=1)).mean()
    loss = shift_cons + 0.5 * interaction + shift_pred
    return loss, {
        "shift_cons": float(shift_cons.detach().item()) if torch.isfinite(shift_cons.detach()).all() else float("nan"),
        "interaction": float(interaction.detach().item()) if torch.isfinite(interaction.detach()).all() else float("nan"),
        "shift_pred": float(shift_pred.detach().item()) if torch.isfinite(shift_pred.detach()).all() else float("nan"),
        "active_tx_domain": float(mask.sum().detach().item()),
    }

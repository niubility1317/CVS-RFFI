"""Pair geometry and falsification diagnostics for CCOI-PA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor
import torch.nn.functional as F

from .tx_rx_geometry import masked_supcon_loss, pair_masks, tx_rx_rectangle_identity_loss


@dataclass
class ChallengePairMasks:
    positive: Tensor
    negative: Tensor
    valid: Tensor


@dataclass
class CCOILossOutput:
    loss: Tensor
    positive_count: int
    negative_count: int
    anchor_count: int


def _pair_cosine(q_summary: Tensor) -> Tensor:
    if q_summary.ndim != 2:
        raise ValueError("q_summary must have shape [N,D]")
    q_norm = F.normalize(torch.nan_to_num(q_summary.float()), dim=1, eps=1e-6)
    return q_norm @ q_norm.t()


def ordinary_pair_masks(y_tx: Tensor, domain: Tensor) -> ChallengePairMasks:
    base = pair_masks(y_tx, domain)
    positive = base["same_tx_cross_domain"]
    negative = base["same_domain_cross_tx"]
    return ChallengePairMasks(positive=positive, negative=negative, valid=positive | negative)


def challenge_pair_masks(
    q_summary: Tensor,
    y_tx: Tensor,
    domain: Tensor,
    min_cosine: float = 0.70,
) -> ChallengePairMasks:
    """Build proxy challenge-matched M2 pairs without semantic-content claims."""

    base = pair_masks(y_tx, domain)
    matched = _pair_cosine(q_summary) >= float(min_cosine)
    positive = base["same_tx_cross_domain"] & matched
    negative = base["same_domain_cross_tx"] & matched
    return ChallengePairMasks(positive=positive, negative=negative, valid=positive | negative)


def ccoi_supcon_loss(
    theta: Tensor,
    masks: ChallengePairMasks,
    temperature: float = 0.12,
) -> CCOILossOutput:
    loss = masked_supcon_loss(
        theta,
        positive_mask=masks.positive,
        valid_mask=masks.valid,
        temperature=float(temperature),
    )
    positive_count = int(torch.triu(masks.positive, diagonal=1).sum().detach().item())
    negative_count = int(torch.triu(masks.negative, diagonal=1).sum().detach().item())
    anchor_count = int((masks.positive.sum(dim=1) > 0).sum().detach().item())
    return CCOILossOutput(
        loss=loss,
        positive_count=positive_count,
        negative_count=negative_count,
        anchor_count=anchor_count,
    )


def ccoi_did_loss(theta: Tensor, y_tx: Tensor, domain: Tensor):
    """Reuse the existing TX/RX rectangle identity difference-in-differences."""

    return tx_rx_rectangle_identity_loss(theta, y_tx, domain)


def _summary(x: Tensor) -> Tensor:
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x.mean(dim=1)
    raise ValueError("response/q must have shape [N,D] or [N,T,D]")


def _relation_mean(distance: Tensor, relation: Tensor) -> tuple[float, int]:
    upper = torch.triu(relation, diagonal=1)
    count = int(upper.sum().detach().item())
    if count == 0:
        return float("nan"), 0
    return float(distance[upper].mean().detach().item()), count


def conditional_distance_diagnostics(
    response: Tensor,
    q_summary: Tensor,
    y_tx: Tensor,
    domain: Tensor,
    min_cosine: float = 0.70,
) -> Dict[str, float | int]:
    """Report d1/d2/d3 with NaN for relations absent from the batch.

    d1: same TX, cross receiver, matched challenge;
    d2: same TX, cross receiver, unmatched challenge;
    d3: cross TX, same receiver, matched challenge.
    """

    response_s = _summary(response).float()
    q_s = _summary(q_summary).float()
    if response_s.size(0) != q_s.size(0):
        raise ValueError("response and q must have the same batch size")
    y = y_tx.view(-1).long().to(response_s.device)
    dom = domain.view(-1).long().to(response_s.device)
    if y.numel() != response_s.size(0) or dom.numel() != response_s.size(0):
        raise ValueError("labels must match response batch size")

    valid = (y >= 0).view(-1, 1) & (dom >= 0).view(-1, 1)
    valid = valid & valid.t()
    same_tx = y.view(-1, 1).eq(y.view(1, -1))
    same_domain = dom.view(-1, 1).eq(dom.view(1, -1))
    challenge_match = _pair_cosine(q_s) >= float(min_cosine)
    distance = torch.cdist(response_s, response_s, p=2)

    d1, n1 = _relation_mean(distance, valid & same_tx & ~same_domain & challenge_match)
    d2, n2 = _relation_mean(distance, valid & same_tx & ~same_domain & ~challenge_match)
    d3, n3 = _relation_mean(distance, valid & ~same_tx & same_domain & challenge_match)
    return {
        "d1": d1,
        "d1_count": n1,
        "d2": d2,
        "d2_count": n2,
        "d3": d3,
        "d3_count": n3,
    }


__all__ = [
    "CCOILossOutput",
    "ChallengePairMasks",
    "ccoi_did_loss",
    "ccoi_supcon_loss",
    "challenge_pair_masks",
    "conditional_distance_diagnostics",
    "ordinary_pair_masks",
]

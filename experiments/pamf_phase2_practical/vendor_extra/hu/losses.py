from __future__ import annotations

import torch
import torch.nn.functional as F


def similarity_loss(tx_features: torch.Tensor, rx_features: torch.Tensor) -> torch.Tensor:
    if tx_features.ndim != 2 or rx_features.ndim != 2 or tx_features.shape != rx_features.shape:
        raise ValueError("tx_features and rx_features must have identical [batch, feature] shapes")
    return torch.linalg.matrix_norm(tx_features.transpose(0, 1).matmul(rx_features), ord="fro")


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=1)
    return -(probabilities * probabilities.clamp_min(torch.finfo(logits.dtype).eps).log()).sum(dim=1).mean()


def feature_separation_loss(
    outputs: dict[str, torch.Tensor],
    tx_labels: torch.Tensor,
    rx_labels: torch.Tensor,
    *,
    lambda_similarity: float = 1.0,
    lambda_tx_entropy: float = 1.0,
    lambda_rx_entropy: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    tx_ce = F.cross_entropy(outputs["tx_logits"], tx_labels)
    rx_ce = F.cross_entropy(outputs["rx_logits"], rx_labels)
    sim = similarity_loss(outputs["tx_features"], outputs["rx_features"])
    tx_entropy = entropy_loss(outputs["tx_logits"])
    rx_entropy = entropy_loss(outputs["rx_logits"])
    total = tx_ce + rx_ce + lambda_similarity * sim + lambda_tx_entropy * tx_entropy + lambda_rx_entropy * rx_entropy
    terms = {
        "tx_ce": tx_ce.detach(),
        "rx_ce": rx_ce.detach(),
        "similarity": sim.detach(),
        "tx_entropy": tx_entropy.detach(),
        "rx_entropy": rx_entropy.detach(),
        "total": total.detach(),
    }
    return total, terms

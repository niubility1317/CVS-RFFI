from __future__ import annotations

import torch
import torch.nn.functional as F


def similarity_loss(tx_features: torch.Tensor, rx_features: torch.Tensor) -> torch.Tensor:
    """Paper-style Frobenius norm of C = X_tx^T X_rx."""
    if tx_features.ndim != 2 or rx_features.ndim != 2:
        raise ValueError("features must be 2-D tensors")
    if tx_features.shape[0] != rx_features.shape[0]:
        raise ValueError("feature batches must have the same length")
    c = tx_features.t().matmul(rx_features)
    return torch.linalg.matrix_norm(c, ord="fro")


def correlation_penalty(tx_features: torch.Tensor, rx_features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    del eps
    return similarity_loss(tx_features, rx_features)


def entropy_loss(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1).mean()


def feature_separation_loss(
    outputs: dict[str, torch.Tensor],
    tx_labels: torch.Tensor,
    rx_labels: torch.Tensor,
    *,
    lambda_similarity: float = 1.0,
    lambda_tx_entropy: float = 1.0,
    lambda_rx_entropy: float = 1.0,
    lambda_correlation: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if lambda_correlation is not None:
        lambda_similarity = lambda_correlation
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

from __future__ import annotations

import torch
import torch.nn.functional as F


def receiver_center_loss(z_rx: torch.Tensor, rx_label: torch.Tensor) -> torch.Tensor:
    return receiver_style_transfer_center_loss(z_rx, rx_label)


def receiver_style_transfer_center_loss(z_rx: torch.Tensor, rx_label: torch.Tensor) -> torch.Tensor:
    """DRIFT receiver-specific style regularizer.

    The paper describes the receiver-specific branch as a communication-style
    representation: samples captured by the same receiver should cluster around
    a receiver style center even when transmitter labels differ.
    """
    losses = []
    for rx in torch.unique(rx_label):
        mask = rx_label == rx
        if int(mask.sum()) <= 1:
            continue
        feat = z_rx[mask]
        center = feat.mean(dim=0, keepdim=True)
        losses.append((feat - center).square().sum(dim=1).mean())
    if not losses:
        return z_rx.new_zeros(())
    return torch.stack(losses).mean()


def negative_mse_separation(z_tx: torch.Tensor, z_rx: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    dim = min(z_tx.size(1), z_rx.size(1))
    a = z_tx[:, :dim]
    b = z_rx[:, :dim]
    if normalize:
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
    return -torch.mean(torch.sum((a - b) ** 2, dim=1))


def compute_drift_loss(
    outputs: dict,
    tx_label: torch.Tensor,
    rx_label: torch.Tensor,
    *,
    lambda_grl: float = 1.0,
    lambda_center: float = 0.01,
    lambda_mse: float = 0.02,
    use_grl: bool = True,
    use_center: bool = True,
    use_mse: bool = True,
    normalize_features_for_mse: bool = True,
):
    loss_ce_tx = F.cross_entropy(outputs["tx_logits"], tx_label)
    loss_ce_rx = F.cross_entropy(outputs["rx_logits"], rx_label)
    loss_grl = F.cross_entropy(outputs["domain_logits"], rx_label)
    loss_center = receiver_center_loss(outputs["z_rx"], rx_label)
    loss_mse = negative_mse_separation(outputs["z_tx"], outputs["z_rx"], normalize=normalize_features_for_mse)
    total = loss_ce_tx + loss_ce_rx
    if use_grl:
        total = total + float(lambda_grl) * loss_grl
    if use_center:
        total = total + float(lambda_center) * loss_center
    if use_mse:
        total = total + float(lambda_mse) * loss_mse
    return {
        "loss": total,
        "loss_ce_tx": loss_ce_tx,
        "loss_ce_rx": loss_ce_rx,
        "loss_grl": loss_grl,
        "loss_center": loss_center,
        "loss_mse": loss_mse,
    }

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ReceiverCenterEMA(nn.Module):
    """Cross-batch receiver center memory for DRIFT center loss."""

    def __init__(self, num_receivers: int, feature_dim: int, momentum: float = 0.95):
        super().__init__()
        if int(num_receivers) <= 0:
            raise ValueError("num_receivers must be positive.")
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive.")
        momentum = float(momentum)
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        self.num_receivers = int(num_receivers)
        self.feature_dim = int(feature_dim)
        self.momentum = momentum
        self.register_buffer("centers", torch.zeros(self.num_receivers, self.feature_dim))
        self.register_buffer("initialized", torch.zeros(self.num_receivers, dtype=torch.bool))
        self.register_buffer("seen_counts", torch.zeros(self.num_receivers, dtype=torch.long))

    def forward(self, z_rx: torch.Tensor, rx_label: torch.Tensor) -> torch.Tensor:
        if z_rx.dim() != 2:
            raise ValueError(f"ReceiverCenterEMA expects z_rx [B,D], got {tuple(z_rx.shape)}")
        if z_rx.size(1) != self.feature_dim:
            raise ValueError(f"ReceiverCenterEMA feature_dim={self.feature_dim}, got {z_rx.size(1)}")
        labels = rx_label.to(device=z_rx.device, dtype=torch.long)
        if labels.numel() != z_rx.size(0):
            raise ValueError("rx_label length must match z_rx batch size.")
        if int(labels.min().item()) < 0 or int(labels.max().item()) >= self.num_receivers:
            raise ValueError("rx_label contains an out-of-range receiver id.")

        target = self.centers.index_select(0, labels).to(dtype=z_rx.dtype).clone()
        detached = z_rx.detach()
        unique_labels = torch.unique(labels).detach().cpu().tolist()
        batch_means: list[tuple[int, torch.Tensor, int]] = []
        for rx in unique_labels:
            rx_i = int(rx)
            mask = labels == rx_i
            batch_mean = detached[mask].mean(dim=0)
            batch_means.append((rx_i, batch_mean, int(mask.sum().item())))
            if not bool(self.initialized[rx_i].item()):
                target[mask] = batch_mean.to(dtype=z_rx.dtype)

        loss = (z_rx - target.detach()).square().sum(dim=1).mean()
        with torch.no_grad():
            for rx_i, batch_mean, count in batch_means:
                if bool(self.initialized[rx_i].item()):
                    updated = self.momentum * self.centers[rx_i] + (1.0 - self.momentum) * batch_mean.to(
                        dtype=self.centers.dtype
                    )
                    self.centers[rx_i].copy_(updated)
                else:
                    self.centers[rx_i].copy_(batch_mean.to(dtype=self.centers.dtype))
                    self.initialized[rx_i] = True
                self.seen_counts[rx_i] += int(count)
        return loss


def receiver_center_loss(
    z_rx: torch.Tensor,
    rx_label: torch.Tensor,
    *,
    center_mode: str = "batch",
    center_memory: ReceiverCenterEMA | None = None,
) -> torch.Tensor:
    if center_mode == "batch":
        return receiver_style_transfer_center_loss(z_rx, rx_label)
    if center_mode == "ema":
        if center_memory is None:
            raise ValueError("center_memory is required when center_mode='ema'.")
        return center_memory(z_rx, rx_label)
    raise ValueError(f"Unsupported DRIFT center mode: {center_mode}")


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


def negative_mse_separation(
    z_tx: torch.Tensor,
    z_rx: torch.Tensor,
    normalize: bool = False,
    reduction: str = "sum",
    cap: float = 0.0,
) -> torch.Tensor:
    dim = min(z_tx.size(1), z_rx.size(1))
    a = z_tx[:, :dim]
    b = z_rx[:, :dim]
    if normalize:
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
    diff_sq = (a - b) ** 2
    if reduction == "sum":
        dist = torch.sum(diff_sq, dim=1)
    elif reduction == "mean":
        dist = torch.mean(diff_sq, dim=1)
    else:
        raise ValueError(f"Unsupported mse reduction: {reduction}")
    if float(cap) > 0.0:
        dist = torch.clamp(dist, max=float(cap))
    return -torch.mean(dist)


def feature_norm_penalty(z_tx: torch.Tensor, z_rx: torch.Tensor, target: float = 0.0) -> torch.Tensor:
    if float(target) > 0.0:
        tx = (torch.linalg.vector_norm(z_tx, dim=1) - float(target)).square().mean()
        rx = (torch.linalg.vector_norm(z_rx, dim=1) - float(target)).square().mean()
        return tx + rx
    return z_tx.square().mean() + z_rx.square().mean()


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
    normalize_features_for_mse: bool = False,
    mse_reduction: str = "sum",
    mse_cap: float = 0.0,
    lambda_feature_norm: float = 0.0,
    feature_norm_target: float = 0.0,
    center_mode: str = "batch",
    center_memory: ReceiverCenterEMA | None = None,
):
    loss_ce_tx = F.cross_entropy(outputs["tx_logits"], tx_label)
    loss_ce_rx = F.cross_entropy(outputs["rx_logits"], rx_label)
    loss_grl = F.cross_entropy(outputs["domain_logits"], rx_label)
    loss_center = receiver_center_loss(
        outputs["z_rx"],
        rx_label,
        center_mode=center_mode,
        center_memory=center_memory,
    )
    loss_mse = negative_mse_separation(
        outputs["z_tx"],
        outputs["z_rx"],
        normalize=normalize_features_for_mse,
        reduction=mse_reduction,
        cap=mse_cap,
    )
    loss_feature_norm = feature_norm_penalty(
        outputs["z_tx"],
        outputs["z_rx"],
        target=feature_norm_target,
    )
    total = loss_ce_tx + loss_ce_rx
    if use_grl:
        total = total + float(lambda_grl) * loss_grl
    if use_center:
        total = total + float(lambda_center) * loss_center
    if use_mse:
        total = total + float(lambda_mse) * loss_mse
    if float(lambda_feature_norm) > 0.0:
        total = total + float(lambda_feature_norm) * loss_feature_norm
    result = {
        "loss": total,
        "loss_ce_tx": loss_ce_tx,
        "loss_ce_rx": loss_ce_rx,
        "loss_grl": loss_grl,
        "loss_center": loss_center,
        "loss_mse": loss_mse,
        "loss_feature_norm": loss_feature_norm,
        "z_tx_norm": torch.linalg.vector_norm(outputs["z_tx"].detach(), dim=1).mean(),
        "z_rx_norm": torch.linalg.vector_norm(outputs["z_rx"].detach(), dim=1).mean(),
    }
    if center_memory is not None:
        result["center_initialized_count"] = center_memory.initialized.float().sum().to(outputs["z_rx"].device)
    return result

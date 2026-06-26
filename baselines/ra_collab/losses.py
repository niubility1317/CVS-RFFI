from __future__ import annotations

import torch
import torch.nn.functional as F


def ra_collab_adversarial_loss(outputs: dict, tx_label: torch.Tensor, rx_label: torch.Tensor, rx_weight: float = 1.0):
    loss_tx = F.cross_entropy(outputs["tx_logits"], tx_label)
    loss_rx = F.cross_entropy(outputs["rx_logits"], rx_label)
    return {"loss": loss_tx + float(rx_weight) * loss_rx, "loss_tx": loss_tx, "loss_rx": loss_rx}

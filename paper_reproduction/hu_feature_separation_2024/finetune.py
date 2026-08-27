from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .model import FeatureSeparationNet


def tx_finetune_parameters(model: FeatureSeparationNet) -> Iterable[nn.Parameter]:
    """Tune the default shared encoder plus TX path while fully freezing RX state."""
    for module in (model.rx_branch, model.rx_classifier):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return list(model.encoder.parameters()) + list(model.tx_branch.parameters()) + list(model.tx_classifier.parameters())


def fine_tune_tx_step(
    model: FeatureSeparationNet,
    iq: torch.Tensor,
    tx_labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """One supervised target-receiver fine-tuning step requiring no RX labels."""
    model.train()
    model.rx_branch.eval()
    model.rx_classifier.eval()
    optimizer.zero_grad(set_to_none=True)
    tx_ce = F.cross_entropy(model(iq)["tx_logits"], tx_labels)
    tx_ce.backward()
    optimizer.step()
    return {"tx_ce": float(tx_ce.detach())}

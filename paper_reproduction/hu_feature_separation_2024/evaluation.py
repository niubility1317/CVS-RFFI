from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

from .model import FeatureSeparationNet

def evaluation_matrix_names() -> tuple[str, ...]:
    """Paper experiment families, independent of external dataset adapters."""
    return (
        "no_new_receiver",
        "cross_receiver",
        "cross_date",
        "channel_augmentation",
        "loss_ablation",
        "few_shot_25_per_tx",
    )


@torch.no_grad()
def evaluate_matrix(
    model: FeatureSeparationNet,
    conditions: Mapping[str, Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
) -> dict[str, dict[str, float]]:
    if tuple(conditions) != evaluation_matrix_names():
        raise ValueError("conditions must follow every paper evaluation family in order")
    model.eval()
    results: dict[str, dict[str, float]] = {}
    for name, batches in conditions.items():
        correct = total = 0
        for iq, tx_labels, _ in batches:
            prediction = model(iq)["tx_logits"].argmax(dim=1)
            correct += int(prediction.eq(tx_labels).sum())
            total += int(tx_labels.numel())
        if not total:
            raise ValueError(f"condition {name} has no examples")
        results[name] = {"tx_accuracy": correct / total}
    return results

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import torch

from .augmentation import augment_iq
from .losses import feature_separation_loss
from .method_config import load_method_config
from .model import FeatureSeparationNet
from .preprocess import preprocess_iq


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_accuracy: float
    best_state_dict: dict[str, torch.Tensor]
    method_metadata: dict[str, object]


@torch.no_grad()
def transmitter_accuracy(model: FeatureSeparationNet, batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
    model.eval()
    correct = total = 0
    for iq, tx_labels, _ in batches:
        prediction = model(iq)["tx_logits"].argmax(dim=1)
        correct += int(prediction.eq(tx_labels).sum())
        total += int(tx_labels.numel())
    return correct / total if total else 0.0


def fit_feature_separation(
    model: FeatureSeparationNet,
    training_batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    validation_batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    max_epochs: int = 200,
    early_stopping_patience: int = 20,
    generator: torch.Generator | None = None,
) -> TrainingResult:
    if max_epochs <= 0 or early_stopping_patience <= 0:
        raise ValueError("max_epochs and early_stopping_patience must be positive")
    train_rows, validation_rows = list(training_batches), list(validation_batches)
    if not train_rows or not validation_rows:
        raise ValueError("training and validation batches must be nonempty")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=10)
    best_epoch, best_accuracy, stale_epochs = 0, -1.0, 0
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, max_epochs + 1):
        model.train()
        for iq, tx_labels, rx_labels in train_rows:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(augment_iq(preprocess_iq(iq), generator=generator))
            loss, _ = feature_separation_loss(
                outputs,
                tx_labels,
                rx_labels,
                lambda_similarity=1.0,
                lambda_tx_entropy=1.0,
                lambda_rx_entropy=1.0,
            )
            loss.backward()
            optimizer.step()
        validation_accuracy = transmitter_accuracy(model, validation_rows)
        scheduler.step(validation_accuracy)
        if validation_accuracy > best_accuracy:
            best_epoch, best_accuracy, stale_epochs = epoch, validation_accuracy, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break
    return TrainingResult(
        best_epoch=best_epoch,
        best_validation_accuracy=best_accuracy,
        best_state_dict=best_state,
        method_metadata=load_method_config().method_metadata(),
    )

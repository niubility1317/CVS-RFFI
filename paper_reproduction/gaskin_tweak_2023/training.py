from __future__ import annotations

import torch
import torch.nn.functional as F
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable
from torch import nn

from .method_config import load_method_config
from .triplet import batch_hard_triplet_loss


def shared_triplet_loss(
    encoder: nn.Module,
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    """Run all three triplet inputs through one shared encoder instance."""
    anchor_embedding = encoder(anchor)
    positive_embedding = encoder(positive)
    negative_embedding = encoder(negative)
    positive_distance = torch.linalg.vector_norm(anchor_embedding - positive_embedding, dim=1)
    negative_distance = torch.linalg.vector_norm(anchor_embedding - negative_embedding, dim=1)
    return F.relu(positive_distance - negative_distance + margin).mean()


def build_sgd_optimizer(encoder: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.9)


@dataclass(frozen=True)
class TweakTrainingResult:
    best_epoch: int
    learning_rate: float
    best_validation_loss: float
    best_state_dict: dict[str, torch.Tensor]
    method_metadata: dict[str, object]


@torch.no_grad()
def _validation_loss(encoder: nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> float:
    encoder.eval()
    values = [float(batch_hard_triplet_loss(encoder(iq), labels)) for iq, labels in batches]
    if not values:
        raise ValueError("validation batches must be nonempty")
    return sum(values) / len(values)


def fit_tweak(
    encoder: nn.Module,
    training_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    validation_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    max_epochs: int = 100,
) -> TweakTrainingResult:
    """Search the frozen learning-rate grid and retain the lowest-loss checkpoint."""
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    train_rows, validation_rows = list(training_batches), list(validation_batches)
    if not train_rows or not validation_rows:
        raise ValueError("training and validation batches must be nonempty")
    metadata = load_method_config().method_metadata()
    learning_rates = metadata["unpublished_defaults"]["learning_rate_grid"]["value"]
    initial_state = deepcopy(encoder.state_dict())
    best_epoch, best_learning_rate, best_loss = 0, 0.0, float("inf")
    best_state: dict[str, torch.Tensor] = {}
    for learning_rate in learning_rates:
        encoder.load_state_dict(initial_state)
        optimizer = build_sgd_optimizer(encoder, float(learning_rate))
        for epoch in range(1, max_epochs + 1):
            encoder.train()
            for iq, labels in train_rows:
                optimizer.zero_grad(set_to_none=True)
                loss = batch_hard_triplet_loss(encoder(iq), labels)
                loss.backward()
                optimizer.step()
            validation_loss = _validation_loss(encoder, validation_rows)
            if validation_loss < best_loss:
                best_epoch, best_learning_rate, best_loss = epoch, float(learning_rate), validation_loss
                best_state = deepcopy(encoder.state_dict())
    return TweakTrainingResult(
        best_epoch=best_epoch,
        learning_rate=best_learning_rate,
        best_validation_loss=best_loss,
        best_state_dict=best_state,
        method_metadata=metadata,
    )

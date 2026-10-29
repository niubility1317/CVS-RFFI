"""Core MoPC-HR equations without dataset or device-specific assumptions.

The default path follows equations (7)-(22) of the paper. The optional
``official_code_dot_softmax`` correction mode exists only to make the public
reference implementation auditable; it is not the paper-faithful default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MoPCHRLoss:
    total: torch.Tensor
    cross_entropy: torch.Tensor
    prototype_augmentation: torch.Tensor
    hierarchical_regularization: torch.Tensor


def compute_class_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_ids: Sequence[int] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one mean feature prototype per requested class (paper eqs. 7/9)."""
    if features.ndim != 2:
        raise ValueError("features must have shape [samples, feature_dim]")
    if labels.ndim != 1 or labels.numel() != features.size(0):
        raise ValueError("labels must have one entry per feature")
    if features.size(0) == 0:
        raise ValueError("features must not be empty")

    ids = torch.unique(labels, sorted=True) if class_ids is None else torch.as_tensor(class_ids, device=labels.device)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("class_ids must be a non-empty one-dimensional sequence")

    prototypes = []
    for class_id in ids:
        selected = features[labels == class_id]
        if selected.size(0) == 0:
            raise ValueError(f"class {int(class_id)} has no features")
        prototypes.append(selected.mean(dim=0))
    return torch.stack(prototypes), ids.to(dtype=labels.dtype)


def prototype_augmentation(
    prototypes: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    num_samples: int,
    noise_std: float = 0.05,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample stored prototypes and add N(0, 0.05) noise (paper eq. 15)."""
    if prototypes.ndim != 2:
        raise ValueError("prototypes must have shape [classes, feature_dim]")
    if class_ids.ndim != 1 or class_ids.numel() != prototypes.size(0):
        raise ValueError("class_ids must align with prototypes")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    indices = torch.randint(prototypes.size(0), (num_samples,), device=prototypes.device, generator=generator)
    noise = torch.randn(
        (num_samples, prototypes.size(1)),
        dtype=prototypes.dtype,
        device=prototypes.device,
        generator=generator,
    ) * noise_std
    return prototypes[indices] + noise, class_ids.to(device=prototypes.device)[indices]


def correct_old_prototypes(
    old_prototypes: torch.Tensor,
    new_prototypes_previous_model: torch.Tensor,
    new_prototypes_current_model: torch.Tensor,
    *,
    alpha: float = 0.97,
    similarity_mode: str = "paper_cosine",
) -> torch.Tensor:
    """Apply momentum prototype correction from paper equations (10)-(14)."""
    if old_prototypes.ndim != 2:
        raise ValueError("old_prototypes must have shape [old_classes, feature_dim]")
    if new_prototypes_previous_model.shape != new_prototypes_current_model.shape:
        raise ValueError("new prototype tensors from previous/current models must match")
    if new_prototypes_previous_model.ndim != 2:
        raise ValueError("new prototypes must have shape [new_classes, feature_dim]")
    if old_prototypes.size(1) != new_prototypes_previous_model.size(1):
        raise ValueError("old and new prototypes must share feature_dim")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    if similarity_mode == "paper_cosine":
        similarity = F.normalize(old_prototypes, dim=1) @ F.normalize(new_prototypes_previous_model, dim=1).t()
    elif similarity_mode == "official_code_dot_softmax":
        similarity = torch.softmax(old_prototypes @ new_prototypes_previous_model.t(), dim=1)
    else:
        raise ValueError("similarity_mode must be paper_cosine or official_code_dot_softmax")

    new_deviation = new_prototypes_current_model - new_prototypes_previous_model
    estimated_old_deviation = similarity @ new_deviation
    return alpha * old_prototypes + (1.0 - alpha) * estimated_old_deviation


def _named_parameters(
    parameters: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
) -> list[tuple[str, torch.Tensor]]:
    if isinstance(parameters, Mapping):
        return list(parameters.items())
    return [(str(index), value) for index, value in enumerate(parameters)]


def hierarchical_regularization(
    current_parameters: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    previous_parameters: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    *,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    """Layer-decayed squared L2 penalty from paper equations (19)-(21)."""
    if lambda_max < 0:
        raise ValueError("lambda_max must be non-negative")
    current = _named_parameters(current_parameters)
    previous = _named_parameters(previous_parameters)
    if not current or len(current) != len(previous):
        raise ValueError("current and previous parameters must have the same non-zero length")
    if [name for name, _ in current] != [name for name, _ in previous]:
        raise ValueError("current and previous parameter names must match in order")

    total = current[0][1].new_zeros(())
    layer_count = len(current)
    for index, ((_, current_value), (_, previous_value)) in enumerate(zip(current, previous)):
        if current_value.shape != previous_value.shape:
            raise ValueError("current and previous parameter shapes must match")
        layer_lambda = lambda_max * (1.0 - index / layer_count)
        total = total + layer_lambda * torch.sum((current_value - previous_value.detach()) ** 2)
    return total


def mopc_hr_incremental_objective(
    current_logits: torch.Tensor,
    current_labels: torch.Tensor,
    prototype_logits: torch.Tensor,
    prototype_labels: torch.Tensor,
    current_parameters: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    previous_parameters: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    *,
    beta: float = 1.0,
    lambda_max: float = 1.0,
) -> MoPCHRLoss:
    """Paper equation (22): CE + prototype replay CE + beta * HR.

    Knowledge distillation is intentionally absent: it is not part of equation
    (22), and the official trainer computes but does not optimize its KD term.
    """
    if beta < 0:
        raise ValueError("beta must be non-negative")
    cross_entropy = F.cross_entropy(current_logits, current_labels)
    prototype_loss = F.cross_entropy(prototype_logits, prototype_labels)
    regularization = hierarchical_regularization(
        current_parameters,
        previous_parameters,
        lambda_max=lambda_max,
    )
    total = cross_entropy + prototype_loss + beta * regularization
    return MoPCHRLoss(total, cross_entropy, prototype_loss, regularization)

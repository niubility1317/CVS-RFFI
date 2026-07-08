from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ZeroBiasCosineClassifier(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, norm_mag: float = 5.0) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.norm_mag = float(norm_mag)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized_features = F.normalize(features, dim=1)
        normalized_weights = F.normalize(self.weight, dim=1)
        return self.norm_mag * (normalized_features @ normalized_weights.T) + self.norm_mag


class CSILClassifier(nn.Module):
    """Zero-bias CSIL classifier with stage-wise channel expansion."""

    def __init__(self, *, input_dim: int, embedding_dim: int, num_classes: int, stage_id: int = 0) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_classes <= 0:
            raise ValueError("embedding_dim and num_classes must be positive")
        self.input_dim = int(input_dim)
        self.stage_id = int(stage_id)
        self.embedding = nn.Linear(self.input_dim, int(embedding_dim))
        self.classifier = ZeroBiasCosineClassifier(int(embedding_dim), int(num_classes))
        self.register_buffer("classifier_train_mask", torch.ones_like(self.classifier.weight), persistent=False)
        self.register_buffer("embedding_train_mask", torch.ones_like(self.embedding.weight), persistent=False)
        self.register_buffer("embedding_bias_train_mask", torch.ones_like(self.embedding.bias), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedding(x))

    def expand_for_stage(self, *, new_classes: int, added_embedding_dim: int, stage_id: int) -> None:
        if new_classes <= 0 or added_embedding_dim <= 0:
            raise ValueError("new_classes and added_embedding_dim must be positive")
        old_embedding = self.embedding
        old_classifier = self.classifier
        old_dim = int(old_embedding.out_features)
        new_dim = old_dim + int(added_embedding_dim)
        old_classes = int(old_classifier.out_features)
        total_classes = old_classes + int(new_classes)
        device = old_embedding.weight.device
        dtype = old_embedding.weight.dtype

        expanded_embedding = nn.Linear(self.input_dim, new_dim).to(device=device, dtype=dtype)
        with torch.no_grad():
            expanded_embedding.weight[:old_dim, :] = old_embedding.weight
            expanded_embedding.bias[:old_dim] = old_embedding.bias
            nn.init.kaiming_uniform_(expanded_embedding.weight[old_dim:, :], a=5**0.5)
            expanded_embedding.bias[old_dim:].zero_()

        expanded_classifier = ZeroBiasCosineClassifier(
            new_dim, total_classes, norm_mag=old_classifier.norm_mag
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            expanded_classifier.weight.zero_()
            expanded_classifier.weight[:old_classes, :old_dim] = old_classifier.weight
            nn.init.kaiming_uniform_(expanded_classifier.weight[old_classes:, old_dim:], a=5**0.5)

        self.embedding = expanded_embedding
        self.classifier = expanded_classifier
        classifier_mask = torch.zeros_like(self.classifier.weight)
        classifier_mask[old_classes:, old_dim:] = 1.0
        embedding_mask = torch.zeros_like(self.embedding.weight)
        embedding_mask[old_dim:, :] = 1.0
        embedding_bias_mask = torch.zeros_like(self.embedding.bias)
        embedding_bias_mask[old_dim:] = 1.0
        self.register_buffer("classifier_train_mask", classifier_mask, persistent=False)
        self.register_buffer("embedding_train_mask", embedding_mask, persistent=False)
        self.register_buffer("embedding_bias_train_mask", embedding_bias_mask, persistent=False)
        self.stage_id = int(stage_id)

    def apply_gradient_masks(self) -> None:
        if self.classifier.weight.grad is not None:
            self.classifier.weight.grad.mul_(self.classifier_train_mask)
        if self.embedding.weight.grad is not None:
            self.embedding.weight.grad.mul_(self.embedding_train_mask)
        if self.embedding.bias.grad is not None:
            self.embedding.bias.grad.mul_(self.embedding_bias_train_mask)


def csil_masked_sgd_step(
    model: CSILClassifier,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
    state: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Apply the paper-style mask to the full SGD+L2+momentum update.

    The official MATLAB code masks the velocity update, not only the raw
    gradient. This prevents old CSIL blocks from drifting through L2 decay or
    stale momentum.
    """
    state = {} if state is None else dict(state)
    masks = {
        "embedding.weight": model.embedding_train_mask,
        "embedding.bias": model.embedding_bias_train_mask,
        "classifier.weight": model.classifier_train_mask,
    }
    next_state: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            mask = masks.get(name)
            if mask is None:
                mask = torch.ones_like(parameter)
            mask = mask.to(device=parameter.device, dtype=parameter.dtype)
            previous_velocity = state.get(name, torch.zeros_like(parameter)).to(
                device=parameter.device, dtype=parameter.dtype
            )
            raw_update = parameter.grad + float(weight_decay) * parameter
            velocity = float(momentum) * previous_velocity + float(lr) * raw_update
            velocity = velocity * mask
            parameter -= velocity
            next_state[name] = velocity.detach().clone()
    return next_state

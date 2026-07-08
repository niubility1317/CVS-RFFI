from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SixBlockConv1DEncoder(nn.Module):
    """Six Conv1D-BN-MaxPool blocks described in the OSC-FSCIL SEI paper."""

    def __init__(
        self,
        *,
        input_channels: int = 2,
        embedding_dim: int = 128,
        channels: tuple[int, int, int, int, int, int] = (32, 64, 96, 128, 160, 192),
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if len(channels) != 6:
            raise ValueError("channels must define exactly six convolution blocks")

        layers: list[nn.Module] = []
        in_channels = input_channels
        padding = kernel_size // 2
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(channels[-1], embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("input must have shape [batch, channels, length]")
        z = self.features(x.float())
        z = self.pool(z).squeeze(-1)
        return self.projection(z)


class CosineClassifier(nn.Module):
    def __init__(self, *, embedding_dim: int, num_classes: int, scale: float = 1.0) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_classes <= 0:
            raise ValueError("embedding_dim and num_classes must be positive")
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        self.scale = float(scale)
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, features: torch.Tensor, weight_override: torch.Tensor | None = None) -> torch.Tensor:
        weights = self.weight if weight_override is None else weight_override
        if features.ndim != 2 or weights.ndim != 2:
            raise ValueError("features and weights must be rank-2")
        if features.size(1) != weights.size(1):
            raise ValueError("feature dimension mismatch")
        return self.scale * (F.normalize(features, dim=1) @ F.normalize(weights, dim=1).t())


def class_mean_weights(features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2:
        raise ValueError("features must have shape [num_samples, feature_dim]")
    if labels.ndim != 1 or labels.numel() != features.size(0):
        raise ValueError("labels must have one value per feature")
    class_ids = torch.unique(labels.detach().cpu(), sorted=True).to(device=labels.device)
    weights = []
    for class_id in class_ids:
        mask = labels == class_id
        weights.append(features[mask].mean(dim=0))
    return F.normalize(torch.stack(weights, dim=0), dim=1), class_ids


def concat_classifier_weights(old_weights: torch.Tensor, new_weights: torch.Tensor) -> torch.Tensor:
    if old_weights.ndim != 2 or new_weights.ndim != 2:
        raise ValueError("classifier weights must be rank-2")
    if old_weights.size(1) != new_weights.size(1):
        raise ValueError("classifier weight dimension mismatch")
    return torch.cat([old_weights, new_weights], dim=0)

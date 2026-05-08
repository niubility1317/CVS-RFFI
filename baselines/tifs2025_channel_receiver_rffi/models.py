from __future__ import annotations

import torch
from torch import nn

from baselines.common.resnet2d import ResNet2DEncoder


class ResNetRFF(nn.Module):
    """ResNet-style spectrogram RFF classifier used by the TIFS2025 baseline."""

    def __init__(self, num_classes: int, in_channels: int = 1, feature_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.feature_extractor = ResNet2DEncoder(
            in_channels=in_channels,
            feature_dim=feature_dim,
            channels=(32, 32, 64, 64),
            blocks_per_stage=(1, 1, 1, 1),
            dropout=dropout,
        )
        self.classifier = nn.Linear(feature_dim, int(num_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor):
        z = self.forward_features(x)
        logits = self.classifier(z)
        return logits, z


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 256, projection_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SiameseRFF(nn.Module):
    """Shared-weight two-branch wrapper. Inference uses `backbone` only."""

    def __init__(self, backbone: ResNetRFF):
        super().__init__()
        self.backbone = backbone

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        logits1, z1 = self.backbone(x1)
        logits2, z2 = self.backbone(x2)
        return logits1, logits2, z1, z2

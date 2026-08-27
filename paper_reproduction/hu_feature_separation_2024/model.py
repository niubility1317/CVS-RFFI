from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .representation import build_fusion_representation


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = ChannelAttention(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.attention(self.bn2(self.conv2(x)))
        return F.leaky_relu(x + residual, inplace=True)


class AttentionResNet18(nn.Module):
    """Figure-6-style attention ResNet18 encoder yielding a 512-D shared feature."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
        )
        self.layer1 = self._layer(16, 16, stride=1)
        self.layer2 = self._layer(16, 32, stride=2)
        self.layer3 = self._layer(32, 64, stride=2)
        self.layer4 = self._layer(64, 128, stride=2)
        self.layer5 = self._layer(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(256, 512)
        self.feature_attention = nn.Sequential(nn.Linear(512, 512), nn.Sigmoid())

    @staticmethod
    def _layer(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(ResidualBlock(in_channels, out_channels, stride), ResidualBlock(out_channels, out_channels))

    def forward(self, fusion: torch.Tensor) -> torch.Tensor:
        x = fusion.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.project(self.pool(x).flatten(1))
        return x * self.feature_attention(x)


class FeatureSeparationNet(nn.Module):
    """Shared encoder plus the paper's transmitter and receiver branches."""

    def __init__(self, *, num_tx: int, num_rx: int) -> None:
        super().__init__()
        self.encoder = AttentionResNet18()
        self.tx_branch = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(inplace=True))
        self.rx_branch = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(inplace=True))
        self.tx_classifier = nn.Linear(256, num_tx)
        self.rx_classifier = nn.Linear(256, num_rx)

    def forward(self, iq_or_fusion: torch.Tensor) -> dict[str, torch.Tensor]:
        if iq_or_fusion.ndim != 3 or iq_or_fusion.shape[-1] != 256 or iq_or_fusion.shape[1] not in (2, 3):
            raise ValueError("input must have shape [batch, 2 or 3, 256]")
        fusion = build_fusion_representation(iq_or_fusion) if iq_or_fusion.shape[1] == 2 else iq_or_fusion
        shared = self.encoder(fusion)
        tx_features = self.tx_branch(shared)
        rx_features = self.rx_branch(shared)
        return {
            "shared": shared,
            "tx_features": tx_features,
            "rx_features": rx_features,
            "tx_logits": self.tx_classifier(tx_features),
            "rx_logits": self.rx_classifier(rx_features),
        }


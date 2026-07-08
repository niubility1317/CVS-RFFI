from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _to_iq_channels(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3 and x.shape[1] == 2:
        return x
    if x.ndim == 3 and x.shape[-1] == 2:
        return x.transpose(1, 2)
    if x.ndim == 4 and x.shape[1] == 1 and x.shape[-1] == 2:
        return x.squeeze(1).transpose(1, 2)
    raise ValueError("expected IQ input shaped [batch,2,length], [batch,length,2], or [batch,1,length,2]")


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class DADDAFeatureExtractor(nn.Module):
    """ResNet18-style 1-D IQ feature extractor G_f; not a layer-for-layer Fig. 3 replica."""

    def __init__(self, *, feature_dim: int = 128, base_channels: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.in_channels = base_channels
        self.layer1 = self._make_layer(base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(base_channels * 8, feature_dim)
        self.out_channels = base_channels * 8

    def _make_layer(self, out_channels: int, *, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock1D(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = _to_iq_channels(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_map = self.layer4(x)
        global_features = self.projection(self.pool(feature_map).flatten(1))
        return global_features, feature_map


class DADDAMultiscaleExtractor(nn.Module):
    """1-D multiscale G_m approximation of the four-branch Fig. 4 module."""

    def __init__(self, *, in_channels: int, multiscale_dim: int = 128) -> None:
        super().__init__()
        branch_dim = max(int(multiscale_dim) // 4, 1)
        self.branch1 = nn.Sequential(nn.Conv1d(in_channels, branch_dim, kernel_size=1), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(branch_dim, branch_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(branch_dim, branch_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Linear(branch_dim * 4, int(multiscale_dim))

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        branches = [self.branch1(feature_map), self.branch2(feature_map), self.branch3(feature_map), self.branch4(feature_map)]
        fused = torch.cat(branches, dim=1)
        pooled = F.adaptive_avg_pool1d(fused, 1).flatten(1)
        return self.projection(pooled)


class DADDAClassifier(nn.Module):
    """Two fully connected classifier G_l; paper reports hidden widths 512 and 128."""

    def __init__(self, *, in_dim: int, num_classes: int, hidden1: int = 512, hidden2: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DADDANet(nn.Module):
    """DADDA network: G_f feature extractor, G_m multiscale extractor, and G_l classifier."""

    def __init__(
        self,
        *,
        num_classes: int,
        feature_dim: int = 128,
        multiscale_dim: int = 128,
        base_channels: int = 16,
        classifier_hidden1: int = 512,
        classifier_hidden2: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_extractor = DADDAFeatureExtractor(feature_dim=feature_dim, base_channels=base_channels)
        self.multiscale_extractor = DADDAMultiscaleExtractor(
            in_channels=self.feature_extractor.out_channels,
            multiscale_dim=multiscale_dim,
        )
        self.classifier = DADDAClassifier(
            in_dim=multiscale_dim,
            num_classes=self.num_classes,
            hidden1=classifier_hidden1,
            hidden2=classifier_hidden2,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        global_features, feature_map = self.feature_extractor(x)
        local_features = self.multiscale_extractor(feature_map)
        return {
            "global_features": global_features,
            "local_features": local_features,
            "logits": self.classifier(local_features),
        }

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["logits"]

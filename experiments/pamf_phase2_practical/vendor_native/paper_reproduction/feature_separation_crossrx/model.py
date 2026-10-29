from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def build_wisig_fusion_representation(iq: torch.Tensor, out_length: int = 256) -> torch.Tensor:
    """Build the paper's WiSig fusion input: I/Q time channels plus a Welch-style PSD channel."""
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError("iq must have shape [batch, 2, length]")
    if iq.shape[-1] != out_length:
        iq = F.interpolate(iq, size=out_length, mode="linear", align_corners=False)
    complex_iq = torch.complex(iq[:, 0], iq[:, 1])
    spectrum = torch.fft.fft(complex_iq, n=out_length, dim=-1)
    psd = torch.log1p(spectrum.abs().pow(2))
    psd = (psd - psd.mean(dim=-1, keepdim=True)) / psd.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.cat([iq, psd.unsqueeze(1)], dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(self.pool(x))


class ResidualAttentionBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = ChannelAttention(out_channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class AttentionResNet18Encoder(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
        )
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_dim)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualAttentionBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualAttentionBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class FeatureSeparationNet(nn.Module):
    """WiSig feature-separation network with shared encoder and TX/RX branches."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        input_length: int = 256,
        embedding_dim: int = 128,
        branch_dim: int = 64,
        num_tx: int,
        num_rx: int,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        del hidden_channels
        if input_channels != 3:
            raise ValueError("Feature Separation paper input must be 3 channels: I, Q, Welch PSD")
        if input_length != 256:
            raise ValueError("Feature Separation paper input length must be 256 for WiSig")
        self.encoder = AttentionResNet18Encoder(embedding_dim=embedding_dim)
        self.tx_branch = nn.Sequential(
            nn.Linear(embedding_dim, branch_dim),
            nn.BatchNorm1d(branch_dim),
            nn.ReLU(inplace=True),
        )
        self.rx_branch = nn.Sequential(
            nn.Linear(embedding_dim, branch_dim),
            nn.BatchNorm1d(branch_dim),
            nn.ReLU(inplace=True),
        )
        self.tx_classifier = nn.Linear(branch_dim, num_tx)
        self.rx_classifier = nn.Linear(branch_dim, num_rx)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("input must have shape [batch, 3, 256]")
        if x.shape[1] == 2:
            x = build_wisig_fusion_representation(x, out_length=256)
        if x.shape[1] != 3 or x.shape[2] != 256:
            raise ValueError("input must have shape [batch, 3, 256]")
        shared = self.encoder(x)
        tx_features = self.tx_branch(shared)
        rx_features = self.rx_branch(shared)
        return {
            "shared": shared,
            "tx_features": tx_features,
            "rx_features": rx_features,
            "tx_logits": self.tx_classifier(tx_features),
            "rx_logits": self.rx_classifier(rx_features),
            "tx_from_rx_logits": self.tx_classifier(rx_features),
            "rx_from_tx_logits": self.rx_classifier(tx_features),
        }

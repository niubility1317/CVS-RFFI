from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class BasicBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class ResNet2DEncoder(nn.Module):
    """Compact ResNet-style encoder for spectrogram/CIS inputs `[B,C,F,T]`."""

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 256,
        channels: Sequence[int] = (32, 64, 128),
        blocks_per_stage: Sequence[int] = (2, 2, 2),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        in_ch = channels[0]
        stages = []
        for stage_i, (out_ch, n_blocks) in enumerate(zip(channels, blocks_per_stage)):
            stride = 1 if stage_i == 0 else 2
            blocks = [BasicBlock2D(in_ch, out_ch, stride=stride)]
            in_ch = out_ch
            for _ in range(1, int(n_blocks)):
                blocks.append(BasicBlock2D(in_ch, out_ch, stride=1))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_ch, self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"ResNet2DEncoder expects [B,C,F,T], got {tuple(x.shape)}")
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

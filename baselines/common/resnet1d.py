from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act(out)


class ResNet1DEncoder(nn.Module):
    """Small ResNet18-style 1D encoder for IQ tensors shaped `[B, C, L]`."""

    def __init__(
        self,
        input_channels: int = 2,
        embedding_dim: int = 512,
        channels: Sequence[int] = (64, 128, 256, 512),
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        in_ch = channels[0]
        stages = []
        for stage_i, (out_ch, n_blocks) in enumerate(zip(channels, blocks_per_stage)):
            stride = 1 if stage_i == 0 else 2
            blocks = [BasicBlock1D(in_ch, out_ch, stride=stride)]
            in_ch = out_ch
            for _ in range(1, int(n_blocks)):
                blocks.append(BasicBlock1D(in_ch, out_ch, stride=1))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_ch, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"ResNet1DEncoder expects [B,C,L], got {tuple(x.shape)}")
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

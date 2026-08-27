from __future__ import annotations

import torch
from torch import nn


class TweakEncoder(nn.Module):
    """The paper's 2x128 IQ to 12-dimensional metric encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 128, kernel_size=7),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=7),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(256),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 24, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 12),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or tuple(iq.shape[1:]) != (2, 128):
            raise ValueError("iq must have shape [batch, 2, 128]")
        return self.embedding(self.features(iq))


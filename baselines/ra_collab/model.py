from __future__ import annotations

import torch
from torch import nn

from baselines.common.grl import gradient_reverse
from baselines.common.resnet2d import ResNet2DEncoder


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RACollabRFFI(nn.Module):
    """RA-Collab GRL-based spectrogram/CIS classifier."""

    def __init__(
        self,
        num_tx: int,
        num_rx: int,
        in_channels: int = 1,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_extractor = ResNet2DEncoder(in_channels=in_channels, feature_dim=feature_dim, dropout=dropout)
        self.tx_classifier = ClassifierHead(feature_dim, int(num_tx), hidden_dim, dropout)
        self.rx_classifier = ClassifierHead(feature_dim, int(num_rx), hidden_dim, dropout)

    def forward(self, x: torch.Tensor, grl_lambda: float = 1.0, return_rx: bool = True):
        feature = self.feature_extractor(x)
        tx_logits = self.tx_classifier(feature)
        out = {"feature": feature, "tx_logits": tx_logits}
        if return_rx:
            out["rx_logits"] = self.rx_classifier(gradient_reverse(feature, grl_lambda))
        return out

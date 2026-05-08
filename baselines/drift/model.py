from __future__ import annotations

import torch
from torch import nn

from baselines.common.grl import gradient_reverse
from baselines.common.resnet1d import MLPClassifier, ResNet1DEncoder


class DRIFTModel(nn.Module):
    """Feature split + GRL + receiver-specific branch model."""

    def __init__(
        self,
        num_tx: int,
        num_rx: int,
        input_channels: int = 2,
        embedding_dim: int = 512,
        split_dim: int = 256,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        if int(split_dim) <= 0 or int(split_dim) >= int(embedding_dim):
            raise ValueError("split_dim must be in (0, embedding_dim).")
        self.embedding_dim = int(embedding_dim)
        self.split_dim = int(split_dim)
        self.encoder = ResNet1DEncoder(input_channels=input_channels, embedding_dim=embedding_dim, dropout=dropout)
        rx_dim = self.embedding_dim - self.split_dim
        self.tx_classifier = MLPClassifier(self.split_dim, int(num_tx), classifier_hidden_dim, dropout)
        self.rx_classifier = MLPClassifier(rx_dim, int(num_rx), classifier_hidden_dim, dropout)
        self.domain_discriminator = MLPClassifier(self.split_dim, int(num_rx), classifier_hidden_dim, dropout)

    def forward(self, x: torch.Tensor, grl_lambda: float = 1.0):
        z = self.encoder(x)
        z_tx = z[:, : self.split_dim]
        z_rx = z[:, self.split_dim :]
        return {
            "z": z,
            "z_tx": z_tx,
            "z_rx": z_rx,
            "tx_logits": self.tx_classifier(z_tx),
            "rx_logits": self.rx_classifier(z_rx),
            "domain_logits": self.domain_discriminator(gradient_reverse(z_tx, grl_lambda)),
        }

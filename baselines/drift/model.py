from __future__ import annotations

import torch
from torch import nn

from baselines.drift.architecture import (
    DRIFTResNet18_1DEncoder,
    DRIFTThreeLayerClassifier,
    DRIFTTwoLayerClassifier,
)
from baselines.drift.grl import gradient_reverse


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
        encoder_use_projection: bool = False,
        domain_discriminator_layers: int = 2,
    ):
        super().__init__()
        if int(split_dim) <= 0 or int(split_dim) >= int(embedding_dim):
            raise ValueError("split_dim must be in (0, embedding_dim).")
        if int(domain_discriminator_layers) not in (2, 3):
            raise ValueError("domain_discriminator_layers must be 2 or 3.")
        self.embedding_dim = int(embedding_dim)
        self.split_dim = int(split_dim)
        self.domain_discriminator_layers = int(domain_discriminator_layers)
        self.encoder = DRIFTResNet18_1DEncoder(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
            use_projection=encoder_use_projection,
        )
        rx_dim = self.embedding_dim - self.split_dim
        self.tx_classifier = DRIFTThreeLayerClassifier(self.split_dim, int(num_tx), classifier_hidden_dim, dropout)
        self.rx_classifier = DRIFTThreeLayerClassifier(rx_dim, int(num_rx), classifier_hidden_dim, dropout)
        discriminator_cls = DRIFTTwoLayerClassifier if self.domain_discriminator_layers == 2 else DRIFTThreeLayerClassifier
        self.domain_discriminator = discriminator_cls(self.split_dim, int(num_rx), classifier_hidden_dim, dropout)

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

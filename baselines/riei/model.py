from __future__ import annotations

import torch
from torch import nn

from baselines.common.resnet1d import MLPClassifier, ResNet1DEncoder


class RIEIModel(nn.Module):
    """Receiver-independent emitter identification via feature disentanglement."""

    def __init__(
        self,
        num_emitters: int,
        num_receivers: int,
        input_channels: int = 2,
        feature_dim: int = 512,
        emitter_feature_dim: int | None = None,
        receiver_feature_dim: int | None = None,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        feature_dim = int(feature_dim)
        emitter_feature_dim = int(emitter_feature_dim or feature_dim // 2)
        receiver_feature_dim = int(receiver_feature_dim or (feature_dim - emitter_feature_dim))
        if emitter_feature_dim + receiver_feature_dim != feature_dim:
            raise ValueError("feature_dim must equal emitter_feature_dim + receiver_feature_dim.")
        self.emitter_feature_dim = emitter_feature_dim
        self.receiver_feature_dim = receiver_feature_dim
        self.fed = ResNet1DEncoder(input_channels=input_channels, embedding_dim=feature_dim, dropout=dropout)
        self.ec = MLPClassifier(emitter_feature_dim, int(num_emitters), classifier_hidden_dim, dropout)
        self.rc = MLPClassifier(receiver_feature_dim, int(num_receivers), classifier_hidden_dim, dropout)
        self.rx_to_emitter_space = (
            nn.Identity()
            if receiver_feature_dim == emitter_feature_dim
            else nn.Linear(receiver_feature_dim, emitter_feature_dim)
        )
        self.emitter_to_rx_space = (
            nn.Identity()
            if emitter_feature_dim == receiver_feature_dim
            else nn.Linear(emitter_feature_dim, receiver_feature_dim)
        )

    def forward(self, x: torch.Tensor):
        z = self.fed(x)
        z_e, z_r = torch.split(z, [self.emitter_feature_dim, self.receiver_feature_dim], dim=1)
        return {
            "z_e": z_e,
            "z_r": z_r,
            "emitter_logits": self.ec(z_e),
            "receiver_logits": self.rc(z_r),
            "cross_emitter_logits": self.ec(self.rx_to_emitter_space(z_r)),
            "cross_receiver_logits": self.rc(self.emitter_to_rx_space(z_e)),
        }

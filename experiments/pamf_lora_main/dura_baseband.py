"""Run both identity-MAP stages in the tracked LoRa baseband coordinates."""

import torch

from .dura_identity_action import IdentityActionDura


class BasebandDura(IdentityActionDura):
    def forward(self, observed, quality, _rotation):
        return super().forward(observed, quality, torch.ones_like(observed))

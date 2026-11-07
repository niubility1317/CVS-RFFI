"""Normalized full-window DURA correction to preserve decoder gradients."""
from torch import nn

from experiments.phase1_dura_orthogonal.large_context_dura import LargeContextDura, WIDTH, LAYERS


class StableLargeContextDura(LargeContextDura):
    def __init__(self, width=WIDTH, layers=LAYERS):
        super().__init__(width=width, layers=layers)
        self.global_context = nn.Sequential(self.global_context, nn.LayerNorm(width))
        self.decode = nn.Sequential(
            nn.Conv1d(width + 64, 128, 5, padding=2), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv1d(128, 64, 5, padding=2), nn.GroupNorm(8, 64), nn.GELU(),
        )

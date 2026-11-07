"""Normalized direct time-frequency recovery without a DURA dependency."""
from torch import nn

from experiments.phase1_dura_orthogonal.large_direct_timefreq import LargeDirectTimeFreq, WIDTH, LAYERS


class StableLargeDirectTimeFreq(LargeDirectTimeFreq):
    def __init__(self, width=WIDTH, layers=LAYERS):
        super().__init__(width=width, layers=layers)
        self.transformer = nn.Sequential(self.transformer, nn.LayerNorm(width))
        self.decode = nn.Sequential(
            nn.Conv1d(width + 96, 192, 5, padding=2), nn.GroupNorm(8, 192), nn.GELU(),
            nn.Conv1d(192, 96, 5, padding=2), nn.GroupNorm(8, 96), nn.GELU(),
        )

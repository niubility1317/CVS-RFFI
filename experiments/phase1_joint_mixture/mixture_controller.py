"""Source-trained correction of physical posterior weights; output is IQ, not a label."""
import torch
from torch import nn
from torch.nn import functional as F

FEATURES = 347
COMPONENTS = 6

class MixtureController(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(COMPONENTS * FEATURES, 256), nn.LayerNorm(256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, COMPONENTS))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, log_weights, states, scale=1.0):
        n = len(features)
        if features.shape != (n, COMPONENTS, FEATURES) or log_weights.shape != (n, COMPONENTS) or states.shape != (n, COMPONENTS, 512):
            raise ValueError('Matched six-component source posterior required')
        correction = self.network(features.reshape(n, -1))
        weights = F.softmax(log_weights + float(scale) * correction, dim=1)
        wave = (states * weights.unsqueeze(-1)).sum(1).reshape(n, 2, 256)
        return wave, weights, correction

"""Remove redundant dense fusion while retaining native IQ/PA representations."""
import math
import torch
from torch import nn
from experiments.cvs_residual_identity.model import build as residual_build

VARIANTS = ('balanced_fusion', 'signed_balanced_fusion')

class BalancedFusion(nn.Module):
    def __init__(self, dimension, has_circularity=True, drop=0.0):
        super().__init__()
        self.dimension = dimension
        self.has_circularity = has_circularity
        self.time_norm = nn.LayerNorm(dimension)
        self.frequency_norm = nn.LayerNorm(dimension)
        self.frequency_gain = nn.Parameter(torch.full((dimension,), math.atanh(0.5)))
        self.circularity_gain = nn.Parameter(torch.zeros(dimension)) if has_circularity else None
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        expected = 2 * self.dimension + int(self.has_circularity)
        if x.ndim != 2 or x.shape[1] != expected:
            raise ValueError('Native time/frequency/circularity schema changed')
        t, f = x[:, :self.dimension], x[:, self.dimension:2*self.dimension]
        joint = self.time_norm(t) + self.frequency_gain.tanh() * self.frequency_norm(f)
        if self.has_circularity:
            joint = joint + x[:, -1:] * self.circularity_gain.tanh()
        return self.drop(joint)

def build(variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered balanced identity architecture')
    model = residual_build('residual_fusion')
    b = model.id_backbone
    if not b.use_time_path or not b.use_freq_path:
        raise ValueError('Balanced fusion requires both native identity branches')
    old = b.fuse
    b.fuse = BalancedFusion(b.emb_dim, b.use_circularity, old[2].p)
    if variant == 'signed_balanced_fusion':
        # Projection and dropout remain; remove the one-sided amplitude clipping.
        b.pa_proj[1] = nn.Identity()
    return model

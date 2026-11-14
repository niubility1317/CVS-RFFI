"""Experimental MoPC bridge with ProtoNet's common logit offset removed.

Only old-class probabilities and decisions are equivalent at the B boundary.
The subsequent expanded C model has a different feature parameterization and
must be recorded as a new adapter, not mixed silently with prior results.
"""
import copy

import torch
from torch import nn


def require_equivalent_b(baseline, alternative):
    if baseline.shape != alternative.shape or not bool(torch.isfinite(baseline).all()) or \
       not bool(torch.isfinite(alternative).all()):
        raise ValueError('Invalid old B logits')
    if not torch.equal(baseline.argmax(1), alternative.argmax(1)):
        raise ValueError('Affine bridge changes old B decision')
    center = lambda z: z - z.mean(dim=1, keepdim=True)
    if not torch.allclose(center(baseline), center(alternative), rtol=1e-4, atol=1e-4) or \
       not torch.allclose(baseline.softmax(1), alternative.softmax(1), rtol=1e-4, atol=1e-4):
        raise ValueError('Affine bridge changes old B probabilities')


class ProtoMoPCAffine(nn.Module):
    def __init__(self, base, prototypes):
        super().__init__()
        if prototypes.ndim != 2 or len(prototypes) == 0 or not bool(torch.isfinite(prototypes).all()):
            raise ValueError('B prototypes must be finite and nonempty')
        self.base = copy.deepcopy(base).eval()
        rows, dim = prototypes.shape
        self.fc = nn.Linear(dim, rows).to(prototypes)
        with torch.no_grad():
            self.fc.weight.copy_(2 * prototypes)
            self.fc.bias.copy_(-prototypes.square().sum(1))

    def forward(self, x):
        z = self.base(x)['features']
        return {'features': z, 'tx_logits': self.fc(z)}

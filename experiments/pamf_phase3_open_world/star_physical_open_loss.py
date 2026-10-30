"""Pre-unit two-view physical preservation and labeled cross-RX/day contrast."""
from __future__ import annotations

import torch
from torch.nn import functional as F


DIMENSION = 64
TEMPERATURE = .1
EPS = 1e-4


def _features(z, name):
    if (not torch.is_tensor(z) or z.ndim != 2 or z.shape[0] < 2 or z.shape[1] != DIMENSION
            or z.dtype != torch.float32 or not bool(torch.isfinite(z).all())):
        raise ValueError('Finite FP32[B>=2,64] features required: '+name)


def physical(z0, z1):
    """VICReg arithmetic on pre-unit features; no identity labels or negatives."""
    _features(z0, 'first view'); _features(z1, 'second view')
    if z0.shape != z1.shape or z0.device != z1.device:
        raise ValueError('Both views must have aligned shapes and the same device')
    with torch.autocast(device_type=z0.device.type, enabled=False):
        invariance = F.mse_loss(z0, z1)
        std = .5 * (F.relu(1-torch.sqrt(z0.var(dim=0, unbiased=True)+EPS)).mean()
                    + F.relu(1-torch.sqrt(z1.var(dim=0, unbiased=True)+EPS)).mean())
        centered0 = z0-z0.mean(dim=0); centered1 = z1-z1.mean(dim=0)
        cov0 = centered0.T @ centered0 / (len(z0)-1)
        cov1 = centered1.T @ centered1 / (len(z1)-1)
        off_diagonal = ~torch.eye(DIMENSION, dtype=torch.bool, device=z0.device)
        covariance = (cov0[off_diagonal].square().sum()+cov1[off_diagonal].square().sum()) / DIMENSION
        total = 25*invariance+25*std+covariance
    return dict(total=total, invariance=invariance, std=std, covariance=covariance)


def cross_rx_supervised(z, y, rx, day, temperature=TEMPERATURE):
    """Average positive negative-log-probability, then average all anchors.

    Only same-TX pairs differing in both RX and day are positives. Other
    same-TX pairs are absent from the denominator; different TX are negatives.
    The returned positive count is directed, and every anchor needs a positive.
    """
    _features(z, 'unit source L')
    if type(temperature) is not float or temperature != TEMPERATURE:
        raise ValueError('The fixed source contrast temperature is 0.1')
    for name, value in (('y', y), ('rx', rx), ('day', day)):
        if (not torch.is_tensor(value) or value.dtype != torch.int64 or value.shape != (len(z),)
                or value.device != z.device):
            raise ValueError('Aligned int64 source labels on the feature device required: '+name)
    if bool(((y < 0) | (y >= 6)).any()) or bool((rx < 0).any()) or bool((day < 0).any()):
        raise ValueError('Only legal old-six labels and nonnegative source domain IDs allowed')
    if not torch.allclose(z.norm(dim=1), torch.ones(len(z), device=z.device), atol=1e-5, rtol=1e-5):
        raise ValueError('Source contrast requires unit open64 features')
    same = y[:, None] == y[None, :]
    positive = same & (rx[:, None] != rx[None, :]) & (day[:, None] != day[None, :])
    counts = positive.sum(dim=1)
    if bool(counts.eq(0).any()):
        raise ValueError('Every source L anchor needs a same-TX cross-RX-and-day positive')
    allowed = positive | ~same
    with torch.autocast(device_type=z.device.type, enabled=False):
        similarity = z @ z.T / temperature
        denominator = torch.logsumexp(similarity.masked_fill(~allowed, -torch.inf), dim=1)
        positive_mean = (similarity*positive).sum(dim=1)/counts
        total = (denominator-positive_mean).mean()
    return dict(total=total, positive_pair_count=int(counts.sum()), anchors=len(z))

"""Source-only residual metric learning and frozen six-class distance aggregates."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import star_relative_core as relative
from .star_local_support_core import _features


ARMS = ('identity', 'compact_oe', 'compact_oe_structure')
METHODS = ('minimum_relative', 'predicted_identity_mahalanobis')
PRIMARY_ARM = 'compact_oe_structure'
PRIMARY_METHOD = 'predicted_identity_mahalanobis'
TRAIN_STEPS = 1000
PARAMETERS = 20704
RECEIVERS = (1, 3, 4, 6, 8)
METRIC_FIELDS = ('class_means', 'background_mean', 'within_whitener', 'background_whitener')


class MetricNet(nn.Module):
    """A zero-residual initialization preserves the original unit features."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(160, 64)
        self.fc2 = nn.Linear(64, 160)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or not len(x)
                or x.shape[1] != 160 or not x.is_floating_point() or not torch.isfinite(x).all()):
            raise ValueError('MetricNet requires finite floating [N,160] features')
        return F.normalize(x + self.fc2(F.relu(self.fc1(x))), dim=1, eps=1e-12)


def _tensor(value, name):
    if (not isinstance(value, torch.Tensor) or value.ndim != 2 or not len(value)
            or value.shape[1] != 160 or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError('Finite floating [N,160] tensor required: ' + name)
    return value


def losses(z_L, y_L, rx_L, z_OE, x_OE, arm):
    """Cross-RX positives, known-prototype OE exclusion, and optional Gram loss."""
    if arm not in ARMS[1:]:
        raise ValueError('Only the two fixed training arms consume losses')
    known, outlier, original = _tensor(z_L, 'z_L'), _tensor(z_OE, 'z_OE'), _tensor(x_OE, 'x_OE')
    if (len(outlier) < 2 or original.shape != outlier.shape
            or any(a.dtype != known.dtype or a.device != known.device for a in (outlier, original))):
        raise ValueError('OE roles require at least two aligned rows with the known dtype/device')
    for value, name in ((y_L, 'y_L'), (rx_L, 'rx_L')):
        if (not isinstance(value, torch.Tensor) or value.shape != (len(known),)
                or value.dtype != torch.int64 or value.device != known.device):
            raise ValueError('Aligned int64 labels on the feature device required: ' + name)
    if not torch.equal(torch.unique(y_L), torch.arange(6, device=known.device)):
        raise ValueError('Visible source L must contain precisely the six old classes')
    if not torch.isin(rx_L, torch.tensor(RECEIVERS, device=known.device)).all():
        raise ValueError('Only the five source RX labels are allowed')
    same_class = y_L[:, None] == y_L[None, :]
    positive = same_class & (rx_L[:, None] != rx_L[None, :])
    negative = ~same_class
    count = positive.sum(dim=1)
    if not (count > 0).all() or not negative.any(dim=1).all():
        raise ValueError('Every L anchor needs cross-RX positives and other-class negatives')
    similarities = known @ known.T / .1
    denominator = similarities.masked_fill(~(positive | negative), -torch.inf).logsumexp(dim=1, keepdim=True)
    contrast = torch.where(positive, denominator - similarities, 0.).sum(dim=1).div(count).mean()
    prototypes = F.normalize(torch.stack([known[y_L == cls].mean(dim=0) for cls in range(6)]), dim=1, eps=1e-12)
    repulsion = (outlier @ prototypes.T).max(dim=1).values.relu().square().mean()

    def gram_loss():
        original_units = F.normalize(original.detach(), dim=1, eps=1e-12)
        difference = outlier @ outlier.T - original_units @ original_units.T
        off_diagonal = ~torch.eye(len(outlier), device=known.device, dtype=torch.bool)
        return difference[off_diagonal].square().mean()

    if arm == 'compact_oe_structure':
        structure = gram_loss()
        total = contrast + 10 * repulsion + 10 * structure
    else:
        with torch.no_grad():
            structure = gram_loss()
        total = contrast + 10 * repulsion
    return dict(total=total, known_contrast=contrast, OE_repulsion=repulsion, OE_structure=structure)


def _labels(value, n):
    labels = np.asarray(value)
    if (labels.shape != (n,) or labels.dtype.kind not in 'iu'
            or set(np.unique(labels).tolist()) != set(range(6))):
        raise ValueError('Aligned visible source labels must contain precisely old classes 0..5')
    return labels


def sample_indices(L_y, L_rx, OE_count, rng):
    """Draw four per old-TX/source-RX cell and 128 OE rows, without replacement."""
    labels = np.asarray(L_y)
    if labels.ndim != 1:
        raise ValueError('One-dimensional L labels required')
    labels = _labels(labels, len(labels))
    receivers = np.asarray(L_rx)
    if (receivers.shape != labels.shape or receivers.dtype.kind not in 'iu'
            or not np.isin(receivers, RECEIVERS).all()):
        raise ValueError('Aligned integer source RX labels required')
    if (isinstance(OE_count, (bool, np.bool_)) or not isinstance(OE_count, (int, np.integer))
            or OE_count < 128 or not isinstance(rng, np.random.Generator)):
        raise ValueError('At least 128 OE records and an explicit NumPy Generator required')
    groups = [np.flatnonzero((labels == cls) & (receivers == rx))
              for cls in range(6) for rx in RECEIVERS]
    if any(len(group) < 4 for group in groups):
        raise ValueError('Each of the thirty source TX/RX cells needs at least four records')
    known = np.concatenate([rng.choice(group, size=4, replace=False) for group in groups])
    return known.astype(np.int64, copy=False), rng.choice(OE_count, size=128, replace=False).astype(np.int64)


def fit_metric(L_metric_units, L_y):
    """Keep four Gaussian aggregates fitted exclusively to source L units."""
    units, norm = _features(L_metric_units, np.float32)
    labels = _labels(L_y, len(units))
    if len(units) <= 6:
        raise ValueError('Within covariance requires N>6')
    units = units / norm[:, None]
    means = np.stack([units[labels == cls].mean(axis=0) for cls in range(6)])
    background = units.mean(axis=0)
    return dict(class_means=means, background_mean=background,
                within_whitener=relative._whitener(units - means[labels], len(units) - 6),
                background_whitener=relative._whitener(units - background, len(units) - 1))


def _metric(metric):
    shapes = ((6, 160), (160,), (160, 160), (160, 160))
    if not isinstance(metric, dict) or set(metric) != set(METRIC_FIELDS):
        raise ValueError('Only the four frozen Gaussian aggregates are allowed')
    for name, shape in zip(METRIC_FIELDS, shapes):
        array = np.asarray(metric[name])
        if array.shape != shape or array.dtype != np.float64 or not np.isfinite(array).all():
            raise ValueError('Finite FP64 aggregate with fixed shape required: ' + name)
    return metric


def metric_scores(metric_units, closed, metric):
    """Row-local minimum RMD and negative default-predicted-class MD."""
    units, norm = _features(metric_units, np.float32)
    units = units / norm[:, None]
    prediction = np.asarray(closed)
    if (prediction.shape != (len(units),) or prediction.dtype.kind not in 'iu'
            or not np.isin(prediction, range(6)).all()):
        raise ValueError('Aligned frozen default-head predictions in old classes 0..5 required')
    fitted = _metric(metric)
    centered = units - fitted['background_mean']
    white = np.einsum('nd,kd->nk', centered, fitted['within_whitener'], optimize=False)
    centers = np.einsum('nd,kd->nk', fitted['class_means'] - fitted['background_mean'],
                        fitted['within_whitener'], optimize=False)
    distances = np.column_stack([np.einsum('nd,nd->n', white - center, white - center, optimize=False)
                                 for center in centers])
    background = np.einsum('nd,kd->nk', centered, fitted['background_whitener'], optimize=False)
    background_distance = np.einsum('nd,nd->n', background, background, optimize=False)
    result = np.column_stack((background_distance - distances.min(axis=1),
                              -distances[np.arange(len(units)), prediction]))
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite frozen metric scores')
    return result

"""Differentiable temporary source-episode classifier on deployed joint160.

Support labels define this episode's class rows; query features are scored one
packet at a time. There is no persistent classifier, optimizer or pseudo-label
state. This training proxy does not implement Phase2 adaptation or access data.
"""
import torch
from torch.nn import functional as F

FEATURE_DIM = 160
NUM_CLASSES = 6
NORMALIZATION_EPSILON = 1e-4
RIDGE = 0.1
LOGIT_SCALE = 10.0


def _features(value, name):
    if not torch.is_tensor(value) or value.ndim != 2 or value.shape[1] != FEATURE_DIM or len(value) == 0:
        raise ValueError(name + ' must be nonempty features[N,160]')
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(name + ' must use float32 or float64')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + ' contains nonfinite features')


def _labels(value, count, device, name):
    if not torch.is_tensor(value) or value.dtype != torch.long or value.shape != (count,):
        raise ValueError(name + ' must be long labels[N]')
    if value.device != device:
        raise ValueError(name + ' device differs from its features')
    if bool(((value < 0) | (value >= NUM_CLASSES)).any()):
        raise ValueError(name + ' must lie in the six registered source classes')


def ridge_logits(support, support_y, query):
    """Return six-class logits with gradients through support, query and solve.

    The dual system is (S S^T + 0.1 I) A = one_hot(support_y, 6).
    Scoring uses 10 Q S^T A, without an inverse or a stored class-weight matrix.
    Normalization is row-wise; neither fitting nor scoring uses query aggregates.
    """
    _features(support, 'support')
    _features(query, 'query')
    if support.device != query.device or support.dtype != query.dtype:
        raise ValueError('Support/query must have the same device and dtype')
    _labels(support_y, len(support), support.device, 'support_y')
    # Native training may surround the objective with autocast. Keep this fixed
    # linear system in the supplied FP32/FP64 precision without detaching it.
    with torch.autocast(device_type=support.device.type, enabled=False):
        s = F.normalize(support, dim=1, eps=NORMALIZATION_EPSILON)
        q = F.normalize(query, dim=1, eps=NORMALIZATION_EPSILON)
        targets = F.one_hot(support_y, NUM_CLASSES).to(dtype=s.dtype)
        system = s @ s.T + RIDGE * torch.eye(len(s), device=s.device, dtype=s.dtype)
        coefficients = torch.linalg.solve(system, targets)
        return LOGIT_SCALE * (q @ s.T @ coefficients)


def episode_loss(support, support_y, query, query_y):
    """Query cross-entropy for a legal source-only training episode."""
    _features(query, 'query')
    _labels(query_y, len(query), query.device, 'query_y')
    return F.cross_entropy(ridge_logits(support, support_y, query), query_y)

"""ECRS cross-RX discriminative mathematics adapted to A1 labeled identity views.

No physical estimator, new encoder, sampler, or unlabeled TX labels are used.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def _metadata(value, name, rows, device):
    if value is None:
        raise ValueError(f"cross-RX requires {name} metadata")
    value = torch.as_tensor(value, device=device)
    if value.ndim != 1 or value.numel() != rows:
        raise ValueError(f"{name} must have shape [B]")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError(f"{name} must contain integer metadata")
    return value.long()


def _check_z(z):
    if not torch.is_tensor(z) or z.ndim != 2 or not z.is_floating_point() or z.shape[1] == 0:
        raise ValueError("identity features must be floating [B,D] with D>0")


def cross_rx_triplet_loss(z, labels, rx, day, view, label_mask=None, margin=.2):
    """All legal triplets, averaged per anchor, using O(B squared) storage.

    Positives have the same TX and a different RX. Negatives have a different
    TX and the same RX/day/view. Missing metadata entries (<0) are excluded.
    An empty legal set returns a detached zero, leaving unrelated grads absent.
    """
    _check_z(z)
    if not math.isfinite(float(margin)) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    rows = z.shape[0]
    labels, rx, day, view = [_metadata(value, name, rows, z.device) for value, name in
                             ((labels, "labels"), (rx, "rx"), (day, "day"), (view, "view"))]
    valid = (labels >= 0) & (rx >= 0) & (day >= 0) & (view >= 0)
    if label_mask is not None:
        label_mask = torch.as_tensor(label_mask, device=z.device)
        if label_mask.ndim != 1 or label_mask.numel() != rows:
            raise ValueError("label_mask must have shape [B]")
        valid &= label_mask.bool()
    with torch.autocast(device_type=z.device.type, enabled=False):
        legal = valid[:, None] & valid[None, :]
        positive = legal & (labels[:, None] == labels[None, :]) & (rx[:, None] != rx[None, :])
        negative = legal & (labels[:, None] != labels[None, :])
        for meta in (rx, day, view):
            negative &= meta[:, None] == meta[None, :]
        pc, nc = positive.sum(1), negative.sum(1)
        anchors = (pc > 0) & (nc > 0)
        count = int(anchors.sum().item())
        if not count:
            return z.detach().new_zeros((), dtype=torch.float32), 0
        # Invalid rows contribute neither values nor gradients, even if malformed.
        unit = F.normalize(torch.where(valid[:, None], z.float(), 0.), dim=-1)
        dist = 1 - unit @ unit.T
        sorted_neg = dist.masked_fill(~negative, float("inf")).sort(dim=1).values
        prefix = F.pad(sorted_neg.masked_fill(~torch.isfinite(sorted_neg), 0).cumsum(1), (1, 0))
        threshold = (dist + float(margin)).contiguous()
        active = torch.searchsorted(sorted_neg.contiguous(), threshold, right=False)
        hinge_sum = active * threshold - prefix.gather(1, active)
        per_anchor = (hinge_sum * positive).sum(1) / (pc * nc).clamp_min(1)
        return per_anchor[anchors].mean(), count


def labeled_cross_rx_objective(z_clean, labels, rx, day, *, z_leo=None,
                               leo_applied=False, scope="clean", weight=.05,
                               margin=.2, num_classes=6):
    """Reuse only supplied labeled clean and actually applied LEO identity views."""
    _check_z(z_clean)
    if scope not in {"clean", "clean_leo"}:
        raise ValueError("scope must be clean or clean_leo")
    if not math.isfinite(float(weight)) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    if not math.isfinite(float(margin)) or margin < 0 or int(num_classes) < 1:
        raise ValueError("margin and num_classes must be valid")
    rows = z_clean.shape[0]
    labels, rx, day = [_metadata(value, name, rows, z_clean.device) for value, name in
                       ((labels, "labels"), (rx, "rx"), (day, "day"))]
    zero = z_clean.detach().new_zeros((), dtype=torch.float32)
    logs = dict(configured=float(weight > 0), executed=0., valid_anchors=0,
                clean_count=rows, leo_count=0, raw_loss=zero, weighted_loss=zero)
    if weight == 0:
        return zero, logs
    use_leo = scope == "clean_leo" and bool(leo_applied)
    z, view = z_clean, torch.zeros(rows, device=z_clean.device, dtype=torch.long)
    if use_leo:
        _check_z(z_leo)
        if z_leo.shape != z_clean.shape or z_leo.device != z_clean.device or z_leo.dtype != z_clean.dtype:
            raise ValueError("applied LEO must match clean identity shape, dtype and device")
        z = torch.cat((z_clean, z_leo))
        labels, rx, day = (value.repeat(2) for value in (labels, rx, day))
        view = torch.cat((view, torch.ones_like(view)))
        logs["leo_count"] = rows
    raw, count = cross_rx_triplet_loss(z, labels, rx, day, view,
                                      label_mask=labels < int(num_classes), margin=margin)
    weighted = raw * float(weight)
    logs.update(executed=float(count > 0), valid_anchors=count, raw_loss=raw.detach(), weighted_loss=weighted.detach())
    return weighted, logs

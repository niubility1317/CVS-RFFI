"""Tensor-only source membership and domain-balanced protected160 geometry."""
from __future__ import annotations

import torch
from torch.nn import functional as F


DIMENSION = 160
CLASS_COUNT = 6
TEMPERATURE = .1
UNIT_EPS = 1e-12


def _matrix(value, width, name):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.ndim != 2
            or value.shape[0] == 0 or value.shape[1] != width or not bool(torch.isfinite(value).all())):
        raise ValueError(name + ' must be a nonempty finite floating tensor [N,' + str(width) + ']')
    return value


def _long(value, count, device, name):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64
            or value.shape != (count,) or value.device != device):
        raise ValueError(name + ' must be aligned int64 on the feature device')
    return value


def _old_labels(value, count, device, name):
    _long(value, count, device, name)
    if bool(((value < 0) | (value >= CLASS_COUNT)).any()):
        raise ValueError(name + ' must contain only visible source old labels 0..5')
    return value


def query_prototypes(support, support_y, support_rx, support_day, query_rx, query_day):
    """Per-query unit means using different RX AND day support for every class.

    Unit-or-zero support is accepted. An exactly zero mean remains zero. Empty
    legal support for any query/class raises rather than adding same-domain rows.
    Gradients are retained through the legal support features.
    """
    _matrix(support, DIMENSION, 'support')
    n, device = len(support), support.device
    _old_labels(support_y, n, device, 'support_y')
    _long(support_rx, n, device, 'support_rx')
    _long(support_day, n, device, 'support_day')
    if not isinstance(query_rx, torch.Tensor) or query_rx.ndim != 1 or len(query_rx) == 0:
        raise ValueError('Nonempty public query RX/day vectors required')
    _long(query_rx, len(query_rx), device, 'query_rx')
    _long(query_day, len(query_rx), device, 'query_day')
    norms = support.norm(dim=1)
    if not bool(((norms == 0) | torch.isclose(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)).all()):
        raise ValueError('Support features must be unit or exact-zero')
    domain_mask = (query_rx[:, None] != support_rx[None]) & (query_day[:, None] != support_day[None])
    classes = torch.arange(CLASS_COUNT, device=device)
    legal = domain_mask[:, None, :] & (support_y[None, None, :] == classes[None, :, None])
    counts = legal.sum(dim=2)
    if bool((counts == 0).any()):
        raise ValueError('Every query/class requires at least one different-RX-and-day support')
    means = legal.to(support.dtype) @ support / counts[:, :, None]
    return F.normalize(means, dim=-1, eps=UNIT_EPS)


def member_loss(query_relation, query_visible_old_y, pseudo_known_classes):
    """Balanced set BCE over only 2/3/4/5 selected visible old-class slots."""
    _matrix(query_relation, CLASS_COUNT, 'query_relation')
    n, device = len(query_relation), query_relation.device
    _old_labels(query_visible_old_y, n, device, 'query_visible_old_y')
    if not isinstance(pseudo_known_classes, torch.Tensor) or pseudo_known_classes.ndim != 1:
        raise ValueError('A tensor of 2..5 unique pseudo-known old classes is required')
    k = len(pseudo_known_classes)
    _old_labels(pseudo_known_classes, k, device, 'pseudo_known_classes')
    if not 2 <= k <= 5 or pseudo_known_classes.unique().numel() != k:
        raise ValueError('Exactly 2..5 unique pseudo-known old-class slots required')
    known = (query_visible_old_y[:, None] == pseudo_known_classes[None]).any(dim=1)
    known_count = int(known.sum())
    if known_count == 0 or known_count == n:
        raise ValueError('Both simulated known and unknown L query groups are required')
    logits = query_relation[:, pseudo_known_classes].max(dim=1).values
    known_loss = F.softplus(-logits[known]).mean()
    unknown_loss = F.softplus(logits[~known]).mean()
    return dict(loss=.5 * (known_loss + unknown_loss), known_loss=known_loss, unknown_loss=unknown_loss,
                known_count=known_count, unknown_count=n - known_count, selected_class_count=k)


def oe_member_loss(oe_relation):
    """Outside-six membership supervision, with no OE identity or K inputs."""
    _matrix(oe_relation, CLASS_COUNT, 'oe_relation')
    return dict(loss=F.softplus(oe_relation.max(dim=1).values).mean(), count=len(oe_relation))


def soft_neighbor_loss(student, teacher, rx, day, opaque_ids):
    """Domain-balanced KL(teacher || student) within a single source role.

    Teacher and student cosine distributions use the same public mask and
    inverse eligible candidate-cell counts. Anchor means are then averaged
    equally across all nonempty RX/day cells. L/OE averaging is the caller's
    responsibility; this function creates no persistent state or hard labels.
    """
    _matrix(student, DIMENSION, 'student')
    _matrix(teacher, DIMENSION, 'teacher')
    if teacher.shape != student.shape or teacher.device != student.device or teacher.dtype != student.dtype:
        raise ValueError('Teacher/student must have aligned shape, dtype and device')
    n, device = len(student), student.device
    for name, value in (('rx', rx), ('day', day), ('opaque_ids', opaque_ids)):
        _long(value, n, device, name)
    _, cells = torch.unique(torch.stack((rx, day), dim=1), dim=0, return_inverse=True)
    cell_count = int(cells.max()) + 1
    eligible = (rx[:, None] != rx[None]) & (day[:, None] != day[None]) & (opaque_ids[:, None] != opaque_ids[None])
    cell_matrix = F.one_hot(cells, num_classes=cell_count).to(student.dtype)
    eligible_per_cell = eligible.to(student.dtype) @ cell_matrix
    candidate_counts = eligible.sum(dim=1)
    candidate_cells = (eligible_per_cell > 0).sum(dim=1)
    if bool((candidate_counts < 2).any()) or bool((candidate_cells < 2).any()):
        raise ValueError('Every anchor requires at least two eligible candidates in at least two RX/day cells')
    logweight = -eligible_per_cell[:, cells].clamp_min(1).to(student.dtype).log()
    teacher_unit = F.normalize(teacher.detach(), dim=1, eps=UNIT_EPS)
    student_unit = F.normalize(student, dim=1, eps=UNIT_EPS)
    teacher_logits = (teacher_unit @ teacher_unit.T / TEMPERATURE + logweight).masked_fill(~eligible, -torch.inf)
    student_logits = (student_unit @ student_unit.T / TEMPERATURE + logweight).masked_fill(~eligible, -torch.inf)
    probability = teacher_logits.softmax(dim=1)
    # Clear invalid log entries before subtraction: 0*(-inf-(-inf)) is NaN.
    teacher_log = teacher_logits.log_softmax(dim=1).masked_fill(~eligible, 0.)
    student_log = student_logits.log_softmax(dim=1).masked_fill(~eligible, 0.)
    per_anchor = (probability * (teacher_log - student_log)).sum(dim=1)
    totals = torch.zeros(cell_count, dtype=student.dtype, device=device).scatter_add(0, cells, per_anchor)
    domain_means = totals / torch.bincount(cells, minlength=cell_count).to(student.dtype)
    return dict(loss=domain_means.mean(), anchor_count=n, domain_count=cell_count,
                candidate_count_min=int(candidate_counts.min()), candidate_count_max=int(candidate_counts.max()),
                candidate_domain_count_min=int(candidate_cells.min()), candidate_domain_count_max=int(candidate_cells.max()))

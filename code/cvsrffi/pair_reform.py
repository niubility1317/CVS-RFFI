"""Source training pair objectives with separate physical/classification semantics.

Safety is a local sufficient condition for a fixed bias-free cosine head only.
It is not a guarantee for unlabeled examples or receiver generalization. All
objectives use a fixed batch denominator; teacher targets and weights stop grad.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _unit(x: Tensor) -> Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-8)


def _weight(x: Tensor, batch: int) -> Tensor:
    x = x.detach().float()
    if x.shape != (batch,):
        raise ValueError('weights must have shape [B]')
    if not torch.isfinite(x).all() or ((x < 0) | (x > 1)).any():
        raise ValueError('weights must be finite in [0, 1]')
    return x


def physical_reliability(values: Tensor, valid: Tensor, *, unknown_policy: str = 'neutral') -> tuple[Tensor, Tensor]:
    """Return physical weights and known-quality flags, never a cache/label score.

    Unknown policy 'neutral' assigns 0.5; 'zero' suppresses the objective. Neither
    policy is a measurement or a claim about the unknown signal's quality.
    """
    if unknown_policy not in ('neutral', 'zero'):
        raise ValueError('unknown_policy must be neutral or zero')
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError('physical values and validity must have shape [B]')
    values = values.detach().float()
    known = valid.detach().bool() & torch.isfinite(values) & (values >= 0) & (values <= 1)
    default = .5 if unknown_policy == 'neutral' else 0.
    return torch.where(known, values, torch.full_like(values, default)), known


@torch.no_grad()
def cosine_safety_radius(teacher_features: Tensor, class_weights: Tensor, labels: Tensor,
                         alpha: float = .5, *, head_kind: str = 'bias_free_cosine',
                         bias: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Distance to nearest class plane under ONE detached normalized weight set.

    Caller must obtain class_weights from the actual fixed cosine classifier;
    a nonlinear head's last matrix is not a compatible substitute. A positive
    common logit scale does not affect this geometry. Ties/zero normals/incorrect
    anchors have zero radius and are invalid, including any nonfinite input.
    """
    if head_kind != 'bias_free_cosine' or bias is not None:
        raise ValueError('safety geometry requires a bias-free cosine head')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one')
    if teacher_features.ndim != 2 or class_weights.ndim != 2 or class_weights.shape[1] != teacher_features.shape[1]:
        raise ValueError('features [B,D] and class weights [C,D] required')
    if class_weights.shape[0] < 2 or labels.shape != teacher_features.shape[:1]:
        raise ValueError('at least two classes and labels [B] required')
    if labels.dtype not in (torch.int32, torch.int64) or ((labels < 0) | (labels >= class_weights.shape[0])).any():
        raise ValueError('labels must be valid integer class indices')
    labels = labels.long()
    finite_t = torch.isfinite(teacher_features).all(-1)
    finite_w = torch.isfinite(class_weights).all()
    t = _unit(torch.nan_to_num(teacher_features.detach(), nan=0., posinf=0., neginf=0.))
    w = _unit(torch.nan_to_num(class_weights.detach(), nan=0., posinf=0., neginf=0.))
    delta = w[labels, None, :] - w[None, :, :]
    lengths = delta.norm(dim=-1)
    other = torch.arange(w.shape[0], device=w.device)[None, :] != labels[:, None]
    signed = (delta * t[:, None, :]).sum(-1) / lengths.clamp_min(1e-8)
    gamma = signed.masked_fill(~other, float('inf')).amin(-1)
    valid = (finite_t & finite_w & (teacher_features.float().norm(dim=-1) > 1e-8)
             & (class_weights.float().norm(dim=-1) > 1e-8).all()
             & ((lengths > 1e-8) | ~other).all(-1)
             & ((t @ w.T).argmax(-1) == labels) & torch.isfinite(gamma) & (gamma > 0))
    return torch.where(valid, alpha * gamma, torch.zeros_like(gamma)), valid


def safe_pair_loss(student: Tensor, teacher: Tensor, class_weights: Tensor, labels: Tensor,
                   r_phys: Tensor, alpha: float = .5, *, head_kind: str = 'bias_free_cosine',
                   bias: Tensor | None = None) -> Tensor:
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError('paired features must have matching [B,D] shape')
    radius, valid = cosine_safety_radius(teacher, class_weights, labels, alpha, head_kind=head_kind, bias=bias)
    # Remove invalid anchors before arithmetic: zero times NaN is still NaN.
    anchor = torch.where(valid[:, None], teacher.detach(), torch.zeros_like(teacher))
    distance = (_unit(student) - _unit(anchor)).square().sum(-1)
    return (_weight(r_phys, student.shape[0]) * valid * F.relu(distance - radius.square())).mean()


def point_pair_loss(student: Tensor, teacher: Tensor, r_phys: Tensor, tolerance: float = 0.) -> Tensor:
    """Class-independent squared unit-distance hinge, including unlabeled pairs."""
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError('paired features must have matching [B,D] shape')
    if not 0 <= tolerance <= 2:
        raise ValueError('unit-feature distance tolerance must be in [0,2]')
    distance = (_unit(student) - _unit(teacher.detach())).square().sum(-1)
    return (_weight(r_phys, student.shape[0]) * F.relu(distance - tolerance ** 2)).mean()


@torch.no_grad()
def asymmetric_teacher_target(clean_teacher: Tensor, leo_teacher: Tensor | None = None,
                              leo_mix: float | Tensor = 0.) -> Tensor:
    """Clean-dominant target; optional LEO must be an actual teacher observation."""
    clean = _unit(clean_teacher)
    mix = torch.as_tensor(leo_mix, device=clean.device, dtype=clean.dtype).detach()
    if not torch.isfinite(mix).all() or ((mix < 0) | (mix > .5)).any():
        raise ValueError('LEO teacher mixture must lie in [0,.5]')
    if mix.ndim == 1:
        if mix.shape != clean.shape[:1]:
            raise ValueError('LEO mixture must be scalar or [B]')
        mix = mix[:, None]
    elif mix.ndim != 0:
        raise ValueError('LEO mixture must be scalar or [B]')
    if leo_teacher is None:
        if (mix != 0).any():
            raise ValueError('nonzero mixture requires real LEO teacher output')
        return clean
    if leo_teacher.shape != clean_teacher.shape:
        raise ValueError('teacher feature shapes must match')
    return _unit((1 - mix) * clean + mix * _unit(leo_teacher))


def asymmetric_pair_loss(student_leo: Tensor, clean_teacher: Tensor, r_phys: Tensor,
                         leo_teacher: Tensor | None = None, leo_mix: float | Tensor = 0.) -> Tensor:
    return point_pair_loss(student_leo, asymmetric_teacher_target(clean_teacher, leo_teacher, leo_mix), r_phys)


def _probabilities(p: Tensor) -> Tensor:
    p = p.detach().float()
    if p.ndim != 2 or p.shape[1] < 2 or not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError('real teacher probabilities must be finite nonnegative [B,C]')
    if not torch.allclose(p.sum(-1), torch.ones_like(p[:, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError('teacher probabilities must sum to one')
    return p


def classification_confidence(teacher_probs: Tensor, other_teacher_probs: Tensor | None = None,
                              *, confidence_power: float = 1., margin_power: float = 1.,
                              js_scale: float = 1.) -> Tensor:
    """Heuristic classification weight, not calibrated correctness probability.

    Normalized excess confidence times top-two margin, with optional exp(-s*JS).
    Uniform outputs always have zero weight; physical/cache scores never enter.
    """
    if (not all(math.isfinite(x) for x in (confidence_power, margin_power, js_scale))
            or confidence_power <= 0 or margin_power <= 0 or js_scale < 0):
        raise ValueError('confidence/margin powers must be positive and JS scale nonnegative')
    p = _probabilities(teacher_probs)
    top = p.topk(2, dim=-1).values
    excess = ((top[:, 0] - 1 / p.shape[1]) / (1 - 1 / p.shape[1])).clamp(0, 1)
    q = excess.pow(confidence_power) * (top[:, 0] - top[:, 1]).pow(margin_power)
    if other_teacher_probs is not None:
        other = _probabilities(other_teacher_probs)
        if other.shape != p.shape:
            raise ValueError('teacher probability shapes must match')
        midpoint = .5 * (p + other)
        logmid = midpoint.clamp_min(1e-12).log()
        js = .5 * ((p * (p.clamp_min(1e-12).log() - logmid)).sum(-1)
                   + (other * (other.clamp_min(1e-12).log() - logmid)).sum(-1))
        q = q * torch.exp(-js_scale * js.clamp_min(0))
    return q.detach()


def unified_soft_ce(student_logits: Tensor, teacher_probs: Tensor, r_phys: Tensor, q_cls: Tensor) -> Tensor:
    """One real teacher soft target, weighted by r_phys*q_cls, divided by B."""
    target = _probabilities(teacher_probs)
    if student_logits.shape != target.shape:
        raise ValueError('student logits and real teacher target must have matching shape')
    batch = student_logits.shape[0]
    ce = -(target * F.log_softmax(student_logits.float(), dim=-1)).sum(-1)
    return (_weight(r_phys, batch) * _weight(q_cls, batch) * ce).mean()

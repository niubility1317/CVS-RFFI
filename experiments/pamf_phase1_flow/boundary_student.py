"""Bounded source-trained student for amortized classifier-boundary corrections."""
import torch
from torch import nn

from .boundary_source_oracle import projected_targeted_steps


class BoundedBoundaryStudent(nn.Module):
    """Predict a bounded correction from received IQ and a frozen DURA anchor."""
    def __init__(self, field, *, epsilon=0.05):
        super().__init__()
        if not 0 < epsilon <= 1:
            raise ValueError('Positive finite perturbation radius required')
        self.field = field
        self.epsilon = float(epsilon)

    def forward(self, received, anchor):
        if received.shape != anchor.shape or received.ndim != 3 or received.shape[1:] != (2,256):
            raise ValueError('One received and corresponding anchor IQ window required')
        proposed = self.field.velocity(anchor, received, received.new_zeros(len(received)))
        radius = self.epsilon * anchor.flatten(1).norm(dim=1)
        magnitude = proposed.flatten(1).norm(dim=1).clamp_min(1e-12)
        scale = (radius / magnitude).clamp(max=1)
        return anchor + proposed * scale[:,None,None]


def distillation_loss(student, oracle, anchor, wrong, *, hard_weight=8.0):
    if student.shape != oracle.shape or student.shape != anchor.shape or wrong.shape != (len(anchor),):
        raise ValueError('Aligned source student, target, anchor and mismatch mask required')
    if wrong.dtype != torch.bool or hard_weight < 1:
        raise ValueError('Boolean mismatch mask and valid hard weighting required')
    energy = anchor.detach().square().mean(tuple(range(1,anchor.ndim))).clamp_min(1e-8)
    per_row = (student-oracle.detach()).square().mean(tuple(range(1,anchor.ndim))) / energy
    weights = torch.where(wrong, hard_weight, 1.0).to(per_row)
    return (per_row * weights).sum() / weights.sum()


def make_source_oracle(teacher, anchor, clean, *, epsilon=0.05, steps=12):
    """Generate privileged targets only during source training; no TX labels."""
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Frozen evaluation-mode teacher required')
    def classify(x):
        return teacher.classify_features(teacher.features(x))
    with torch.no_grad():
        target = classify(clean).argmax(1)
        before = classify(anchor).argmax(1)
        wrong = before != target
    result = anchor.detach().clone()
    if wrong.any():
        radius_norm = anchor[wrong].detach().flatten(1).norm(dim=1).clamp_min(1e-8)
        result[wrong] = projected_targeted_steps(classify,anchor[wrong],target[wrong],radius_norm,
                                                 epsilon=epsilon,steps=steps)
    return result, wrong

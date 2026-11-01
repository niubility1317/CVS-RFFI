"""Class-specific bounded proposals with a verified abstention gate."""
import torch
from torch import nn
from torch.nn import functional as F

from .boundary_source_oracle import projected_targeted_steps


def source_oracle(teacher, anchor, target, valid, *, epsilon=.05, steps=12):
    """Generate input-only targets for valid source mismatches; never use at inference."""
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Frozen evaluation-mode teacher required')
    if target.shape != (len(anchor),) or valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError('Aligned source targets and boolean validity required')
    def classify(x):
        return teacher.classify_features(teacher.features(x))
    with torch.no_grad():
        mismatch = valid & (classify(anchor).argmax(1) != target)
    oracle = anchor.detach().clone()
    reached = torch.zeros_like(valid)
    if mismatch.any():
        norm = anchor[mismatch].detach().flatten(1).norm(dim=1).clamp_min(1e-8)
        oracle[mismatch] = projected_targeted_steps(classify, anchor[mismatch], target[mismatch], norm,
                                                     epsilon=epsilon, steps=steps)
        with torch.no_grad():
            reached[mismatch] = classify(oracle[mismatch]).argmax(1) == target[mismatch]
    return oracle, reached


class ClassHypothesis(nn.Module):
    """Predict a class hypothesis and a bounded candidate for a chosen class."""
    def __init__(self, *, feature_dim=160, classes=6, width=64, layers=3, heads=4, epsilon=.05):
        super().__init__()
        if not 0 < epsilon <= 1:
            raise ValueError('Positive bounded radius required')
        self.feature_dim, self.classes, self.epsilon = feature_dim, classes, float(epsilon)
        self.input = nn.Conv1d(4, width, 4, stride=4)
        self.position = nn.Parameter(torch.randn(1, 64, width) * .01)
        self.condition = nn.Sequential(nn.Linear(feature_dim + classes, width * 2), nn.SiLU(),
                                       nn.Linear(width * 2, width))
        block = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0., activation='gelu',
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.posterior = nn.Linear(width, classes)
        self.output = nn.Linear(width, classes * 8)
        nn.init.zeros_(self.posterior.weight)
        nn.init.zeros_(self.posterior.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, received, anchor, features, logits, target=None):
        n = len(received)
        if received.ndim != 3 or received.shape[1:] != (2, 256) or anchor.shape != received.shape:
            raise ValueError('Aligned received and anchor IQ windows required')
        if features.shape != (n, self.feature_dim) or logits.shape != (n, self.classes):
            raise ValueError('One frozen teacher feature and score vector per record required')
        context = torch.cat((F.normalize(features.detach(), dim=1), F.softmax(logits.detach(), dim=1)), dim=1)
        h = self.input(torch.cat((anchor, received), dim=1)).transpose(1, 2)
        h = self.encoder(h + self.position + self.condition(context)[:, None, :])
        posterior_logits = logits.detach() + self.posterior(h.mean(dim=1))
        chosen = posterior_logits.argmax(1) if target is None else target
        if chosen.shape != (n,) or chosen.dtype != torch.long or ((chosen < 0) | (chosen >= self.classes)).any():
            raise ValueError('One valid integer candidate class per record required')
        all_proposals = self.output(h).reshape(n, 64, self.classes, 2, 4)
        gather = chosen[:, None, None, None, None].expand(n, 64, 1, 2, 4)
        proposed = all_proposals.gather(2, gather).squeeze(2).permute(0, 2, 1, 3).reshape_as(received)
        radius = self.epsilon * anchor.flatten(1).norm(dim=1)
        magnitude = proposed.flatten(1).norm(dim=1).clamp_min(1e-12)
        return posterior_logits, anchor + proposed * (radius / magnitude).clamp(max=1)[:, None, None]


def select_verified(anchor, candidate, posterior, baseline_class, candidate_class, *, threshold=.95):
    """Return the candidate only after confidence, disagreement, and teacher checks."""
    if anchor.shape != candidate.shape or posterior.ndim != 2 or len(posterior) != len(anchor):
        raise ValueError('Aligned candidates and class probabilities required')
    if baseline_class.shape != (len(anchor),) or candidate_class.shape != (len(anchor),) or not 0 <= threshold <= 1:
        raise ValueError('Aligned class decisions and valid threshold required')
    confidence, proposed = posterior.max(dim=1)
    accepted = (confidence >= threshold) & (proposed != baseline_class) & (candidate_class == proposed)
    return torch.where(accepted[:, None, None], candidate, anchor), accepted

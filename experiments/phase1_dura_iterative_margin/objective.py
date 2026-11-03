"""Source-only confidence-gated recognition margin relative to a frozen DURA anchor."""
import torch
from torch.nn import functional as F

from experiments.phase1_dura_safe_memory.objective import protected_source_objective


def relative_margin_penalty(output_logits, anchor_logits, clean_logits, groups):
    """Protect correct anchors and move wrong anchors toward source pseudo-targets.

    Clean and anchor logits are source-training references only. They and the
    discrete target are detached; inference needs only the resulting network.
    """
    n = len(output_logits)
    if (n < 1 or output_logits.shape != (n, 6) or anchor_logits.shape != (n, 6)
            or clean_logits.shape != (n, 6) or groups.shape != (n,)):
        raise ValueError('Matching six-class logits and one receiver group per record are required')
    if not all(torch.isfinite(x).all() for x in (output_logits, anchor_logits, clean_logits)):
        raise FloatingPointError('Recognition margin requires finite logits')
    with torch.no_grad():
        teacher = clean_logits.detach()
        confidence, target = F.softmax(teacher, dim=1).max(dim=1)
        source_anchor = anchor_logits.detach()
        mask = F.one_hot(target, 6).bool()
        anchor_margin = (source_anchor.gather(1, target[:, None]).squeeze(1)
                         - source_anchor.masked_fill(mask, -torch.inf).max(dim=1).values)
        anchor_correct = source_anchor.argmax(dim=1) == target
        target_margin = torch.where(anchor_correct, anchor_margin.clamp(max=2.),
                                    torch.full_like(anchor_margin, .5))
        eligible = confidence >= .95
    candidate_margin = (output_logits.gather(1, target[:, None]).squeeze(1)
                        - output_logits.masked_fill(mask, -torch.inf).max(dim=1).values)
    violations = F.relu(target_margin-candidate_margin)
    per_group = [violations[(groups == group) & eligible].mean()
                 for group in groups.unique() if ((groups == group) & eligible).any()]
    return torch.stack(per_group).mean() if per_group else output_logits.sum()*0.


def recognition_protected_objective(reader, output, anchor, clean, clean_features, groups):
    protected, terms = protected_source_objective(reader, output, anchor, clean, clean_features, groups)
    with torch.no_grad():
        clean_logits = reader.classify_features(clean_features.detach())
        anchor_logits = reader.classify_features(reader.features(anchor.detach()))
    output_logits = reader.classify_features(reader.features(output))
    margin = relative_margin_penalty(output_logits, anchor_logits, clean_logits, groups)
    loss = protected + .1*margin
    if not torch.isfinite(loss):
        raise FloatingPointError('Finite recognition-protected source loss required')
    return loss, dict(terms, relative_recognition_margin=margin)

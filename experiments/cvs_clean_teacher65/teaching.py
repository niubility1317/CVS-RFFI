"""One batch-normalized hidden-U KL family; only teacher provenance differs."""
import torch
from torch.nn import functional as F

ARMS = ('ZERO_KL', 'EVOLVING_AS_EMA', 'FROZEN_AS20_EMA', 'FROZEN_NOAS20_EMA')
TEACHERS = ARMS[1:]


def weight(arm, local_epoch, policy):
    if arm not in ARMS or local_epoch not in range(1, 11):
        raise ValueError('Only registered teacher arm and new E1..10')
    if (policy.get('tau'), policy.get('max_weight'), policy.get('ramp_epochs')) != (.95, .16, 5):
        raise ValueError('Fixed confidence mass and five-local-epoch ramp required')
    return 0. if arm == 'ZERO_KL' else .16 * min(1., local_epoch / 5.)


def _statistics(logits, q, per_sample):
    p = logits.detach().softmax(1)
    confidence, pseudo = p.max(1)
    positive = q > 0
    return dict(
        q_mean=float(q.mean()), q_mass=float(q.sum()), positive_q=int(positive.sum()),
        pmax_mean=float(confidence.mean()),
        pmax_quantiles=[float(v) for v in torch.quantile(confidence, confidence.new_tensor([0., .1, .5, .9, 1.]))],
        q_quantiles=[float(v) for v in torch.quantile(q, q.new_tensor([0., .1, .5, .9, 1.]))],
        q_histogram=[int((q == 0).sum()), int(((q > 0) & (q <= .25)).sum()),
                     int(((q > .25) & (q <= .5)).sum()), int(((q > .5) & (q <= .75)).sum()),
                     int((q > .75).sum())],
        pseudo_counts=torch.bincount(pseudo, minlength=6).cpu().tolist(),
        positive_pseudo_counts=torch.bincount(pseudo[positive], minlength=6).cpu().tolist(),
        KL=float((q * per_sample).mean()), unweighted_KL=float(per_sample.mean()))


def teaching_loss(student, teachers, hidden_y, arm, local_epoch, policy):
    """Always calculate all three teacher readouts, including the zero-KL arm.

    q weights only this extra KL. The original U mask and every labeled As CE
    are outside this function and are never changed by its confidence rule.
    """
    coefficient = weight(arm, local_epoch, policy)
    if hidden_y.dtype != torch.long or hidden_y.ndim != 1 or not bool((hidden_y == -1).all()):
        raise ValueError('U TX must remain hidden -1 sentinels')
    n = len(hidden_y)
    if n == 0 or student.shape != (n, 6) or not bool(torch.isfinite(student).all()):
        raise ValueError('Finite original six-class student logits required')
    if set(teachers) != set(TEACHERS):
        raise ValueError('All three matched teacher readouts are required')
    values, details, predicted = {}, {}, {}
    for name in TEACHERS:
        logits = teachers[name]
        if logits.shape != student.shape or not bool(torch.isfinite(logits).all()):
            raise ValueError('Matched finite default-head teacher logits required')
        p = logits.detach().softmax(1)
        q = ((p.max(1).values - .95) / .05).clamp(0., 1.)
        per_sample = F.kl_div(F.log_softmax(student, 1), p, reduction='none').sum(1)
        values[name] = (q * per_sample).mean()
        details[name] = _statistics(logits, q.detach(), per_sample.detach())
        predicted[name] = p.argmax(1)
    selected = TEACHERS[0] if arm == 'ZERO_KL' else arm
    loss = coefficient * values[selected]
    record = dict(arm=arm, selected_teacher=selected, weight=coefficient,
                  KL=float(values[selected].detach()), weighted_KL=float(loss.detach()),
                  q_mean=details[selected]['q_mean'], positive_q=details[selected]['positive_q'],
                  teachers=details,
                  teacher_disagreement={a + '__' + b: float((predicted[a] != predicted[b]).float().mean())
                                        for i, a in enumerate(TEACHERS) for b in TEACHERS[i + 1:]},
                  U_TX_truth_read=False, urban_confidence_filter=False,
                  old_U_masks_unchanged=True, temperature=1., selected_count_normalization=False)
    return loss, record

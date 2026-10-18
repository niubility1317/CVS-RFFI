"""As-only probability marginal CE over the two existing dropout logits.

No temperature, feature constraint, label gate, KL change, or new forward.
"""
import math

import torch

MODES = ('MEAN_CE', 'PROBABILITY_MEAN_CE')
SMOOTHING = .01
CLASSES = 6


def _validate(first, second, labels, mode, quality_groups):
    if mode not in MODES:
        raise ValueError('Unregistered As CE aggregation mode')
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise ValueError('Two default-head logit tensors required')
    if first.ndim != 2 or first.shape[1] != CLASSES or len(first) == 0 or first.shape != second.shape:
        raise ValueError('Two matching nonempty Nx6 default-head logits required')
    if not first.is_floating_point() or second.dtype != first.dtype or second.device != first.device:
        raise ValueError('Matching floating logit dtype/device required')
    if not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all()):
        raise ValueError('Nonfinite default-head logits')
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long or labels.shape != (len(first),) or labels.device != first.device or not bool(((labels >= 0) & (labels < CLASSES)).all()):
        raise ValueError('True legal source TX labels required')
    if quality_groups is not None and (not isinstance(quality_groups, torch.Tensor) or quality_groups.dtype != torch.long or quality_groups.shape != labels.shape or quality_groups.device != first.device or not bool(((quality_groups >= 0) & (quality_groups < 4)).all())):
        raise ValueError('Original four-bin source quality groups required')


def _soft_ce(log_probability, labels):
    # Target = (1-eps)*one_hot(y) + eps/K, including eps/K on y itself.
    true_log_probability = log_probability.gather(1, labels[:, None]).squeeze(1)
    return -(1. - SMOOTHING) * true_log_probability - (SMOOTHING / CLASSES) * log_probability.sum(1)


def _probability_mean_log(log0, log1):
    # This is logsumexp(log0,log1)-log(2). Factoring the shared maximum
    # keeps extreme logits stable and makes identical log probabilities exact.
    maximum = torch.maximum(log0, log1)
    offsets = torch.stack((log0 - maximum, log1 - maximum), 0)
    return maximum + (torch.logsumexp(offsets, 0) - math.log(2.))


def _metrics(log0, log1, pooled, labels, mean_per, probability_per, mode):
    count = len(labels)
    if not count:
        return dict(count=0, mean_CE=None, probmean_CE=None, Jensen_gap=None, used_CE=None,
                    mask_disagreement_count=0, true_probability_mask0=None,
                    true_probability_mask1=None, true_probability_pooled=None,
                    mask0_correct=0, mask1_correct=0, pooled_correct=0,
                    pooled_vs_mask0=dict(repaired=0, new_errors=0, both_correct=0, both_wrong=0),
                    pooled_vs_mask1=dict(repaired=0, new_errors=0, both_correct=0, both_wrong=0))
    pred0, pred1, pooled_pred = log0.argmax(1), log1.argmax(1), pooled.argmax(1)
    correct0, correct1, correct_pool = pred0 == labels, pred1 == labels, pooled_pred == labels
    comparisons = []
    for correct in (correct0, correct1):
        comparisons.append(dict(repaired=int((~correct & correct_pool).sum()),
                                new_errors=int((correct & ~correct_pool).sum()),
                                both_correct=int((correct & correct_pool).sum()),
                                both_wrong=int((~correct & ~correct_pool).sum())))
    mean, probability = float(mean_per.mean()), float(probability_per.mean())
    return dict(count=count, mean_CE=mean, probmean_CE=probability,
                Jensen_gap=mean - probability, used_CE=probability if mode == 'PROBABILITY_MEAN_CE' else mean,
                mask_disagreement_count=int((pred0 != pred1).sum()),
                true_probability_mask0=float(log0.gather(1, labels[:, None]).exp().mean()),
                true_probability_mask1=float(log1.gather(1, labels[:, None]).exp().mean()),
                true_probability_pooled=float(pooled.gather(1, labels[:, None]).exp().mean()),
                mask0_correct=int(correct0.sum()), mask1_correct=int(correct1.sum()), pooled_correct=int(correct_pool.sum()),
                pooled_vs_mask0=comparisons[0], pooled_vs_mask1=comparisons[1])


def terms(first, second, labels, mode, quality_groups=None):
    """Return the live, unscaled As CE correction and detached diagnostics.

    The caller applies the existing .25 As CE coefficient.  Existing symmetric
    KL stays outside this helper.  quality_groups only partitions diagnostics.
    """
    _validate(first, second, labels, mode, quality_groups)
    log0, log1 = first.log_softmax(1), second.log_softmax(1)
    pooled = _probability_mean_log(log0, log1)
    if not bool(torch.isfinite(pooled).all()):
        raise ValueError('Nonfinite normalized default-head probabilities')
    mean_per = .5 * (_soft_ce(log0, labels) + _soft_ce(log1, labels))
    probability_per = _soft_ce(pooled, labels)
    difference = probability_per.mean() - mean_per.mean()
    delta = difference if mode == 'PROBABILITY_MEAN_CE' else difference * 0.
    with torch.no_grad():
        report = _metrics(log0, log1, pooled, labels, mean_per, probability_per, mode)
        report.update(mode=mode, label_smoothing=SMOOTHING,
                      probability_rule='arithmetic mean of two softmax probabilities; stable logsumexp',
                      prediction_reference='each mask default-head argmax separately; no mean-logit reference',
                      group_stats=[])
        if quality_groups is not None:
            for group in range(4):
                selected = quality_groups == group
                values = _metrics(log0[selected], log1[selected], pooled[selected], labels[selected],
                                  mean_per[selected], probability_per[selected], mode)
                report['group_stats'].append(dict(group=group, **values))
    return delta, report

"""Source probability summaries; hidden-U paths never accept TX labels."""
import torch

from . import design as d


def _probabilities(q):
    if q.ndim != 2 or q.shape[1] != d.CLASSES or len(q) == 0:
        raise ValueError('Nonempty six-class probabilities required')
    if q.requires_grad or not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError('Detached finite probabilities required')
    if not torch.allclose(q.sum(1), torch.ones(len(q), device=q.device), atol=1e-5):
        raise ValueError('Probabilities must sum to one')


def _safe_rate(numerator, denominator):
    return numerator / denominator if denominator else None


def calibration(q, y=None):
    """Fixed 10 equal-width bins, without fitting a temperature or threshold."""
    _probabilities(q)
    confidence, prediction = q.max(1)
    high = confidence >= d.THRESHOLD
    result = dict(count=len(q), mean_confidence=float(confidence.mean()),
                  mean_entropy=float(-(q * q.clamp_min(1e-12).log()).sum(1).mean()),
                  prediction_counts=torch.bincount(prediction, minlength=d.CLASSES).tolist(),
                  high_confidence_count=int(high.sum()),
                  high_confidence_prediction_counts=torch.bincount(prediction[high], minlength=d.CLASSES).tolist(),
                  threshold=d.THRESHOLD, threshold_fitted=False)
    if y is None:
        result.update(accuracy=None, NLL=None, ECE=None, high_confidence_error=None,
                      reason='U_s TX labels hidden; no correctness calculation')
        return result
    if y.shape != (len(q),) or y.dtype != torch.long or y.min() < 0 or y.max() >= d.CLASSES:
        raise ValueError('Legal labeled source TX values required')
    correct = prediction == y
    bins = []; ece = 0.
    indices = (confidence * 10).long().clamp(max=9)
    for k in range(10):
        mask = indices == k; count = int(mask.sum())
        accuracy = float(correct[mask].float().mean()) if count else None
        mean = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(q) * abs(accuracy - mean)
        bins.append(dict(bin=k, lower=k / 10, upper=(k + 1) / 10,
                         upper_inclusive=k == 9, count=count,
                         accuracy=accuracy, mean_confidence=mean))
    errors = int((high & ~correct).sum()); selected = int(high.sum())
    result.update(accuracy=float(correct.float().mean()), correct_count=int(correct.sum()),
                  error_count=int((~correct).sum()),
                  NLL=float(-q[torch.arange(len(q), device=q.device), y].clamp_min(1e-12).log().mean()),
                  ECE=ece, calibration_bins=bins,
                  high_confidence_error=dict(count=errors, selected=selected,
                                             error_rate=_safe_rate(errors, selected)))
    return result


def selection_comparison(own, other):
    """Return fixed masks and the untrained average-probability peer target."""
    if other.shape != (2, len(own), d.CLASSES):
        raise ValueError('Exactly two preregistered peer models required')
    _probabilities(own)
    for probabilities in other:
        _probabilities(probabilities)
    peer = other.mean(0)
    own_confidence, own_prediction = own.max(1)
    peer_confidence, peer_prediction = peer.max(1)
    peer_agree = other[0].argmax(1) == other[1].argmax(1)
    own_high = own_confidence >= d.THRESHOLD
    peer_high = peer_confidence >= d.THRESHOLD
    masks = dict(all=torch.ones(len(own), dtype=torch.bool, device=own.device), own_high=own_high,
                 peer_agree=peer_agree, peer_high=peer_high,
                 peer_agree_high=peer_agree & peer_high,
                 own_high_peer_agree_high=own_high & peer_agree & peer_high)
    return peer, masks, own_prediction, peer_prediction


def summarize(own, other, rx, day, role, y=None):
    if role not in d.ROLE_COUNTS or role == 'U_s' and y is not None:
        raise ValueError('U_s TX labels forbidden; source role required')
    if role != 'U_s' and y is None:
        raise ValueError('L_s/V diagnostic labels required')
    if rx.shape != (len(own),) or day.shape != rx.shape:
        raise ValueError('Source RX/day metadata differs')
    peer, masks, own_prediction, peer_prediction = selection_comparison(own, other)
    agreement = own_prediction == peer_prediction

    def group(mask):
        if not mask.any():
            raise ValueError('Empty diagnostic group')
        targets = None if y is None else y[mask]
        result = dict(count=int(mask.sum()), own=calibration(own[mask], targets),
                      fixed_peer=calibration(peer[mask], targets),
                      own_peer_agreement_count=int((agreement & mask).sum()),
                      own_peer_agreement=float(agreement[mask].float().mean()), selection={})
        for name, selected_mask in masks.items():
            selected = mask & selected_mask; count = int(selected.sum())
            row = dict(count=count, coverage=count / int(mask.sum()),
                       own_peer_agreement_count=int((agreement & selected).sum()),
                       own_prediction_counts=torch.bincount(own_prediction[selected], minlength=d.CLASSES).tolist(),
                       peer_prediction_counts=torch.bincount(peer_prediction[selected], minlength=d.CLASSES).tolist())
            if y is not None:
                own_error = own_prediction != y; peer_error = peer_prediction != y
                total_errors = int((own_error & mask).sum())
                covered = int((own_error & selected).sum())
                corrected = int((own_error & ~peer_error & selected).sum())
                harmed = int((~own_error & peer_error & selected).sum())
                row.update(own_error_count=covered, peer_error_count=int((peer_error & selected).sum()),
                           own_error_coverage=_safe_rate(covered, total_errors),
                           peer_corrected_own_errors=corrected, peer_harmed_own_correct=harmed,
                           net_correct_change=corrected - harmed)
            else:
                row.update(own_error_count=None, peer_error_count=None,
                           reason='Hidden U_s TX labels; counts are predictions only')
            result['selection'][name] = row
        return result

    groups = {f'RX{r}_day{dv}': group((rx == r) & (day == dv))
              for r in sorted(rx.unique().tolist()) for dv in sorted(day.unique().tolist())
              if ((rx == r) & (day == dv)).any()}
    labeled_tx = None if y is None else {str(tx): group(y == tx) for tx in sorted(y.unique().tolist())}
    return dict(role=role, summary=group(torch.ones(len(own), dtype=torch.bool)),
                RX_day=groups, labeled_TX=labeled_tx, U_accuracy=None if role == 'U_s' else 'not applicable',
                threshold=d.THRESHOLD, threshold_fitted=False,
                peer_selection='all three seed models fixed before source diagnostics',
                target_access=False, U_truth_read=False, performance_gate=False)


def validate_role(cache, role, expected_ids):
    """Check source role identity using the reused contract, not new data hashes."""
    if role == 'U_s':
        from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled
        validate_unlabeled(cache, expected_ids)
    elif role in ('L_s', 'V'):
        if set(cache) != {'x', 'y', 'rx', 'day', 'ids'}:
            raise ValueError('L_s/V legal raw source schema required')
        if cache['y'].shape != (len(cache['ids']),) or cache['y'].dtype != torch.long:
            raise ValueError('L_s/V labels differ')
        if cache['x'].shape != (len(cache['ids']), 2, 256) or cache['x'].dtype != torch.float32:
            raise ValueError('Source IQ shape/type differs')
        if len(set(cache['ids'])) != len(cache['ids']) or sorted(cache['ids']) != sorted(expected_ids):
            raise ValueError('Source role IDs differ')
        if not torch.isfinite(cache['x']).all():
            raise ValueError('Finite source IQ required')
    else:
        raise ValueError('Forbidden role')
    if set(cache['rx'].tolist()) != set(d.RX) or set(cache['day'].tolist()) != set(d.DAYS):
        raise ValueError('Current legal source RX/day required')
    if role != 'U_s' and set(cache['y'].tolist()) != set(range(d.CLASSES)):
        raise ValueError('Current six labeled source TX values required')
    return len(cache['ids'])

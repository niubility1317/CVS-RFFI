"""Truth-side class-balanced metrics; no fitting, threshold choice, or IO.

Scores are knownness (larger means more acceptable). One source-frozen
threshold is used for operating metrics. Curves summarize fixed scores only;
their points must never become deployment thresholds. Required class rosters
are supplied only by the independent scorer. Missing coverage is None, not 0.
"""
from __future__ import annotations

import numpy as np


def _harmonic(values):
    if any(v is None for v in values):
        return None
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Metric components must be finite proportions or None')
    if any(v == 0 for v in values):
        return 0.
    return float(len(values) / sum(1. / v for v in values))


def open_world_h(old_accuracy, new_accuracy, later_rejection):
    """Project descriptive three-component H, never a substitute for gates."""
    return _harmonic((old_accuracy, new_accuracy, later_rejection))


def _vector(value, name, numeric=False):
    a = np.asarray(value)
    if a.ndim != 1 or (numeric and (a.dtype.kind not in 'fiu' or not np.isfinite(a).all())):
        raise ValueError('One-dimensional aligned finite values required: ' + name)
    if not numeric and a.dtype.kind not in 'iuUS':
        raise ValueError('Explicit integer or string labels required: ' + name)
    return a


def _population(labels, roster):
    roster = tuple(roster)
    if not roster or len(set(roster)) != len(roster) or not set(labels.tolist()).issubset(set(roster)):
        raise ValueError('Nonempty unique required roster must contain every label')
    groups = [(c, labels == c) for c in roster]
    complete = all(np.any(mask) for _, mask in groups)
    weights = np.zeros(len(labels), np.float64)
    if complete:
        for _, mask in groups:
            weights[mask] = 1. / (len(roster) * int(mask.sum()))
    return groups, complete, weights


def _curves(known_scores, unknown_scores, correct, known_weights, unknown_weights):
    scores = np.concatenate((known_scores, unknown_scores)).astype(np.float64)
    wk = np.concatenate((known_weights, np.zeros(len(unknown_scores))))
    wu = np.concatenate((np.zeros(len(known_scores)), unknown_weights))
    wc = np.concatenate((known_weights * correct, np.zeros(len(unknown_scores))))
    order = np.argsort(-scores, kind='stable')
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(order) - 1]
    fpr = np.r_[0., np.cumsum(wu[order])[ends]]
    tpr = np.r_[0., np.cumsum(wk[order])[ends]]
    ccr = np.r_[0., np.cumsum(wc[order])[ends]]
    return float(np.clip(np.trapz(tpr, fpr), 0., 1.)), float(np.clip(np.trapz(ccr, fpr), 0., 1.))


def open_set_metrics(known_y, known_prediction, known_scores, unknown_y, unknown_scores,
                     threshold, known_classes, unknown_classes, baseline_prediction=None):
    """Evaluate one source-fixed operating point and macro-weighted curves.

    known_prediction is the pre-rejection identity choice. It need not be
    correct and may identify a wrong anonymous class. Unknown acceptance is
    always an error regardless of the predicted identity. A value exactly
    equal to threshold is accepted, including all ties.
    """
    ky = _vector(known_y, 'known labels')
    kp = _vector(known_prediction, 'known identity predictions')
    ks = _vector(known_scores, 'known scores', True)
    uy = _vector(unknown_y, 'unknown labels')
    us = _vector(unknown_scores, 'unknown scores', True)
    if (len(ky) != len(kp) or len(ky) != len(ks) or len(uy) != len(us)
            or not np.isscalar(threshold) or not np.isfinite(threshold)):
        raise ValueError('Aligned vectors and one finite frozen threshold required')
    base = None if baseline_prediction is None else _vector(baseline_prediction, 'baseline prediction')
    if base is not None and len(base) != len(ky):
        raise ValueError('Baseline identity predictions must align with known records')
    kg, kc, kw = _population(ky, known_classes)
    ug, uc, uw = _population(uy, unknown_classes)
    ka, ur, correct = ks >= threshold, us < threshold, kp == ky
    mean = lambda values, groups, complete: float(np.mean([values[m].mean() for _, m in groups])) if complete else None
    acceptance = mean(ka, kg, kc)
    accuracy = mean(ka & correct, kg, kc)
    rejection = mean(ur, ug, uc)
    baseline = mean(base == ky, kg, kc) if base is not None else None
    auroc, oscr = _curves(ks, us, correct, kw, uw) if kc and uc else (None, None)
    scalar = lambda v: v.item() if isinstance(v, np.generic) else v
    known = [dict(label=scalar(c), count=int(m.sum()), accepted=int(ka[m].sum()),
                  correct_accepted=int((ka & correct)[m].sum()),
                  baseline_correct=int((base[m] == ky[m]).sum()) if base is not None else None) for c, m in kg]
    unknown = [dict(label=scalar(c), count=int(m.sum()), rejected=int(ur[m].sum())) for c, m in ug]
    return dict(known_count=len(ky), unknown_count=len(uy), known_per_class=known, unknown_per_class=unknown,
        coverage=dict(known_complete=kc, unknown_complete=uc, complete=kc and uc),
        known_macro_acceptance=acceptance, known_macro_frr=None if acceptance is None else 1. - acceptance,
        known_macro_open_accuracy=accuracy, unknown_macro_rejection=rejection,
        unknown_macro_false_acceptance=None if rejection is None else 1. - rejection,
        original_F0_macro_closed_accuracy=baseline,
        original_closed_minus_new_open=None if baseline is None or accuracy is None else baseline - accuracy,
        H_OS=_harmonic((accuracy, rejection)), auroc_known_macro=auroc, oscr_auc_macro=oscr,
        threshold=float(threshold), averaging='equal per required TX, not per sample',
        curves='macro-weighted fixed-score diagnostics; not deployment threshold selection')

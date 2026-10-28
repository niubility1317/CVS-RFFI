"""Global source-known risk thresholds with intact score ties; no unknown input."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .scores import METHODS as STATIC_METHODS


METHODS = tuple(STATIC_METHODS) + ('linear_oe', 'mlp16_oe')
RECEIVERS = (1, 3, 4, 6, 8)
FRR_LIMIT = Fraction(1, 25)
LOSS_LIMIT = Fraction(3, 200)
SCHEMA = 'phase3_source_known_risk_calibration_v1'


def _inputs(ids, y, closed_prediction, knownness, rx_i, method_names):
    ids, y, closed, values, rx = map(np.asarray, (ids, y, closed_prediction, knownness, rx_i))
    n = len(ids) if ids.ndim else 0
    if (not n or ids.shape != (n,) or ids.dtype.kind != 'U' or np.any(np.char.strip(ids) == '')
            or len(set(ids.tolist())) != n):
        raise ValueError('Unique nonempty known-source Unicode IDs are required')
    for name, array in (('y', y), ('closed_prediction', closed), ('rx_i', rx)):
        if array.shape != (n,) or array.dtype.kind not in 'iu':
            raise ValueError('Aligned known-source integer vector required: ' + name)
    if (np.any(y >= 6) or np.any(y < 0) or np.any(closed >= 6) or np.any(closed < 0)
            or set(rx.tolist()) != set(RECEIVERS)):
        raise ValueError('Only six known classes and all five original source RX are allowed')
    if values.shape != (n, len(method_names)) or values.dtype.kind != 'f' or not np.isfinite(values).all():
        raise ValueError('Aligned finite known-source scores must match the declared methods')
    for receiver in RECEIVERS:
        if set(y[rx == receiver].tolist()) != set(range(6)):
            raise ValueError('Every source RX must contain all six known classes')
    return y.astype(np.int64), closed.astype(np.int64), values.astype(np.float64), rx


def _macro(numerator, denominator):
    return sum((Fraction(int(n), int(d)) for n, d in zip(numerator, denominator)), Fraction()) / 6


def _feasible(rejected, rejected_correct, total):
    return all(_macro(r, n) <= FRR_LIMIT and _macro(c, n) <= LOSS_LIMIT
               for r, c, n in zip(rejected, rejected_correct, total))


def _group(rejected, rejected_correct, total, closed_correct):
    frr, loss = _macro(rejected, total), _macro(rejected_correct, total)
    baseline = _macro(closed_correct, total)
    checks = dict(macro_frr_le_04=frr <= FRR_LIMIT, macro_correct_rejection_loss_le_015=loss <= LOSS_LIMIT)
    return dict(known_count=int(total.sum()), class_counts=total.tolist(),
        closed_correct_count=int(closed_correct.sum()), closed_correct_by_class=closed_correct.tolist(),
        rejected_count=int(rejected.sum()), rejected_by_class=rejected.tolist(),
        rejected_correct_count=int(rejected_correct.sum()), rejected_correct_by_class=rejected_correct.tolist(),
        macro_frr=float(frr), macro_correct_rejection_loss=float(loss),
        closed_macro_accuracy=float(baseline), accepted_macro_accuracy=float(baseline - loss),
        checks=checks, constraints_passed=all(checks.values()))


def _one(y, closed, scores, rx):
    # Sorting and block accumulation are O(N log N). Candidate checks inspect
    # only the 6x6 aggregate table, never rescan samples at each threshold.
    order = np.argsort(scores, kind='stable')
    levels, sizes = np.unique(scores[order], return_counts=True)
    block = np.repeat(np.arange(len(levels)), sizes)
    labels, receivers = y[order], np.searchsorted(RECEIVERS, rx[order]) + 1
    correct = closed[order] == labels
    counts = np.zeros((len(levels), 6, 6), dtype=np.int64)
    correct_counts = np.zeros_like(counts)
    np.add.at(counts, (block, 0, labels), 1)
    np.add.at(counts, (block, receivers, labels), 1)
    np.add.at(correct_counts, (block[correct], 0, labels[correct]), 1)
    np.add.at(correct_counts, (block[correct], receivers[correct], labels[correct]), 1)
    prefix = np.concatenate((np.zeros((1, 6, 6), dtype=np.int64), np.cumsum(counts, axis=0)), axis=0)
    correct_prefix = np.concatenate((np.zeros((1, 6, 6), dtype=np.int64), np.cumsum(correct_counts, axis=0)), axis=0)
    total, total_correct = prefix[-1], correct_prefix[-1]
    with np.errstate(over='ignore', invalid='ignore'):
        sentinel = float(np.nextafter(np.float64(levels[-1]), np.inf))
    finite_sentinel = bool(np.isfinite(sentinel))
    # If max is the largest representable float, no finite all-reject candidate
    # exists. It is omitted rather than returning an infinite deployment scalar.
    candidates = np.r_[levels, sentinel] if finite_sentinel else levels
    left, right = 0, len(candidates) - 1
    while left < right:
        middle = (left + right + 1) // 2
        if _feasible(prefix[middle], correct_prefix[middle], total):
            left = middle
        else:
            right = middle - 1
    chosen = left
    overall = _group(prefix[chosen, 0], correct_prefix[chosen, 0], total[0], total_correct[0])
    by_rx = {str(receiver): _group(prefix[chosen, j], correct_prefix[chosen, j], total[j], total_correct[j])
             for j, receiver in enumerate(RECEIVERS, start=1)}
    return dict(threshold=float(candidates[chosen]), candidate_count=len(candidates), score_unique_count=len(levels),
        finite_all_reject_candidate_included=finite_sentinel, selected_candidate_index=int(chosen),
        next_candidate_threshold=float(candidates[chosen + 1]) if chosen + 1 < len(candidates) else None,
        next_candidate_passed=False if chosen + 1 < len(candidates) else None,
        overall=overall, by_receiver=by_rx,
        constraints_passed=overall['constraints_passed'] and all(g['constraints_passed'] for g in by_rx.values()))


def calibrate_known(ids, y, closed_prediction, knownness, rx_i, method_names=None):
    """Choose one global scalar per method using known source V labels only.

    FRR is the mean of rejected/class count. Accuracy loss is the mean of
    rejected originally-correct/class count, not divided by correct count.

    The default remains the original seven methods. An explicit fixed STATIC5
    order supports embedding-increment controls without changing risk budgets.
    """
    if method_names is None:
        names = METHODS
    elif (isinstance(method_names, (list, tuple)) and all(isinstance(n, str) for n in method_names)
          and tuple(method_names) in (METHODS, tuple(STATIC_METHODS))):
        names = tuple(method_names)
    else:
        raise ValueError('Only the fixed original seven or static-five method order is allowed')
    y, closed, values, rx = _inputs(ids, y, closed_prediction, knownness, rx_i, names)
    methods = {name: _one(y, closed, values[:, j], rx) for j, name in enumerate(names)}
    return dict(schema=SCHEMA, method_names=list(names), known_count=len(y), class_count=6,
        receivers=list(RECEIVERS), thresholds=np.asarray([methods[name]['threshold'] for name in names], dtype=np.float64),
        limits=dict(known_macro_frr=.04, known_macro_correct_rejection_loss=.015), methods=methods,
        constraints_passed=all(method['constraints_passed'] for method in methods.values()),
        acceptance_rule='knownness >= one global threshold per method; score ties remain intact',
        risk_scope='empirical risk on the same known-source calibration samples',
        unknown_inputs_used=False, source_known_labels_used=True)

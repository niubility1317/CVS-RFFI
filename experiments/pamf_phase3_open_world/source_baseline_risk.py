"""Maximum global thresholds under empirical original-baseline source risk."""
from __future__ import annotations

from fractions import Fraction
import numpy as np

from .source_risk_calibration import METHODS, RECEIVERS, _inputs, _macro


SCHEMA = 'phase3_source_original_baseline_risk_v1'
FRR = Fraction(1, 20)
TOTAL_LOSS = Fraction(1, 50)
ENCODER_LOSS = Fraction(1, 200)


def _metrics(rejected, correct_rejected, total, new_correct, original_correct):
    frr = _macro(rejected, total)
    original = _macro(original_correct, total)
    closed = _macro(new_correct, total)
    rejection = _macro(correct_rejected, total)
    loss = original - closed + rejection
    return dict(known_count=int(total.sum()), class_counts=total.tolist(),
        rejected_count=int(rejected.sum()), rejected_by_class=rejected.tolist(),
        rejected_correct_count=int(correct_rejected.sum()), rejected_correct_by_class=correct_rejected.tolist(),
        original_correct_by_class=original_correct.tolist(), new_closed_correct_by_class=new_correct.tolist(),
        macro_frr=float(frr), original_closed_macro_accuracy=float(original),
        new_closed_macro_accuracy=float(closed), new_open_macro_accuracy=float(closed - rejection),
        encoder_closed_loss=float(original - closed), correct_rejection_loss=float(rejection),
        original_closed_minus_new_open=float(loss), encoder_protection_passed=original - closed <= ENCODER_LOSS,
        constraints_passed=frr <= FRR and loss <= TOTAL_LOSS)


def _one(y, new, original, scores, rx):
    order = np.argsort(scores, kind='stable')
    levels, sizes = np.unique(scores[order], return_counts=True)
    block = np.repeat(np.arange(len(levels)), sizes)
    labels = y[order]; groups = np.searchsorted(RECEIVERS, rx[order]) + 1
    counts = np.zeros((len(levels), 6, 6), dtype=np.int64)
    correct = np.zeros_like(counts)
    new_ok = new[order] == labels
    np.add.at(counts, (block, 0, labels), 1); np.add.at(counts, (block, groups, labels), 1)
    np.add.at(correct, (block[new_ok], 0, labels[new_ok]), 1)
    np.add.at(correct, (block[new_ok], groups[new_ok], labels[new_ok]), 1)
    prefix = np.concatenate((np.zeros((1, 6, 6), dtype=np.int64), np.cumsum(counts, axis=0)))
    correct_prefix = np.concatenate((np.zeros((1, 6, 6), dtype=np.int64), np.cumsum(correct, axis=0)))
    original_correct = np.zeros((6, 6), dtype=np.int64)
    old_ok = original[order] == labels
    np.add.at(original_correct, (np.zeros(int(old_ok.sum()), dtype=np.int64), labels[old_ok]), 1)
    np.add.at(original_correct, (groups[old_ok], labels[old_ok]), 1)
    total, new_correct = prefix[-1], correct_prefix[-1]
    fixed_loss = [_macro(old, n) - _macro(new_count, n) for old, new_count, n in zip(original_correct, new_correct, total)]
    def feasible(index):
        return all(_macro(rejected, n) <= FRR and base + _macro(correct_rejected, n) <= TOTAL_LOSS
            for rejected, correct_rejected, n, base in zip(prefix[index], correct_prefix[index], total, fixed_loss))
    with np.errstate(over='ignore'):
        sentinel = float(np.nextafter(np.float64(levels[-1]), np.inf))
    candidates = np.r_[levels, sentinel] if np.isfinite(sentinel) else levels
    left, right = 0, len(candidates) - 1
    no_rejection_feasible = feasible(0)
    if no_rejection_feasible:
        while left < right:
            middle = (left + right + 1) // 2
            if feasible(middle): left = middle
            else: right = middle - 1
    # If even no rejection is infeasible, retain every sample and report failure.
    # No threshold can repair an encoder whose closed loss already exceeds .02.
    chosen = left
    rows = [_metrics(prefix[chosen, g], correct_prefix[chosen, g], total[g], new_correct[g], original_correct[g]) for g in range(6)]
    return dict(threshold=float(candidates[chosen]), candidate_count=len(candidates),
        selected_candidate_index=int(chosen), no_rejection_feasible=no_rejection_feasible,
        next_candidate_threshold=float(candidates[chosen + 1]) if chosen + 1 < len(candidates) else None,
        next_candidate_passed=feasible(chosen + 1) if chosen + 1 < len(candidates) else None,
        overall=rows[0], by_receiver={str(rx): rows[j] for j, rx in enumerate(RECEIVERS, 1)},
        constraints_passed=all(row['constraints_passed'] for row in rows),
        encoder_protection_passed=all(row['encoder_protection_passed'] for row in rows))


def calibrate_known(ids, y, closed_new, closed_original, knownness, rx_i):
    """Use known V only; keep ties and all original-RX constraints intact."""
    y, new, values, rx = _inputs(ids, y, closed_new, knownness, rx_i, METHODS)
    original = np.asarray(closed_original)
    if original.shape != y.shape or original.dtype.kind not in 'iu' or np.any(original < 0) or np.any(original >= 6):
        raise ValueError('Aligned original six-class source baseline predictions required')
    methods = {name: _one(y, new, original, values[:, j], rx) for j, name in enumerate(METHODS)}
    return dict(schema=SCHEMA, method_names=list(METHODS), known_count=len(y), class_count=6, receivers=list(RECEIVERS),
        thresholds=np.asarray([methods[name]['threshold'] for name in METHODS], dtype=np.float64), methods=methods,
        limits=dict(known_macro_frr=.05, original_closed_minus_new_open=.02), encoder_closed_loss_limit=.005,
        constraints_passed=all(row['constraints_passed'] for row in methods.values()),
        encoder_protection_passed=all(row['encoder_protection_passed'] for row in methods.values()),
        acceptance_rule='knownness >= one maximum global threshold per method; ties intact',
        source_known_labels_used=True, unknown_inputs_used=False,
        risk_scope='empirical risk on the same known V calibration rows; not an independent guarantee')

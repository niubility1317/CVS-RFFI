"""D92 successor: support-only regularized discriminants and frozen-DG protection.

No optimizer, query fitting, source sample access or class-specific constants.
The old registry prefix is registration metadata, never a query role.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy.special import logsumexp


def _rows(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] not in (256, 288) or not np.isfinite(x).all():
        raise ValueError('Expected finite identity160 + FFT96 features')
    return x[:, :256]


def _features(x, fft_weight):
    x = _rows(x)
    blocks = []
    for part, weight in ((x[:, :160], 1.0), (x[:, 160:256], fft_weight)):
        norm = np.linalg.norm(part, axis=1, keepdims=True)
        blocks.append(weight * part / np.maximum(norm, 1e-12))
    return np.concatenate(blocks, axis=1) / np.sqrt(1 + fft_weight ** 2)


def _logprob(x):
    return x - logsumexp(x, axis=1, keepdims=True)


def protect_old(scores, frozen_logits, old_count, protection):
    """Preserve support-estimated old/new mass; mix only conditional old identity.

Both heads see the same individual input. No prototype is updated or fitted.
With no new classes and protection=1, argmax equals frozen Phase1 exactly.
"""
    scores = np.asarray(scores, dtype=np.float64)
    frozen_logits = np.asarray(frozen_logits, dtype=np.float64)
    if (frozen_logits.shape != (len(scores), old_count)
            or not np.isfinite(frozen_logits).all() or not 0 <= protection <= 1):
        raise ValueError('Frozen logits/protection contract mismatch')
    result = _logprob(scores)
    if protection == 0:
        return result
    mass = logsumexp(result[:, :old_count], axis=1, keepdims=True)
    dg = _logprob(frozen_logits)
    if protection == 1:
        result[:, :old_count] = mass + dg
    else:
        old = result[:, :old_count] - mass
        result[:, :old_count] = mass + np.logaddexp(
            np.log1p(-protection) + old, np.log(protection) + dg)
    return result


def _fit_head(x, y, class_count, old_count, ridge):
    means = np.stack([x[y == c].mean(0) for c in range(class_count)])
    k = len(x) // class_count
    if k == 1:
        # No within-class variance is identifiable from one physical shot.
        # Cosine prototypes avoid treating tiny sample norms as evidence.
        means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
        return means * 10.0, np.zeros(class_count)
    residual = x - means[y]
    # Each task contributes equally; every class within a task uses one formula.
    weights = np.empty(len(x), dtype=np.float64)
    for indices in (np.arange(old_count), np.arange(old_count, class_count)):
        if len(indices):
            mask = np.isin(y, indices)
            weights[mask] = 1.0 / (len(indices) * (k - 1))
    if old_count < class_count:
        weights *= 0.5
    residual *= np.sqrt(weights[:, None])
    cov = residual.T @ residual
    scale = max(float(np.trace(cov)) / x.shape[1], 1e-8)
    # Spherical shrinkage stays positive definite in low-rank small-K regimes.
    cov += ridge * scale * np.eye(x.shape[1])
    coefficient = np.linalg.solve(cov, means.T).T
    intercept = -0.5 * np.sum(means * coefficient, axis=1)
    coefficient -= coefficient.mean(0, keepdims=True)
    intercept -= intercept.mean()
    # A global scale limits LDA overconfidence, estimated from support residuals.
    temperature = max(1.0, float(np.sqrt(np.mean((residual @ coefficient.T) ** 2))))
    return coefficient / temperature, intercept / temperature


@dataclass(frozen=True)
class SupportCVState:
    classes: tuple
    old_count: int
    fft_weight: float
    ridge: float
    protection: float
    coefficient: np.ndarray
    intercept: np.ndarray
    audit: dict

    def score(self, features, frozen_logits):
        x = _features(features, self.fft_weight)
        scores = x @ self.coefficient.T + self.intercept
        return protect_old(scores, frozen_logits, self.old_count, self.protection)


def fit_support_cv(*, support_features, support_labels, support_logits,
                   classes, old_count=6, k1_fft_weight=0.0,
                   k1_protection=1.0, fft_grid=(0.0, 0.5, 1.0, 4.0),
                   ridge_grid=(0.1, 1.0, 10.0), protection_grid=(0.0, 0.5, 1.0)):
    """Select one head using stratified support folds, then refit all support.

K=1 uses fixed source-development defaults, never fake augmented validation.
K>=2 holds out one physical support/class in up to three deterministic folds.
    All intermediate states are discarded. Ranking maximizes held-support H
    with an old-accuracy constraint under the same registered-class competition.
"""
    started = time.perf_counter()
    x = _rows(support_features)
    y = np.asarray(support_labels)
    logits = np.asarray(support_logits, dtype=np.float64)
    classes = tuple(map(str, classes))
    c = len(classes)
    if (c < 2 or len(set(classes)) != c or not 1 <= old_count <= c
            or y.shape != (len(x),) or y.dtype.kind not in 'iu'
            or set(y.tolist()) != set(range(c))
            or logits.shape != (len(x), old_count) or not np.isfinite(logits).all()):
        raise ValueError('Support registry/labels/logits contract mismatch')
    counts = np.bincount(y, minlength=c)
    if counts.min() < 1 or np.any(counts != counts[0]):
        raise ValueError('Equal positive physical K required')
    k = int(counts[0])
    if (not np.isfinite(k1_fft_weight) or k1_fft_weight < 0
            or not 0 <= k1_protection <= 1
            or not fft_grid or not ridge_grid or not protection_grid
            or any(not np.isfinite(v) or v < 0 for v in fft_grid)
            or any(not np.isfinite(v) or v <= 0 for v in ridge_grid)
            or any(not np.isfinite(v) or not 0 <= v <= 1 for v in protection_grid)):
        raise ValueError('Invalid fixed hyperparameter grid')
    # Canonical content order makes fold assignment independent of input order.
    # Identical duplicates remain equivalent. Physical-ID disjointness is the
    # caller's capsule obligation, not inferred from embedding equality.
    order = np.concatenate([np.asarray(sorted(np.flatnonzero(y == cls),
        key=lambda i: x[i].tobytes())) for cls in range(c)])
    x, y, logits = x[order], y[order], logits[order]
    trace = []
    if k == 1:
        choice = (float(k1_fft_weight), 1.0, float(k1_protection))
    else:
        folds = [np.arange(c) * k + i for i in range(min(k, 3))]
        held_y = np.concatenate([y[ix] for ix in folds])
        held_logits = np.concatenate([logits[ix] for ix in folds])
        old = held_y < old_count
        dg_accuracy = float(np.mean(held_logits[old].argmax(1) == held_y[old]))
        reference = []
        identity = _features(x, 0.0)
        for held in folds:
            keep = np.ones(len(x), dtype=bool); keep[held] = False
            means = np.stack([identity[keep & (y == cls)].mean(0) for cls in range(c)])
            means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
            reference.append(identity[held] @ means.T)
        reference_old = (float(np.mean(np.concatenate(reference).argmax(1)[old] == held_y[old]))
                         if c > old_count else dg_accuracy)
        for fw in fft_grid:
            transformed = _features(x, fw)
            for ridge in ridge_grid:
                predictions = []
                for held in folds:
                    keep = np.ones(len(x), dtype=bool); keep[held] = False
                    coef, intercept = _fit_head(transformed[keep], y[keep], c, old_count, ridge)
                    predictions.append(transformed[held] @ coef.T + intercept)
                raw = np.concatenate(predictions)
                for protection in protection_grid:
                    scores = protect_old(raw, held_logits, old_count, protection)
                    correct = scores.argmax(1) == held_y
                    old_acc = float(np.mean(correct[old]))
                    new_acc = float(np.mean(correct[~old])) if np.any(~old) else None
                    loss = -scores[np.arange(len(held_y)), held_y]
                    balanced = float(0.5 * (loss[old].mean() + loss[~old].mean())
                                     if np.any(~old) else loss.mean())
                    trace.append(dict(fft_weight=float(fw), ridge=float(ridge),
                        protection=float(protection), balanced_nll=balanced,
                        old_accuracy=old_acc, new_accuracy=new_acc,
                        harmonic=(2 * old_acc * new_acc / max(old_acc + new_acc, 1e-12)
                                  if new_acc is not None else old_acc),
                        frozen_dg_old_accuracy=dg_accuracy,
                        reference_old_accuracy=reference_old,
                        old_deficit=max(0.0, reference_old - 0.01 - old_acc)))
        # Joint competition may make the DG constraint unattainable. In that
        # case minimize measured deficit first; report it without claiming safety.
        chosen = min(trace, key=lambda t: (t['old_deficit'], -t['harmonic'], t['balanced_nll'],
                     t['fft_weight'], -t['ridge'], -t['protection']))
        choice = tuple(chosen[key] for key in ('fft_weight', 'ridge', 'protection'))
    fw, ridge, protection = choice
    coefficient, intercept = _fit_head(_features(x, fw), y, c, old_count, ridge)
    coefficient.setflags(write=False); intercept.setflags(write=False)
    audit = dict(method='D92-SCV-v1', k=k, feature_dim=256, support_rows=len(x),
        query_rows_used_for_fit=0, encoder_updated=False, source_runtime_access=False,
        selection='fixed_source_defaults' if k == 1 else 'support_physical_holdout',
        selected=dict(fft_weight=fw, ridge=ridge, protection=protection),
        trace=trace, fit_seconds=time.perf_counter() - started,
        persistent_state_bytes=coefficient.nbytes + intercept.nbytes,
        optimizer_steps=0, loss=None, loss_reason='closed_form; validation NLL in trace')
    return SupportCVState(classes, old_count, fw, ridge, protection,
                          coefficient, intercept, audit)

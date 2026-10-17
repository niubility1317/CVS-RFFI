"""Source scalar transport, not an identity or receiver classifier."""
from itertools import permutations
import numpy as np
from .design import RX


def cube(values, tx, rx, day, per_cell):
    values = np.asarray(values, dtype=np.float64)
    tx, rx, day = (np.asarray(v) for v in (tx, rx, day))
    n = 90 * per_cell
    if values.shape != (n,) or any(a.shape != (n,) for a in (tx, rx, day)):
        raise ValueError('Complete role scalar vectors required')
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1 + 1e-12).any():
        raise ValueError('Finite bounded circularity required')
    if set(tx) != set(range(6)) or set(rx) != set(RX) or set(day) != {1, 2, 3}:
        raise ValueError('All original source TX/RX/day required')
    means = np.empty((5, 3, 6))
    records = []
    for i, receiver in enumerate(RX):
        for j, date in enumerate((1, 2, 3)):
            for t in range(6):
                a = values[(rx == receiver) & (day == date) & (tx == t)]
                if len(a) != per_cell:
                    raise ValueError('Exact original70L or300V per source cell required')
                means[i, j, t] = a.mean()
                records.append(dict(RX=receiver, day=date, TX=t, count=per_cell,
                                    mean=float(a.mean()), std=float(a.std(ddof=1))))
    return means, records


def evaluate(L, V):
    L, V = (np.asarray(a, dtype=np.float64) for a in (L, V))
    if L.shape != (5, 3, 6) or V.shape != L.shape or not np.isfinite(L).all() or not np.isfinite(V).all():
        raise ValueError('All90 L and V cell means required')
    train_contrast = L - L.mean(axis=0, keepdims=True)
    validation_contrast = V - V.mean(axis=0, keepdims=True)
    wrong = [p for p in permutations(range(5)) if all(i != j for i, j in enumerate(p))]
    squared_baseline, squared_error, wrong_sse, folds = [], [], np.zeros(44), []
    for date in range(3):
        for tx in range(6):
            # No V statistic enters this fit. Every source RX is always retained.
            fit = train_contrast[:, [d for d in range(3) if d != date]][:, :, [t for t in range(6) if t != tx]]
            prediction = fit.mean(axis=(1, 2))
            actual = validation_contrast[:, date, tx]
            b, e = actual ** 2, (actual - prediction) ** 2
            squared_baseline.append(b)
            squared_error.append(e)
            wrong_sse += np.array([np.sum((actual - prediction[list(p)]) ** 2) for p in wrong])
            gain = 1 - float(e.sum() / b.sum()) if b.sum() > 1e-16 else None
            folds.append(dict(TX=tx, day=date + 1, fit_L_cells=50, evaluation_V_cells=5,
                              baseline_SSE=float(b.sum()), RX_SSE=float(e.sum()), risk_gain=gain,
                              prediction=prediction.tolist(), V_RX_contrast=actual.tolist()))
    b, e = np.array(squared_baseline), np.array(squared_error)
    base, error = float(b.sum()), float(e.sum())
    gain = 1 - error / base if base > 1e-16 else None
    per_RX = {str(r): dict(baseline_SSE=float(b[:, i].sum()), RX_SSE=float(e[:, i].sum()),
                          risk_gain=1 - float(e[:, i].sum() / b[:, i].sum()) if b[:, i].sum() > 1e-16 else None)
              for i, r in enumerate(RX)}
    positive = sum(f['risk_gain'] is not None and f['risk_gain'] > 0 for f in folds)
    conditions = dict(nonzero_baseline=base > 1e-16,
                      relative_risk_gain=gain is not None and gain >= .20,
                      positive_folds=positive >= 15,
                      each_RX_nonnegative=all(v['RX_SSE'] <= v['baseline_SSE'] + 1e-20 for v in per_RX.values()))
    return dict(decision='SOURCE_ASSOCIATION_CONFIRMED_ONLY' if all(conditions.values()) else 'CLOSED_FIXED_L_TO_V_SOURCE_BASIS',
                conditions=conditions, baseline_SSE=base, RX_SSE=error, risk_gain=gain,
                positive_folds=positive, per_RX=per_RX, folds=folds,
                wrong_RX=[dict(permutation=list(p), SSE=float(s), risk_gain=1-float(s)/base if base>1e-16 else None)
                          for p, s in zip(wrong, wrong_sse)],
                wrong_RX_is_promotion_condition=False, V_used_for_fit=False,
                L_RX_means=L.mean((1, 2)).tolist(), V_RX_means=V.mean((1, 2)).tolist(),
                source_RX_holdout=None, model_constructed=False, IQ_operator_implemented=False,
                Ag_proved=False, joint_goal_complete=False, target_access=False)

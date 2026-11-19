"""Choose a REPL dose using only paired held-out source predictions."""
from fractions import Fraction


NATIVE_WEIGHT = 6.0
ALTERNATIVES = (4.5, 3.0)


def choose_weight(folds, results):
    """Return a dose and its exact held-out evidence, with strong native fallback.

    Each fold names its held-out groups. A result for each group is
    (same-sample count, wrong-to-right count, right-to-wrong count) versus native.
    """
    if len(folds) < 2:
        return dict(weight=NATIVE_WEIGHT, decision='insufficient_source_folds', candidates={})
    expected = {(fold, group) for fold, groups in folds.items() for group in groups}
    if not expected or any(not groups for groups in folds.values()):
        raise ValueError('Every source fold must specify its held-out groups')
    if set(results) != set(ALTERNATIVES) or any(set(rows) != expected for rows in results.values()):
        raise ValueError('Both alternatives require the complete preregistered source matrix')

    candidates = {}
    top = Fraction(0), NATIVE_WEIGHT
    for weight in ALTERNATIVES:
        scores = results[weight]
        fold_gain = {fold: 0 for fold in folds}
        fold_n = {fold: 0 for fold in folds}
        for key in sorted(expected):
            n, gain, damage = scores[key]
            if (any(type(value) is not int for value in (n, gain, damage)) or n <= 0
                    or gain < 0 or damage < 0 or gain + damage > n
                    or any(other[key][0] != n for other in results.values())):
                raise ValueError('Paired source populations and nonnegative correction counts required')
            fold = key[0]
            fold_gain[fold] += gain - damage
            fold_n[fold] += n
        deltas = {fold: Fraction(fold_gain[fold], n) for fold, n in fold_n.items()}
        macro = sum(deltas.values()) / len(deltas)
        eligible = all(delta > 0 for delta in deltas.values()) and all(gain >= damage for n, gain, damage in scores.values())
        candidates[weight] = dict(eligible=eligible, fold_net=fold_gain, fold_n=fold_n,
                                  macro_delta=float(macro), worst_group_net=min(g-d for _, g, d in scores.values()))
        if eligible:
            top = max(top, (macro, weight))
    return dict(weight=top[1], decision='source_improvement' if top[1] != NATIVE_WEIGHT else 'retain_native',
                candidates=candidates)

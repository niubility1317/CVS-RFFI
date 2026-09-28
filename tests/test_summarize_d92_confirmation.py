"""Synthetic complete-matrix checks; no target results are loaded."""
from itertools import product
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from summarize_d92_confirmation import summarize


SEEDS = range(2026092701, 2026092705)
RECEIVERS = ['19-1', '8-14', '8-7']
SCENES = ['practical_high', 'practical_mid', 'practical_low_urban']
SUPPORT_SEEDS = range(2026092711, 2026092716)


def complete_scores(values):
    """Populate all 3,600 cells per method and the 36 frozen-DG records."""
    rows = []
    for method, seed, rx, scene, k, new_count, support_seed in product(
            ['D92', 'D92-SCV-v1'], SEEDS, RECEIVERS, SCENES,
            [1, 5, 10, 20], [0, 2, 5, 10, 20], SUPPORT_SEEDS):
        old, new = values(method, seed)
        # Old-only cells intentionally differ so their accidental inclusion is visible.
        old = old if new_count else 0.99
        new = new if new_count else None
        harmonic = 2 * old * new / (old + new) if new is not None else None
        rows.append(dict(method=method, model_seed=seed, receiver=rx, scenario=scene,
            k=k, new_count=new_count, support_seed=support_seed,
            accuracy=(6 * old + new_count * new) / (6 + new_count) if new_count else old,
            old_accuracy=old, new_accuracy=new, harmonic_mean=harmonic,
            macro_f1=(old + new) / 2 if new_count else old,
            old_macro_f1=old, new_macro_f1=new, old_floor=old, new_floor=new,
            forgetting=0.99 - old if new_count else None))
    for seed, rx, scene in product(SEEDS, RECEIVERS, SCENES):
        rows.append(dict(method='frozen_dg', model_seed=seed, receiver=rx, scenario=scene,
            k=1, new_count=0, support_seed=SUPPORT_SEEDS.start,
            accuracy=0.8, old_accuracy=0.8, new_accuracy=None, harmonic_mean=None,
            macro_f1=0.8, old_macro_f1=0.8, new_macro_f1=None,
            old_floor=0.8, new_floor=None, forgetting=None))
    return dict(status='SCORED', results=rows, selection_feedback_forbidden=True,
                claim_scope='Synthetic regression data; no real target observations')


@pytest.mark.parametrize('missing_method', ['D92', 'D92-SCV-v1', 'frozen_dg'])
def test_complete_coverage_rejects_missing_rows(missing_method):
    data = complete_scores(lambda method, seed: (0.5, 0.5))
    assert len(data['results']) == 7236
    index = next(i for i, row in enumerate(data['results']) if row['method'] == missing_method)
    data['results'].pop(index)
    with pytest.raises(ValueError, match='Incomplete paired matrix|Wrong final coverage'):
        summarize(data)


def test_h_is_mean_of_cell_h_and_excludes_old_only_rows():
    def values(method, seed):
        if method == 'D92-SCV-v1':
            return 0.6, 0.6
        return (0.9, 0.1) if seed % 2 else (0.1, 0.9)
    result = summarize(complete_scores(values))
    assert result['rows'] == 7236
    for k in [1, 5, 10, 20]:
        pair = {r['method']: r for r in result['tables']['per_k'] if r['k'] == k}
        baseline, candidate = pair['D92'], pair['D92-SCV-v1']
        assert baseline['cells'] == candidate['cells'] == 720
        assert baseline['old_accuracy'] == pytest.approx(0.5)
        assert baseline['new_accuracy'] == pytest.approx(0.5)
        assert baseline['harmonic_mean'] == pytest.approx(0.18)
        assert baseline['harmonic_mean'] != pytest.approx(
            2 * baseline['old_accuracy'] * baseline['new_accuracy'] /
            (baseline['old_accuracy'] + baseline['new_accuracy']))
        assert candidate['harmonic_mean'] == pytest.approx(0.6)
        comparison = next(r for r in result['comparisons'] if r['k'] == k)
        assert comparison['delta']['harmonic_mean'] == pytest.approx(0.42)
    assert result['preregistered_guard_pass'] is True


@pytest.mark.parametrize('baseline,candidate,failing_guard', [
    ((0.2, 0.8), (0.5, 0.75), 'new_guard'),
    ((0.8, 0.2), (0.75, 0.5), 'old_guard'),
])
def test_h_gain_cannot_hide_either_side_regression(baseline, candidate, failing_guard):
    result = summarize(complete_scores(lambda method, seed: baseline if method == 'D92' else candidate))
    for comparison in result['comparisons']:
        assert comparison['h_improved'] is True
        assert comparison[failing_guard] is False
        other = 'new_guard' if failing_guard == 'old_guard' else 'old_guard'
        assert comparison[other] is True
        assert comparison['delta']['harmonic_mean'] == pytest.approx(0.28)
    assert result['preregistered_guard_pass'] is False
    assert result['new_improved_every_k'] is (failing_guard == 'old_guard')

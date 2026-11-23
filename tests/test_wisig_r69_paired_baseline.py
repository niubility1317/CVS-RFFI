from unittest.mock import patch

import numpy as np
import pytest

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_repl_view_shuffle import dura_baseline as baseline


def test_full_view_wrappers_pair_by_id_and_require_the_same_clean(tmp_path):
    parent = tmp_path / 'reference'
    clean = np.arange(6 * 2 * 256, dtype=np.float32).reshape(6, 2, 256)
    ids = np.asarray([f'p{i}' for i in range(len(clean))])
    parent.mkdir()
    np.save(parent / 'clean.npy', clean)
    np.savez(parent / 'index.npz', ids=ids)
    manifest = dict(status='VALIDATED_ONCE', channel='residual/post_sync/noeq',
                    count=6, scenes=list(baseline.VIEWS[1:]), truth_read=False,
                    namespace='seed-domain', session='fixed-session',
                    clean_ref=str(parent / 'clean.npy'), evidence=[])
    write(parent / 'manifest.json', manifest)

    # Source view arrays use their own opaque-ID ordering and positions.
    ordering = np.asarray([4, 5, 1, 2, 0, 3])
    union = tmp_path / 'union.npy'
    np.save(union, clean[ordering])
    tests = {}
    for task, positions in (('cross_rx', [1, 3, 5, 2]), ('cross_both', [0, 4])):
        wrapper = tmp_path / task
        wrapper.mkdir()
        np.savez(wrapper / 'index.npz', ids=ids[ordering[positions]], positions=positions)
        write(wrapper / 'manifest.json', dict(manifest, classes=['a'], count=len(positions),
              source_target_disjoint=True, clean_ref=str(union)))
        tests[task] = {'views_root': str(wrapper)}
    run = tmp_path / 'run'
    run.mkdir()
    rt = dict(residual_reference=str(parent), run_root=str(run))
    # Change one existing IQ row while preserving the ID order.
    mismatch = clean[ordering]
    mismatch[1, 0, 0] += 1
    np.save(union, mismatch)
    with patch.object(baseline, 'COUNTS', dict(cross_rx=4, cross_both=2)), \
         patch.object(baseline, 'historical', return_value=dict(tests=tests, classes=['a'])):
        with pytest.raises(ValueError, match='clean IQ differ'):
            baseline.build_full(rt)
    assert not (run / 'full_views' / 'cross_rx' / 'manifest.json').exists()
    assert read(parent / 'manifest.json') == manifest


def test_truth_scoring_waits_for_each_route_and_seed():
    from experiments.cric_dura_comparison import evaluation

    configs = []
    def fail_last(c):
        configs.append(c)
        if c == ('full', baseline.SEEDS[-1]):
            raise ValueError('last row incomplete')

    with patch.object(baseline, 'score_config', side_effect=lambda _, route, seed: (route, seed)), \
         patch.object(evaluation, 'preflight', side_effect=fail_last), \
         patch.object(evaluation, 'score') as scorer, \
         patch.object(evaluation, 'tasks', evaluation.tasks):
        with pytest.raises(ValueError, match='last row incomplete'):
            baseline.score_all({})
    assert configs == [(route, seed) for route in baseline.ROUTES for seed in baseline.SEEDS]
    scorer.assert_not_called()

import numpy as np
import pytest

from experiments.cric_dura_comparison.io import write
from experiments.pamf_unrolled_restore import wisig_preamble_dura_target as target
from experiments.pamf_unrolled_restore.wisig_preamble_dura_score import acceptance, read_decisions, score, score_partition


def test_candidate_fit_gate_does_not_access_a_checkpoint_early(tmp_path, monkeypatch):
    fitted = tmp_path/'runs'/target.ORIGIN
    fitted.mkdir(parents=True)
    path = tmp_path/'releases'/f'{target.ORIGIN}-release'/'runtime.json'
    write(path, dict(run_id=target.ORIGIN, code_commit=target.ORIGIN_CODE, run_root=str(fitted)))
    args = {'project': str(tmp_path)}

    def premature_load(*_args, **_kwargs):
        pytest.fail('Checkpoint opened before the source fit was frozen')

    monkeypatch.setattr(target.frozen, 'load_models', premature_load)
    with pytest.raises(ValueError, match='must be frozen'):
        target.frozen_candidate(args)


def test_target_eligibility_uses_frozen_fit_without_source_accuracy(tmp_path, monkeypatch):
    fitted = tmp_path/'runs'/target.ORIGIN
    write(fitted/'preamble-complete.json', {'status': 'SOURCE_FROZEN'})
    path = tmp_path/'releases'/f'{target.ORIGIN}-release'/'runtime.json'
    write(path, dict(run_id=target.ORIGIN, code_commit=target.ORIGIN_CODE, run_root=str(fitted)))
    model, gate = object(), (0.2, 0.4)
    monkeypatch.setattr(target.frozen, 'load_models', lambda *_args, **_kwargs: iter(((model, gate),)))
    assert target.frozen_candidate({'project': str(tmp_path)}) == (model, gate)


def test_truth_stays_closed_before_predictions_freeze(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    write(tmp_path/'state.json', {'status': 'PREDICTING'})
    with pytest.raises(ValueError, match='Truth remains closed'):
        score({'run_root': str(tmp_path)})


def make_predictions(tmp_path, ids, *, mismatch=False):
    origin = tmp_path/'candidate'
    baseline = tmp_path/'baseline'
    route, split = 'full', 'cross_rx'
    candidate = origin/route/split
    write(candidate/'complete.json', dict(status='PREDICTIONS_FROZEN', route=route, split=split,
          source_run=target.ORIGIN, count=len(ids), seeds=list(target.SEEDS), modes=list(target.MODES),
          views=list(target.VIEWS), truth_read=False, active={view: 0 for view in target.VIEWS}))
    zero = np.zeros((len(target.SEEDS), len(ids)), np.uint8)
    values = {f'{mode}__{view}': zero for view in target.VIEWS for mode in target.MODES}
    np.savez(candidate/'decisions.npz', ids=ids, **values)
    for seed in target.SEEDS:
        prior = baseline/route/str(seed)/'predictions'
        write(prior/'complete.json', dict(run_id=target.BASELINE, status='ALL_PREDICTIONS_COMPLETE',
              row_id=f'{route}-s{seed}', truth_read=False, query_fit=False, readouts=list(target.MODES[:2])))
        dest = prior/split
        dest.mkdir()
        methods = {f'{mode}__{view}': np.zeros(len(ids), dtype=np.int64)
                   for view in target.VIEWS for mode in target.MODES[:2]}
        if mismatch and seed == target.SEEDS[-1]:
            methods['R69__clean'][0] = 1
        np.savez(dest/'predictions.npz', ids=ids, **methods)
    return origin, baseline, route, split


@pytest.mark.parametrize('changed', [False, True])
def test_before_truth_all_three_replayed_baselines_match(tmp_path, changed):
    ids = np.array(['rx-day-one', 'rx-day-two'])
    inputs = make_predictions(tmp_path, ids, mismatch=changed)
    if changed:
        with pytest.raises(ValueError, match='Archived and replayed'):
            read_decisions(*inputs, ids)
    else:
        aligned, counts = read_decisions(*inputs, ids)
        assert aligned[target.MODES[-1]].shape == (3, 7, 2) and all(count == 0 for count in counts.values())


def test_main_score_keeps_clean_threshold_and_both_routes():
    shape = (len(target.SEEDS), len(target.VIEWS), 4)
    good = np.zeros(shape, dtype=np.uint8)
    baseline = good.copy()
    baseline[:, 1:, 0] = 1
    correct = np.zeros(4, dtype=np.uint8)
    groups = np.array(['rx1', 'rx1', 'rx2', 'rx2'])
    prod = score_partition(dict(raw=baseline, R69=baseline, preamble_gated=good), correct, groups,
                           np.random.default_rng(17))
    matrix = {route: {'combined': prod} for route in target.ROUTES}
    results = acceptance(matrix)
    assert results['passed'] and results['by_route']['full'] == dict(six_scene_net=18, clean_net=0, passed=True)
    assert prod['by_seed'][str(target.SEEDS[0])]['R69']['six_practical']['by_receiver']['rx1']['net'] == 6
    prod['combined_seeds']['R69']['clean']['overall']['net'] = -1
    assert not acceptance(matrix)['passed']

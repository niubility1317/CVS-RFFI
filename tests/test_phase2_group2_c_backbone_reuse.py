import json
from pathlib import Path

import numpy as np
import pytest

from experiments.pamf_phase2_practical.group2_c_backbone_reuse import hu_aux_folder, reuse_pilot_prediction, score_with_pilot_reuse


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(tmp_path):
    old = tmp_path/'pilot'; old.mkdir()
    checkpoint = tmp_path/'sources'/'source'/'orthogonal-s2026092701'/'source.pt'
    expected = {'split_id': 'a'*24, 'registered_classes': list(range(26)), 'query_count': 2}
    row = dict(expected, k=5, new_count=20, stage='C', arm='poster_ft__orthogonal__C_SOURCE',
        input_path='P0', prediction='p.npz', resources={'B_fit_shared_across_N': True,
            'B_and_C': {'source_checkpoint': str(checkpoint)}})
    np.savez(old/'p.npz', query_indices=np.array([2, 3]), predicted_indices=np.array([0, 25]),
             old_restricted_indices=np.array([0, 5]))
    put(old/'complete.json', dict(status='PREDICTIONS_COMPLETE', truth_read=False, query_fit=False,
                                   rows=1, capsule_id='cap'))
    put(old/'predictions.json', dict(capsule='capsules/P01/MIX01/1-1', capsule_id='cap',
        profile='poster_ft', c_method='orthogonal', model_seed=2026092701, rows=[row]))
    rt = {'c_source_roots': {'orthogonal': str(tmp_path/'sources')}}
    job = dict(reuse_prediction=str(old), reuse_score=str(tmp_path/'pilot-score'),
        capsule='capsules/P01/MIX01/1-1', b_method='poster_ft', c_method='orthogonal', seed=2026092701)
    return rt, job, expected, old


def test_auxiliary_roots_are_selected_by_exact_model_seed():
    rt = {'hu_aux_roots_by_seed': {'2026092701': '/aux/one', '2026092702': '/aux/two'}}
    assert hu_aux_folder(rt, 'mopc_hr', 2026092702) == Path('/aux/two/source/mopc_hr-s2026092702')
    with pytest.raises(ValueError, match='Hu source'):
        hu_aux_folder(rt, 'mopc_hr', 2026092703)


def test_pilot_prediction_is_reused_and_its_b_state_uncertainty_is_explicit(tmp_path):
    rt, job, spec, old = fixture(tmp_path)
    out = tmp_path/'out'; out.mkdir()
    row = reuse_pilot_prediction(rt, job, {'capsule_id': 'cap'}, spec, np.array([2, 3]), out)
    assert (out/'p.npz').read_bytes() == (old/'p.npz').read_bytes()
    assert row['reuse']['B_state_across_N'] == 'not_verified_against_new_B'
    assert row['resources']['B_fit_shared_across_N'] is None
    assert row['reuse']['prediction_source'] == str(old)
    assert row['resources']['B_and_C']['source_checkpoint'].endswith('source.pt')


def test_pilot_reuse_rejects_mismatched_source_or_query(tmp_path):
    rt, job, spec, old = fixture(tmp_path)
    out = tmp_path/'out'; out.mkdir()
    with pytest.raises(ValueError, match='query'):
        reuse_pilot_prediction(rt, job, {'capsule_id': 'cap'}, spec, np.array([1, 3]), out)
    rt['c_source_roots']['orthogonal'] = '/wrong/source'
    with pytest.raises(ValueError, match='checkpoint'):
        reuse_pilot_prediction(rt, job, {'capsule_id': 'cap'}, spec, np.array([2, 3]), out)


def test_truth_last_score_reuses_old_row_without_rescoring_it(tmp_path, monkeypatch):
    rt, job, spec, old = fixture(tmp_path)
    root = tmp_path/'run'; rt['run_root'] = str(root)
    put(root/'predictions_complete.json', {'status': 'ALL_PREDICTIONS_FIXED'})
    out = tmp_path/'prediction'; out.mkdir()
    row = reuse_pilot_prediction(rt, job, {'capsule_id': 'cap'}, spec, np.array([2, 3]), out)
    new = dict(row, split_id='b'*24, k=1, prediction='new.npz'); new.pop('reuse')
    np.savez(out/'new.npz', predicted_indices=np.array([0]))
    put(out/'complete.json', dict(status='PREDICTIONS_COMPLETE', truth_read=False, rows=2, capsule_id='cap'))
    put(out/'predictions.json', dict(capsule=job['capsule'], capsule_id='cap', model_seed=job['seed'], rows=[new, row]))
    oldscore = Path(job['reuse_score'])
    oldrow = dict(row, metrics={'H_macro': .4}); oldrow.pop('reuse')
    put(oldscore/'complete.json', {'status': 'SCORED_COMPLETE', 'rows': 1, 'truth_last': True})
    put(oldscore/'metrics.json', {'capsule_id': 'cap', 'model_seed': job['seed'],
        'rows': [oldrow], 'truth_last': True, 'target_feedback': False})
    put(oldscore/'common_query_metrics.json', {'rows': [oldrow], 'truth_last': True,
        'target_feedback': False, 'scope': 'intersection', 'common_query_counts': {'20': 1}})
    calls = []
    def score(rt_arg, task):
        inp = Path(task['prediction']); rows = json.loads((inp/'predictions.json').read_text())['rows']
        calls.extend(r['prediction'] for r in rows)
        put(Path(task['output'])/'metrics.json', {'capsule_id': 'cap', 'model_seed': job['seed'],
            'rows': [dict(r, metrics={'H_macro': .7}) for r in rows], 'truth_last': True, 'target_feedback': False})
        put(Path(task['output'])/'complete.json', {'status': 'SCORED_COMPLETE', 'rows': 1, 'truth_last': True})
    def common(rt_arg, task, prediction, index, output):
        put(output/'common_query_metrics.json', {'rows': [dict(r, metrics={'H_macro': .7}) for r in index['rows']],
            'truth_last': True, 'target_feedback': False, 'scope': 'intersection', 'common_query_counts': {'20': 1}})
    monkeypatch.setattr('experiments.pamf_phase2_practical.episodic_score.run', score)
    monkeypatch.setattr('experiments.pamf_phase2_practical.milkr_r42_suite.common_query_score', common)
    job.update(output=str(out), key='job')
    score_with_pilot_reuse(rt, job)
    assert calls == ['new.npz']
    metrics = json.loads((root/'scores/job/metrics.json').read_text())
    assert [r['metrics']['H_macro'] for r in metrics['rows']] == [.7, .4]
    assert metrics['rows'][1]['reuse']['B_state_across_N'] == 'not_verified_against_new_B'
    assert metrics['rows'][1]['resources']['B_fit_shared_across_N'] is None
    assert json.loads((root/'scores/job/complete.json').read_text())['rows'] == 2


def test_reused_score_cannot_be_read_before_global_prediction_barrier(tmp_path):
    root = tmp_path/'run'
    put(root/'predictions_complete.json', {'status': 'NOT_COMPLETE'})
    with pytest.raises(ValueError, match='Truth barrier'):
        score_with_pilot_reuse({'run_root': str(root)}, {'output': '/missing'})

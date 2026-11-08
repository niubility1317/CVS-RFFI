import json

import pytest

from experiments.pamf_phase2_practical.group2_schedule import closed_prediction, _score
from experiments.pamf_phase2_practical.group2_resume import method_subset_jobs
from experiments.pamf_phase2_practical import group2_resume


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding='utf-8')


def test_subset_marker_cannot_pass_as_full_matrix_and_requires_saved_b(tmp_path):
    job = dict(key='p-s1-c0', profile='protonet_cda', seed=1, capsule='capsules/P/M/R',
               methods=['orthogonal', 'lora'], points=[[5, 20]], output=str(tmp_path), save_B_for_mopc=True)
    rows = [dict(k=5, new_count=20, arm='protonet_cda__'+m, prediction=m+'.npz') for m in job['methods']]
    _write(tmp_path/'predictions.json', dict(capsule=job['capsule'], capsule_id='cap', model_seed=1,
                                             profile=job['profile'], rows=rows))
    _write(tmp_path/'complete.json', dict(status='SUBSET_PREDICTIONS_FIXED', truth_read=False,
                                          query_fit=False, capsule_id='cap', rows=2))
    for method in job['methods']:(tmp_path/(method+'.npz')).touch()
    with pytest.raises(ValueError, match='Missing saved B state'):
        closed_prediction(job, 2)
    (tmp_path/'B-K5.pt').touch()
    closed_prediction(job, 2)
    with pytest.raises(ValueError, match='Prediction not frozen'):
        closed_prediction(dict(job, save_B_for_mopc=False), 2)
    with pytest.raises(ValueError, match='subset cannot be scored'):
        _score({}, job)


def test_only_unstarted_nonradio_jobs_are_transformed_without_mutation():
    jobs = [dict(key='p', profile='protonet_cda', methods=['mopc_hr', 'orthogonal', 'lora']),
            dict(key='d', profile='dadda_sda', methods=['mopc_hr', 'orthogonal', 'lora'])]
    result = method_subset_jobs(jobs)
    assert [x['methods'] for x in result] == [['orthogonal', 'lora'], ['orthogonal', 'lora']]
    assert all(x['save_B_for_mopc'] for x in result)
    assert jobs[0]['methods'] == ['mopc_hr', 'orthogonal', 'lora']
    with pytest.raises(ValueError, match='Invalid method-subset job'):
        method_subset_jobs([dict(jobs[0], profile='radionet_ada_knn')])


def test_subset_dispatch_never_writes_full_prediction_or_score_barrier(tmp_path, monkeypatch):
    parent = tmp_path/'parent'; (parent/'launches').mkdir(parents=True)
    parent_runtime = str(tmp_path/'parent.json')
    base = dict(data_root='data', profiles=['protonet_cda', 'dadda_sda', 'radionet_ada_knn'])
    previous = dict(base, run_root=str(parent), runtime_file=parent_runtime)
    current = dict(base, run_root=str(tmp_path/'new'), previous_run_root=str(parent),
                   previous_runtime_file=parent_runtime, runtime_file='new.json',
                   deferred_failure_keys=['failed'], cancelled_profiles=['radionet_ada_knn'],
                   gpu_limits={'0': 2}, last_profiles=[])
    records = {parent_runtime: previous, str(parent/'worker_state.json'):
               dict(status='FAILED', active=[], failed=[dict(key='failed')]), str(parent/'owner.json'): {}}
    monkeypatch.setattr(group2_resume, 'validate_runtime', lambda rt: None)
    monkeypatch.setattr(group2_resume, 'read', lambda path: records[str(path)])
    jobs = [dict(key='p', profile='protonet_cda', methods=['mopc_hr','orthogonal','lora'], points=[[5, 20]]),
            dict(key='r', profile='radionet_ada_knn', methods=['mopc_hr','orthogonal','lora'], points=[[5, 20]])]
    monkeypatch.setattr(group2_resume, 'classify_chain', lambda *a: ([], [dict(key='failed')], jobs))
    monkeypatch.setattr(group2_resume, '_smoke_gpu', lambda rt: '0')
    monkeypatch.setattr(group2_resume.subprocess, 'Popen', lambda *a, **k: type('P', (), {'pid': 123, 'wait': lambda s: 0})())
    seen = []
    monkeypatch.setattr(group2_resume, '_queue', lambda rt, j, action: seen.append((action, j)))
    monkeypatch.setattr(group2_resume, 'closed_prediction', lambda *a: None)
    writes = {}
    monkeypatch.setattr(group2_resume, 'write', lambda path, value, **kw: writes.update({path.name: value}))
    group2_resume.dispatch(current, method_subset=True)
    assert len(seen) == 1 and seen[0][0] == 'worker'
    assert seen[0][1][0]['methods'] == ['orthogonal', 'lora'] and seen[0][1][0]['save_B_for_mopc']
    assert writes['resume_manifest.json']['scope'] == 'ORTHOGONAL_LORA_ONLY_MOPC_PENDING'
    assert writes['subset_predictions_complete.json']['status'] == 'SUBSET_FIXED_MOPC_PENDING'
    assert writes['parent.json']['scope'] == 'ORTHOGONAL_LORA_ONLY_MOPC_PENDING'
    assert 'predictions_complete.json' not in writes and 'complete.json' not in writes


def test_subset_reservation_prevents_both_entrypoints_and_preserves_old_claim(tmp_path):
    parent = tmp_path/'r10'; parent.mkdir()
    rt = dict(run_root=str(tmp_path/'subset'), runtime_file='subset.json')
    group2_resume.reserve_owner(parent, rt, True)
    assert not (tmp_path/'.resume_successors'/'r10.json').exists()
    with pytest.raises(FileExistsError):
        group2_resume.reserve_owner(parent, dict(rt, runtime_file='other.json'), False)
    with pytest.raises(FileExistsError):
        group2_resume.reserve_owner(parent, rt, True)
    original = tmp_path/'.resume_successors'/'older.json'
    original.parent.mkdir(exist_ok=True)
    original.write_text('{"preserve": true}', encoding='utf-8')
    (tmp_path/'older').mkdir()
    with pytest.raises(ValueError, match='already has a successor'):
        group2_resume.reserve_owner(tmp_path/'older', rt, True)
    assert original.read_text(encoding='utf-8') == '{"preserve": true}'

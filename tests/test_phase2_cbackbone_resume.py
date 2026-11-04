import json
from pathlib import Path

import pytest

from experiments.pamf_phase2_practical import group2_c_backbone_resume as resume
from experiments.pamf_phase2_practical.group2_c_backbone_fast_takeover import link_closed_predictions
from experiments.pamf_phase2_practical.group2_c_backbone_schedule import _queue


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def test_excluded_partial_is_preserved_but_not_copied(tmp_path):
    old = tmp_path/'old'
    partial = old/'predictions'/'partial'
    partial.mkdir(parents=True)
    (partial/'fixed.npz').write_bytes(b'previous prediction')
    jobs = [{'key': 'partial', 'output': str(tmp_path/'new/predictions/partial')}]
    assert link_closed_predictions(old, jobs, excluded_indices={0}) == set()
    assert (partial/'fixed.npz').read_bytes() == b'previous prediction'
    assert not Path(jobs[0]['output']).exists()
    with pytest.raises((ValueError, FileNotFoundError)):
        link_closed_predictions(old, jobs)


def test_excluded_job_is_not_counted_complete_or_scored(tmp_path):
    rt = {'run_root': str(tmp_path), 'gpu_limits': {'0': 1}, 'score_processes': 1, 'poll_seconds': 1}
    _queue(rt, [{'key': 'closed'}, {'key': 'partial'}], 'worker', reused_indices={0}, excluded_indices={1})
    state = json.loads((tmp_path/'worker_state.json').read_text())
    assert state['status'] == 'PARTIAL' and state['completed'] == 1 and state['total'] == 2
    assert state['excluded'] == 1 and state['pending'] == 0
    assert not (tmp_path/'predictions_complete.json').exists()
    with pytest.raises(ValueError):
        _queue(rt, [{'key': 'x'}], 'score', excluded_indices={0})


def test_resume_never_opens_truth_barrier_with_quarantined_jobs(tmp_path, monkeypatch):
    oldroot, root = tmp_path/'old', tmp_path/'new'
    old = {'run_root': str(oldroot), 'runtime_file': '/old/runtime'}
    rt = {'parent_runtime_file': str(tmp_path/'old.json'), 'run_root': str(root), 'runtime_file': '/new/runtime',
          'excluded_failed_indices': [1]}
    put(tmp_path/'old.json', old)
    put(oldroot/'worker_state.json', {'status': 'FAILED', 'completed': 1, 'active': [], 'failed': [{'index': 1}]})
    jobs = [{'key': 'closed'}, {'key': 'partial'}]
    monkeypatch.setattr(resume, 'validate_runtime', lambda x: None)
    monkeypatch.setattr(resume, 'check_parent_runtime', lambda x, y: None)
    monkeypatch.setattr(resume, '_old_processes', lambda x: [])
    monkeypatch.setattr(resume, 'tasks', lambda x: jobs)
    monkeypatch.setattr(resume, '_smoke', lambda x, y: None)
    monkeypatch.setattr(resume, 'link_closed_predictions', lambda *a, **k: {0})
    called = []
    monkeypatch.setattr(resume, '_queue', lambda *a, **k: called.append((a[2], k)))
    resume.dispatch(rt)
    assert called == [('worker', {'reused_indices': {0}, 'excluded_indices': {1}})]
    assert json.loads((root/'resume_complete.json').read_text())['status'] == 'AWAITING_QUARANTINED_JOBS'
    assert not (root/'predictions_complete.json').exists() and not (root/'score_state.json').exists()


def test_resume_refuses_changed_failure_inventory(tmp_path, monkeypatch):
    oldroot = tmp_path/'old'
    put(tmp_path/'old.json', {'run_root': str(oldroot), 'runtime_file': '/old/runtime'})
    put(oldroot/'worker_state.json', {'status': 'FAILED', 'active': [], 'failed': [{'index': 2}]})
    monkeypatch.setattr(resume, 'validate_runtime', lambda x: None)
    monkeypatch.setattr(resume, 'check_parent_runtime', lambda x, y: None)
    with pytest.raises(ValueError, match='inventory'):
        resume.dispatch({'parent_runtime_file': str(tmp_path/'old.json'), 'excluded_failed_indices': [1]})


def test_existing_successor_claim_blocks_later_resume(tmp_path, monkeypatch):
    oldroot = tmp_path/'old'
    put(tmp_path/'old.json', {'run_root': str(oldroot), 'runtime_file': '/old/runtime'})
    put(oldroot/'worker_state.json', {'status': 'FAILED', 'completed': 0, 'active': [], 'failed': [{'index': 0}]})
    put(tmp_path/'.resume_successors/old.json', {'successor_root': 'previous-successor'})
    monkeypatch.setattr(resume, 'validate_runtime', lambda x: None)
    monkeypatch.setattr(resume, 'check_parent_runtime', lambda x, y: None)
    monkeypatch.setattr(resume, '_old_processes', lambda x: [])
    monkeypatch.setattr(resume, 'tasks', lambda x: [{'key': 'partial'}])
    with pytest.raises(FileExistsError):
        resume.dispatch({'parent_runtime_file': str(tmp_path/'old.json'),
                         'run_root': str(tmp_path/'other'), 'runtime_file': '/new/runtime',
                         'excluded_failed_indices': [0]})
    assert not (tmp_path/'other').exists()

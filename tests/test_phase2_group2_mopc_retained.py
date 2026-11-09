import json
from pathlib import Path

import pytest

from experiments.pamf_phase2_practical import group2_mopc_retained as retained


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def make_parent(tmp_path, monkeypatch):
    root = tmp_path/'r13'
    for name in ('jobs', 'logs', 'launches', 'predictions'):
        (root/name).mkdir(parents=True)
    desired = []
    for i in range(3):
        key = f'job-{i}'
        next_job = dict(key=key, profile='dadda_sda', seed=7, capsule='capsules/P/M/R',
                        methods=['mopc_hr'], points=[(1, 2)],
                        output=str(tmp_path/'next/predictions'/key),
                        load_B_for_mopc=str(tmp_path/'r11/predictions'/key), trace_registration=True)
        previous = dict(next_job, output=str(root/'predictions'/key))
        write(root/'jobs'/f'{i:05d}.json', previous)
        desired.append(next_job)
    for i in (0, 1):
        (root/'predictions'/f'job-{i}').mkdir()
        write(root/'launches'/f'worker-{i:05d}.json', {'pid': i+10})
    (root/'predictions/job-0/complete.json').touch()
    write(root/'predictions/job-0/predictions.json',
          {'rows': [dict(k=1, resources={'B_state_ref': str(Path(desired[0]['load_B_for_mopc'])/'B-K1.pt')})]})
    (root/'predictions/job-1/partial.npz').touch()
    (root/'logs/worker-00001.log').write_text('REGISTRATION_FAILURE nonfinite\n', encoding='utf-8')
    checked = []
    monkeypatch.setattr(retained, 'closed_prediction', lambda job, count: checked.append((job['key'], count)))
    state = dict(status='FAILED', completed=1, total=3, active=[],
                 failed=[dict(index=1, key='job-1', exit_code=1)])
    return root, state, desired, checked


def test_partition_preserves_completed_and_failed_and_only_returns_unstarted(tmp_path, monkeypatch):
    root, state, desired, checked = make_parent(tmp_path, monkeypatch)
    result = retained.partition_retained_mopc(root, state, desired)
    assert [job['key'] for job in result['completed']] == ['job-0']
    assert [job['key'] for job in result['quarantined']] == ['job-1']
    assert [job['key'] for job in result['pending']] == ['job-2']
    assert result['completed'][0]['output'] == str(root/'predictions/job-0')
    assert result['pending'][0]['load_B_for_mopc'] == desired[2]['load_B_for_mopc']
    assert checked == [('job-0', 1)]
    assert (root/'predictions/job-1/partial.npz').is_file()


def test_ambiguous_started_job_cannot_be_redispatched(tmp_path, monkeypatch):
    root, state, desired, _ = make_parent(tmp_path, monkeypatch)
    write(root/'launches/worker-00002.json', {'pid': 999})
    with pytest.raises(ValueError, match='Unclassified'):
        retained.partition_retained_mopc(root, state, desired)


def test_worker_log_without_launch_record_is_not_unstarted(tmp_path, monkeypatch):
    root, state, desired, _ = make_parent(tmp_path, monkeypatch)
    (root/'logs/worker-00002.log').write_text('', encoding='utf-8')
    with pytest.raises(ValueError, match='Unclassified'):
        retained.partition_retained_mopc(root, state, desired)


def test_parent_or_saved_b_mismatch_is_rejected(tmp_path, monkeypatch):
    root, state, desired, _ = make_parent(tmp_path, monkeypatch)
    desired[2]['load_B_for_mopc'] = str(tmp_path/'other-B')
    with pytest.raises(ValueError, match='binding'):
        retained.partition_retained_mopc(root, state, desired)
    desired[2]['load_B_for_mopc'] = str(tmp_path/'r11/predictions/job-2')
    state['active'] = [dict(index=1, pid=5)]
    with pytest.raises(ValueError, match='drained'):
        retained.partition_retained_mopc(root, state, desired)


def test_completed_wrong_saved_b_is_rejected(tmp_path, monkeypatch):
    root, state, desired, _ = make_parent(tmp_path, monkeypatch)
    write(root/'predictions/job-0/predictions.json',
          {'rows': [dict(k=1, resources={'B_state_ref': 'wrong'})]})
    with pytest.raises(ValueError, match='saved B binding'):
        retained.partition_retained_mopc(root, state, desired)


def test_different_kn_points_are_rejected(tmp_path, monkeypatch):
    root, state, desired, _ = make_parent(tmp_path, monkeypatch)
    desired[2]['points'] = [(1, 5)]
    with pytest.raises(ValueError, match='binding'):
        retained.partition_retained_mopc(root, state, desired)


def test_unknown_parent_output_and_nonmatching_failure_are_rejected(tmp_path, monkeypatch):
    root, state, desired, _ = make_parent(tmp_path, monkeypatch)
    (root/'predictions/unknown').mkdir()
    with pytest.raises(ValueError, match='Unclassified'):
        retained.partition_retained_mopc(root, state, desired)
    (root/'predictions/unknown').rmdir()
    (root/'logs/worker-00001.log').write_text('different failure', encoding='utf-8')
    with pytest.raises(ValueError, match='Unclassified retained failure'):
        retained.partition_retained_mopc(root, state, desired)

import pytest
from experiments.pamf_phase2_practical import group2_priority as priority


def test_peer_identity_requires_exact_runtime_cwd_and_start():
    peer = {'pid': 3, 'start_ticks': 99, 'release_root': '/release', 'runtime_file': '/release/runtime.json'}
    info = {'pid': 3, 'start_ticks': 99, 'cwd': '/release', 'state': 'S',
            'argv': ['python', '-m', 'experiments.pamf_phase2_practical.group2_c_backbone_resume',
                     '--runtime', '/release/runtime.json']}
    assert priority.peer_matches(info, peer)
    assert not priority.peer_matches(dict(info, start_ticks=100), peer)
    assert not priority.peer_matches(dict(info, cwd='/other'), peer)
    assert not priority.peer_matches(dict(info, argv=info['argv'][:-1]+['/other/runtime.json']), peer)


def test_only_verified_stopped_peer_is_resumed(monkeypatch):
    monkeypatch.setattr(priority.signal, 'SIGCONT', 18, raising=False)
    peer = {'pid': 3, 'start_ticks': 99, 'release_root': '/release', 'runtime_file': '/release/runtime.json'}
    info = {'pid': 3, 'start_ticks': 99, 'cwd': '/release', 'state': 'T',
            'argv': ['python', '-m', 'experiments.pamf_phase2_practical.group2_c_backbone_resume',
                     '--runtime', '/release/runtime.json']}
    calls = []
    monkeypatch.setattr(priority, '_proc', lambda pid: info)
    monkeypatch.setattr(priority.os, 'kill', lambda pid, sig: calls.append((pid, sig)))
    assert priority.resume_peer(peer)
    assert calls == [(3, priority.signal.SIGCONT)]
    calls.clear()
    info['start_ticks'] = 101
    assert not priority.resume_peer(peer) and not calls


def test_completed_c_workers_are_verified_before_b_can_start(tmp_path, monkeypatch):
    root = tmp_path/'c'
    (root/'jobs').mkdir(parents=True)
    (root/'jobs'/'00007.json').write_text('{"output":"saved"}')
    peer = {'pid': 3, 'start_ticks': 99, 'release_root': '/release', 'runtime_file': '/release/runtime.json'}
    monkeypatch.setattr(priority, '_proc', lambda pid: {'start_ticks': 5, 'state': 'Z'})
    checked = []
    monkeypatch.setattr(priority, 'prediction_closed', lambda job: checked.append(job['output']))
    assert priority.workers_drained({7: {'pid': 4, 'start_ticks': 5}}, root)
    assert checked == ['saved']
    monkeypatch.setattr(priority, '_proc', lambda pid: {'start_ticks': 5, 'state': 'R'})
    assert not priority.workers_drained({7: {'pid': 4, 'start_ticks': 5}}, root)


def test_invalid_c_output_prevents_priority_transfer(tmp_path, monkeypatch):
    root = tmp_path/'c'
    (root/'jobs').mkdir(parents=True)
    (root/'jobs'/'00007.json').write_text('{}')
    monkeypatch.setattr(priority, '_proc', lambda pid: None)
    def invalid(job):
        raise ValueError('not closed')
    monkeypatch.setattr(priority, 'prediction_closed', invalid)
    with pytest.raises(ValueError, match='not closed'):
        priority.workers_drained({7: {'pid': 4, 'start_ticks': 5}}, root)


@pytest.mark.parametrize('failure', ['reaped_C_failure', 'B_validation_failure'])
def test_preparation_failure_restores_c_without_claiming_b(tmp_path, monkeypatch, failure):
    import json
    import os
    from types import SimpleNamespace
    monkeypatch.setattr(priority.signal, 'SIGCONT', 18, raising=False)
    monkeypatch.setattr(priority.signal, 'SIGSTOP', 19, raising=False)
    root = tmp_path/'c'
    (root/'jobs').mkdir(parents=True)
    old = {'run_root': str(root), 'release_root': '/release', 'runtime_file': str(tmp_path/'c.json')}
    (tmp_path/'c.json').write_text(json.dumps(old))
    (root/'worker_state.json').write_text(json.dumps({'status': 'RUNNING', 'failed': [], 'active': []}))
    (root/'jobs'/'00007.json').write_text('{}')
    peer = dict(old, pid=3, start_ticks=99)
    info = dict(peer, cwd='/release', state='S', argv=['python', '-m',
                'experiments.pamf_phase2_practical.group2_c_backbone_resume', '--runtime', old['runtime_file']])
    rt = {'priority_peer': peer, 'run_root': str(tmp_path/'b'), 'previous_run_root': str(tmp_path/'b_old'),
          'runtime_file': str(tmp_path/'b.json'), 'poll_seconds': 1}
    monkeypatch.setattr(priority, '_proc', lambda pid: {'start_ticks': 10} if pid == os.getpid() else info)
    monkeypatch.setattr(priority, '_children', lambda *args: {})
    monkeypatch.setattr(priority, '_launched_indices', lambda old: {7} if failure == 'reaped_C_failure' else set())
    monkeypatch.setattr(priority, 'b_processes', lambda runtime: [])
    monkeypatch.setattr(priority.os, 'kill', lambda pid, sig: info.update(state='T' if sig == 19 else 'S'))
    def popen(*args, **kwargs):
        (tmp_path/'b_coordination'/'watcher.json').write_text('{}')
        return SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(priority.subprocess, 'Popen', popen)
    started = []
    if failure == 'reaped_C_failure':
        monkeypatch.setattr(priority, 'resume_b', lambda rt: started.append(True))
    else:
        from experiments.pamf_phase2_practical import group2_resume
        def invalid_runtime(rt):
            raise ValueError('launched output did not close')
        monkeypatch.setattr(group2_resume, 'validate_runtime', invalid_runtime)
    def invalid(job):
        raise ValueError('launched output did not close')
    monkeypatch.setattr(priority, 'prediction_closed', invalid)
    with pytest.raises(ValueError, match='launched output did not close'):
        priority.dispatch(rt)
    assert not started and info['state'] == 'S'
    assert not (tmp_path/'.resume_successors'/'b_old.json').exists()


def test_watcher_spawn_failure_does_not_claim_b_parent(tmp_path, monkeypatch):
    import json
    import os
    old = {'run_root': str(tmp_path/'c'), 'release_root': '/release', 'runtime_file': str(tmp_path/'c.json')}
    (tmp_path/'c').mkdir()
    (tmp_path/'c.json').write_text(json.dumps(old))
    (tmp_path/'c'/'worker_state.json').write_text(json.dumps({'status': 'RUNNING', 'failed': []}))
    peer = dict(old, pid=3, start_ticks=99)
    info = dict(peer, cwd='/release', state='S', argv=['python', '-m',
                'experiments.pamf_phase2_practical.group2_c_backbone_resume', '--runtime', old['runtime_file']])
    monkeypatch.setattr(priority, '_proc', lambda pid: {'start_ticks': 10} if pid == os.getpid() else info)
    def failed(*args, **kwargs):
        raise OSError('watcher failed')
    monkeypatch.setattr(priority.subprocess, 'Popen', failed)
    rt = {'priority_peer': peer, 'run_root': str(tmp_path/'b'), 'previous_run_root': str(tmp_path/'b_old'),
          'runtime_file': str(tmp_path/'b.json'), 'poll_seconds': 1}
    with pytest.raises(OSError, match='watcher failed'):
        priority.dispatch(rt)
    assert info['state'] == 'S'
    assert not (tmp_path/'.resume_successors'/'b_old.json').exists()

import importlib.util
import json
from pathlib import Path
import signal
import pytest

spec = importlib.util.spec_from_file_location('pause_controls', Path(__file__).resolve().parents[1] / 'tools/pause_separate_controls.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture(autouse=True)
def posix_constants(monkeypatch):
    # Only the mocked kill implementation below consumes these on Windows.
    monkeypatch.setattr(signal, 'SIGSTOP', 19, raising=False)
    monkeypatch.setattr(signal, 'SIGCONT', 18, raising=False)


def setup(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    dest = run / 'pause'
    dest.mkdir(parents=True)
    dispatcher = dict(pid=10, start='1', state='S', cwd='/release', argv=['dispatch', str(run)])
    worker = dict(pid=20, start='2', state='S', cwd='/release', argv=['train', '--output', str(run / 'PURE_SIM')])
    native = dict(pid=30, start='3', state='S', cwd='/release', argv=['native'])
    processes = {x['pid']: x for x in (dispatcher, worker, native)}
    checkpoint = dest / 'checkpoint'
    checkpoint.write_text('retained')
    row = run / 'PURE_SIM'
    row.mkdir()
    (row / 'actions.jsonl').write_text('{}\n' * 3)
    manifest = dict(run=str(run), dispatcher=dict(dispatcher), rows={'PURE_SIM':dict(process=dict(worker), checkpoint=str(checkpoint), step=2)})
    (dest / 'manifest.json').write_text(json.dumps(manifest))
    (dest / 'restore_validation.json').write_text(json.dumps(dict(status='PASS',rows={'PURE_SIM':dict(status='PASS',checkpoint=str(checkpoint))})))
    state = dict(release='/release', rows={'PURE_SIM':dict(family='pure_game',status='RUNNING'),
        'NATIVE':dict(family='native_baseline',status='RUNNING'), 'QUEUED':dict(family='pure_game',status='QUEUED')})
    (run / 'pipeline_state.json').write_text(json.dumps(state))
    calls = []
    pending = set()
    def kill(pid, sig):
        calls.append((pid,sig))
        if sig == signal.SIGSTOP: processes[pid]['state']='T'
        elif sig == signal.SIGTERM: pending.add(pid)
        elif sig == signal.SIGCONT:
            if pid in pending: processes.pop(pid,None)
            elif pid in processes: processes[pid]['state']='S'
    monkeypatch.setattr(m,'identity',lambda pid:processes.get(pid))
    monkeypatch.setattr(m.os,'kill',kill)
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    monkeypatch.setattr(m,'live_workers',lambda *args:[dict(worker)])
    return run,dest,processes,calls


def test_only_verified_pure_worker_stopped(tmp_path,monkeypatch):
    run,dest,processes,calls=setup(tmp_path,monkeypatch)
    m.stop(run,dest)
    assert set(processes)=={30}
    assert all(pid!=30 for pid,sig in calls)
    state=json.loads((run/'pipeline_state.json').read_text())
    assert state['rows']['NATIVE']['status']=='RUNNING'
    assert state['rows']['QUEUED']['status']=='HELD_USER_PAUSE'
    assert state['rows']['PURE_SIM']['status']=='PAUSED_RECOVERABLE'


def test_unrecorded_worker_aborts_without_termination(tmp_path,monkeypatch):
    run,dest,processes,calls=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'live_workers',lambda *args:[dict(pid=99,argv=['train','--output',str(run/'PURE_OTHER')])])
    with pytest.raises(RuntimeError,match='Unrecorded'):m.stop(run,dest)
    assert set(processes)=={10,20,30}
    assert processes[10]['state']=='S'
    assert not any(sig==signal.SIGTERM for pid,sig in calls)


def test_actions_read_failure_unfreezes_worker(tmp_path,monkeypatch):
    run,dest,processes,calls=setup(tmp_path,monkeypatch)
    (run/'PURE_SIM/actions.jsonl').unlink()
    with pytest.raises(FileNotFoundError):m.stop(run,dest)
    assert processes[20]['state']=='S'
    assert 30 in processes
    assert json.loads((dest/'manifest.json').read_text())['phase']=='PARTIAL_PAUSE_FAILED'

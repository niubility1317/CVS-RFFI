import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('pair_dispatch', Path(__file__).parents[1] / 'tools/pair_matrix_dispatch.py')
dispatch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dispatch)


def test_gpu_choice_respects_foreign_training_and_one_owned_job():
    gpu = {0: {'free_mib': 9000, 'training_pids': {10}},
           1: {'free_mib': 20000, 'training_pids': {11, 12}},
           2: {'free_mib': 8000, 'training_pids': set()}}
    assert dispatch.choose_gpu([0, 1, 2], gpu, []) == 0
    assert dispatch.choose_gpu([0, 1, 2], gpu, [{'gpu': 0, 'pid': 20}]) is None


def test_assigned_gpu_obeys_configured_caps_and_memory():
    gpu = {0: {'free_mib': 9000, 'training_pids': {1, 2, 3, 4}},
           1: {'free_mib': 20000, 'training_pids': set()}}
    active = [{'gpu': 0, 'pid': 10}, {'gpu': 0, 'pid': 11}]
    assert dispatch.choose_gpu([0], gpu, active, max_new_per_gpu=3, max_per_gpu=8) == 0
    assert dispatch.choose_gpu([0], gpu, active + [{'gpu': 0, 'pid': 12}], max_new_per_gpu=3, max_per_gpu=8) is None
    gpu[0]['free_mib'] = 8191
    assert dispatch.choose_gpu([0], gpu, [], max_new_per_gpu=3, max_per_gpu=8) is None


def test_pending_selection_can_skip_busy_assigned_gpu_without_reassigning():
    snapshot = {0: {'free_mib': 0, 'training_pids': set()}, 1: {'free_mib': 9000, 'training_pids': set()}}
    rows = [{'row_id': 'A', 'assigned_gpu': 0}, {'row_id': 'B', 'assigned_gpu': 1}]
    manifest = {'gpus': [0, 1], 'max_new_per_gpu': 3, 'max_per_gpu': 8, 'min_free_mib': 8192}
    assert dispatch.choose_pending_row(rows, manifest, snapshot, []) == (1, 1)


def test_exit_zero_requires_epoch200_and_checkpoint(tmp_path):
    assert not dispatch.validate_training_artifacts(tmp_path)[0]
    (tmp_path / 'final_ssdg.pth').write_bytes(b'checkpoint')
    metrics = tmp_path / 'metrics_epoch.jsonl'
    metrics.write_text(json.dumps({'epoch': 199}) + '\n', encoding='utf-8')
    assert not dispatch.validate_training_artifacts(tmp_path)[0]
    metrics.write_text(json.dumps({'epoch': 200}) + '\n', encoding='utf-8')
    assert dispatch.validate_training_artifacts(tmp_path)[0]


def test_fingerprint_groups_same_exception_but_not_different_errors():
    a = dispatch.failure_fingerprint('Traceback\nRuntimeError: out of memory on GPU 1\n', 1)
    b = dispatch.failure_fingerprint('Traceback\nRuntimeError: out of memory on GPU 6\n', 1)
    c = dispatch.failure_fingerprint('Traceback\nValueError: invalid shape\n', 1)
    assert a == b and a != c


def test_manifest_rejects_outputs_outside_runroot(tmp_path):
    code = tmp_path / 'code'; code.mkdir()
    checkpoint = code / 'init.pth'; checkpoint.write_bytes(b'x')
    root = tmp_path / 'run'
    manifest = dict(run_id='run', code_root=str(code), output_root=str(root),
        checkpoint=str(checkpoint), python=sys.executable, gpus=[0], max_active=24,
        max_new_per_gpu=3, max_per_gpu=8, min_free_mib=8192,
        rows=[dict(row_id='A', assigned_gpu=0, argv=[sys.executable, 'train.py', '--output_dir', str(root/'A')],
                   environment={}, cwd=str(code))], smoke_argv=[sys.executable, 'smoke.py'])
    dispatch.validate_manifest(manifest)
    manifest['rows'][0]['argv'][-1] = str(tmp_path / 'escape')
    with pytest.raises(ValueError, match='output_dir'):
        dispatch.validate_manifest(manifest)


def test_atomic_root_creation_never_reuses_existing_directory(tmp_path):
    root = tmp_path / 'run'
    dispatch.create_runroot(root)
    (root / 'preserve').write_text('existing', encoding='utf-8')
    with pytest.raises(FileExistsError):
        dispatch.create_runroot(root)
    assert (root / 'preserve').read_text() == 'existing'


def test_repeated_failure_stops_new_rows_but_waits_for_healthy(monkeypatch, tmp_path):
    import os
    code = tmp_path / 'code'; code.mkdir()
    checkpoint = code / 'init.pth'; checkpoint.write_bytes(b'x')
    root = tmp_path / 'run'
    rows = [dict(row_id=name, assigned_gpu=index % 3, argv=[sys.executable, 'train.py', '--output_dir', str(root/name)],
                 environment={}, cwd=str(code)) for index, name in enumerate(('healthy', 'bad1', 'bad2', 'never'))]
    manifest = dict(run_id='run', code_root=str(code), output_root=str(root), checkpoint=str(checkpoint),
                    python=sys.executable, gpus=[0, 1, 2], max_active=8, rows=rows,
                    smoke_argv=[sys.executable, 'smoke.py'])
    path = code / 'manifest.json'; path.write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(flock=lambda *a: None, LOCK_EX=1, LOCK_NB=2))
    monkeypatch.setattr(dispatch, 'os', SimpleNamespace(name='posix', getpid=os.getpid, fsync=os.fsync, replace=os.replace))
    monkeypatch.setattr(dispatch.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(dispatch, 'gpu_snapshot', lambda: {
        gpu: {'free_mib': 9000, 'training_pids': set()} for gpu in range(3)})
    launched = []
    def start(argv, cwd, environment, gpu, runroot):
        runroot.mkdir()
        log = runroot / 'process.log'; log.write_text('RuntimeError: same failure 1\n')
        name = runroot.name; launched.append(name)
        class Process:
            calls = 0
            def poll(self):
                self.calls += 1
                if name == 'healthy' and self.calls < 6:
                    return None
                return 1 if name.startswith('bad') else 0
        return Process(), dict(pid=len(launched), cwd=cwd, gpu=gpu, argv=argv,
                               runroot=str(runroot), log=str(log), status='RUNNING')
    monkeypatch.setattr(dispatch, '_start', start)
    monkeypatch.setattr(dispatch, 'validate_training_artifacts', lambda root: (True, 'TRAINING_COMPLETE_SOURCE_ONLY'))
    assert dispatch.run(path) == 1
    state = json.loads((root / 'status.json').read_text())
    assert launched == ['smoke', 'healthy', 'bad1', 'bad2']
    assert state['rows'][0]['status'] == 'TRAINING_COMPLETE_SOURCE_ONLY'
    assert state['unlaunched_rows'] == ['never']
    assert state['status'] == 'STOPPED_NEW_ROWS'

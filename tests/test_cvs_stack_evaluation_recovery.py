import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.cvs_phase1_stack import evaluation_recovery as e


@pytest.fixture
def parent(tmp_path):
    root = tmp_path / 'runs' / 'source-run'
    release = tmp_path / 'releases' / 'source-release'
    rows = [dict(row_id='r2-a-s1'), dict(row_id='r3-b-s1')]
    d = SimpleNamespace(BASE=str(root), PROJECT=str(tmp_path), RUN='source-run',
                        RELEASE='source-release', STAGES=['r2', 'r3'], rows=lambda: rows,
                        read=lambda p: json.loads(Path(p).read_text(encoding='utf-8')))
    def validate(value, stage):
        if value != dict(stage=stage, status='SOURCE_FROZEN', target_access=False):
            raise ValueError('Invalid source freeze')
    d.validate_freeze = validate
    active = dict(pid=101, start_ticks=77, cwd=str(release),
                  argv=['python', '-u', '-m', 'experiments.cvs_phase1_stack.recover'],
                  run_id=d.RUN, release=d.RELEASE)
    e.write(root / 'dispatcher_active.json', active)
    e.write(root / 'all_sources_frozen.json', dict(status='ALL_SOURCE_FROZEN', run_id=d.RUN,
            rows=[r['row_id'] for r in rows], stages=d.STAGES, target_access=False))
    for stage in d.STAGES:
        e.write(root / (stage + '_source_frozen.json'), dict(stage=stage, status='SOURCE_FROZEN', target_access=False))
    failures = [dict(row_id=rows[0]['row_id'], exit_code=1, error='Worker failed or missing valid completion')]
    e.write(root / 'queue_state.json', dict(phase='predict', kind='predict', active=[],
            pending=[rows[1]['row_id']], failures=failures, controller_pid=101, controller_release=d.RELEASE))
    e.write(root / 'failure.json', dict(status='FAILED', no_retry=True, release=d.RELEASE,
            error=repr(RuntimeError('Failed rows preserved; no retry ' + repr(failures)))))
    rid = rows[0]['row_id']
    log = tmp_path / 'logs' / d.RUN / ('predict-' + rid + '.log')
    log.parent.mkdir(parents=True)
    log.write_text('Traceback (most recent call last):\n  File "' + str(release / 'experiments/cvs_phase1_stack/predict.py')
                   + '", line 29, in predict\n'
                   + "    if contract[k]!=expected_contract[k]:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH '+k)\n"
                   + "KeyError: 'classes'\n", encoding='utf-8')
    e.write(root / 'launch_predict.json', dict(rows=[dict(row_id=rid, pid=202, cwd=str(release),
            argv=['python', '-u', '-m', 'experiments.cvs_phase1_stack.predict', '--config',
                  str(root / 'configs' / ('predict-' + rid + '.json'))],
            log=str(log), controller_release=d.RELEASE)]))
    return d


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_exact_prequery_failure_opens_gate_without_mutating_parent(parent):
    before = snapshot(Path(parent.PROJECT))
    assert e.parent_ready(parent, lambda pid: None)
    assert snapshot(Path(parent.PROJECT)) == before


def test_live_controller_does_not_read_freezes_or_query_even_if_failure_exists(parent):
    original_read = parent.read
    active = original_read(Path(parent.BASE) / 'dispatcher_active.json')
    reads = []
    def read(path):
        reads.append(Path(path).name)
        if reads != ['dispatcher_active.json']:
            raise AssertionError('Read beyond active controller while training')
        return active
    parent.read = read
    assert not e.parent_ready(parent, lambda pid: active)
    assert reads == ['dispatcher_active.json']


@pytest.mark.parametrize('corruption', [
    'missing_freeze', 'partial_freeze', 'invalid_stage', 'source_failure', 'failure_fingerprint',
    'different_exception', 'different_traceback', 'prediction_artifact', 'unlaunched_prediction_artifact',
    'scoring_artifact', 'worker_alive', 'missing_failure', 'duplicate_launch', 'missing_pending',
])
def test_gate_rejects_other_failure_or_target_contact(parent, corruption):
    root = Path(parent.BASE)
    process = lambda pid: None
    if corruption == 'missing_freeze':
        (root / 'all_sources_frozen.json').unlink()
    elif corruption == 'partial_freeze':
        data = parent.read(root / 'all_sources_frozen.json'); data['rows'].pop()
        e.write(root / 'all_sources_frozen.json', data)
    elif corruption == 'invalid_stage':
        e.write(root / 'r3_source_frozen.json', {})
    elif corruption in {'source_failure', 'missing_pending'}:
        data = parent.read(root / 'queue_state.json')
        if corruption == 'source_failure': data['phase'] = 'r3'
        else: data['pending'] = []
        e.write(root / 'queue_state.json', data)
    elif corruption == 'failure_fingerprint':
        data = parent.read(root / 'failure.json'); data['error'] = repr(KeyError('classes'))
        e.write(root / 'failure.json', data)
    elif corruption in {'different_exception', 'different_traceback'}:
        log = Path(parent.read(root / 'launch_predict.json')['rows'][0]['log'])
        text = log.read_text(encoding='utf-8')
        text = text.replace("KeyError: 'classes'", "KeyError: 'other'") if corruption == 'different_exception' else text.replace('contract[k]', 'query[k]')
        log.write_text(text, encoding='utf-8')
    elif corruption in {'prediction_artifact', 'unlaunched_prediction_artifact'}:
        rid = parent.rows()[0 if corruption == 'prediction_artifact' else 1]['row_id']
        e.write(root / rid / 'prediction' / 'resolved_config.json', {})
    elif corruption == 'scoring_artifact':
        e.write(root / 'phase1_scored_results.json', {})
    elif corruption == 'worker_alive':
        process = lambda pid: dict(pid=202) if pid == 202 else None
    elif corruption == 'missing_failure':
        (root / 'failure.json').unlink()
    elif corruption == 'duplicate_launch':
        data = parent.read(root / 'launch_predict.json'); data['rows'] *= 2
        e.write(root / 'launch_predict.json', data)
    before = snapshot(Path(parent.PROJECT))
    with pytest.raises((ValueError, FileNotFoundError)):
        e.parent_ready(parent, process)
    assert snapshot(Path(parent.PROJECT)) == before


def test_pid_reuse_after_original_controller_exit_is_not_a_live_original(parent):
    assert e.parent_ready(parent, lambda pid: dict(start_ticks=999) if pid == 101 else None)


def test_new_run_exclusive_creation_preserves_existing_failure(tmp_path, monkeypatch):
    root = tmp_path / 'evaluation'
    e.write(root / 'failure.json', {'preserve': True})
    monkeypatch.setattr(e, 'BASE', root)
    before = snapshot(root)
    with pytest.raises(FileExistsError):
        e.dispatch()
    assert snapshot(root) == before


def test_mixed_interfaces_keep_source_paths_and_score_only_after_queue(tmp_path, monkeypatch):
    from experiments.cvs_phase1_stack import predict
    base = tmp_path / 'evaluation'
    base.mkdir()
    monkeypatch.setattr(e, 'BASE', base)
    monkeypatch.setattr(predict, 'PREDICTION_BASE', str(base / 'mixed'))
    source = tmp_path / 'source-run'
    row = dict(row_id='r2-bridge-s1', model_seed=1, stage='r2', arm='bridge')
    config = dict(row, output_root=str(tmp_path / 'original-r2' / 'source'))
    source_path = source / 'configs' / ('source-' + row['row_id'] + '.json')
    e.write(source_path, config)
    calls = []
    def validate(value):
        assert value == config
    d = SimpleNamespace(BASE=str(source), PROJECT=str(tmp_path), ROOT=e.ROOT,
                        CAPSULE='query-capsule-path', TRUTH='truth-path', rows=lambda: [row],
                        read=lambda p: json.loads(Path(p).read_text(encoding='utf-8')), validate=validate)
    def queue(design, jobs, kind, phase, adoptions):
        assert (design.BASE, design.RUN, design.ROOT) == (str(base / 'mixed'), e.EVAL_RUN, e.ROOT)
        assert (kind, phase, adoptions) == ('predict', 'predict', {})
        p = d.read(jobs[0]['config'])
        assert p == dict(row_id=row['row_id'], source_config=str(source_path),
                         source_output=config['output_root'], p1_capsule=d.CAPSULE,
                         output_root=str(base / 'mixed' / row['row_id'] / 'prediction'))
        calls.append('predictions-fixed')
    q = SimpleNamespace(queue=queue)
    def run(cmd, cwd, check):
        assert calls[0] == 'predictions-fixed'
        assert cwd == e.ROOT and check
        calls.append(cmd[2])
        spec = d.read(base / 'mixed' / 'evaluation_spec.json')
        assert spec['run_id'] == e.EVAL_RUN and spec['runtime_root'] == str(base / 'mixed')
    monkeypatch.setattr(e.subprocess, 'run', run)
    before = snapshot(source)
    e.mixed(d, q)
    assert calls == ['predictions-fixed', 'comparison_suite.score', 'experiments.cvs_phase1_stack.analyze']
    assert d.read(base / 'mixed' / 'completion.json') == e.COMPLETE
    assert snapshot(source) == before


@pytest.mark.parametrize('gate_fails', [False, True])
def test_dispatch_waits_without_query_then_sequences_mixed_and_seven(tmp_path, monkeypatch, gate_fails):
    from experiments import cvs_phase1_stack as package
    root = tmp_path / 'runs' / e.EVAL_RUN
    release = tmp_path / 'releases' / e.RELEASE
    release.mkdir(parents=True)
    (release / 'release_commit.txt').write_text('test-commit\n')
    monkeypatch.setattr(e, 'PROJECT', str(tmp_path))
    monkeypatch.setattr(e, 'BASE', root)
    monkeypatch.setattr(e, 'ROOT', release)
    d = SimpleNamespace(ROOT=release, RUN='source-run',
                        read=lambda p: json.loads(Path(p).read_text(encoding='utf-8')))
    q = SimpleNamespace(proc=lambda pid: dict(pid=pid))
    monkeypatch.setattr(package, 'dispatch', d)
    monkeypatch.setattr(package, 'capacity16', q)
    calls = []
    def gate(*args):
        calls.append('gate')
        assert d.read(root / 'state.json')['target_read'] is False
        if gate_fails:
            raise ValueError('Unsupported original failure')
        return len(calls) > 1
    monkeypatch.setattr(e, 'parent_ready', gate)
    def sleep(seconds):
        assert seconds <= 30
        assert not (root / 'mixed').exists()
        calls.append('sleep')
    monkeypatch.setattr(e.time, 'sleep', sleep)
    def mixed(*args):
        assert d.read(root / 'parent_gate.json')['status'] == 'VERIFIED'
        calls.append('mixed')
        e.write(root / 'mixed' / 'completion.json', e.COMPLETE)
    monkeypatch.setattr(e, 'mixed', mixed)
    def seven():
        assert d.read(root / 'mixed' / 'completion.json') == e.COMPLETE
        calls.append('seven')
        e.write(root / 'seven_views' / 'completion.json', dict(e.COMPLETE, views=['clean']))
    monkeypatch.setattr(package, 'sixscene_after', SimpleNamespace(BASE=root / 'seven_views',
                        RELEASE=e.RELEASE, VIEWS=['clean'], dispatch=seven), raising=False)
    if gate_fails:
        with pytest.raises(ValueError, match='Unsupported'):
            e.dispatch()
        assert calls == ['gate']
        assert d.read(root / 'failure.json')['status'] == 'FAILED'
        assert not (root / 'mixed').exists()
    else:
        e.dispatch()
        assert calls == ['gate', 'sleep', 'gate', 'mixed', 'seven']
        assert d.read(root / 'completion.json')['status'] == 'ANALYZED'

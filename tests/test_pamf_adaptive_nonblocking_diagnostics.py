import json

from experiments.pamf_branch_training import prepare as prepare_module
from experiments.pamf_branch_training import runner
from experiments.pamf_repl_completion import runner as resource_runner


def test_failed_source_diagnostic_does_not_cancel_frozen_predictions_or_scores(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    source = root / 'source/complete.json'
    diag = root / 'diagnostic/complete.json'
    pred = root / 'predictions/complete.json'
    weak = root / 'weak_predictions/complete.json'
    score = root / 'scores/complete.json'
    weak_score = root / 'weak_scores/complete.json'
    jobs = [
        dict(id='train', kind='train', config='unused', completion=str(source), successful_statuses=['DONE'], dependencies=[]),
        dict(id='diag', kind='source_diagnostic', config='unused', completion=str(diag), successful_statuses=['DONE'], dependencies=[str(source)], nonblocking=True),
        dict(id='predict', kind='predict', config='unused', completion=str(pred), successful_statuses=['DONE'], dependencies=[str(source)]),
        dict(id='weak_predict', kind='weak_predict', config='unused', completion=str(weak), successful_statuses=['DONE'], dependencies=[str(source)]),
        dict(id='score', kind='score', config='unused', completion=str(score), successful_statuses=['DONE'], dependencies=[str(pred), str(weak)]),
        dict(id='weak_score', kind='weak_score', config='unused', completion=str(weak_score), successful_statuses=['DONE'], dependencies=[str(pred), str(weak)]),
    ]
    monkeypatch.setattr(prepare_module, 'prepare', lambda rt: jobs)
    monkeypatch.setattr(runner, 'admits', lambda *args: True)
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    monkeypatch.setattr(resource_runner, 'foreign_CPU_score', lambda _: False)
    class Process:
        def __init__(self, code):
            self.pid, self.code, self.returncode = 1234, code, code
        def wait(self):
            return self.code
        def poll(self):
            return self.code
    def launch(rt, job, gpu):
        if job['kind'] == 'smoke':
            path = root / 'smoke.json'
            data, code = {'status': 'PASS', 'checks': [{'peak_allocated_bytes': 1}]}, 0
        elif job['kind'] == 'source_diagnostic':
            path, data, code = None, None, 1
        else:
            path, data, code = __import__('pathlib').Path(job['completion']), {'status': 'DONE'}, 0
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding='utf-8')
        return Process(code), dict(status='RUNNING', pid=1234, gpu=gpu)
    monkeypatch.setattr(runner, 'launch', launch)
    runner.dispatch(dict(run_root=str(root), code_commit='test', launch_owner='test',
                         allowed_gpus=[0], runtime_file='unused'))
    state = json.loads((root / 'pipeline_state.json').read_text(encoding='utf-8'))
    assert state['status'] == 'PARTIAL_TECHNICAL_FAILURE'
    assert state['jobs']['diag']['status'] == 'FAILED'
    assert state['nonblocking_failed_jobs'] == ['diag'] and not state['stop_new_dispatch']
    assert all(state['jobs'][job]['status'] == 'COMPLETE' for job in ('predict', 'weak_predict', 'score', 'weak_score'))

"""Synthetic orchestration checks; no network, GPU or source data."""
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from experiments.phase1_joint_particle import runner


def _exercise(fail_predict):
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root/'release_commit.txt').write_text('example-commit', encoding='ascii')
        cache = root/'source-cache.pt'
        cache.write_bytes(b'fixed-source-handle')
        source, ground = 'source-unit', 'ground-unit'
        calls = []

        class FakeProcess:
            next_pid = 400

            def __init__(self, command, *, cwd, env, stdin, stdout, stderr):
                del cwd, stdin, stdout, stderr
                self.pid = FakeProcess.next_pid
                FakeProcess.next_pid += 1
                module = command[command.index('-m')+1]
                mode = ('source_smoke' if '--smoke' in command else 'source_train') if module.endswith('.train') else command[command.index(module)+1]
                calls.append((mode, env['CUDA_VISIBLE_DEVICES']))
                if mode == 'source_smoke':
                    out = Path(command[command.index('--out')+1])
                    out.mkdir(parents=True)
                    (out/'smoke.json').write_text(json.dumps({'status':'SOURCE_TRAINING_SMOKE_PASS', 'F_weights_read':0}), encoding='utf8')
                if mode == 'predict' and not fail_predict:
                    out = Path(command[command.index('--out')+1])/'flow/test_prediction'
                    out.mkdir(parents=True)
                    (out/'complete.json').write_text('{}', encoding='utf8')
                if mode == 'score':
                    assert env['CUDA_VISIBLE_DEVICES'] == ''
                    out = Path(command[command.index('--out')+1])/'flow/test_prediction/complete.json'
                    assert out.is_file(), 'score must follow complete predictions'
                self.returncode = 7 if mode == 'predict' and fail_predict else 0

            def wait(self):
                return self.returncode

        try:
            os.chdir(root)
            with patch.object(runner, 'wait_for_gpu', lambda _: None), patch.object(runner.subprocess, 'Popen', FakeProcess):
                if fail_predict:
                    try:
                        runner.execute(root, source, ground, 'example-commit', cache, 3)
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError('Failed prediction must abort')
                else:
                    runner.execute(root, source, ground, 'example-commit', cache, 3)
            status = json.loads((root/'runs'/(source+'-controller')/'status.json').read_text(encoding='utf8'))
            return calls, status, (root/'runs'/(source+'-controller')/'failure.json').exists()
        finally:
            os.chdir(previous)


def test_complete_flow_scores_only_after_prediction_on_cpu():
    calls, status, failure = _exercise(False)
    assert calls == [('source_smoke','3'), ('smoke','3'), ('source_train','3'), ('predict','3'), ('score','')]
    assert status['state'] == 'GROUND_SCORE_COMPLETE' and not failure


def test_prediction_failure_does_not_score():
    calls, status, failure = _exercise(True)
    assert [mode for mode, _ in calls] == ['source_smoke','smoke','source_train','predict']
    assert status['state'] == 'FAILED' and failure


def test_unregistered_release_rejected_before_capacity_check():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root/'release_commit.txt').write_text('other', encoding='ascii')
        try:
            os.chdir(root)
            with patch.object(runner, 'wait_for_gpu', side_effect=AssertionError('must not query GPU')):
                try:
                    runner.execute(root, 'source-unit', 'ground-unit', 'expected', root/'cache', 3)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Wrong release identity must fail')
        finally:
            os.chdir(previous)

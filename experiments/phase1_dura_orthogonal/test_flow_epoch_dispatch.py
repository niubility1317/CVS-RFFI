import json
import sys

import pytest

from experiments.phase1_dura_orthogonal import controller, gpu_capacity_queue


@pytest.mark.parametrize('enabled', [False, True])
def test_queue_forwards_epoch_time_option(monkeypatch, tmp_path, enabled):
    cache = tmp_path/'cache.pt'
    cache.touch()
    argv = ['queue', '--root', str(tmp_path), '--source', 'source', '--ground', 'ground',
            '--commit', 'test', '--cache', str(cache), '--gpu', '0', '--cuda-graph-flow',
            '--objective', 'wide_conditional_flow', '--architecture', 'conditional_iq_flow']
    if enabled:
        argv.append('--epoch-flow-times')
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(gpu_capacity_queue.subprocess, 'check_output',
                        lambda cmd, **kwargs: '0, GPU-A, 5000\n' if '--query-gpu=index,uuid,memory.free' in cmd else '')
    commands = []
    class Child:
        pid = 123
        def wait(self): return 0
    monkeypatch.setattr(gpu_capacity_queue.subprocess, 'Popen', lambda cmd, **kwargs: (commands.append(cmd) or Child()))
    gpu_capacity_queue.main()
    assert ('--epoch-flow-times' in commands[0]) == enabled


@pytest.mark.parametrize('enabled', [False, True])
def test_controller_forwards_epoch_time_option_to_smoke_and_train(monkeypatch, tmp_path, enabled):
    argv = ['controller', '--root', str(tmp_path), '--source', 'source', '--ground', 'ground',
            '--commit', 'test', '--device', 'cuda:0', '--deterministic', '--cuda-graph-flow',
            '--objective', 'wide_conditional_flow', '--architecture', 'conditional_iq_flow']
    if enabled:
        argv.append('--epoch-flow-times')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    monkeypatch.setattr(sys, 'argv', argv)
    commands = []
    def fake_run(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            output = tmp_path/'runs/source-training-smoke'
            output.mkdir(parents=True)
            (output/'smoke.json').write_text(json.dumps({'status': 'SOURCE_TRAINING_SMOKE_PASS', 'F_weights_read': 0}), encoding='utf8')
            return type('Result', (), {'returncode': 0})()
        raise RuntimeError('test stops before training')
    monkeypatch.setattr(controller.subprocess, 'run', fake_run)
    with pytest.raises(RuntimeError, match='test stops'):
        controller.main()
    assert len(commands) == 2
    assert all(('--epoch-flow-times' in command) == enabled for command in commands)

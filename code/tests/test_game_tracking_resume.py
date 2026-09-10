"""End-to-end deterministic CORE90 resume and audit-only noninterference.

Use the actual restored backbone/objective/optimizer/prototype/EMA path with
synthetic legal source roles. Only the checkpoint boundary is interrupted;
the declared two-epoch training contract never changes between invocations.
"""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cvsrffi.game_tracking import runtime
from cvsrffi.game_tracking.config import parse_args


def _args(output, *, audit=True, resume=None):
    argv = ['--output_dir', str(output), '--game_synthetic', '--device', 'cpu',
            '--epochs', '2', '--batch_size', '18', '--num_workers', '0',
            '--game_max_steps_per_epoch', '1', '--game_probe_steps', '2',
            '--game_audit_interval', '1', '--game_audit_samples_per_capture', '2',
            '--game_skip_final_eval']
    if not audit:
        argv.append('--game_no_audit')
    if resume is not None:
        argv.extend(['--game_resume', str(resume)])
    args = parse_args(argv)
    assert args.use_ema_teacher and not args.amp
    assert args.game_control == 'off' and args.game_curriculum == 'fixed'
    return args


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _equal(left, right, path='state'):
    """Exact recursive comparison, including NumPy RNG arrays and NaN floats."""
    if torch.is_tensor(left):
        assert torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            _equal(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _equal(a, b, f'{path}[{index}]')
    elif isinstance(left, float) and np.isnan(left):
        assert isinstance(right, float) and np.isnan(right), path
    else:
        assert left == right, path


def _training_states_equal(left, right):
    for key in ('model', 'ema', 'optimizer', 'prototype', 'scaler', 'solver', 'satellite_generator'):
        _equal(left[key], right[key], key)
    for key in ('python', 'numpy', 'cpu', 'cuda'):
        _equal(getattr(left['rng'], key), getattr(right['rng'], key), f'rng.{key}')
    assert left['epoch'] == right['epoch'] == 2
    assert left['step'] == right['step'] == 2
    assert left['encoder_version'] == right['encoder_version'] == 2
    assert left['ema'] is not None
    assert left['optimizer']['state']
    assert torch.count_nonzero(left['prototype']['class_count']) > 0


@pytest.fixture(scope='module')
def continuous_audited_run(tmp_path_factory):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        output = tmp_path_factory.mktemp('core90-continuous-audited')
        assert runtime.train(_args(output)) == 0
        payload = _load(output / 'final_ssdg.pth')
        assert payload['auditor']['calls'] == 2
        yield output, payload
    finally:
        torch.set_num_threads(old_threads)


def test_epoch_checkpoint_resume_matches_uninterrupted_actual_training(continuous_audited_run, tmp_path, monkeypatch):
    _, continuous = continuous_audited_run
    interrupted_dir = tmp_path / 'interrupted'
    checkpoint = runtime.checkpoint
    signature = inspect.signature(checkpoint)

    class StopAfterEpochOne(RuntimeError):
        pass

    def stop_after_persisting(*args, **kwargs):
        checkpoint(*args, **kwargs)
        call = signature.bind(*args, **kwargs).arguments
        if call['epoch'] == 1 and Path(call['path']).name == 'latest_ssdg.pth':
            raise StopAfterEpochOne('test interruption after durable E1 checkpoint')

    with monkeypatch.context() as patch:
        patch.setattr(runtime, 'checkpoint', stop_after_persisting)
        with pytest.raises(StopAfterEpochOne, match='durable E1'):
            runtime.train(_args(interrupted_dir))
    saved_path = interrupted_dir / 'latest_ssdg.pth'
    partial = _load(saved_path)
    assert partial['epoch'] == partial['step'] == 1
    assert partial['args']['epochs'] == 2
    assert partial['auditor']['calls'] == 1
    assert not (interrupted_dir / 'final_ssdg.pth').exists()
    assert runtime.train(_args(interrupted_dir, resume=saved_path)) == 0
    resumed = _load(interrupted_dir / 'final_ssdg.pth')
    _training_states_equal(continuous, resumed)
    assert continuous['next_audit'] == resumed['next_audit'] == 2
    _equal(continuous['auditor'], resumed['auditor'], 'auditor')
    _equal(continuous['source_info'], resumed['source_info'], 'source_info')
    rows = [json.loads(line) for line in (interrupted_dir / 'game_actions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [row['epoch'] for row in rows] == [1, 2]
    assert all(row['accepted'] for row in rows)


def test_audit_only_matches_no_audit_actual_training_trajectory(continuous_audited_run, tmp_path):
    audited_dir, audited = continuous_audited_run
    no_audit_dir = tmp_path / 'no-audit'
    assert runtime.train(_args(no_audit_dir, audit=False)) == 0
    no_audit = _load(no_audit_dir / 'final_ssdg.pth')
    _training_states_equal(audited, no_audit)
    assert no_audit['auditor']['calls'] == 0
    assert not (no_audit_dir / 'game_audit.jsonl').exists()
    audited_rows = [json.loads(line) for line in (audited_dir / 'game_actions.jsonl').read_text(encoding='utf-8').splitlines()]
    plain_rows = [json.loads(line) for line in (no_audit_dir / 'game_actions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(audited_rows) == len(plain_rows) == 2
    for monitored, baseline in zip(audited_rows, plain_rows):
        for key in ('loss', 'grad_norm', 'accepted', 'field_evaluations', 'satellite_scenario', 'sample_count'):
            _equal(monitored[key], baseline[key], f'trajectory.step{monitored["step"]}.{key}')

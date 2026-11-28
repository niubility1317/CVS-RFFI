import pytest
import importlib
import sys
import numpy as np
import torch

from experiments.pamf_action_restore.operators import Action
from experiments.pamf_lora_main.dura_real_gain_fusion import PhysicsGain, RealSpectralGain
from experiments.pamf_lora_main.w24_native_spectral_innovation import SpectralInnovation
from experiments.pamf_lora_main.dura_real_gain_screen import paired
from experiments.pamf_lora_main.dura_real_gain_fit import activate_wisig_archive
from experiments.pamf_lora_main import dura_real_gain_run as run
from experiments.cric_dura_comparison.io import read, write


@pytest.mark.parametrize('width', [32, 1024])
def test_zero_gain_preserves_prior_and_source_loss_trains_gain(width):
    torch.manual_seed(13)
    y = torch.randn(2, 8 * width, dtype=torch.complex64)
    p = y * 0.7
    h = torch.ones(2, 1, dtype=torch.complex64)
    action = Action(h, torch.ones_like(y), torch.zeros_like(y))
    gain = PhysicsGain(width)
    result = gain(y, p, action)
    torch.testing.assert_close(result, p, rtol=0, atol=0)
    result.abs().square().mean().backward()
    gradient = gain.net[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0


def test_output_depends_only_on_the_corresponding_packet():
    torch.manual_seed(21)
    y = torch.randn(2, 256, dtype=torch.complex64)
    prior = torch.randn_like(y)
    gain = PhysicsGain(32)
    torch.nn.init.normal_(gain.net[-1].weight, std=0.02)
    full = Action(torch.ones(2, 1, dtype=y.dtype), torch.ones_like(y), torch.zeros_like(y))
    one = Action(torch.ones(1, 1, dtype=y.dtype), torch.ones_like(y[:1]), torch.zeros_like(y[:1]))
    torch.testing.assert_close(gain(y, prior, full)[:1], gain(y[:1], prior[:1], one), rtol=1e-5, atol=1e-6)


def test_wisig_real_gain_starts_at_prior_and_can_train():
    torch.manual_seed(31)
    y = torch.randn(3, 8, 32, dtype=torch.complex64)
    prior = .7 * y
    gain = RealSpectralGain(32)
    update = gain(y, prior)
    assert torch.count_nonzero(update) == 0
    (prior + update).abs().square().mean().backward()
    assert gain.net[-1].weight.grad.norm() > 0


def test_adaptable_real_gain_matches_original_lora_formula():
    torch.manual_seed(41)
    original = SpectralInnovation(phase_aware=True)
    replacement = RealSpectralGain(1024)
    torch.nn.init.normal_(original.gain[-1].weight, std=0.02)
    replacement.net.load_state_dict(original.gain.state_dict())
    y = torch.randn(2, 8, 1024, dtype=torch.complex64)
    prior = torch.randn_like(y)
    torch.testing.assert_close(replacement(y, prior), original(y, prior), rtol=1e-6, atol=1e-6)


def test_screen_requires_gain_in_both_predefined_halves():
    truth = np.zeros(8, dtype=np.int64)
    baseline = np.ones(8, dtype=np.int64)
    permutation = np.arange(8)
    imbalanced = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    balanced = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    assert not paired(imbalanced, baseline, truth, permutation, 1)['passes']
    assert paired(balanced, baseline, truth, permutation, 1)['passes']


def test_failed_fit_does_not_abandon_other_dataset(monkeypatch, tmp_path):
    root = tmp_path / 'run'
    runtime = tmp_path / 'runtime.json'
    write(runtime, {'run_root': str(root)})
    monkeypatch.setattr(run, 'ROWS', ('lora', 'wisig', 'wisig_real_gain'))
    monkeypatch.setattr(run, 'free_gpus', lambda: [0, 1, 2])
    monkeypatch.setattr(run, 'sleep', lambda _: None)
    started = []

    class Child:
        pid = 42

        def __init__(self, args, **kwargs):
            action = args[-3] if 'dura_real_gain_screen' in args[3] else 'fit'
            row = args[-1] if action != 'fit' else args[-2]
            self.code = 1 if action == 'fit' and row == 'lora' else 0
            started.append((action, row))
            if action == 'score':
                write(root / row / 'screen-score.json', {'promote': False})

        def poll(self):
            return self.code

        def wait(self):
            return self.code

    monkeypatch.setattr(run.subprocess, 'Popen', Child)
    run.dispatch(runtime)
    state = read(root / 'state.json')
    assert state['status'] == 'PARTIAL'
    assert state['screens'] == {'wisig': False}
    assert 'lora' in state['failures'] and 'lora_screen' in state['failures']
    assert ('predict', 'wisig') in started and ('score', 'wisig') in started
    assert ('predict', 'lora') not in started


def test_archived_wisig_dependencies_take_precedence(monkeypatch, tmp_path):
    import experiments
    import experiments.pamf_lora_main as package
    import experiments.cric_dura_comparison as common

    archived = tmp_path / 'experiments' / 'pamf_lora_main'
    archived.mkdir(parents=True)
    (archived / 'refine_runner.py').write_text('def select_gpu(): return 17\n', encoding='utf-8')
    current = tmp_path / 'current'
    current.mkdir()
    (current / 'refine_runner.py').write_text('unrelated = True\n', encoding='utf-8')
    monkeypatch.setattr(package, '__path__', [str(current)])
    monkeypatch.setattr(common, '__path__', list(common.__path__))
    monkeypatch.setattr(experiments, '__path__', list(experiments.__path__))
    monkeypatch.delitem(sys.modules, 'experiments.pamf_lora_main.refine_runner', raising=False)
    try:
        activate_wisig_archive(tmp_path)
        loaded = importlib.import_module('experiments.pamf_lora_main.refine_runner')
        assert loaded.select_gpu() == 17
    finally:
        sys.modules.pop('experiments.pamf_lora_main.refine_runner', None)


def test_frozen_lora_is_screened_without_retraining(monkeypatch, tmp_path):
    root = tmp_path / 'run'
    runtime = tmp_path / 'runtime.json'
    write(runtime, {'run_root': str(root), 'lora_frozen_root': str(tmp_path / 'old')})
    monkeypatch.setattr(run, 'ROWS', ('wisig', 'wisig_real_gain'))
    monkeypatch.setattr(run, 'free_gpus', lambda: [0, 1, 2])
    monkeypatch.setattr(run, 'sleep', lambda _: None)
    started = []

    class Child:
        pid = 43

        def __init__(self, args, **kwargs):
            action = args[-3] if 'dura_real_gain_screen' in args[3] else 'fit'
            row = args[-1] if action != 'fit' else args[-2]
            started.append((action, row))
            if action == 'score':
                write(root / row / 'screen-score.json', {'promote': False})

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(run.subprocess, 'Popen', Child)
    run.dispatch(runtime)
    assert ('fit', 'lora') not in started
    assert ('predict', 'lora') in started and ('score', 'lora') in started
    assert read(root / 'state.json')['status'] == 'SOURCE_SCREEN_SCORED'

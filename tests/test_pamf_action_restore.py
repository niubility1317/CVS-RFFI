import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.pamf_action_restore.operators import Action, soft_map, real_dot
from experiments.pamf_action_restore.models import ActionRestorer, ActionEstimator
from experiments.pamf_action_restore.data import teacher, clean_teacher
from experiments.pamf_action_restore.runtime import validate_payload
from experiments.pamf_action_restore import design as d
from experiments.pamf_action_restore.evaluation import prediction_arrays, probe_error
from experiments.pamf_operator_restore.operators import filter_matrix, real_operator, real_pair, complex_pair
from leo_practical.channel import Config, receiver_for_session
from leo_practical.residual import ResidualChannel

torch.set_num_threads(2)


def complex_random(*shape):
    return torch.complex(torch.randn(*shape, dtype=torch.float64), torch.randn(*shape, dtype=torch.float64))


def test_fft_causal_reflect_action_and_real_adjoint():
    torch.manual_seed(1)
    h, a, b = complex_random(2, 7), complex_random(2, 19), complex_random(2, 19)
    x, v = complex_random(2, 19), complex_random(2, 19)
    action = Action(h, a, b)
    for i in range(2):
        matrix = filter_matrix(h[i].numpy(), 19)
        expected = a[i].numpy()*(matrix@x[i].numpy())+b[i].numpy()*(matrix@x[i].numpy()).conj()
        np.testing.assert_allclose(action(x)[i].numpy(), expected, atol=1e-12)
    torch.testing.assert_close(real_dot(action(x), v), real_dot(x, action.adjoint(v)), atol=1e-11, rtol=1e-11)


def test_soft_map_matches_dense_real_system_and_remains_soft():
    torch.manual_seed(2)
    h, a, b = complex_random(1, 4), complex_random(1, 11), .1*complex_random(1, 11)
    action = Action(h, a, b)
    observed, prior = complex_random(1, 11), complex_random(1, 11)
    noise = torch.tensor([.7], dtype=torch.float64)
    matrix = filter_matrix(h[0].numpy(), 11)
    real = real_operator(a[0].numpy()[:, None]*matrix, b[0].numpy()[:, None]*matrix.conj())
    expected = np.linalg.solve(real.T@real+.7/.3*np.eye(22), real.T@real_pair(observed[0].numpy())+.7/.3*real_pair(prior[0].numpy()))
    restored = soft_map(action, observed, prior, noise, .3, steps=40)
    np.testing.assert_allclose(restored[0].numpy(), complex_pair(expected), atol=1e-8)
    assert not torch.allclose(action(restored), observed)
    before = (action(prior)-observed).abs().square().sum()+.7/.3*(prior-prior).abs().square().sum()
    six = soft_map(action, observed, prior, noise, .3, steps=6)
    after = (action(six)-observed).abs().square().sum()+.7/.3*(six-prior).abs().square().sum()
    assert after < before
    with pytest.raises(ValueError):
        soft_map(action, observed, prior, torch.tensor([0.]), .3)


@pytest.mark.parametrize('scene', ['practical_high', 'practical_low_suburban', 'practical_mid_urban', 'practical_low_urban'])
def test_compact_source_teacher_replays_native_action_without_engine_change(scene):
    cfg = Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route='residual', mode='post_sync')
    rx = receiver_for_session(cfg, 2026101911, 'rx1:day1:eq1')
    clean = np.random.default_rng(9).normal(size=(2, 256)).astype(np.float32)/2**.5
    arrays, meta = teacher(clean, cfg, 99, rx)
    h = torch.complex(torch.tensor(arrays['kernel'][0]), torch.tensor(arrays['kernel'][1]))[None]
    c = torch.tensor(arrays['coefficients'])
    action = Action(h, torch.complex(c[0], c[1])[None], torch.complex(c[2], c[3])[None])
    signal = action(torch.complex(torch.tensor(clean[0]), torch.tensor(clean[1]))[None])[0].numpy()
    observed, details, trace = ResidualChannel(cfg, 99, rx).process(clean[0]+1j*clean[1], return_traces=True)
    np.testing.assert_array_equal(arrays['received'][0]+1j*arrays['received'][1], observed)
    np.testing.assert_allclose(signal, trace['pre_agc_signal']*10**(details['agc_gain_db']/20), atol=1e-6)
    assert meta['replay_error'] < 1e-9 and 1e-8 <= arrays['variance'] <= 1


def test_teacher_rejects_unsupported_route_and_clean_teacher_is_identity():
    clean = np.zeros((2, 256), np.float32)
    clean[0, 12] = 1
    arrays, _ = clean_teacher(clean)
    assert arrays['kernel'][0, 0] == 1 and arrays['coefficients'][0].min() == 1
    cfg = Config(fs_hz=25e6, scenario='practical_high', processing_route='full')
    with pytest.raises(ValueError):
        teacher(clean, cfg, 1, receiver_for_session(cfg, 2, 'source'))


def test_network_probe_training_has_no_F_TX_or_oracle_forward_inputs_and_frozen_parent():
    model = ActionRestorer(nn.Conv1d(2, 2, 1))
    model.train()
    assert not model.coarse.training and not any(p.requires_grad for p in model.coarse.parameters())
    received = torch.randn(3, 2, 256)
    _, action, variance = model.estimate(received)
    target = Action(action.kernel.detach().roll(1, -1), action.main.detach(), action.image.detach())
    loss = probe_error(action, target, [torch.randn(3, 256, dtype=torch.complex64) for _ in range(2)]).mean()+variance.log().square().mean()
    loss.backward()
    assert all(p.grad is None for p in model.coarse.parameters())
    assert model.estimator.global_head.weight.grad.abs().sum() > 0
    assert model.estimator.core.out.weight.grad.abs().sum() > 0
    assert list(inspect.signature(model.forward).parameters) == ['received']
    assert list(inspect.signature(ActionEstimator.forward).parameters) == ['self', 'received', 'coarse']
    assert torch.isfinite(model(received)).all()


def test_checkpoint_negative_permissions_lineage_and_budget():
    from experiments.pamf_residual_restore import design as p
    payload = dict(schema='learned_action_checkpoint_v1', config=d.configuration(),
        training_TX_labels_read=0, training_F_access=False, target_training=False, epoch=20, updates=4940,
        parent=dict(run=p.PARENT_RUN, commit=p.PARENT_COMMIT, candidate=p.COARSE_NAME,
            selector='source_waveform_A', training_TX_labels_read=0, target_training=False))
    validate_payload(payload)
    for key, value in [('training_TX_labels_read', 1), ('training_F_access', True), ('target_training', True), ('updates', 4941)]:
        bad = copy.deepcopy(payload); bad[key] = value
        with pytest.raises(ValueError):
            validate_payload(bad)
    bad = copy.deepcopy(payload); bad['parent']['selector'] = 'target_best'
    with pytest.raises(ValueError):
        validate_payload(bad)


def test_prediction_completion_and_per_view_coverage_precede_source_label_open(tmp_path):
    with pytest.raises(FileNotFoundError):
        prediction_arrays(tmp_path, np.array(['a', 'b']))
    (tmp_path/'complete.json').write_text(json.dumps(dict(status='SOURCE_PREDICTIONS_COMPLETE', views=['clean'],
        readouts=['identity', 'coarse', 'learned_operator'], count=2, TX_labels_read=0, target_reads=0,
        state_updates=0, head_fit_updates=0)), encoding='utf-8')
    with pytest.raises(ValueError):
        prediction_arrays(tmp_path, np.array(['a', 'b']))


def test_subprocess_entry_and_fixed_source_design():
    result = subprocess.run([sys.executable, '-m', 'experiments.pamf_action_restore.runner', '--help'], cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=''), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--target' not in result.stdout
    config = d.configuration()
    assert config['updates'] == 4940 and config['L_exposures'] == 126000 and config['U_exposures'] == 1134000
    assert config['training_TX_labels_read'] == config['target_reads'] == 0 and not config['training_F_access']
    assert config['capacity_lock'] == 'receiver_residual_capacity4.lock'

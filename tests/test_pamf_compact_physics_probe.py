from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_compact_physics_probe.analysis import action, diagnostics, validate_estimate, summarize
from experiments.pamf_compact_physics_probe import runner
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_restoration.runner import array_tensor, tensor_array
from experiments.pamf_restoration.models import complex_iq, pair


def fixture(count=3):
    rng = np.random.default_rng(40)
    clean = rng.normal(size=(count, 2, 256)).astype(np.float32)
    kernel = np.zeros((count, 2, 64), np.float32)
    kernel[:, 0, 0] = 1
    coefficients = np.zeros((count, 4, 256), np.float32)
    coefficients[:, 0] = 1
    estimate = dict(coarse=clean.copy(), kernel=kernel, coefficients=coefficients, variance=np.full(count, 1e-8))
    oracle = {k: v.copy() for k, v in estimate.items() if k != 'coarse'}
    probes = rng.normal(size=(4, count, 256))+1j*rng.normal(size=(4, count, 256))
    return estimate, oracle, clean, probes


def test_numpy_causal_reflect_composite_action_matches_torch():
    estimate, _, clean, _ = fixture()
    rng = np.random.default_rng(2026102040)
    estimate['kernel'] = rng.normal(size=(3, 2, 64)).astype(np.float32)
    estimate['coefficients'] = rng.normal(size=(3, 4, 256)).astype(np.float32)
    coefficients = array_tensor(estimate['coefficients'], 'cpu')
    operator = Action(complex_iq(array_tensor(estimate['kernel'], 'cpu')),
        torch.complex(coefficients[:, 0], coefficients[:, 1]),
        torch.complex(coefficients[:, 2], coefficients[:, 3]))
    expected_pair = tensor_array(pair(operator(complex_iq(array_tensor(clean, 'cpu')))))
    expected = expected_pair[:, 0]+1j*expected_pair[:, 1]
    assert np.allclose(action(estimate, clean[:, 0]+1j*clean[:, 1]), expected, rtol=2e-5, atol=2e-5)


def test_exact_composite_recovery_has_zero_error_and_clean_noise_floor_is_explicit():
    estimate, oracle, clean, probes = fixture()
    values = diagnostics(estimate, oracle, clean, clean, probes)
    for key in ('coarse_clean_nmse', 'action_clean_nmse', 'action_probe_nmse', 'log_variance_error',
                'oracle_action_residual_over_native_variance'):
        # FFT roundoff / the clean synthetic 1e-8 variance floor is ~1e-23.
        assert np.allclose(values[key], 0, atol=1e-20)
    assert np.array_equal(values['native_real_variance'], np.full(3, 1e-8))


def test_kernel_gain_gauge_does_not_create_false_operator_error():
    estimate, oracle, clean, probes = fixture()
    estimate['kernel'] *= 2
    estimate['coefficients'] *= .5
    values = diagnostics(estimate, oracle, clean, clean, probes)
    assert np.allclose(values['action_probe_nmse'], 0, atol=1e-25)


def test_noise_variance_has_real_axis_units_not_complex_energy_or_scaled_prior_units():
    estimate, oracle, clean, probes = fixture()
    estimate['variance'] = np.full(3, .25)
    oracle['variance'] = np.full(3, .25)
    noisy = clean.copy()
    noisy[:, 0] += .5
    noisy[:, 1] += .5
    values = diagnostics(estimate, oracle, clean, noisy, probes)
    assert np.allclose(values['estimated_action_residual_real_mse'], .25, atol=1e-7)
    assert np.allclose(values['oracle_action_residual_over_native_variance'], 1., atol=1e-6)


@pytest.mark.parametrize('key', ['kernel', 'coefficients', 'coarse', 'variance'])
def test_missing_context_component_is_rejected(key):
    estimate, _, _, _ = fixture()
    estimate.pop(key)
    with pytest.raises(ValueError):
        validate_estimate(estimate, 3)


@pytest.mark.parametrize('bad', [0., -1., float('nan')])
def test_invalid_variance_never_reaches_oracle_statistics(bad):
    estimate, _, _, _ = fixture()
    estimate['variance'][0] = bad
    with pytest.raises(ValueError):
        validate_estimate(estimate, 3)


def test_unknown_candidate_rejected_before_checkpoint_open(monkeypatch):
    monkeypatch.setattr(runner, 'load_original', lambda *args: pytest.fail('Checkpoint opened'))
    with pytest.raises(ValueError):
        runner.model_for(SimpleNamespace(), 'target_best')


def test_incomplete_predictions_stop_before_clean_oracle_open(monkeypatch, tmp_path):
    args = SimpleNamespace(output=tmp_path)
    monkeypatch.setattr(runner, 'inputs', lambda _: (tmp_path, np.arange(1800), np.ones(1800), np.ones(1800)))
    def missing(*_):
        raise ValueError('Missing frozen context')
    monkeypatch.setattr(runner, 'load_estimates', missing)
    monkeypatch.setattr(runner.np, 'load', lambda *a, **k: pytest.fail('Oracle opened before estimates complete'))
    with pytest.raises(ValueError, match='Missing frozen context'):
        runner.score(args)


def test_source_group_statistics_preserve_all_rows_and_scenes():
    estimate, oracle, clean, probes = fixture()
    values = diagnostics(estimate, oracle, clean, clean, probes)
    result = summarize(values, 'dense_half', 'practical_low_urban', np.array([1, 3, 3]), np.array([1, 2, 3]))
    assert len(result) == 1+2+3
    assert result[0]['count'] == 3
    assert all(r['candidate'] == 'dense_half' and r['view'] == 'practical_low_urban' for r in result)

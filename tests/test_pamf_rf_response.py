from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_rf_response import design as d, evaluation as e
from experiments.pamf_rf_response.models import directions, oracle_response, response_metrics
from experiments.pamf_rf_response.runtime import validate_args
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_restoration.models import complex_iq, pair


def sample():
    return torch.randn((4, 2, 256), generator=torch.Generator().manual_seed(42))


def identity(count):
    ones = torch.ones((count, 256), dtype=torch.complex64)
    return Action(torch.ones((count, 1), dtype=torch.complex64), ones, torch.zeros_like(ones))


def test_directions_are_batch_independent_orthogonal_and_match_energy():
    clean = sample()
    x = complex_iq(clean)
    whole = directions(clean, [6, 10, 3, 15], .01)
    for name, (delta, valid) in whole.items():
        assert valid.all()
        torch.testing.assert_close((delta.conj()*x).sum(-1), torch.zeros(4, dtype=torch.complex64), atol=3e-6, rtol=0)
        torch.testing.assert_close(delta.abs().square().sum(-1)/x.abs().square().sum(-1), torch.full((4,), .0001))
        reversed_part = directions(clean[[3, 1]], [15, 10], .01)[name][0]
        torch.testing.assert_close(reversed_part, delta[[3, 1]], atol=0, rtol=0)
    torch.testing.assert_close(whole['pa_phase'][0], 1j*whole['pa_amplitude'][0])


def test_zero_clean_is_explicitly_invalid_not_division_by_zero():
    for delta, valid in directions(torch.zeros(2, 2, 256), [0, 1], .01).values():
        assert not valid.any()
        assert torch.count_nonzero(delta) == 0


@pytest.mark.parametrize('ripple', [0., 1e-6])
def test_constant_and_nearly_constant_complex_iq_do_not_create_fake_directions(ripple):
    time = torch.arange(256, dtype=torch.float32)
    real = 1.1+ripple*torch.cos(time*.15)
    imaginary = .3+ripple*torch.sin(time*.15)
    clean = torch.stack((real, imaginary))[None].repeat(2, 1, 1)
    values = directions(clean, [0, 1], .01)
    for name in d.DIRECTIONS[:-1]:
        delta, valid = values[name]
        assert not valid.any(), name
        assert torch.count_nonzero(delta) == 0, name
    assert values['random_control'][1].all()


@pytest.mark.parametrize('amplitude', [0, -.01, .011, float('nan')])
def test_unregistered_scale_rejected(amplitude):
    with pytest.raises(ValueError):
        directions(sample(), [0, 1, 2, 3], amplitude)


def test_unknown_panel_index_rejected():
    with pytest.raises(ValueError):
        directions(sample(), [-1, 1, 2, 3], .01)


def test_oracle_matches_independent_diagonal_ridge_inverse():
    clean = sample()
    delta = complex_iq(clean)*.01
    factor = torch.linspace(.3, 1.2, 256)[None].expand(4, -1)
    action = Action(torch.ones((4, 1), dtype=torch.complex64), factor.to(torch.complex64), torch.zeros_like(delta))
    variance = torch.tensor([1e-6, 1e-5, 1e-4, 1e-3])
    actual, residual = oracle_response(action, action(delta), variance)
    expected = delta*factor.square()/(factor.square()+variance[:, None]/SCALE**2)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    assert (residual < 1e-4).all()


def test_central_difference_gain_curvature_and_numpy_recompute():
    clean = sample()
    delta, valid = directions(clean, [0, 1, 2, 3], .01)['pa_amplitude']
    base = torch.zeros_like(clean)
    curvature = pair(delta)*.1
    plus, minus = pair(delta)*.6+curvature, -pair(delta)*.6+curvature
    variance = torch.full((4,), 1e-6)
    residual = torch.zeros(4)
    metrics = response_metrics(plus, minus, base, delta, delta, delta, variance, valid, residual)
    torch.testing.assert_close(metrics['gain'], torch.full((4,), .6))
    torch.testing.assert_close(metrics['response_nmse'], torch.full((4,), .16))
    torch.testing.assert_close(metrics['curvature_nmse'], torch.full((4,), .01))
    assert metrics['observable'].all()
    arrays = dict(plus=plus.numpy(), minus=minus.numpy(), oracle=pair(delta).numpy(),
                  observation_delta=pair(delta).numpy(), CG_residual=residual.numpy())
    independent = e.numpy_metrics(arrays, base.numpy(), pair(delta).numpy(), valid.numpy(), variance.numpy())
    for key in e.METRICS:
        np.testing.assert_allclose(independent[key], metrics[key].numpy(), rtol=1e-5, atol=1e-6)


def test_unobservable_and_unconverged_directions_remain_reported():
    clean = sample()
    delta, valid = directions(clean, [0, 1, 2, 3], .01)['iq_image']
    values = response_metrics(pair(delta), -pair(delta), torch.zeros_like(clean), delta, delta,
        delta, torch.full((4,), 100.), valid, torch.ones(4))
    assert not values['observable'].any()
    record = e.aggregate({k: v.numpy() for k, v in values.items()}, np.ones(4, bool))
    assert record['count'] == record['valid_count'] == 4
    assert record['observable_count'] == 0 and record['observable_gain_median'] is None


def test_nonfinite_input_rejected():
    clean = sample()
    clean[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        directions(clean, [0, 1, 2, 3], .01)


def test_parent_and_run_checks_before_any_io():
    args = SimpleNamespace(output=Path(d.RUN), parent_run=Path(d.PARENT),
                           parent_action=Path(d.ACTION), reference_run=Path(d.REFERENCE))
    validate_args(args)
    args.parent_run = Path('target_ranked_or_unknown')
    with pytest.raises(ValueError):
        validate_args(args)


@pytest.mark.parametrize('field,value', [('cells', 69), ('F_access', 1), ('target_reads', 1),
                                       ('TX_truth_reads', 1), ('source_V_state_updates', 1)])
def test_incomplete_or_illegal_completion_rejected(field, value):
    record = dict(status='SOURCE_COUNTERFACTUALS_COMPLETE', count=1800, views=list(e.VIEWS),
        rows=list(d.ROWS), cells=70, parent=d.PARENT, parent_commit=d.PARENT_COMMIT,
        candidate=d.CANDIDATE, TX_truth_reads=0, F_access=0, target_reads=0,
        optimizer_updates=0, source_V_state_updates=0, oracle_not_deployable=True)
    record[field] = value
    with pytest.raises(ValueError):
        e.complete_contract(record, 1800)


def test_source_truth_stays_closed_until_every_counterfactual_complete(monkeypatch, tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(e, 'source_inputs', lambda args: (tmp_path, np.arange(1800), None, None))
    monkeypatch.setattr(e, 'load_predictions', lambda *args: (_ for _ in ()).throw(ValueError('incomplete')))
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('Truth or arrays opened too early'))
    with pytest.raises(ValueError, match='incomplete'):
        e.score(SimpleNamespace(output=tmp_path))


def test_priority_is_not_target_promotion_or_TX_authenticity():
    records = []
    for direction in d.DIRECTIONS:
        records.append(dict(view='practical_low_urban', row=e.row_name(direction, .01), dimension='overall', value='all',
            observable_count=101, observable_response_nmse_median=.5, observable_gain_median=.5,
            observable_curvature_nmse_median=.01))
    value = e.interpret(records)
    assert value['supports_RF_response_training'] and not value['target_promotion']
    assert value['training_effect_unproven'] and value['real_TX_fidelity_unproven'] and not value['goal_complete']


def test_missing_primary_source_group_rejected():
    with pytest.raises(ValueError):
        e.interpret([])

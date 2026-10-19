import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from experiments.pamf_uncertainty_restore.models import weighted_map, variance_nll, UncertaintyRestorer
from experiments.pamf_uncertainty_restore.runtime import validate_payload, validate_calibration, calibration_indices
from experiments.pamf_uncertainty_restore.controller import source_qualification
from experiments.pamf_uncertainty_restore.evaluation import reference, prediction_arrays, VIEWS
from experiments.pamf_uncertainty_restore import design as d
from experiments.pamf_action_restore.operators import Action, soft_map
from experiments.pamf_operator_restore.operators import filter_matrix, real_operator, real_pair, complex_pair
from experiments.pamf_restoration.models import complex_iq

torch.set_num_threads(2)


def random_complex(*shape):
    return torch.complex(torch.randn(*shape, dtype=torch.float64), torch.randn(*shape, dtype=torch.float64))


def test_weighted_real_MAP_matches_dense_heteroscedastic_system():
    torch.manual_seed(3)
    h, a, b = random_complex(1, 4), random_complex(1, 10), .1*random_complex(1, 10)
    action = Action(h, a, b)
    observed, prior = random_complex(1, 10), random_complex(1, 10)
    variance = torch.linspace(.1, .8, 10, dtype=torch.float64)[None]
    matrix = filter_matrix(h[0].numpy(), 10)
    real = real_operator(a[0].numpy()[:, None]*matrix, b[0].numpy()[:, None]*matrix.conj())
    weight = np.diag(1/np.tile(variance[0].numpy(), 2))
    expected = np.linalg.solve(real.T@weight@real+np.eye(20)/.3,
        real.T@weight@real_pair(observed[0].numpy())+real_pair(prior[0].numpy())/.3)
    actual = weighted_map(action, observed, prior, variance, .3, 40)
    np.testing.assert_allclose(actual[0].numpy(), complex_pair(expected), atol=1e-9)


def test_scalar_and_column_variance_agree_with_existing_soft_update():
    torch.manual_seed(4)
    action = Action(random_complex(2, 3), random_complex(2, 13), .2*random_complex(2, 13))
    y, prior = random_complex(2, 13), random_complex(2, 13)
    variance = torch.tensor([.3, .6], dtype=torch.float64)
    expected = soft_map(action, y, prior, variance, .2, 6)
    torch.testing.assert_close(weighted_map(action, y, prior, variance, .2, 6), expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(weighted_map(action, y, prior, variance[:, None], .2, 6), expected, atol=1e-10, rtol=1e-10)


def test_six_step_soft_objective_decreases_and_gradients_stay_finite():
    torch.manual_seed(5)
    action = Action(random_complex(2, 4), random_complex(2, 12), .1*random_complex(2, 12))
    y = random_complex(2, 12); prior = random_complex(2, 12).requires_grad_()
    variance = torch.rand(2, 12, dtype=torch.float64)+.2
    result = weighted_map(action, y, prior, variance, .1, 6)
    before = ((action(prior)-y).abs().square()/variance).sum()
    after = ((action(result)-y).abs().square()/variance).sum()+((result-prior).abs().square()/.1).sum()
    assert after < before
    result.abs().square().mean().backward()
    assert torch.isfinite(prior.grad).all()
    with pytest.raises(ValueError):
        weighted_map(action, y, prior, torch.zeros(2, 12))
    with pytest.raises(ValueError):
        weighted_map(action, y, prior, torch.ones(2, 11))


class TinyPhysics(nn.Module):
    def __init__(self):
        super().__init__(); self.coarse = nn.Conv1d(2, 2, 1, bias=False)
        with torch.no_grad():
            self.coarse.weight.copy_(.9*torch.eye(2)[:, :, None])

    def estimate(self, value):
        base = self.coarse(value); z = complex_iq(value)
        h = torch.zeros(len(value), 4, dtype=z.dtype); h[:, 0] = 1
        return base, Action(h, torch.ones_like(z), torch.zeros_like(z)), torch.full((len(value),), .001)


def test_reconstruction_and_proper_variance_gradients_are_separate():
    model = UncertaintyRestorer('uncertain_map_prior', TinyPhysics(), .01).train()
    r = torch.randn(2, 2, 256); x = r+.3
    output, variance, action, _ = model.training_outputs(r)
    (output-x).square().mean().backward()
    assert all(p.grad is None for p in model.confidence.parameters())
    assert model.prior.out.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.frozen.parameters()) and not model.frozen.training
    model.zero_grad(set_to_none=True)
    _, variance, action, _ = model.training_outputs(r)
    variance_nll(action, r, x, variance).mean().backward()
    assert all(p.grad is None for p in model.prior.parameters())
    assert model.confidence.network.out.weight.grad.abs().sum() > 0
    assert list(inspect.signature(model.forward).parameters) == ['received']


def test_direct_control_has_no_action_forward_and_one_IQ_output():
    coarse = nn.Conv1d(2, 2, 1)
    model = UncertaintyRestorer('direct_control', coarse).train()
    r = torch.randn(2, 2, 256); output, variance, action, base = model.training_outputs(r)
    assert output.shape == r.shape and variance is None and action is None
    torch.testing.assert_close(output, base)
    assert not coarse.training and not any(p.requires_grad for p in coarse.parameters())
    physics = UncertaintyRestorer('global_floor', TinyPhysics(), .01)
    assert not any(p.requires_grad for p in physics.parameters()) and torch.isfinite(physics(r)).all()


def calibration():
    return dict(status='SOURCE_EXCESS_FROZEN', count=8192, seed=d.CALIBRATION_SEED,
        source_V_reads=0, TX_labels_read=0, F_outputs_read=0, target_reads=0, optimizer_updates=0,
        fitted_scalars=1, excess_variance=.01, L_exposures=820, U_exposures=7372)


def test_calibration_uses_training_IQ_only_and_checkpoint_permissions_are_enforced():
    value = calibration(); validate_calibration(value)
    assert len(set(calibration_indices(63000))) == 8192
    with pytest.raises(ValueError):
        validate_calibration(dict(value, source_V_reads=1))
    with pytest.raises(ValueError):
        validate_calibration(dict(value, U_exposures=7371))
    payload = dict(schema='uncertainty_checkpoint_v1', config=d.configuration(), kind='uncertain_map_prior',
        training_TX_labels_read=0, training_F_access=False, target_training=False, calibration=value,
        epoch=20, updates=4940, parent=dict(coarse_run=d.COARSE_RUN, coarse_commit=d.COARSE_COMMIT,
        action_run=d.ACTION_RUN, action_commit=d.ACTION_COMMIT, data_contract='original7/63/30', training_TX_F_reads=0, target_training=False))
    validate_payload(payload)
    for key in ('training_F_access', 'target_training'):
        with pytest.raises(ValueError):
            validate_payload(dict(payload, **{key: True}))
    bad = copy.deepcopy(payload); bad['parent']['action_commit'] = 'unknown'
    with pytest.raises(ValueError):
        validate_payload(bad)


def source_summary(mean, accuracy, clean=98.5, reference=False):
    name = 'r03_reference' if reference else 'restored'
    views = {scene: {name: dict(waveform=dict(iq_nmse=mean))} for scene in VIEWS}
    views['clean'][name]['waveform']['iq_nmse'] = .001
    records = [dict(view='practical_low_urban', readout=name, dimension='overall', value='all', accuracy=accuracy),
               dict(view='clean', readout=name, dimension='overall', value='all', accuracy=clean)]
    records += [dict(view='practical_low_urban', readout=name, dimension='TX', value=str(i), accuracy=accuracy) for i in range(6)]
    return dict(status='ANALYZED', role='source', count=1800, all_predictions_before_truth=True, views=views, records=records)


def test_fixed_source_waveform_and_identity_thresholds_require_all_TX_groups():
    direct, uncertain, old = source_summary(.03, 80), source_summary(.028, 81), source_summary(.029, 80, reference=True)
    decision = source_qualification(direct, uncertain, old)
    assert decision['qualified'] and decision['target_candidates'] == ['direct_control', 'uncertain_map_prior']
    bad = copy.deepcopy(uncertain); bad['records'][-1]['accuracy'] = 78.9
    assert not source_qualification(direct, bad, old)['qualified']
    assert not source_qualification(direct, source_summary(.031, 82), old)['qualified']
    with pytest.raises(ValueError):
        source_qualification(direct, dict(uncertain, role='target'), old)


def test_unqualified_source_decision_blocks_target_before_IQ_is_opened(tmp_path):
    (tmp_path/'source_decision.json').write_text(json.dumps(dict(status='SOURCE_DECISION_FROZEN', qualified=False, target_candidates=[])), encoding='utf-8')
    args = SimpleNamespace(output=tmp_path, kind='uncertain_map_prior', formal_reference=tmp_path/'forbidden')
    with pytest.raises(ValueError):
        reference(args, 'target')


def test_incomplete_or_wrong_role_predictions_cannot_reach_truth(tmp_path):
    ids = np.array(['a', 'b'])
    with pytest.raises(FileNotFoundError):
        prediction_arrays(tmp_path, ids, 'direct_control', 'source')
    (tmp_path/'complete.json').write_text(json.dumps(dict(status='PREDICTIONS_COMPLETE', kind='direct_control',
        role='target', count=2, views=list(VIEWS), readouts=['restored'], TX_truth_reads=0, state_updates=0,
        head_fit_updates=0, F_run='wrong', F_seed=2026092701)), encoding='utf-8')
    with pytest.raises(ValueError):
        prediction_arrays(tmp_path, ids, 'direct_control', 'source')


def test_module_entry_and_fixed_incremental_budget():
    result = subprocess.run([sys.executable, '-m', 'experiments.pamf_uncertainty_restore.runner', '--help'], cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=''), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    config = d.configuration()
    assert config['training_TX_labels_read'] == config['V_persistent_updates'] == 0 and not config['training_F_access']
    assert config['epochs'] == 20 and config['updates'] == 4940 and config['U_exposures'] == 1134000
    assert config['max_simultaneous_GPU_workers'] == 3 and config['capacity_lock'] == 'receiver_residual_capacity4.lock'


def test_prior_initialization_does_not_depend_on_frozen_parent_rng_consumption():
    from experiments.pamf_uncertainty_restore.training import initialize_model
    from experiments.pamf_reception_restore.models import ComplexTFUNet
    from experiments.pamf_action_restore.models import ActionRestorer
    priors = []
    for kind in d.NAMES:
        torch.manual_seed(123)
        frozen = ComplexTFUNet() if kind == 'direct_control' else ActionRestorer(ComplexTFUNet())
        before = next(frozen.parameters()).detach().clone()
        # Frozen parent construction/loading can consume different random draws.
        torch.rand(7 if kind == 'direct_control' else 123)
        model = initialize_model(kind, frozen, .01 if kind == 'uncertain_map_prior' else None)
        assert model.frozen is frozen and torch.equal(before, next(frozen.parameters()))
        priors.append({key: value.clone() for key, value in model.prior.state_dict().items()})
    assert all(torch.equal(priors[0][key], prior[key]) for prior in priors[1:] for key in priors[0])

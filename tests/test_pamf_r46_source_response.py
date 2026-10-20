from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch

from experiments.pamf_r46_source_response import design as d
from experiments.pamf_r46_source_response.metrics import (
    perturbations, response_fields, response_metrics, step_disagreement, power)
from experiments.pamf_r46_source_response import runtime


def iq():
    return torch.linspace(-1, 1, 3 * 2 * 256, dtype=torch.float64).reshape(3, 2, 256)


def test_fixed_source_only_matrix_and_exposures():
    c = d.configuration()
    assert len(c['arms']) == 4 and c['reference_arm'] == d.ARMS[0]
    assert c['diagnostic_IQ_inputs_per_model'] == 6912
    assert c['smoke_IQ_inputs_per_model'] == 2
    assert c['expected_metric_rows'] == 1224
    assert c['new_training_updates'] == c['new_network_versions'] == c['F_weight_reads'] == 0
    assert c['TX_label_reads'] == c['V_reads'] == c['target_reads'] == 0


@pytest.mark.parametrize('probe', d.PROBES)
@pytest.mark.parametrize('epsilon', d.EPSILONS)
def test_equal_input_power_and_common_directions(probe, epsilon):
    x = iq()
    a, b = perturbations(x, 25), perturbations(x, 25)
    assert torch.equal(a[probe][epsilon], b[probe][epsilon])
    assert torch.allclose(power(a[probe][epsilon]), epsilon**2 * power(x), atol=1e-16)
    assert torch.allclose(a[probe][d.EPSILONS[1]], 2 * a[probe][d.EPSILONS[0]])
    if probe == 'phase':
        assert torch.allclose((a[probe][epsilon] * x).sum((1, 2)), torch.zeros(3, dtype=x.dtype), atol=1e-14)


@pytest.mark.parametrize('scale', [0., .5, 1., 2.])
def test_linear_response_has_closed_form_error_and_gain(scale):
    x = iq()
    delta = perturbations(x, 25)['random'][.01]
    t_odd, t_even = response_fields(x + delta, x - delta, x)
    s_odd, s_even = response_fields(scale*(x + delta), scale*(x - delta), scale*x)
    measured = response_metrics(s_odd, s_even, t_odd, t_even, delta, x)
    assert torch.allclose(measured['relative_response_nmse'], torch.full((3,), (scale-1)**2, dtype=x.dtype), atol=1e-11)
    assert torch.allclose(measured['student_response_gain'], torch.full((3,), scale**2, dtype=x.dtype), atol=1e-11)
    assert torch.allclose(measured['teacher_response_gain'], torch.ones(3, dtype=x.dtype), atol=1e-11)
    assert measured['student_even_to_odd'].max() < 1e-20


def test_finite_epsilon_measurement_does_not_claim_exact_derivative():
    x = iq()
    delta = perturbations(x, 25)['phase']
    odd = {e:response_fields((x+v)**3, (x-v)**3, x**3)[0] for e,v in delta.items()}
    mismatch = step_disagreement(odd[.005], odd[.01], .005, .01, x)
    assert torch.all(mismatch > 0)
    linear = {e:response_fields(x+v, x-v, x)[0] for e,v in delta.items()}
    assert step_disagreement(linear[.005], linear[.01], .005, .01, x).max() < 1e-20


def test_constant_reference_and_zero_input_are_finite_and_flag_floor():
    x = torch.zeros(2, 2, 256)
    delta = perturbations(x, 25)['random'][.01]
    zero = torch.zeros_like(x)
    measured = response_metrics(zero, zero, zero, zero, delta, x)
    assert all(torch.isfinite(v).all() for v in measured.values())
    assert torch.equal(measured['teacher_floor_active'], torch.ones(2, dtype=torch.float64))
    assert measured['relative_response_nmse'].sum() == 0


@pytest.mark.parametrize('bad', [torch.zeros(2, 256), torch.zeros(0, 2, 256),
                               torch.zeros(2, 2, 255), torch.zeros(2, 2, 256, dtype=torch.int32),
                               torch.full((2, 2, 256), float('nan'))])
def test_bad_iq_rejected(bad):
    with pytest.raises(ValueError):
        perturbations(bad, 25)


@pytest.mark.parametrize('epsilons', [(0., .01), (-.005, .01), (.01, .005), (.005, float('nan'))])
def test_invalid_steps_rejected(epsilons):
    with pytest.raises(ValueError):
        perturbations(iq(), 25, epsilons=epsilons)


def test_unknown_model_rejected_before_checkpoint_io(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, 'read_json', lambda p:calls.append(p))
    with pytest.raises(ValueError):
        runtime.source_root(SimpleNamespace(), 'target_selected_model')
    assert not calls


def test_source_training_flags_must_be_complete_and_legal():
    value = dict(status='SOURCE_TRAINED', epochs=20, updates=19700, parameters=278180,
        L_exposures=126000, U_exposures=1134000, training_F_access=False,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0)
    runtime.validate_training(value)
    for field in ('TX_labels_read', 'F_weights_read', 'F_outputs_read', 'V_reads', 'target_reads'):
        with pytest.raises(ValueError):
            runtime.validate_training(dict(value, **{field:1}))
        missing = dict(value)
        missing.pop(field)
        with pytest.raises(ValueError):
            runtime.validate_training(missing)


def test_foreign_output_is_untouched_before_data_io(tmp_path, monkeypatch):
    from experiments.pamf_r46_source_response import runner
    output = tmp_path/d.RUN
    output.mkdir()
    marker = output/'existing.txt'
    marker.write_text('preserve')
    monkeypatch.setattr(runner, 'load_panel', lambda args:pytest.fail('source IQ opened'))
    args = SimpleNamespace(output=output, _owns_output=False)
    with pytest.raises(ValueError):
        runner.diagnose(args)
    assert marker.read_text() == 'preserve' and list(output.iterdir()) == [marker]
    assert not args._owns_output


@pytest.mark.parametrize('flag', ['--reader-folder', '--target-truth', '--query-role', '--F-checkpoint'])
def test_forbidden_runtime_inputs_have_no_cli(flag):
    from experiments.pamf_r46_source_response.runner import parser
    with pytest.raises(SystemExit):
        parser().parse_args([flag, 'forbidden'])


def test_state_change_is_detected():
    model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    snapshot = {k:v.clone() for k,v in model.state_dict().items()}
    runtime.check_frozen(model, snapshot)
    model.weight.add_(1)
    with pytest.raises(ValueError):
        runtime.check_frozen(model, snapshot)


def test_complete_synthetic_matrix_uses_common_probes_and_actual_forward_counts(tmp_path, monkeypatch):
    from experiments.pamf_r46_source_response import runner

    class Scale(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.register_buffer('scale', torch.tensor(scale))

        def forward(self, value):
            return self.scale*value

    generator = torch.Generator().manual_seed(25)
    clean = torch.rand((256,2,256),generator=generator)+.1
    panel = dict(clean=clean, received01=1.2*clean, received20=.8*clean)
    models = {arm:Scale(scale).eval().requires_grad_(False)
              for arm,scale in zip(d.ARMS,(1.,.8,.6,.5))}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(runner,'code_commit',lambda:'method')
    monkeypatch.setattr(runner,'validate_sources',lambda args:None)
    monkeypatch.setattr(runner,'load_model',lambda args,arm:(models[arm],dict(
        epoch=d.SOURCE_BUDGET[arm][0],updates=d.SOURCE_BUDGET[arm][1])))
    monkeypatch.setattr(runner,'load_panel',lambda args:(panel,np.arange(256),
        np.asarray(['L_s']*128+['U_s']*128)))
    args = SimpleNamespace(output=tmp_path/d.RUN,expected_commit='method',_owns_output=False)
    runner.diagnose(args)
    result = json.loads((args.output/'results.json').read_text())
    assert result['status'] == 'SOURCE_RESPONSE_COMPLETE'
    metric_rows = 0
    for arm,scale in zip(d.ARMS,(1.,.8,.6,.5)):
        row = result['rows'][arm]
        assert row['actual_IQ_inputs'] == dict(smoke=2,center=768,perturbed=6144)
        assert row['checkpoint_unchanged'] is True
        for view in d.VIEWS:
            measured = row['views'][view]
            groups = list(measured['center'].values())
            for probe in d.PROBES:
                groups.extend(measured['scaling'][probe].values())
                for step in d.EPSILONS:
                    response = measured['responses'][probe][f'{step:.3f}']
                    assert response['all']['count'] == 256 and response['U_s']['count'] == 128
                    error = response['all']['metrics']['relative_response_nmse']['mean']
                    assert error == pytest.approx((scale-1)**2,abs=2e-5)
                    groups.extend(response.values())
            metric_rows += sum(len(g['metrics']) for g in groups)
    assert metric_rows == d.configuration()['expected_metric_rows'] == 1224
    assert all(result[k] == 0 for k in ('F_weight_reads','F_output_reads','TX_label_reads','V_reads','target_reads'))

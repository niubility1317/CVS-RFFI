import argparse
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.pamf_r46_source_fidelity import design as d
from experiments.pamf_r46_source_fidelity.metrics import fidelity, lag_correlations, summarize
from experiments.pamf_r46_source_fidelity.runner import load_panel, parser, select_positions, grouped


@pytest.fixture
def iq():
    return torch.randn(4,2,256, generator=torch.Generator().manual_seed(23))


def test_identity_all_metrics_zero(iq):
    assert all(torch.equal(v, torch.zeros_like(v)) for v in fidelity(iq, iq).values())


def test_zero_input_is_finite():
    values = fidelity(torch.zeros(2,2,256), torch.zeros(2,2,256))
    assert all(torch.isfinite(v).all() for v in values.values())
    assert torch.equal(values['iq_nmse'], torch.zeros(2))


def test_complex_phase_is_visible_but_power_unchanged(iq):
    rotated = torch.stack((-iq[:,1], iq[:,0]),1)
    values = fidelity(rotated, iq)
    torch.testing.assert_close(values['iq_nmse'], torch.full((4,),2.))
    assert all(torch.all(values[f'fft_power_Q{i}_complex_nmse'] > 1.99) for i in range(1,5))
    torch.testing.assert_close(lag_correlations(rotated), lag_correlations(iq))


def test_gain_error_and_relative_iq(iq):
    values = fidelity(2*iq, iq)
    torch.testing.assert_close(values['iq_nmse'], torch.ones(4))
    for i in range(1,5):
        torch.testing.assert_close(values[f'amplitude_Q{i}_complex_gain_error'], torch.ones(4))


def test_tail_damage_stays_in_tail(iq):
    damaged = iq.clone();damaged[:,:,232:] = 0
    values = fidelity(damaged, iq)
    assert not values['prefix232_nmse'].any()
    torch.testing.assert_close(values['tail24_nmse'], torch.ones(4))


@pytest.mark.parametrize('shape', [(2,256),(2,3,256),(2,2,255)])
def test_wrong_iq_shape_rejected(shape, iq):
    with pytest.raises(ValueError):
        fidelity(torch.zeros(shape), iq)


def test_differentiable_analytic_metrics(iq):
    candidate = (.8*iq).requires_grad_()
    loss = sum(v.mean() for v in fidelity(candidate, iq).values())
    loss.backward()
    assert torch.isfinite(candidate.grad).all() and candidate.grad.abs().sum() > 0


def test_nonfinite_rejected(iq):
    iq[0,0,0] = float('nan')
    with pytest.raises(FloatingPointError):
        fidelity(iq, torch.zeros_like(iq))


def test_positions_uniform_and_role_balanced():
    roles = np.array(['U_s','L_s']*9)
    got = select_positions(roles,4)
    assert got.tolist() == [1,5,11,17,0,4,10,16]
    assert len(set(got.tolist())) == 8


@pytest.mark.parametrize('roles', [np.array(['V','L_s','U_s']),np.array(['L_s']*3),np.array(['target','U_s'])])
def test_forbidden_roles_rejected(roles):
    with pytest.raises(ValueError):
        select_positions(roles,1)


def test_too_few_distinct_positions():
    with pytest.raises(ValueError):
        select_positions(np.array(['L_s','U_s']),2)


def test_wrong_cache_name_rejected_before_io(monkeypatch):
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Forbidden IQ IO'))
    with pytest.raises(ValueError,match='before IQ IO'):
        load_panel(argparse.Namespace(parent_action=Path('target'),source_cache=Path('source_cache')))


@pytest.mark.parametrize('flag', ['--reader-folder','--target-truth','--formal-reference','--TX-labels'])
def test_forbidden_cli_handles_absent(flag):
    flags = {f for action in parser()._actions for f in action.option_strings}
    assert flag not in flags


def test_group_partitions_fixed():
    values = dict(iq_nmse=torch.arange(256,dtype=torch.float32))
    got = grouped(values)
    assert got['L_s']['iq_nmse']['mean'] == 63.5
    assert got['U_s']['iq_nmse']['mean'] == 191.5


def test_summary_exact():
    got = summarize(dict(error=torch.tensor([0.,1.,2.])))['error']
    assert got['mean'] == got['median'] == 1 and got['maximum'] == 2


def test_no_training_or_model_selection_in_plan():
    plan = d.configuration()
    assert plan['new_network_versions'] == plan['new_training_updates'] == plan['GPU_workers'] == 0
    assert all(plan[k] == 0 for k in ('F_weight_reads','F_output_reads','TX_label_reads','V_reads','target_reads'))
    assert not plan['model_selection'] and not plan['teacher_selection'] and not plan['target_feedback']
    assert d.ARMS == d.frozen.ARMS and len(d.ARMS) == 3


@pytest.mark.parametrize('name', ['other-owner', d.RUN])
def test_rejected_existing_output_has_no_failure_write(tmp_path, monkeypatch, name):
    from experiments.pamf_r46_source_fidelity import runner
    folder = tmp_path/name;folder.mkdir()
    marker = folder/'preserve.txt';marker.write_text('existing output')
    args = argparse.Namespace(output=folder)
    monkeypatch.setattr(runner, 'parser', lambda:argparse.Namespace(parse_args=lambda:args))
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda n:None)
    monkeypatch.setattr(runner, 'write_json', lambda *a,**k:pytest.fail('Rejected output must stay read-only'))
    with pytest.raises(ValueError, match='immutable diagnostic output'):
        runner.main()
    assert list(folder.iterdir()) == [marker] and marker.read_text() == 'existing output'

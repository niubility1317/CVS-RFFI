from copy import deepcopy
import pytest
import torch
from torch import nn

from cvsrffi.game_tracking.audit_evidence import make_evidence_v2, require_evidence_v2
from cvsrffi.game_tracking.source_audit import (ProbeConfigV2, fit_empirical_lag,
    cross_tx_readout, _fit_v2, _weighted_inputs, isolated_rng)


def run(head, x, y, **kw):
    return fit_empirical_lag(head, x, y, sample_weights=torch.ones(len(y)),
        objective_scope='current_training_head_objective', **kw)


def test_near_optimal_zero_improvement_is_reliable_low():
    head = nn.Linear(2, 2).eval()
    with torch.no_grad():
        head.weight.zero_(); head.bias.zero_()
    x = torch.zeros(8, 2, requires_grad=True)
    y = torch.arange(2).repeat(4)
    out = run(head, x, y, config=ProbeConfigV2(steps=4, min_steps=4))
    assert out.metrics['status'] == 'RELIABLE_LOW_GAP'
    assert out.metrics['gap_raw'] == 0
    assert out.metrics['quality_pass'] and x.grad is None
    assert out.metrics['monitor_before'] is None
    assert len(out.metrics['trajectory']) == 5


def test_underfit_head_recovers_and_preserves_state():
    head = nn.Linear(2, 2).eval()
    with torch.no_grad():
        head.weight.zero_(); head.bias.zero_()
    initial = deepcopy(head.state_dict()); rng = torch.get_rng_state().clone()
    y = torch.arange(2).repeat(8); x = torch.eye(2)[y]*3
    out = run(head, x, y, config=ProbeConfigV2(steps=40, min_steps=4, lr=.05))
    assert out.metrics['status'] == 'RELIABLE_HIGH_GAP'
    assert out.metrics['gap_normalized'] > .2
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, head.state_dict()[k]) for k,v in initial.items())
    assert 'gradient_norm' in out.metrics['trajectory'][0]


def test_incomplete_budget_and_unstable_are_unknown_not_zero():
    y = torch.arange(2).repeat(4); x = torch.eye(2)[y]
    short = run(nn.Linear(2,2).eval(), x, y, config=ProbeConfigV2(steps=0))
    assert short.metrics['status'] == 'BUDGET_INCONCLUSIVE'
    assert short.metrics['gap_normalized'] is None
    broken = run(nn.Linear(2,2).eval(), x*float('nan'), y)
    assert broken.metrics['status'] == 'OPTIMIZATION_FAILURE'
    assert broken.metrics['gap_raw'] is None


def test_train_mode_requires_fixed_random_state_and_surrogate_never_controls():
    head = nn.Sequential(nn.Dropout(.5), nn.Linear(2,2)).train()
    y = torch.arange(2).repeat(4); x = torch.eye(2)[y]
    missing = run(head, x, y)
    assert not missing.metrics['control_ready']
    fixed = {'cpu_rng': torch.get_rng_state().clone()}
    a = run(head, x, y, fixed_head_state=fixed, config=ProbeConfigV2(steps=5,min_steps=4))
    b = run(head, x, y, fixed_head_state=fixed, config=ProbeConfigV2(steps=5,min_steps=4))
    assert a.metrics['trajectory'] == b.metrics['trajectory']
    surrogate = fit_empirical_lag(head, x,y,sample_weights=torch.ones(8),
        objective_scope='eval_clean_surrogate',config=ProbeConfigV2(steps=5,min_steps=4))
    assert not surrogate.metrics['control_ready']


def test_schema_explicitly_rejects_v1():
    with pytest.raises(ValueError, match='v2'):
        require_evidence_v2({'valid':True,'G_lag':0})
    e = make_evidence_v2(observation_id='obs',step=1,encoder_version=1,
                         data_valid=True,coverage_valid=True)
    assert require_evidence_v2(e) is e
    assert e['lag']['gap_normalized'] is None


def test_cross_tx_two_folds_and_monitor_never_selects_endpoint():
    tx = torch.arange(6).repeat_interleave(8)
    y = torch.arange(2).repeat(24)
    x = torch.eye(2)[y]*2
    groups = [f'{int(t)}-{int(c)}' for t,c in zip(tx,y)]
    out = cross_tx_readout(x,y,y,tx,groups,config=ProbeConfigV2(steps=4,min_steps=4),seed=9)
    assert len(out['folds']) == 2
    a,b = out['folds']
    assert a['fit_tx'] == b['monitor_tx']
    for fold in out['folds']:
        assert not set(fold['fit_groups']) & set(fold['monitor_groups'])
        for task in ('domain','rx'):
            for kind in ('linear','mlp'):
                p = fold['readouts'][task][kind]
                assert p['selection'] == 'fixed_endpoint'
                assert p['budget']['actual_steps'] == 4
                assert len(p['trajectory']) == 5


def test_large_lr_finite_divergence_is_failure_and_weighted_target_exact():
    head = nn.Linear(2,2).eval()
    with torch.no_grad():
        head.weight.zero_(); head.bias.zero_()
    x = torch.zeros(4,2); y = torch.tensor([0,0,0,1])
    bad = run(head,x,y,config=ProbeConfigV2(steps=4,min_steps=4,lr=100))
    assert bad.metrics['status'] == 'OPTIMIZATION_FAILURE'
    assert bad.metrics['stop_reason'] == 'fit_instability'
    assert bad.metrics['gap_normalized'] is None
    head.bias.data.copy_(torch.tensor([1.,-1.]))
    weights = torch.tensor([1.,2.,3.,4.])
    out = fit_empirical_lag(head,x,y,sample_weights=weights,objective_scope='current_training_head_objective',
        config=ProbeConfigV2(steps=0))
    expected = (torch.nn.functional.cross_entropy(head(x),y,reduction='none')*weights/weights.sum()).sum()
    assert out.metrics['fit_before'] == pytest.approx(float(expected.detach()))
    with pytest.raises(ValueError,match='nonnegative'):
        fit_empirical_lag(head,x,y,sample_weights=-weights,objective_scope='current_training_head_objective')


def test_monitor_changes_cannot_select_or_backprop_and_transfer_is_separate():
    y = torch.arange(2).repeat(16); x = torch.eye(2)[y]*3
    head = nn.Linear(2,2).eval()
    with torch.no_grad():
        head.weight.zero_(); head.bias.zero_()
    config = ProbeConfigV2(steps=20,min_steps=4,lr=.05)
    bad_monitor = (-x).requires_grad_()
    with isolated_rng(0):
        a = _fit_v2(head,x,y,None,config,'cross_tx_readout',monitor=_weighted_inputs(x,y,None))
        b = _fit_v2(head,x,y,None,config,'cross_tx_readout',monitor=_weighted_inputs(bad_monitor,y,None))
    assert bad_monitor.grad is None
    assert a.metrics['fit_after'] == b.metrics['fit_after']
    assert all(torch.equal(v,b.recovered_head.state_dict()[k]) for k,v in a.recovered_head.state_dict().items())
    tx = torch.arange(6).repeat_interleave(32)
    yy = y.repeat(6)
    xx = x.repeat(6,1)
    first = cross_tx_readout(xx,yy,yy,tx,[f'{int(t)}-{int(c)}' for t,c in zip(tx,yy)],config=config,seed=3)
    held = set(first['folds'][0]['monitor_tx'])
    xx[torch.tensor([int(t) in held for t in tx])] *= -1
    transfer = cross_tx_readout(xx,yy,yy,tx,[f'{int(t)}-{int(c)}' for t,c in zip(tx,yy)],config=config,seed=3)
    p = transfer['folds'][0]['readouts']['domain']['linear']
    assert p['fit_after'] < p['fit_before']
    assert p['monitor_after'] > p['monitor_before']
    assert p['status'] == 'TRANSFER_FAILURE' and not p['quality_pass']
    assert run(head,x,y,config=config).metrics['status'] == 'RELIABLE_HIGH_GAP'


def test_training_bn_dropout_owned_copy_preserves_buffers_and_mixed_modes():
    head = nn.Sequential(nn.BatchNorm1d(2),nn.Dropout(.25),nn.Linear(2,2)).train()
    head[1].eval()
    before = deepcopy(head.state_dict()); rng = torch.get_rng_state().clone()
    y = torch.arange(2).repeat(8); x = torch.eye(2)[y]
    result = run(head,x,y,config=ProbeConfigV2(steps=5,min_steps=4),
        fixed_head_state={'cpu_rng':rng,'module_training':{n:m.training for n,m in head.named_modules()}})
    assert result.metrics['module_training']['1'] is False
    assert all(torch.equal(v,head.state_dict()[k]) for k,v in before.items())
    assert torch.equal(rng,torch.get_rng_state())


def test_objective_scale_zero_inactive_and_scaled_gradient_reported():
    head=nn.Linear(2,2).eval()
    with torch.no_grad():head.weight.zero_();head.bias.zero_()
    y=torch.arange(2).repeat(4);x=torch.eye(2)[y]
    off=run(head,x,y,config=ProbeConfigV2(steps=4,min_steps=4,objective_scale=0.))
    assert off.metrics['status']=='UNAVAILABLE' and not off.metrics['control_ready']
    assert off.metrics['gap_normalized'] is None and off.metrics['budget']['actual_steps']==0
    full=run(head,x,y,config=ProbeConfigV2(steps=4,min_steps=4,objective_scale=1.))
    half=run(head,x,y,config=ProbeConfigV2(steps=4,min_steps=4,objective_scale=.5))
    f,h=full.metrics['trajectory'][0],half.metrics['trajectory'][0]
    assert h['fit_ce']==f['fit_ce']
    assert h['fit_objective']==pytest.approx(.5*f['fit_ce'])
    assert h['gradient_norm']==pytest.approx(.5*f['gradient_norm'])

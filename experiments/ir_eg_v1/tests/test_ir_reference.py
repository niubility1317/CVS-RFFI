from copy import deepcopy
import pytest
import torch
from cvsrffi.game_tracking.solvers import GameSolver
from cvsrffi.game_tracking.state import RNGState
from cvsrffi.xuc_fusion.response_config import resolve,schedule
from ir_fixtures import make_case,run,assert_nested


def test_kappa_zero_direct_EG(monkeypatch):
    a=make_case(kappa=0.);b=make_case(kappa=0.)
    b.solver=GameSolver(b.model,b.opt,mode='extragradient',max_grad_norm=.7)
    import cvsrffi.xuc_fusion.ir_solver as module
    monkeypatch.setattr(module,'capture_head_calls',lambda *a:(_ for _ in ()).throw(AssertionError('new path ran')))
    rng=RNGState.capture();ra=run(a);after=RNGState.capture();rng.restore()
    rb=b.solver.step(b.closure);b.commit()
    assert_nested(a.model.state_dict(),b.model.state_dict());assert_nested(a.opt.state_dict(),b.opt.state_dict())
    assert torch.equal(after.cpu,torch.get_rng_state());assert ra.loss_replay==rb.loss_replay
    assert_nested(a.external.state_dict(),b.external.state_dict())


def test_reference_one_formal_commit():
    a=make_case('reference');result=run(a)
    assert result.accepted and a.external.commits==1 and a.solver.steps==1
    assert all(s['step']==1 for s in a.opt.state.values())
    assert result.field_evaluations==3 and result.telemetry['full_backward_calls']==2


def test_phir_not_directly_committed():
    a=make_case('reference');run(a);trace=a.solver.last_trace
    phir=torch.cat([p.flatten() for p in trace['phip']])+.25*trace['delta']
    assert not torch.equal(phir,torch.cat([p.detach().flatten() for p in a.model.adv_head.parameters()]))


def test_E21_start_without_TR_ramp():
    c=resolve(dict(method='IR_EG'))
    assert not schedule(c,20,0)['active']
    assert schedule(c,21,0)==dict(active=True,strength=.25,base='extragradient')


@pytest.mark.parametrize('all_head',[False,True])
def test_frozen_head_has_none_gradient_and_no_response(all_head):
    a=make_case();frozen=list(a.model.adv_head.parameters()) if all_head else [a.model.adv_head[0].bias]
    for p in frozen:p.requires_grad_(False)
    before=[p.detach().clone() for p in frozen]
    run(a)
    assert_nested(before,[p.detach() for p in frozen])
    assert all(p.grad is None and p not in a.opt.state for p in frozen)


@pytest.mark.parametrize('key,value',[('ir_kappa',True),('ir_cg_max_iterations',True),('ir_cg_max_iterations',3),('br_refresh_interval',0),('ir_origin_scales','legacy_reestimate')])
def test_strict_config(key,value):
    with pytest.raises(ValueError):resolve(dict(method='IR_EG',**{key:value}))

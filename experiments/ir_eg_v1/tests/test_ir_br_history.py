from copy import deepcopy
import pytest
import torch
from cvsrffi.game_tracking.state import RNGState
from cvsrffi.xuc_fusion.br_history import BRHistory,RefreshRequired
from ir_fixtures import make_case,run,assert_nested


def test_cache_is_previous_accepted_final_raw_and_age_one_not_eight():
    a=make_case(method='BR_IR_EG')
    for index in range(1,5):
        a.ctx.pending=None;result=run(a)
        expected=a.solver._layout_order(a.solver.last_trace['corrector'])
        assert_nested(a.solver.history.gradients,expected)
        assert a.solver.history.accepted_index==index
        if index>1:assert result.telemetry['history_age']==1 and not result.telemetry['br_refreshed']
        assert result.telemetry['full_backward_calls']==(2 if index==1 else 1)


def test_refresh_every_eight_and_boundaries_force_refresh():
    a=make_case(method='BR_IR_EG');refresh=[]
    for index in range(1,11):
        a.ctx.pending=None;r=run(a)
        if r.telemetry['br_refreshed']:refresh.append(index)
    assert refresh==[1,9]
    a.ctx.epoch=41;a.ctx.pending=None
    assert run(a).telemetry['br_refresh_reason']=='stage_boundary'


def test_none_activity_requires_refresh():
    a=make_case(method='BR_IR_EG');run(a);h=a.solver.history
    signature=a.solver._signature(a.ctx)
    assert h.refresh_reason(signature,2,None)=='activity_unavailable'
    activity=[g is not None for i,g in enumerate(h.gradients) if i in h.layout.psi_indices]
    activity[0]=not activity[0]
    assert h.refresh_reason(signature,2,activity)=='activity_changed'
    with pytest.raises(RefreshRequired):h.predictor(None,a.solver.last_trace['h0'],signature,2)


@pytest.mark.parametrize('kappa',[0.,.25])
def test_R1_equals_IR(kappa):
    a=make_case(method='BR_IR_EG',refresh=1,kappa=kappa);b=make_case(kappa=kappa)
    for _ in range(3):
        a.ctx.pending=None;b.ctx.pending=None
        rng=RNGState.capture();run(a);rng.restore();run(b)
        assert_nested(a.model.state_dict(),b.model.state_dict(),1e-6,1e-5)
        assert_nested(a.opt.state_dict(),b.opt.state_dict(),1e-6,1e-5)


def test_kappa0_equals_BR_EG_not_full_EG():
    a=make_case(method='BR_IR_EG',kappa=0.);b=make_case(kappa=0.)
    for _ in range(3):
        a.ctx.pending=None;b.ctx.pending=None
        rng=RNGState.capture();r=run(a);rng.restore();run(b)
    assert r.telemetry['full_backward_calls']==1 and r.telemetry['cg_iterations']==0
    assert any(not torch.equal(p,q) for p,q in zip(a.model.parameters(),b.model.parameters()))

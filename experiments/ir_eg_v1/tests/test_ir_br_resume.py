from copy import deepcopy
import pytest
import torch
from cvsrffi.game_tracking.state import RNGState
from cvsrffi.xuc_fusion.resume import read_checkpoint
from ir_fixtures import make_case,run,assert_nested


def test_resume_matches_continuous(tmp_path):
    a=make_case(method='BR_IR_EG');run(a);a.ctx.pending=None;run(a)
    b=make_case(method='BR_IR_EG')
    path=tmp_path/'latest_ssdg.pth'
    torch.save(dict(model=a.model.state_dict(),optimizer=a.opt.state_dict(),solver=a.solver.state_dict(),external=a.external.state_dict()),path)
    # Use the production CPU deserializer: AdamW noncapturable step scalars
    # must stay on CPU; load_state_dict moves only moments to parameter devices.
    saved=read_checkpoint(path)
    b.model.load_state_dict(saved['model']);b.opt.load_state_dict(saved['optimizer'])
    b.solver.load_state_dict(saved['solver']);b.external.load_state_dict(saved['external'])
    for parameter,state in b.opt.state.items():
        assert state['step'].device.type=='cpu'
        assert state['exp_avg'].device==state['exp_avg_sq'].device==parameter.device
    vars(b.ctx).update(deepcopy(vars(a.ctx)))
    for _ in range(3):
        a.ctx.pending=None;b.ctx.pending=None;rng=RNGState.capture()
        run(a);rng.restore();run(b)
        assert_nested(a.model.state_dict(),b.model.state_dict())
        assert_nested(a.opt.state_dict(),b.opt.state_dict())
        assert_nested(a.solver.state_dict(),b.solver.state_dict())


def test_rejected_step_does_not_update_cache():
    a=make_case(method='BR_IR_EG');run(a);before=deepcopy(a.solver.history.state_dict())
    def fail(stage):
        if stage=='after_formal':raise RuntimeError('reject')
    a.solver.failure_injector=fail
    with pytest.raises(RuntimeError):run(a)
    assert_nested(a.solver.history.state_dict(),before)


def test_cache_layout_mismatch_fails():
    a=make_case(method='BR_IR_EG');run(a);state=a.solver.state_dict()
    state['br_history']['layout']='wrong'
    with pytest.raises(ValueError,match='layout'):a.solver.load_state_dict(state)

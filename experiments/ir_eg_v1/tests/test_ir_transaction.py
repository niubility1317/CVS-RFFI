from copy import deepcopy
import pytest
import torch
from cvsrffi.game_tracking.state import RNGState
from ir_fixtures import make_case,run,assert_nested


@pytest.mark.parametrize('stage',['after_origin','after_virtual','after_capture','inside_cg','after_corrector','after_formal','after_external_commit'])
def test_transaction_failure_restores_everything(stage):
    a=make_case();before=(deepcopy(a.model.state_dict()),deepcopy(a.opt.state_dict()),deepcopy(vars(a.ctx)),deepcopy(a.solver.state_dict()),a.external.state_dict())
    rng=RNGState.capture()
    def inject(actual):
        if actual==stage:
            a.model.eval();raise RuntimeError(stage)
    a.solver.failure_injector=inject
    with pytest.raises(RuntimeError,match=stage):run(a)
    for x,y in zip((a.model.state_dict(),a.opt.state_dict(),vars(a.ctx),a.solver.state_dict(),a.external.state_dict()),before):assert_nested(x,y)
    assert a.model.training and torch.equal(rng.cpu,torch.get_rng_state())
    assert not a.model.adv_head._forward_hooks


def test_external_commit_partial_failure_is_atomic():
    a=make_case();old=deepcopy(a.model.state_dict())
    def fail():a.external.commits+=1;a.external.scale=123.;raise RuntimeError('normalizer')
    a.commit=fail
    with pytest.raises(RuntimeError,match='normalizer'):run(a)
    assert a.external.commits==0 and a.external.scale==1. and a.solver.steps==0
    assert_nested(a.model.state_dict(),old)


def test_numerical_failure_same_step_EG(monkeypatch):
    a=make_case();b=make_case(kappa=0.);rng=RNGState.capture()
    import cvsrffi.xuc_fusion.ir_solver as module
    def fail(*a,**kw):raise FloatingPointError('CG injected')
    monkeypatch.setattr(module,'solve_response',fail)
    result=run(a);rng.restore();run(b)
    assert result.telemetry['fallback'] and result.failure_stage=='CG injected'
    assert_nested(a.model.state_dict(),b.model.state_dict());assert_nested(a.opt.state_dict(),b.opt.state_dict())


def test_cached_nonfinite_scalar_with_finite_derivatives_aborts(monkeypatch):
    from cvsrffi.xuc_fusion.ir_objective import ObjectivePacket
    from cvsrffi.game_tracking.solvers import NonFiniteStep
    a=make_case();before=deepcopy(a.model.state_dict());original=ObjectivePacket.assemble;calls=[0]
    def bad(self,leaves):
        result=original(self,leaves);calls[0]+=1
        return result+torch.tensor(float('nan'),device=result.device) if calls[0]==2 else result
    monkeypatch.setattr(ObjectivePacket,'assemble',bad)
    with pytest.raises(NonFiniteStep,match='cached_corrector'):run(a)
    assert a.solver.steps==0 and a.external.commits==0 and not a.opt.state
    assert_nested(a.model.state_dict(),before)

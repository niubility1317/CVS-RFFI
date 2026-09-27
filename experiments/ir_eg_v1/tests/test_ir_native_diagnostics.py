from copy import deepcopy
import math
import torch
from cvsrffi.game_tracking.state import RNGState
from cvsrffi.xuc_fusion.ir_telemetry import native_heavy_diagnostics
from test_ir_native_joint import native_case,step
from ir_fixtures import assert_nested


def test_native_source_diagnostics_no_target_and_no_state_changes():
    case=native_case();case.args.joint['probe_interval']=0
    step(case)
    before=deepcopy(case.model.state_dict());optim=deepcopy(case.opt.state_dict())
    teacher=deepcopy(case.ema.state_dict());dr=deepcopy(case.dr.state_dict())
    rng=RNGState.capture()
    result=native_heavy_diagnostics(case.solver,case.ctx,case.dr)
    assert result['recovery']['steps']==10 and result['source_panel']['role']=='V'
    assert result['controls_training'] is False
    assert result['nonlinear_residual_full']>=0
    assert_nested(before,case.model.state_dict());assert_nested(optim,case.opt.state_dict())
    assert_nested(teacher,case.ema.state_dict())
    after_dr=case.dr.state_dict()
    assert_nested(vars(dr.pop('calibration')),vars(after_dr.pop('calibration')))
    def nan_safe(value):
        if isinstance(value,dict):return {k:nan_safe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [nan_safe(v) for v in value]
        if isinstance(value,float) and math.isnan(value):return 'unchanged_nan'
        return value
    assert_nested(nan_safe(dr),nan_safe(after_dr))
    assert torch.equal(rng.cpu,torch.get_rng_state())
    if rng.cuda:
        assert all(torch.equal(a,b) for a,b in zip(rng.cuda,torch.cuda.get_rng_state_all()))

import torch
from cvsrffi.game_tracking.state import RNGState
from ir_fixtures import make_case,run,assert_nested


def test_cached_reference_actual_state_alignment_ten_steps():
    a=make_case('reference');b=make_case('cached')
    for step in range(10):
        a.ctx.pending=None;b.ctx.pending=None
        rng=RNGState.capture();ra=run(a);after=RNGState.capture();rng.restore();rb=run(b)
        assert abs(ra.loss_replay-rb.loss_replay)<1e-6
        assert_nested(a.solver.last_trace['corrector'],b.solver.last_trace['corrector'],1e-6,1e-5)
        assert_nested(a.model.state_dict(),b.model.state_dict(),1e-6,1e-4)
        assert_nested(a.opt.state_dict(),b.opt.state_dict(),1e-6,1e-4)
        assert_nested(a.external.state_dict(),b.external.state_dict(),1e-6,1e-4)
        assert torch.equal(after.cpu,torch.get_rng_state())
        assert ra.telemetry['model_forward_calls']==6 and rb.telemetry['model_forward_calls']==4
        assert rb.telemetry['full_backward_calls']==2

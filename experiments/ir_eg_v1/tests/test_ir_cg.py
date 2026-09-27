import os
import pytest

@pytest.fixture(autouse=True)
def _acceptance_device():
    device=os.environ.get('IR_ACCEPTANCE_DEVICE','cpu')
    if device not in ('cpu','cuda'):raise ValueError('invalid IR_ACCEPTANCE_DEVICE')
    with torch.device(device):
        yield

import torch
from cvsrffi.xuc_fusion.ir_types import ResponseMetric
from cvsrffi.xuc_fusion.ir_cg import solve_response

def metric(n=5):
    r=torch.ones(n);return ResponseMetric(r,r.sqrt(),torch.arange(n),1.)

def test_CG_two_steps_truncated_is_usable():
    r=solve_response(torch.ones(5),metric(),lambda v:torch.arange(1,6)*v,rtol=1e-12)
    assert r.iterations==2 and r.status=='truncated_usable' and r.relative_residual>1e-12

def test_zero_rhs():
    r=solve_response(torch.zeros(5),metric(),lambda v: (_ for _ in ()).throw(AssertionError()))
    assert r.status=='zero_rhs' and r.iterations==0

def test_zero_h0_nonzero_hp():
    r=solve_response(torch.ones(5),metric(),lambda v: v)
    assert r.status=='early_converged';torch.testing.assert_close(r.solution,-torch.ones(5)/2)

def test_nonfinite_fallback_status():
    r=solve_response(torch.ones(5),metric(),lambda v: v*float('nan'))
    assert r.status=='numerical_failure' and r.reason

def test_G_zero_equals_minus_R_e():
    d=torch.tensor([.2,.4]);r=ResponseMetric(d,d.sqrt(),torch.tensor([0,2]),.5);e=torch.tensor([2.,4.,8.])
    result=solve_response(e,r,lambda v:torch.zeros_like(v))
    delta=torch.zeros_like(e);delta[r.active_indices]=r.sqrt_diagonal*result.solution
    torch.testing.assert_close(delta,torch.tensor([-.4,0.,-3.2]))

def test_nonpositive_operator_fails():
    result=solve_response(torch.ones(5),metric(),lambda v:-2*v)
    assert result.status=='numerical_failure'

import os
import pytest

@pytest.fixture(autouse=True)
def _acceptance_device():
    device=os.environ.get('IR_ACCEPTANCE_DEVICE','cpu')
    if device not in ('cpu','cuda'):raise ValueError('invalid IR_ACCEPTANCE_DEVICE')
    with torch.device(device):
        yield

import copy
import pytest
import torch
from torch import nn
from cvsrffi.xuc_fusion.ir_types import build_layout
from cvsrffi.xuc_fusion.ir_metric import build_response_metric

def setup(history=False,clip=.25,lr=.03,none=False):
    m=nn.Module();m.adv_head=nn.Linear(2,1);opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(.7,.8),eps=1e-5,weight_decay=.2)
    if history:
        for p in m.parameters():p.grad=torch.ones_like(p)*.3
        opt.step()
        # Parameter-local count really differs.
        opt.state[m.adv_head.bias]['step'].add_(2)
    origin=copy.deepcopy(opt.state_dict());h=tuple(None if none and i==0 else torch.full_like(p,.4) for i,p in enumerate(m.parameters()))
    for p,g in zip(m.parameters(),h):p.grad=None if g is None else clip*g
    opt.step();record={'clip_coefficient':clip,'optimizer_after':copy.deepcopy(opt.state_dict())}
    layout=build_layout(m,opt)
    return origin,record,h,layout

def test_first_adam_step_metric():
    args=setup();r=build_response_metric(*args)
    torch.testing.assert_close(r.diagonal,torch.full((3,),.03*.25/(.1+1e-5)))

def test_nonzero_history_and_decay():
    origin,record,h,layout=setup(True);r=build_response_metric(origin,record,h,layout)
    group=origin['param_groups'][0];expected=[]
    for pid,g in zip(group['params'],h):
        state=origin['state'][pid];k=int(state['step'])+1;v=record['optimizer_after']['state'][pid]['exp_avg_sq']/(1-.8**k)
        expected.append((.03*.25*.3/((1-.7**k)*(v.sqrt()+1e-5))).flatten())
    torch.testing.assert_close(r.diagonal,torch.cat(expected))

def test_parameter_local_step_counter():
    r=build_response_metric(*setup(True));assert r.diagonal[0]!=r.diagonal[-1]

def test_clip_coefficient_used_once():
    args=setup(clip=.25);r=build_response_metric(*args);step=1e-3
    # Frozen v and c: independent affine AdamW formula finite difference.
    origin,record,h,layout=args;v=record['optimizer_after']['state'][0]['exp_avg_sq']/(1-.8)
    f=lambda x: -.03*(.3*.25*x)/(.3*(v.sqrt()+1e-5))
    fd=(f(h[0]+step)-f(h[0]-step))/(2*step)
    torch.testing.assert_close(-fd.flatten(),r.diagonal[:2],atol=2e-6,rtol=2e-5)

def test_zero_lr_excluded():assert build_response_metric(*setup(lr=0.)).diagonal.numel()==0

def test_none_gradient_not_zero():
    r=build_response_metric(*setup(none=True));assert r.active_indices.tolist()==[2]

@pytest.mark.parametrize('flag',['amsgrad','maximize','fused','capturable','differentiable'])
def test_unsupported_optimizer_flags(flag):
    origin,record,h,layout=setup();origin['param_groups'][0][flag]=True
    with pytest.raises(ValueError):build_response_metric(origin,record,h,layout)

def test_real_predictor_state_required():
    origin,record,h,layout=setup();record['optimizer_after']['state'][0]['exp_avg_sq'].add_(1.)
    with pytest.raises(ValueError,match='second'):build_response_metric(origin,record,h,layout)

def test_frozen_head_replay_response_and_formal_none_semantics():
    from cvsrffi.xuc_fusion.ir_head import capture_head_calls,head_logits,head_gradient
    from cvsrffi.xuc_fusion.ir_cg import solve_response
    m=nn.Module();m.adv_head=nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Dropout(.2),nn.Linear(3,2))
    m.adv_head[0].bias.requires_grad_(False)
    phi=tuple(m.adv_head.parameters());opt=torch.optim.AdamW([{'params':list(phi)[::-1]}],lr=.01)
    layout=build_layout(m,opt);origin=copy.deepcopy(opt.state_dict())
    before=phi[1].detach().clone()
    with capture_head_calls(m.adv_head,layout.signature,'freeze') as tape:
        native=m.adv_head(torch.randn(4,2))
        tape.register('L',tuple(map(str,range(4))),torch.zeros(4,dtype=torch.long),torch.ones(4)/4,1.)
    torch.testing.assert_close(head_logits(phi,tape.batch().calls[0]),native)
    full=head_gradient(phi,tape.batch()).split([p.numel() for p in phi])
    h0=tuple(g.reshape_as(p) if p.requires_grad else None for g,p in zip(full,phi))
    for p,g in zip(phi,h0):p.grad=g
    opt.step()
    metric=build_response_metric(origin,{'clip_coefficient':1.,'optimizer_after':copy.deepcopy(opt.state_dict())},h0,layout)
    e=torch.ones(sum(p.numel() for p in phi));result=solve_response(e,metric,torch.zeros_like)
    delta=torch.zeros_like(e);delta[metric.active_indices]=metric.sqrt_diagonal*result.solution
    lo=phi[0].numel();hi=lo+phi[1].numel()
    assert torch.count_nonzero(delta[lo:hi])==0
    assert phi[1].grad is None and phi[1] not in opt.state
    assert torch.equal(phi[1],before)

def test_all_none_head_gradients_return_empty_response_on_input_device():
    from cvsrffi.xuc_fusion.ir_cg import solve_response
    origin,record,h,layout=setup();h=tuple(None for _ in h)
    origin['state']={};metric=build_response_metric(origin,record,h,layout)
    e=torch.ones(3);result=solve_response(e,metric,lambda v:(_ for _ in ()).throw(AssertionError()))
    assert result.status=='zero_rhs' and result.solution.numel()==0 and result.solution.device==e.device

import os
import pytest

@pytest.fixture(autouse=True)
def _acceptance_device():
    device=os.environ.get('IR_ACCEPTANCE_DEVICE','cpu')
    if device not in ('cpu','cuda'):raise ValueError('invalid IR_ACCEPTANCE_DEVICE')
    with torch.device(device):
        yield

import pytest
import torch
from torch import nn
from cvsrffi.xuc_fusion.ir_head import capture_head_calls,head_logits,head_gradient

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('p',[0.,.4])
def test_replay_logits_and_gradients(dtype,p):
    torch.manual_seed(42);head=nn.Sequential(nn.Linear(2,4),nn.ReLU(inplace=True),nn.Dropout(p),nn.Linear(4,3)).to(dtype)
    z=torch.randn(5,2,dtype=dtype,requires_grad=True);d=torch.tensor([0,1,2,0,1]);w=torch.full((5,),.2,dtype=dtype)
    with capture_head_calls(head,'layout','ctx') as tape:
        native=head(z);tape.register('L',tuple(map(str,range(5))),d,w,1.)
    call=tape.batch().calls[0];phi=tuple(head.parameters());replay=head_logits(phi,call)
    atol,rtol=(1e-10,1e-8) if dtype==torch.float64 else (1e-6,1e-5)
    torch.testing.assert_close(replay,native,atol=atol,rtol=rtol)
    from dataclasses import replace
    replay_input=head_logits(phi,replace(call,z_detached=z))
    gn=torch.autograd.grad(native.sum(),(z,)+phi,retain_graph=True)
    gr=torch.autograd.grad(replay_input.sum(),(z,)+phi,retain_graph=True)
    for a,b in zip(gn,gr):torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
    hn=torch.autograd.grad(torch.nn.functional.cross_entropy(native,d),phi)
    torch.testing.assert_close(head_gradient(phi,tape.batch()),torch.cat([x.flatten() for x in hn]),atol=atol,rtol=rtol)

def test_zero_activation_does_not_hide_mask():
    head=nn.Sequential(nn.Linear(2,64),nn.ReLU(),nn.Dropout(.5),nn.Linear(64,2))
    with torch.no_grad():head[0].weight.zero_();head[0].bias.zero_()
    with capture_head_calls(head,'l','c') as tape:
        head(torch.zeros(4,2));tape.register('L',('a','b','c','d'),torch.zeros(4,dtype=torch.long),torch.ones(4)/4,1.)
    mask=tape.batch().calls[0].scaled_dropout_mask
    assert (mask==2).any() and (mask==0).any()

def test_L_U_separate_call_rng():
    head=nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Dropout(.5),nn.Linear(16,2))
    with capture_head_calls(head,'l','c') as tape:
        for key,n in [('L',2),('U',5)]:
            head(torch.ones(n,2));tape.register(key,tuple(map(str,range(n))),torch.zeros(n,dtype=torch.long),torch.ones(n)/n,0.)
    assert [c.z_detached.shape[0] for c in tape.batch().calls]==[2,5]

def test_aux_LEO_call_not_in_domain_loss():
    head=nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Dropout(.5),nn.Linear(3,2))
    with capture_head_calls(head,'l','c') as tape:
        head(torch.ones(2,2));head(torch.ones(2,2))
        tape.register('L',('a','b'),torch.zeros(2,dtype=torch.long),torch.ones(2)/2,1.,call_index=0)
    assert len(tape.records)==2 and len(tape.batch().calls)==1

def test_replay_restores_caller_rng():
    head=nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Dropout(.5),nn.Linear(3,2));z=torch.ones(2,2)
    start=torch.get_rng_state();head(z);expected=torch.get_rng_state();torch.set_rng_state(start)
    with capture_head_calls(head,'l','c'):head(z)
    assert torch.equal(expected,torch.get_rng_state())

def test_unsupported_head_rejected():
    with pytest.raises(ValueError,match='head'):
        with capture_head_calls(nn.Linear(2,3),'l','c'):pass

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_dropout_replay_and_rng():
    head=nn.Sequential(nn.Linear(2,16),nn.ReLU(inplace=True),nn.Dropout(.3),nn.Linear(16,3)).cuda();z=torch.randn(7,2,device='cuda')
    rng=torch.cuda.get_rng_state();head(z);expected=torch.cuda.get_rng_state();torch.cuda.set_rng_state(rng)
    with capture_head_calls(head,'l','c') as tape:
        logits=head(z);tape.register('L',tuple(map(str,range(7))),torch.zeros(7,dtype=torch.long,device='cuda'),torch.ones(7,device='cuda')/7,1.)
    assert torch.equal(expected,torch.cuda.get_rng_state())
    torch.testing.assert_close(head_logits(tuple(head.parameters()),tape.batch().calls[0]),logits,atol=1e-6,rtol=1e-5)

def test_capture_hooks_removed_on_failure():
    head=nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Dropout(.5),nn.Linear(3,2))
    with pytest.raises(RuntimeError):
        with capture_head_calls(head,'l','c'):raise RuntimeError('injected')
    assert not head._forward_pre_hooks and not head._forward_hooks and not head[2]._forward_pre_hooks

import os
import pytest

@pytest.fixture(autouse=True)
def _acceptance_device():
    device=os.environ.get('IR_ACCEPTANCE_DEVICE','cpu')
    if device not in ('cpu','cuda'):raise ValueError('invalid IR_ACCEPTANCE_DEVICE')
    with torch.device(device):
        yield

from dataclasses import replace
import torch
from torch import nn
from cvsrffi.xuc_fusion.ir_types import DomainCall,HeadBatch
from cvsrffi.xuc_fusion.ir_head import head_logits,head_loss,ggn_matvec

def fixture():
    torch.manual_seed(9)
    phi=(torch.tensor([[1.,.1],[.2,1.]],dtype=torch.float64),torch.ones(2,dtype=torch.float64),torch.randn(3,2,dtype=torch.float64),torch.zeros(3,dtype=torch.float64))
    calls=[]
    for name,n,alpha in [('L',2,.7),('U',5,.2)]:
        calls.append(DomainCall(name,tuple(map(str,range(n))),torch.rand(n,2,dtype=torch.float64),torch.arange(n)%3,torch.full((n,),alpha/n,dtype=torch.float64),torch.ones(n,2,dtype=torch.float64),1. if name=='L' else 0.))
    return phi,HeadBatch(tuple(calls),'l','c')

def dense(phi,batch):
    total=sum(p.numel() for p in phi);G=torch.zeros(total,total,dtype=torch.float64)
    for call in batch.calls:
        js=torch.func.jacrev(lambda *p:head_logits(p,call),argnums=tuple(range(len(phi))))(*phi)
        J=torch.cat([j.reshape(call.domain.numel(),3,-1) for j in js],-1)
        prob=head_logits(phi,call).softmax(-1)
        for i in range(call.domain.numel()):
            C=torch.diag(prob[i])-prob[i,:,None]*prob[i,None,:]
            G+=call.sample_weight[i]*J[i].T@C@J[i]
    return G

def test_explicit_GGN_symmetry_psd_weight_once():
    phi,batch=fixture();G=dense(phi,batch);v=torch.randn(G.shape[0],dtype=G.dtype)
    torch.testing.assert_close(ggn_matvec(phi,batch,v),G@v,atol=1e-10,rtol=1e-8)
    torch.testing.assert_close(G,G.T,atol=1e-12,rtol=1e-10)
    assert torch.linalg.eigvalsh(G).min()>-1e-12
    doubled=replace(batch,calls=tuple(replace(c,sample_weight=2*c.sample_weight) for c in batch.calls))
    torch.testing.assert_close(ggn_matvec(phi,doubled,v),2*ggn_matvec(phi,batch,v))

def test_nonlinear_GGN_not_declared_full_Hessian():
    phi,batch=fixture();v=torch.randn(sum(p.numel() for p in phi),dtype=torch.float64);leaves=tuple(p.requires_grad_() for p in phi)
    gradients=torch.autograd.grad(head_loss(leaves,batch),leaves,create_graph=True)
    directional=(torch.cat([g.flatten() for g in gradients])*v).sum()
    Hv=torch.cat([g.flatten() for g in torch.autograd.grad(directional,leaves)])
    assert not torch.allclose(Hv,ggn_matvec(phi,batch,v),atol=1e-6)

def test_GGN_does_not_mutate_rng_parameters_masks():
    phi,batch=fixture();before=[p.clone() for p in phi];masks=[c.scaled_dropout_mask.clone() for c in batch.calls]
    rng=torch.get_rng_state();ggn_matvec(phi,batch,torch.ones(sum(p.numel() for p in phi),dtype=torch.float64))
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(p,b) for p,b in zip(phi,before))
    assert all(torch.equal(c.scaled_dropout_mask,b) for c,b in zip(batch.calls,masks))

def test_linear_softmax_last_layer_explicit_kronecker():
    phi,batch=fixture();nfirst=phi[0].numel()+phi[1].numel();G=dense(phi,batch)
    # Only last linear weights/bias vary: analytic multinomial logistic GGN.
    analytic=torch.zeros_like(G[nfirst:,nfirst:])
    for call in batch.calls:
        hidden=torch.relu(torch.nn.functional.linear(call.z_detached,phi[0],phi[1]))*call.scaled_dropout_mask
        p=head_logits(phi,call).softmax(-1)
        for i in range(hidden.shape[0]):
            j=torch.cat((torch.kron(torch.eye(3,dtype=p.dtype),hidden[i:i+1]),torch.eye(3,dtype=p.dtype)),dim=1)
            analytic+=call.sample_weight[i]*j.T@(torch.diag(p[i])-torch.outer(p[i],p[i]))@j
    torch.testing.assert_close(analytic,G[nfirst:,nfirst:],atol=1e-12,rtol=1e-10)

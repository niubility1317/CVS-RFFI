"""Independent transport arithmetic, filling and detached CCD supervision."""
import importlib
import math
import numpy as np
import pytest
import torch
from torch.nn import functional as F


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_common_transport_core')


def centers():return torch.eye(160,dtype=torch.float32)[:6]


def test_scalar_transport_mass_and_multivariate_stationarity():
    c=api();torch.set_num_threads(1)
    q,d=c._sinkhorn(torch.tensor([[.23]],dtype=torch.float64),torch.ones(1,dtype=torch.float64))
    assert q.item()==pytest.approx(1.) and math.exp(d['log_total_mass'])==pytest.approx(math.exp(.23/(.01+2*.5)),rel=1e-6)
    sim=torch.tensor([[.1,.35],[-.2,.12],[.4,-.1]],dtype=torch.float64);beta=torch.tensor([.3,.7],dtype=torch.float64)
    q,d=c._sinkhorn(sim,beta);mass=q*math.exp(d['log_total_mass'])
    derivative=.01*mass.log()-sim+.5*(mass.sum(1)*3).log()[:,None]+.5*(mass.sum(0)/beta).log()[None,:]
    assert derivative.abs().max()<1e-6 and d['iterations']==400 and d['stationarity_residual']<1e-6
    shifted,sd=c._sinkhorn(sim+.2,beta);torch.testing.assert_close(shifted,q,rtol=2e-6,atol=1e-10)
    assert sd['log_total_mass']-d['log_total_mass']==pytest.approx(.2/1.01,abs=1e-6)


def test_adaptive_filling_positive_negative_and_empty_selection():
    c=api();p=centers();beta=torch.full((6,),1/6,dtype=torch.float64)
    weak=torch.eye(160)[8:16];empty=c.assign(weak,p,beta,torch.Generator().manual_seed(1))
    assert not empty['mask'].any() and empty['diagnostics']['fill_branch']=='positive' and empty['diagnostics']['fake_size']==0
    assert empty['diagnostics']['requested_fake_size']==8 and empty['diagnostics']['initial_selected_count']==0
    positive=weak.clone();positive[:2]=p[0]
    filled=c.assign(positive,p,beta,torch.Generator().manual_seed(1))
    assert filled['diagnostics']['positive_count']==2 and filled['diagnostics']['fake_size']==4 and filled['diagnostics']['uot_calls']==2
    negative=c.assign(p[torch.arange(8)%6],p,beta,torch.Generator().manual_seed(1))
    assert negative['diagnostics']['fill_branch']=='negative' and negative['diagnostics']['fake_size']==8 and negative['labels'].shape==(8,)
    torch.testing.assert_close(negative['beta'],.7*beta+.3*torch.tensor(negative['diagnostics']['column_mass'],dtype=torch.float64))
    assert all(not negative[k].requires_grad for k in ('labels','mask','beta'))


def test_CCD_reuses_strong_features_has_no_teacher_gradient_and_empty_is_live_zero(subtests):
    c=api();p=centers().requires_grad_();x=torch.randn(4,160,requires_grad=True);y=torch.randn(4,160,requires_grad=True)
    labels=torch.arange(8)%6;mask=torch.tensor([True,False,True,False,True,False,True,False]);idx=torch.tensor([0,1,4,5])
    assignment=dict(labels=labels,mask=mask);r=c.classification_loss(x,y,p,idx,assignment)
    expected=.5*sum(F.cross_entropy((F.normalize(a,dim=1)@F.normalize(p.detach(),dim=1).T/.1)[mask[idx]],labels[idx][mask[idx]]) for a in (x,y))
    torch.testing.assert_close(r['loss'],expected);r['loss'].backward()
    assert x.grad.norm()>0 and y.grad.norm()>0 and p.grad is None and r['selected_batch_count']==2
    zero=c.classification_loss(x,y,p,idx,dict(labels=labels,mask=torch.zeros(8,dtype=torch.bool)))['loss']
    assert zero.requires_grad and zero.item()==0
    for bad in (torch.zeros(6,dtype=torch.float64),torch.full((6,),float('nan'),dtype=torch.float64)):
        with subtests.test(beta=bad.tolist()),pytest.raises(ValueError):c.assign(torch.eye(160)[:8],p,bad,torch.Generator())


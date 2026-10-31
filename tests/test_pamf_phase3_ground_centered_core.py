"""Centered six-logit evidence and equal-domain source risk mathematics."""
import importlib
import math
import numpy as np
import pytest
import torch
from torch.nn import functional as F


def api():return importlib.import_module('experiments.pamf_phase3_open_world.ground_centered_core')


def test_shared_shift_invariance_uniform_KL_and_zero_sum_gradient():
    m=api();x=torch.tensor([[2.,0.,-2.,1.,-1.,3.],[-1.,1.,0.,2.,-2.,3.]],requires_grad=True)
    expected=-(F.log_softmax(x,dim=1)/6).sum(1)-math.log(6)
    torch.testing.assert_close(m.centered_evidence(x),expected)
    shift=torch.tensor([[8.],[-4.]])
    torch.testing.assert_close(m.centered_evidence(x+shift),m.centered_evidence(x),rtol=1e-6,atol=1e-6)
    m.centered_evidence(x).sum().backward()
    assert x.grad.norm()>0 and bool((x.grad.sum(1).abs()<2e-7).all())
    a=np.asarray(x.detach().tolist(),np.float32);c=m.centered_energy(a)
    assert c.dtype==np.float64 and c.shape==(2,)
    np.testing.assert_allclose(c,m.centered_energy(a+np.asarray([[8.],[-4.]],np.float32)),atol=1e-14,rtol=0)
    assert m.centered_energy(np.full((2,6),30.,np.float32)).tolist()==[0.,0.]


def test_domain_equal_centered_pair_risk_and_teacher_are_exact():
    m=api();torch.manual_seed(31)
    L=torch.randn(4,6,requires_grad=True);OE=torch.randn(3,6,requires_grad=True);teacher=torch.randn(4,6,requires_grad=True)
    y=torch.tensor([0,1,2,3]);ld=torch.tensor([0,0,0,1]);od=torch.tensor([0,0,1])
    result=m.compute(L,y,teacher,ld,OE,od)
    cl=m.centered_evidence(L);co=m.centered_evidence(OE)
    means=[F.softplus(co[od==d][None]-cl[ld==d][:,None]).mean() for d in (0,1)]
    equal=torch.stack(means).mean();global_mean=F.softplus(co[None]-cl[:,None])[ld[:,None]==od[None]].mean()
    ce=F.cross_entropy(L,y);kd=4*F.kl_div(F.log_softmax(L/2,dim=1),F.softmax(teacher.detach()/2,dim=1),reduction='batchmean')
    torch.testing.assert_close(result['total'],ce+kd+equal)
    torch.testing.assert_close(result['OE'],equal);assert not torch.isclose(equal,global_mean)
    assert result['domain_pair_counts']==[6,1]+[0]*13 and result['domain_count']==2 and result['pair_count']==7
    assert len(result['domain_mean_losses'])==len(result['domain_raw_mean_losses'])==15
    result['total'].backward();assert teacher.grad is None and L.grad.norm()>0 and OE.grad.norm()>0
    shifted=m.compute(L.detach()+4,y,teacher.detach(),ld,OE.detach()+8,od)
    torch.testing.assert_close(shifted['OE'],result['OE'].detach(),atol=1e-6,rtol=1e-6)
    assert abs(float(shifted['raw_domain_mean'])-float(result['raw_domain_mean']))>1


def test_input_boundaries_reject_nonfinite_labels_domains_and_extra_unknown_inputs(subtests):
    m=api();L=torch.zeros(2,6);teacher=L.clone();y=torch.tensor([0,1]);d=torch.tensor([0,1])
    for defect in ('nan','dtype','shape','label','negative_label','label_dtype','domain','unmatched','teacher','arm','extra'):
        with subtests.test(defect=defect),pytest.raises((ValueError,TypeError)):
            if defect=='nan':a=L.clone();a[0,0]=float('nan');m.compute(a,y,teacher,d,L,d)
            elif defect=='dtype':m.centered_evidence(L.double())
            elif defect=='shape':m.centered_energy(np.zeros((3,5),np.float32))
            elif defect=='label':m.compute(L,torch.tensor([0,6]),teacher,d,L,d)
            elif defect=='negative_label':m.compute(L,torch.tensor([0,-1]),teacher,d,L,d)
            elif defect=='label_dtype':m.compute(L,y.int(),teacher,d,L,d)
            elif defect=='domain':m.compute(L,y,teacher,torch.tensor([0,15]),L,d)
            elif defect=='unmatched':m.compute(L,y,teacher,d,L,torch.tensor([0,0]))
            elif defect=='teacher':m.compute(L,y,teacher[:1],d,L,d)
            elif defect=='arm':m.compute(L,y,teacher,d,L,d,arm='DMEAN')
            else:m.compute(L,y,teacher,d,L,d,OE_true_TX=['private'])
    with pytest.raises(ValueError):m.centered_energy(np.full((2,6),np.inf,np.float32))

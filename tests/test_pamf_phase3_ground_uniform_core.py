"""Equal-domain uniform OE, with ranking restricted to reported diagnostics."""
import importlib
import math
import pytest
import torch
from torch.nn import functional as F


def api():return importlib.import_module('experiments.pamf_phase3_open_world.ground_uniform_core')


def values():
    L=torch.tensor([[2.,0.,-1.,0.,1.,0.],[0.,3.,0.,-1.,0.,1.],[0.,0.,2.,0.,-1.,0.],[0.,0.,0.,1.,0.,-1.]])
    OE=torch.tensor([[0.,0.,0.,0.,0.,0.],[4.,1.,0.,0.,0.,0.],[-1.,2.,0.,1.,0.,0.]])
    return L,torch.arange(4),L.flip(1).clone(),torch.tensor([0,0,1,1]),OE,torch.tensor([0,1,1])


def test_exact_equal_domain_equation_not_global_mean_or_pair_ranking():
    m=api();L,y,teacher,ld,OE,od=values();a=m.compute(L,y,teacher,ld,OE,od)
    per=-F.log_softmax(OE,dim=1).mean(1)-math.log(6.)
    uniform=(per[0]+per[1:].mean())/2
    ce=F.cross_entropy(L,y);kd=4*F.kl_div(F.log_softmax(L/2,1),F.softmax(teacher/2,1),reduction='batchmean')
    torch.testing.assert_close(a['OE'],uniform);torch.testing.assert_close(a['uniform'],uniform)
    torch.testing.assert_close(a['total'],ce+kd+uniform)
    torch.testing.assert_close(a['domain_uniform_mean'],uniform);torch.testing.assert_close(a['uniform_global_mean'],per.mean())
    assert not torch.isclose(uniform,per.mean()) and not torch.isclose(a['rank'],uniform)
    assert a['domain_uniform_counts']==[1,2]+[0]*13
    assert a['domain_uniform_losses'][:2]==pytest.approx([float(per[0]),float(per[1:].mean())],abs=1e-7)
    assert a['domain_uniform_losses'][2:]==[None]*13 and a['rank_is_diagnostic'] is True
    assert a['OE_objective']=='equal_domain_uniform_posterior' and a['domain_count']==2


def test_shift_invariance_zero_uniform_and_gradients_only_CE_KD_on_L():
    m=api();L,y,teacher,ld,OE,od=values();L.requires_grad_();OE.requires_grad_();teacher.requires_grad_()
    a=m.compute(L,y,teacher,ld,OE,od)
    assert torch.autograd.grad(a['OE'],L,allow_unused=True,retain_graph=True)[0] is None
    expected=F.cross_entropy(L,y)+4*F.kl_div(F.log_softmax(L/2,1),F.softmax(teacher.detach()/2,1),reduction='batchmean')
    actual_grad=torch.autograd.grad(a['total'],L,retain_graph=True)[0]
    torch.testing.assert_close(actual_grad,torch.autograd.grad(expected,L)[0],atol=0,rtol=0)
    a['total'].backward();assert teacher.grad is None and float(OE.grad.norm())>0
    torch.testing.assert_close(OE.grad.sum(1),torch.zeros(3),atol=torch.finfo(torch.float32).eps,rtol=0)
    shifted=m.compute(L.detach(),y,teacher.detach(),ld,OE.detach()+torch.tensor([8.,-4.,16.])[:,None],od)
    torch.testing.assert_close(a['uniform'],shifted['uniform'],atol=2e-7,rtol=0)
    assert abs(float(m.compute(L.detach(),y,teacher.detach(),ld,torch.zeros_like(OE),od)['uniform']))<2e-7
    assert float(a['uniform'].detach())>0 and float(a['rank'].detach())>0


def test_finite_dtype_class_domain_and_private_input_rejections(subtests):
    m=api();L,y,t,ld,OE,od=values()
    bad=OE.clone();bad[0,0]=float('nan')
    cases=[(L.double(),y,t,ld,OE,od),(L,y.float(),t,ld,OE,od),(L,torch.full_like(y,-1),t,ld,OE,od),
        (L,y,t,ld,bad,od),(L,y,t,ld,OE,od.float()),(L,y,t,ld,OE,torch.full_like(od,15)),
        (L,y,t,torch.full_like(ld,2),OE,od),(L,y,t[:3],ld,OE,od)]
    for i,args in enumerate(cases):
        with subtests.test(invalid=i),pytest.raises(ValueError):m.compute(*args)
    with subtests.test(private_OE_truth=True),pytest.raises(TypeError):m.compute(L,y,t,ld,OE,od,OE_y=torch.zeros(3,dtype=torch.long))
    with subtests.test(wrong_arm=True),pytest.raises(ValueError):m.compute(L,y,t,ld,OE,od,'RXPAIR')

"""Analytic and gradient checks for the independent source metric network."""
import importlib
import math

import pytest
import torch
from torch.nn import functional as F


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_metric_residual_model')


def labeled():
    x=torch.zeros(4,160);c=math.sqrt(3)/2;s=.5
    x[:,:2]=torch.tensor([[1.,0.],[c,s],[0.,1.],[-s,c]])
    return x,torch.tensor([0,0,1,1]),torch.tensor([1,3,1,3]),torch.tensor([1,2,1,2])


def test_supervised_KL_matches_closed_form_and_equal_domain_replication():
    m=api();z,y,rx,day=labeled();z.requires_grad_()
    loss=m.supervised_metric(z,y,rx,day)
    c=math.sqrt(3)/2;s=.5
    expected=.5*(math.log1p(math.exp(-(c+s)/.1))+math.log1p(math.exp(-(c-s)/.1)))
    assert float(loss['loss'].detach())==pytest.approx(expected,abs=2e-7)
    loss['loss'].backward();assert z.grad is not None and z.grad.abs().sum()>0
    # Replicate the whole first public cell, keeping both class proportions.
    ix=torch.tensor([0,1,2,3,0,2])
    changed=m.supervised_metric(z.detach()[ix],y[ix],rx[ix],day[ix])
    torch.testing.assert_close(changed['loss'],loss['loss'].detach(),atol=2e-7,rtol=0.)


def test_OE_soft_KL_teacher_stopgrad_and_domain_average():
    m=api();teacher,y,rx,day=labeled();teacher.requires_grad_()
    student=teacher.detach().clone();student[0,1]=.4;student.requires_grad_()
    same=m.oe_structure(teacher,teacher,rx,day)
    assert abs(float(same['loss'].detach()))<2e-7
    out=m.oe_structure(student,teacher,rx,day);assert out['loss']>0
    out['loss'].backward();assert teacher.grad is None and student.grad.abs().sum()>0
    ix=torch.tensor([0,1,2,3,0,2])
    replicated=m.oe_structure(student.detach()[ix],teacher.detach()[ix],rx[ix],day[ix])
    torch.testing.assert_close(replicated['loss'],out['loss'].detach(),atol=2e-7,rtol=0.)


def test_zero_final_initialization_same_seed_first_then_hidden_gradients_no_teacher_grad():
    m=api();torch.set_num_threads(1);rng=torch.get_rng_state().clone();a=m.MetricResidual();b=m.MetricResidual()
    torch.testing.assert_close(torch.get_rng_state(),rng,atol=0,rtol=0)
    for name,p in a.state_dict().items():torch.testing.assert_close(p,b.state_dict()[name],atol=0,rtol=0)
    assert sum(p.numel() for p in a.parameters())==102688
    assert not a.network[-1].weight.any() and not a.network[-1].bias.any()
    u,y,rx,day=labeled();u.requires_grad_();c=torch.randn(4,640,requires_grad=True)
    torch.testing.assert_close(a(u,c),F.normalize(u.detach(),dim=1),atol=0,rtol=0)
    optimizer=torch.optim.AdamW(a.parameters(),lr=1e-4,weight_decay=1e-4)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);m.supervised_metric(a(u,c),y,rx,day)['loss'].backward()
        assert a.network[-1].weight.grad.abs().sum()>0
        if step==0:assert not a.network[0].weight.grad.any()
        else:assert a.network[0].weight.grad.abs().sum()>0
        assert u.grad is None and c.grad is None;optimizer.step()


def test_input_shape_finite_zero_and_missing_positive_coverage(subtests):
    m=api();net=m.MetricResidual();u,y,rx,day=labeled();c=torch.randn(4,640)
    for first,second in ((u[:,:159],c),(u,c[:,:639]),(u.double(),c),(u,torch.zeros_like(c)),(torch.full_like(u,float('nan')),c)):
        with subtests.test(shape=tuple(first.shape)),pytest.raises(ValueError):net(first,second)
    for labels,rr,dd in ((torch.arange(4),rx,day),(y,torch.ones_like(rx),day),(y,rx,torch.ones_like(day))):
        with subtests.test(labels=labels.tolist()),pytest.raises(ValueError):m.supervised_metric(u,labels,rr,dd)
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):assert net(u,c).dtype==torch.float32

"""Tail objectives: domain weighting, difficult pairs and true gradients."""
import importlib
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.ground_tail_core')


def values():
    L=torch.tensor([[6.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.],[3.,0.,0.,0.,0.,0.]],requires_grad=True)
    OE=torch.tensor([[5.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[7.,0.,0.,0.,0.,0.]],requires_grad=True)
    return L,torch.tensor([0,0,0]),L.detach().clone(),torch.tensor([0,0,1]),OE,torch.tensor([0,0,1])


def test_literal_per_domain_mean_and_hard_quarter_have_correct_gradients():
    m=api();args=values();a=m.compute(*args,'DMEAN');b=m.compute(*args,'DTAIL')
    L,y,t,ld,O,od=args
    sl=torch.logsumexp(L,1);so=torch.logsumexp(O,1)
    p0=torch.nn.functional.softplus(torch.stack((so[0]-sl[0],so[1]-sl[0],so[0]-sl[1],so[1]-sl[1])))
    p1=torch.nn.functional.softplus(so[2]-sl[2])
    assert torch.allclose(a['OE'],(p0.mean()+p1)/2)
    assert torch.allclose(b['OE'],(p0.max()+p1)/2)
    assert a['pair_count']==b['pair_count']==5 and a['selected_pair_count']==5 and b['selected_pair_count']==2
    assert a['domain_pair_counts']==[4,1]+[0]*13 and b['domain_selected_counts']==[1,1]+[0]*13
    assert a['domain_count']==2 and b['domain_count']==2 and b['OE']>a['OE']
    b['total'].backward()
    assert torch.isfinite(L.grad).all() and torch.isfinite(O.grad).all()
    assert O.grad[1].abs().sum()==0 and O.grad[0].abs().sum()>0 and O.grad[2].abs().sum()>0
    assert torch.allclose(a['CE'],b['CE']) and torch.allclose(a['KD'],b['KD'])


def test_tail_equal_domain_weights_and_stable_ties():
    m=api();L=torch.zeros((4,6),requires_grad=True);OE=torch.zeros((4,6),requires_grad=True)
    ld=torch.tensor([0,0,1,1]);od=torch.tensor([0,0,1,1]);y=torch.arange(4)
    out=m.compute(L,y,L.detach(),ld,OE,od,'DTAIL');out['OE'].backward()
    assert out['domain_selected_counts'][:2]==[1,1]
    assert OE.grad[0].abs().sum()>0 and OE.grad[1].abs().sum()==0
    assert OE.grad[2].abs().sum()>0 and OE.grad[3].abs().sum()==0
    assert torch.allclose(out['OE'],torch.tensor(np.log(2),dtype=torch.float32))


def test_protocol_negative_inputs_and_shift_invariance(subtests):
    m=api();args=values();a=m.compute(*args,'DTAIL');shift=list(args);shift[0]=args[0]+11;shift[4]=args[4]+11
    assert torch.allclose(a['total'],m.compute(*shift,'DTAIL')['total'],atol=1e-6)
    cases=[(list(args),'RXPAIR'),([*args[:5],torch.tensor([3,3,3])],'DTAIL'),
        ([args[0],torch.tensor([0,0,6]),*args[2:]],'DTAIL'),([args[0].double(),*args[1:]],'DTAIL')]
    bad=list(args);bad[4]=args[4].detach().clone();bad[4][0,0]=float('nan');cases.append((bad,'DTAIL'))
    for j,(v,arm) in enumerate(cases):
        with subtests.test(case=j),pytest.raises(ValueError):m.compute(*v,arm)

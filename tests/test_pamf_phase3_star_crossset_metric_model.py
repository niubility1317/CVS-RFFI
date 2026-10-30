"""Analytic half-prior mass and direct gradients for cross-set separation."""
import importlib
import math
import inspect

import pytest
import torch
from torch.nn import functional as F


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_crossset_metric_model')


def example():
    L=torch.zeros(4,160);L[[0,1],0]=1;L[[2,3],1]=1
    OE=torch.zeros(6,160);OE[:,2]=1
    return L,OE,torch.tensor([0,0,1,1]),torch.tensor([1,3,1,3]),torch.tensor([1,2,1,2]),torch.tensor([1,3,1,3,4,4]),torch.tensor([1,2,1,2,3,3])


def test_hand_KL_half_priors_ignore_candidate_count_and_effective_domain_imbalance():
    m=api();L,OE,y,rx,day,orx,oday=example()
    result=m.crossset_metric(L,OE,y,rx,day,orx,oday)
    # L prior: .25*exp(10) positive + .25 negative; OE prior totals .5,
    # even though it has two effective domains and more eligible rows.
    assert float(result['loss'])==pytest.approx(math.log1p(3*math.exp(-10)),abs=2e-7)
    assert result['candidate_set_priors']==[.5,.5] and result['minimum_OE_candidate_domains']==2
    ix=torch.tensor([0,1,2,3,4,5,4,5,4,5])
    repeated=m.crossset_metric(L,OE[ix],y,rx,day,orx[ix],oday[ix])
    torch.testing.assert_close(result['loss'],repeated['loss'],atol=2e-7,rtol=0.)
    overlap=OE.clone();overlap[:,0]=1;overlap[:,2]=0
    assert m.crossset_metric(L,overlap,y,rx,day,orx,oday)['loss']>result['loss']


def test_crossset_has_real_gradients_for_both_sets_and_no_OE_OE_hard_edges():
    m=api();L,OE,y,rx,day,orx,oday=example();L.requires_grad_();OE.requires_grad_()
    first=m.crossset_metric(L,OE,y,rx,day,orx,oday);first['loss'].backward()
    assert (L.grad.norm(dim=1)>0).all() and (OE.grad.norm(dim=1)>0).all()
    changed=OE.detach().clone();changed[::2,2]=0;changed[::2,3]=1
    assert not torch.equal(OE.detach()@OE.detach().T,changed@changed.T)
    torch.testing.assert_close(m.crossset_metric(L.detach(),changed,y,rx,day,orx,oday)['loss'],first['loss'].detach(),atol=0,rtol=0)
    assert first['OE_OE_hard_negatives'] is False
    assert tuple(inspect.signature(m.crossset_metric).parameters)==('z_L','z_OE','y_L','rx_L','day_L','rx_OE','day_OE')


def test_missing_set_candidates_positive_coverage_and_nonfinite_fail(subtests):
    m=api();args=list(example())
    for defect in ('no_OE','no_L','no_positive','nonfinite','OE_shape'):
        values=[a.clone() for a in args]
        if defect=='no_OE':values[5].fill_(1);values[6].fill_(1)
        elif defect=='no_L':values[3].fill_(1);values[4].fill_(1)
        elif defect=='no_positive':values[2]=torch.arange(4)
        elif defect=='nonfinite':values[1][0,0]=float('nan')
        else:values[1]=values[1][:,:159]
        with subtests.test(defect=defect),pytest.raises(ValueError):m.crossset_metric(*values)
    from experiments.pamf_phase3_open_world.star_metric_residual_model import MetricResidual
    assert m.MetricResidual is MetricResidual and sum(p.numel() for p in m.MetricResidual().parameters())==102688

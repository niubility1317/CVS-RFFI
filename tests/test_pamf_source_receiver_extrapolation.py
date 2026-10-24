from copy import deepcopy
import pytest
import torch
from experiments.pamf_ssl_research.source_receiver_extrapolation import ALPHAS,extrapolate,select_scale
from experiments.pamf_ssl_research.matched_report import SEEDS


def inputs():
    identity=torch.arange(160,dtype=torch.float64)/20
    nuisance=torch.stack([torch.full((160,),x,dtype=torch.float64) for x in (1.,3.,5.)])
    h=(identity+nuisance).requires_grad_()
    return identity,nuisance,h,torch.zeros(3,dtype=torch.long),torch.tensor([True,True,False]),torch.tensor([1,3,4]),torch.tensor([1,0,2]),torch.tensor([True,True,False])


def test_full_identity_is_retained_under_additive_nuisance_extrapolation_and_gradient_is_real():
    mu,n,h,p,a,rx,partner,v=inputs();out=extrapolate(h,p,a,rx,partner,v,.5)
    torch.testing.assert_close(out[0],mu+1.5*n[0]-.5*n[1]);torch.testing.assert_close(out[2],h[2])
    out[0].sum().backward();torch.testing.assert_close(h.grad[0],torch.full((160,),1.5,dtype=torch.float64))
    torch.testing.assert_close(h.grad[1],torch.full((160,),-.5,dtype=torch.float64));assert not h.grad[2].count_nonzero()


@pytest.mark.parametrize('bad',['unaccepted','wrongclass','sameRX','scale'])
def test_illegal_partner_or_unregistered_scale_is_rejected(bad):
    _,_,h,p,a,rx,partner,v=inputs();alpha=.5
    if bad=='unaccepted':a[1]=False
    elif bad=='wrongclass':p[1]=1
    elif bad=='sameRX':rx[1]=rx[0]
    else:alpha=.75
    with pytest.raises(ValueError):extrapolate(h,p,a,rx,partner,v,alpha)


def rows():
    return [dict(model_seed=s,role=role,cases=[dict(alpha=x,selected=900 if role=='U_s' else 650,
        pseudo_agreements=900 if role=='U_s' else 650,true_correct=649 if role=='V' else None,
        pseudo_hard_CE=.01*(1+2*x)) for x in ALPHAS]) for s in SEEDS for role in ('U_s','V')]


def test_scale_uses_the_fixed_worst_seed_source_rules_and_can_falsify_all_candidates():
    r=rows();r[-1]['cases'][-1]['true_correct']=640
    result=select_scale(r);assert result['selected_alpha']==.5
    assert result['decisions'][-1]['passes'] is False and result['U_actual_accuracy'].startswith('UNKNOWN')
    for row in r:
        if row['role']=='U_s':
            for c in row['cases']:c['pseudo_hard_CE']=.01
    result=select_scale(r);assert result['status']=='NO_SOURCE_CANDIDATE' and result['selected_alpha'] is None
    with pytest.raises(ValueError):select_scale(r[:-1])

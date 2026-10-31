import numpy as np
import pytest
import torch
from experiments.pamf_dura_efficiency.signature_parameters import split_parameters,statistics,component_summary,decision


def test_actual_collection_splits_mean_derivatives_with_unused_parameters_and_no_accumulation():
    a=torch.nn.Parameter(torch.tensor([1.,-1.]));unused=torch.nn.Parameter(torch.tensor([4.]));params=[a,unused]
    x=torch.tensor([[1.,2.],[3.,4.]]);physical=(x@a).square();signature=(2*x@a-1).square()
    reference=torch.autograd.grad((physical+signature).mean(),params,retain_graph=True,allow_unused=True)[0]
    p,s,pu,su=split_parameters(physical,signature,params)
    torch.testing.assert_close(p,torch.tensor([-4.,-6.,0.]));torch.testing.assert_close((p+s)[:2],reference)
    assert pu==su==1 and all(v.grad is None for v in params)
    with pytest.raises(ValueError):split_parameters(torch.zeros(2),torch.zeros(3),params)


def test_statistics_use_raw_vectors_and_report_undefined_instead_of_finite_ratio():
    p=np.array([[2.,0.],[0.,1.]]);s=np.array([[0.,.1],[.2,0.]])
    r=statistics(p,s);assert r['SIG_to_physical_ratio']==pytest.approx([.05,.2]) and r['physical_SIG_cosine']==pytest.approx([0.,0.])
    t=p+s
    np.testing.assert_allclose(r['total_physical_cosine'],np.sum(t*p,1)/(np.linalg.norm(t,axis=1)*np.linalg.norm(p,axis=1)))
    z=statistics(np.zeros((1,2)),np.ones((1,2)));assert z['SIG_to_physical_ratio']==[None] and z['ratio_defined']==[False]
    with pytest.raises(ValueError):statistics(p,np.full_like(p,np.inf))


def test_primary_requires_both_views_and_checks_raw_pool_not_mean_of_ratios():
    p=np.ones((5,2));small=component_summary(p,.05*p);large=component_summary(p,.2*p)
    assert decision({'1':small,'20':small})['supports_small_SIG_parameter_contribution']
    r=decision({'1':small,'20':large});assert not r['supports_small_SIG_parameter_contribution'] and not r['automatic_training']
    # Four small individual ratios can coexist with cancellation of physical
    # gradients, leaving a large pooled SIG influence: this must not pass.
    p=np.array([[1.,0.],[1,0],[-1,0],[-1,0],[.01,0]])
    s=np.array([[.05,0],[.05,0],[.05,0],[.05,0],[.001,0]])
    cancelled=component_summary(p,s)
    assert not decision({'1':cancelled,'20':cancelled})['supports_small_SIG_parameter_contribution']

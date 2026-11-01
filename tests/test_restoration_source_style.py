import math
import numpy as np
import pytest
import torch
from experiments.pamf_phase3_restoration import source_style as style


def test_phase_and_gains_have_exact_declared_IQ_action_and_energy():
    torch.manual_seed(8);x=torch.randn(3,2,256)
    y=style.transform(x,'phase90')
    assert torch.equal(y[:,0],-x[:,1]) and torch.equal(y[:,1],x[:,0])
    torch.testing.assert_close(y.square().sum((1,2)),x.square().sum((1,2)))
    for name,g in [('gain070',.7),('gain140',1.4)]:
        torch.testing.assert_close(style.transform(x,name),g*x,atol=0,rtol=0)
    with pytest.raises(ValueError):style.transform(x,'unknown')
    with pytest.raises(ValueError):style.transform(x.double(),'identity')


def test_smooth_filter_uses_reflect_not_wrap_and_transforms_are_per_record_linear():
    x=torch.zeros(1,2,256);x[:,0,-1]=1;y=style.transform(x,'reflect_fir025')
    assert y[0,:,0].abs().sum()==0
    assert y[0,0,-1]==pytest.approx(1/math.sqrt(1+.25**2))
    x.zero_();x[:,0,1]=1;y=style.transform(x,'reflect_fir025')
    assert y[0,0,0]==pytest.approx(.25*.5/math.sqrt(1+.25**2))
    torch.manual_seed(9);a=torch.randn(3,2,256);b=torch.randn_like(a)
    for name in style.STYLES:
        torch.testing.assert_close(style.transform(a+b,name),style.transform(a,name)+style.transform(b,name))
        torch.testing.assert_close(style.transform(a,name)[:1],style.transform(a[:1],name),atol=0,rtol=0)


def test_equivariance_reference_IQ_error_is_invariant_to_phase_and_scalar_gain():
    torch.manual_seed(4);x=torch.randn(4,2,256);estimate=x+.3*torch.randn_like(x)
    base=(estimate-x).square().mean((1,2))/x.square().mean((1,2))
    for name in ['identity','phase90','gain070','gain140']:
        y=style.transform(estimate,name);target=style.transform(x,name)
        torch.testing.assert_close((y-target).square().mean((1,2))/target.square().mean((1,2)),base)


def test_summary_refuses_partial_styles_and_keeps_actual_reference_pair_direction():
    with pytest.raises(ValueError,match='All512source'):style.summarize({'scene':np.asarray(style.SCENES)})
    r={'style':np.repeat(style.STYLES,10240),'scene':np.tile(np.resize(np.asarray(style.SCENES),10240),5),
       'clean_feature_shift':np.zeros(51200),'clean_decision_preservation':np.ones(51200)}
    for method in style.METHODS:
        r[method+'__equivariance_nmse']=np.zeros(51200)
        for path in ['actual','reference']:
            for key in style.METRICS:r[method+'__'+path+'__'+key]=np.ones(51200)
        r[method+'__reference__clean_decision_agreement'][0]=0
    q=style.summarize(r)
    assert q['identity']['ALL']['methods']['DURA_v1']['actual_against_transformed_baseline']==dict(repairs=1,damage=0,net=1)

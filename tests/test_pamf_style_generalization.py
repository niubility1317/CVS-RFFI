from dataclasses import replace
import numpy as np
import pytest
import torch
from experiments.pamf_style_generalization.mixing import ARMS,make_plan,mix_statistics


def source():
    return np.array([1,1,3,3,4,4,6,6]),np.array([0,0,0,0,1,1,1,1]),np.ones(8,dtype=np.bool_)


def test_identity_protection_and_matched_effective_exposure_across_RX_controls():
    rx,pseudo,accepted=source();plans=[make_plan(rx,pseudo,accepted,np.random.default_rng(7),a,p=1.) for a in ARMS]
    assert np.array_equal(plans[0].physical_active,plans[1].physical_active)
    assert torch.equal(plans[0].coefficient,plans[1].coefficient)
    for arm,plan in zip(ARMS,plans):
        active=np.flatnonzero(plan.physical_active);donor=np.array(plan.donor[:8].tolist())
        assert len(active)>0 and np.all(pseudo[active]==pseudo[donor[active]])
        assert np.all((rx[active]==rx[donor[active]]) if arm==ARMS[0] else (rx[active]!=rx[donor[active]]))
        assert np.all(donor[active]!=active)
        assert torch.equal(plan.donor[:8]+8,plan.donor[8:])
        assert plan.donor[:8].max()<8 and plan.donor[8:].min()>=8
        assert torch.equal(plan.coefficient[:8],plan.coefficient[8:]) and torch.equal(plan.active[:8],plan.active[8:])
        assert plan.counts()['paired_IQ_applications_per_layer']==2*len(active)


def test_no_fallback_when_either_RX_regime_has_no_identity_partner():
    rx=np.array([1,3]);pseudo=np.array([0,0]);accepted=np.ones(2,dtype=np.bool_)
    for arm in ARMS:
        plan=make_plan(rx,pseudo,accepted,np.random.default_rng(1),arm,p=1.)
        assert not plan.active.any() and not plan.common_eligible.any()
        x=torch.randn(4,3,16);assert torch.equal(x,mix_statistics(x,plan))


def test_rejected_pseudo_records_are_never_donors_or_recipients_and_labels_permute():
    rx,pseudo,accepted=source();accepted[0]=False
    renamed=np.array([5,2])[pseudo]
    for arm in ARMS:
        a=make_plan(rx,pseudo,accepted,np.random.default_rng(2),arm,p=1.)
        b=make_plan(rx,renamed,accepted,np.random.default_rng(2),arm,p=1.)
        assert not a.physical_active[0] and 0 not in a.donor[a.active].tolist()
        assert torch.equal(a.donor,b.donor) and torch.equal(a.coefficient,b.coefficient)


def test_donor_statistics_detach_and_clean_half_cannot_feed_observed_half():
    rx,pseudo,accepted=source();plan=make_plan(rx,pseudo,accepted,np.random.default_rng(3),ARMS[1],p=1.)
    i=int(np.flatnonzero(plan.physical_active)[0]);donor=int(plan.donor[i])
    x=torch.randn(16,3,16,requires_grad=True);y=mix_statistics(x,plan);y[i].square().mean().backward()
    assert x.grad[i].abs().sum()>0 and torch.count_nonzero(x.grad[donor])==0
    assert torch.count_nonzero(x.grad[8:])==0
    x2=x.detach().clone();x2[8:]+=100.
    assert torch.equal(mix_statistics(x.detach(),plan)[:8],mix_statistics(x2,plan)[:8])


def test_mixed_moments_and_zero_variance_are_finite():
    rx,pseudo,accepted=source();plan=make_plan(rx,pseudo,accepted,np.random.default_rng(3),ARMS[1],p=1.)
    plan=replace(plan,coefficient=torch.full((16,1,1),.5))
    x=torch.arange(16.).reshape(1,1,16).repeat(16,3,1)
    for i in range(16):x[i]=x[i]*(i+1)+i
    y=mix_statistics(x,plan)
    for i in np.flatnonzero(plan.physical_active):
        j=int(plan.donor[i]);expected_mean=(x[i].mean(-1)+x[j].mean(-1))/2
        expected_std=(x[i].std(-1,unbiased=False)+x[j].std(-1,unbiased=False))/2
        assert torch.allclose(y[i].mean(-1),expected_mean,atol=2e-5,rtol=1e-5)
        assert torch.allclose(y[i].std(-1,unbiased=False),expected_std,atol=2e-5,rtol=1e-5)
    assert torch.isfinite(mix_statistics(torch.ones_like(x),plan)).all()


@pytest.mark.parametrize('bad',['RX','pseudo','mask','shape','arm','p','alpha'])
def test_invalid_source_mixing_inputs_rejected(bad):
    rx,pseudo,accepted=source();arm=ARMS[1];p=.5;alpha=.1
    if bad=='RX':rx[0]=2
    elif bad=='pseudo':pseudo[0]=6
    elif bad=='mask':accepted=accepted.astype(np.int64)
    elif bad=='shape':pseudo=pseudo[:1]
    elif bad=='arm':arm='unknown'
    elif bad=='p':p=2.
    else:alpha=0.
    with pytest.raises(ValueError):make_plan(rx,pseudo,accepted,np.random.default_rng(1),arm,p=p,alpha=alpha)

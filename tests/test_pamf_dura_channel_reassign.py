import numpy as np
import torch
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_restoration.models import complex_iq,pair
from experiments.pamf_dura_efficiency.channel_reassign import matched_donors,reassign,qualifies


def test_matched_draws_keep_scene_state_lock_and_separate_RX_without_self_donors():
    rx=np.asarray([1,1,3,3,1,1,4]);day=np.ones(7,np.int64)
    labels={k:np.asarray(['a','a','a','a','b','b','c']) for k in ('scene','state','lock')}
    plan=matched_donors(np.arange(7),rx,day,labels,np.random.default_rng(24))
    assert np.array_equal(plan['positions'],np.arange(4)) and np.array_equal(plan['excluded'],[4,5,6])
    for i,j,k in zip(plan['positions'],plan['within'],plan['cross']):
        assert i!=j and rx[i]==rx[j] and day[i]==day[j] and rx[i]!=rx[k]
        assert all(v[i]==v[j]==v[k] for v in labels.values())


def test_full_real_linear_channel_reassignment_preserves_reference_and_exact_donor_noise():
    generator=torch.Generator().manual_seed(25)
    clean=torch.randn(2,2,16,generator=generator,dtype=torch.float64);before=clean.clone()
    donor=torch.randn(clean.shape,generator=generator,dtype=clean.dtype)
    kernel=torch.tensor([[1,.1j,.05],[.8,.2,.03j]],dtype=torch.complex128)
    main=torch.exp(1j*torch.linspace(0,2,16,dtype=clean.dtype)).expand(2,-1)
    image=torch.full_like(main,.1j);action=Action(kernel,main,image)
    noise=.05*torch.randn(clean.shape,generator=generator,dtype=clean.dtype)
    received=pair(action(complex_iq(donor)))+noise
    result,residual,signal=reassign(clean,donor,received,action)
    assert torch.equal(clean,before)
    torch.testing.assert_close(residual,noise,atol=3e-15,rtol=1e-12)
    torch.testing.assert_close(result-signal,noise,atol=3e-15,rtol=1e-12)


def test_training_qualification_requires_both_effects_both_views_and_coverage():
    row=dict(count=512,IQ_ratio=1.11,action_ratio=1.11,reference_unchanged=True,noise_error_max=1e-14)
    assert qualifies([row,row])
    for key,bad in [('count',479),('IQ_ratio',1.09),('action_ratio',1.09),('reference_unchanged',False),('noise_error_max',1e-9)]:
        assert not qualifies([row,dict(row,**{key:bad})])

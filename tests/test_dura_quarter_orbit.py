import pytest
import torch
from experiments.pamf_dura_efficiency.quarter_orbit import rotate_iq, quarter_orbit, decide


def test_rotation_group_and_inverse_preserve_each_sample_power():
    x=torch.randn(5,2,256,generator=torch.Generator().manual_seed(2))
    for a in range(4):
        assert torch.equal(rotate_iq(rotate_iq(x,a),-a),x)
        torch.testing.assert_close(rotate_iq(x,a).square().sum(1),x.square().sum(1))
        for b in range(4):assert torch.equal(rotate_iq(rotate_iq(x,a),b),rotate_iq(x,a+b))


def test_identity_and_constant_bias_cancellation_without_cross_sample_average():
    x=torch.randn(5,2,256,generator=torch.Generator().manual_seed(3))
    value,routes=quarter_orbit(lambda v:v+v.new_tensor([.3,-.2])[None,:,None],x)
    torch.testing.assert_close(value,x,atol=2e-7,rtol=2e-7)
    assert routes.shape==(4,5,2,256)
    for i in range(5):torch.testing.assert_close(value[i:i+1],quarter_orbit(lambda v:v+v.new_tensor([.3,-.2])[None,:,None],x[i:i+1])[0])


def test_nonlinear_orbit_is_equivariant_and_has_mean_route_error_decomposition():
    x=torch.randn(4,2,256,generator=torch.Generator().manual_seed(4));target=torch.randn_like(x)
    model=lambda v:v+.1*torch.tanh(2*v+v.flip(1))
    value,routes=quarter_orbit(model,x)
    for k in range(4):torch.testing.assert_close(quarter_orbit(model,rotate_iq(x,k))[0],rotate_iq(value,k),atol=3e-7,rtol=3e-7)
    total=(routes-target[None]).square().mean((0,2,3))
    pieces=(value-target).square().mean((1,2))+(routes-value[None]).square().mean((0,2,3))
    torch.testing.assert_close(total,pieces)


def test_two_cached_views_and_full_clean_preservation_are_required():
    cases=lambda iq,correct:dict(native=dict(iq_nmse=1.,teacher_correct=490),orbit=dict(iq_nmse=iq,teacher_correct=correct))
    groups=lambda:dict(all=dict(count=512,cases=cases(.9,496)),**{'scene:clean':dict(count=80,cases=cases(.9,490)),
      **{'scene:'+str(i):dict(count=72,cases=cases(.9,490)) for i in range(6)}})
    summary={'clean':dict(all=dict(count=512,cases=cases(.9,490))),'1':groups(),'20':groups()}
    assert decide(summary)['eligible']
    summary['clean']['all']['cases']['orbit']['teacher_correct']=489
    assert not decide(summary)['eligible']
    with pytest.raises(ValueError):decide({'1':groups()})

import pytest
import torch
from experiments.pamf_dura_efficiency.signature_cone import common_step,common_signature_steps
from tests.test_dura_signature_iq import Reader
from experiments.pamf_dura_efficiency.signature_iq import signature_features


def test_common_derivative_and_exact_finite_IQ_step_for_conflicting_gradients():
    torch.manual_seed(91);clean=torch.randn(50,2,8);value=clean+.2*torch.randn_like(clean)
    IQ=2*(value-clean)/(16*clean.square().mean((1,2))[:,None,None])
    gradient=-.8*IQ+.5*IQ.flatten(1).norm(dim=1)[:,None,None]*torch.randn_like(IQ)
    step,info=common_step(value,clean,torch.ones(50),gradient)
    assert (info['cosine']<0).any() and info['active'].all()
    assert (info['common_IQ_first_order']>=-1e-7).all() and (info['common_feature_first_order']>=-1e-7).all()
    assert (((value+step-clean)**2).sum((1,2))<=((value-clean)**2).sum((1,2))*(1+1e-6)).all()
    assert (gradient*step).sum((1,2)).max()<=1e-7


def test_zero_or_exactly_opposed_derivatives_do_not_force_a_step():
    clean=torch.ones(3,2,8);value=clean+.1;g=2*(value-clean)/16
    for loss,gradient in [(torch.ones(3),-g),(torch.ones(3),torch.zeros_like(g)),(torch.zeros(3),g)]:
        step,_=common_step(value,clean,loss,gradient);torch.testing.assert_close(step,torch.zeros_like(step),atol=0,rtol=0)
    step,_=common_step(clean,clean,torch.ones(3),g);torch.testing.assert_close(step,torch.zeros_like(step),atol=0,rtol=0)
    with pytest.raises(ValueError):common_step(value,clean,torch.ones(2),g)


def test_four_steps_keep_same_IQ_ball_and_anchor_error_without_F_updates():
    torch.manual_seed(92);reader=Reader().eval().requires_grad_(False);anchor=torch.randn(6,2,8)
    clean=anchor+.1*torch.randn_like(anchor);target=signature_features(reader,clean).detach()
    before={k:v.clone() for k,v in reader.state_dict().items()}
    value,history,info=common_signature_steps(reader,anchor,target,clean)
    assert history.shape==(4,6) and info['cosine'].shape==(4,6)
    assert ((value-anchor).square().sum((1,2))/anchor.square().sum((1,2))<=.00010001).all()
    assert ((value-clean).square().sum((1,2))<=(anchor-clean).square().sum((1,2))*(1+3e-5)+1e-10).all()
    assert all(p.grad is None for p in reader.parameters()) and all(torch.equal(v,before[k]) for k,v in reader.state_dict().items())

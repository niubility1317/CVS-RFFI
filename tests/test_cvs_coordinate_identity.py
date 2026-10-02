import math
import pytest
import torch
from experiments.cvs_coordinate_identity.model import build,VARIANTS,rotate
from experiments.cvs_synchronized_identity.model import build as synchronized_build


@pytest.mark.parametrize('variant',VARIANTS)
def test_alignment_with_retained_coordinate_reconstructs_original_even_across_branch(variant):
    model=build(variant).double();torch.manual_seed(4)
    x=torch.randn(3,2,256,dtype=torch.float64)
    for hz in (0.,500000.,800000.,1250000.):
        original=rotate(x,torch.full((3,),2*math.pi*hz/25000000,dtype=x.dtype))
        aligned,omega,_,_=model.coordinates(original)
        torch.testing.assert_close(rotate(aligned,omega),original,atol=2e-14,rtol=2e-14)
    assert model.contract()['whole_affine_phase_invariant'] is False


@pytest.mark.parametrize('variant',VARIANTS)
def test_initial_conditioner_exactly_preserves_synchronized_core_and_bound(variant):
    torch.manual_seed(7);model=build(variant).eval();x=torch.randn(2,2,256)
    old=synchronized_build('synchronized_equivariant' if variant=='coordinate_equivariant' else 'synchronized_gauge').eval()
    old.core.load_state_dict(model.core.state_dict(),strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model(x),old(x),atol=0,rtol=0)
        model.conditioner.weight.fill_(100);model.conditioner.bias.fill_(-10)
        d=model.diagnostics(x)
    assert .75<=d['gain_min']<=d['gain_max']<=1.25
    assert sum(p.numel() for p in model.conditioner.parameters())==640
    assert sum(p.numel() for p in model.parameters())==(203193 if variant=='coordinate_equivariant' else 164865)


@pytest.mark.parametrize('variant',VARIANTS)
def test_nonzero_conditioner_preserves_global_phase_and_packet_independence(variant):
    torch.manual_seed(11);model=build(variant).eval();model.conditioner.weight.data.normal_(0,.1)
    x=torch.randn(3,2,256)
    # Global phase uses a constant rotation, not a linear phase ramp.
    r,i=x.unbind(1);phi=.37;y=torch.stack((r*math.cos(phi)-i*math.sin(phi),r*math.sin(phi)+i*math.cos(phi)),1)
    with torch.no_grad():
        torch.testing.assert_close(model(x),model(y),atol=1e-4,rtol=1e-4)
        torch.testing.assert_close(model(x[:1]),model(x)[:1],atol=3e-5,rtol=3e-5)


@pytest.mark.parametrize('variant',VARIANTS)
def test_ce_trains_conditioner_and_zero_correlation_has_finite_input_gradient(variant):
    torch.manual_seed(19);model=build(variant).train();x=torch.randn(4,2,256,requires_grad=True)
    loss=torch.nn.functional.cross_entropy(model(x),torch.arange(4));loss.backward()
    assert model.conditioner.weight.grad is not None and model.conditioner.weight.grad.abs().sum()>0
    assert model.conditioner.bias.grad is not None and model.conditioner.bias.grad.abs().sum()>0
    assert torch.isfinite(x.grad).all() and torch.isfinite(loss)
    model.zero_grad(set_to_none=True)
    zero=torch.zeros(2,2,256,requires_grad=True);loss=torch.nn.functional.cross_entropy(model(zero),torch.tensor([0,1]));loss.backward()
    assert torch.isfinite(zero.grad).all()
    assert not any('domain' in name for name,_ in model.named_modules())

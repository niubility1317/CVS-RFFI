import copy
import torch
import pytest
from experiments.cvs_crossphase_identity.model import build,VARIANTS,CrossChannelPhaseReadout,crossphase_contract
from experiments.cvs_energy_identity.model import build as prior_build
from experiments.cvs_equivariant_identity.model import InvariantReadout

torch.set_num_threads(2)

def rotate(z,theta):
    r,i=z.unbind(1);c,s=torch.cos(theta),torch.sin(theta)
    return torch.stack((c*r-s*i,s*r+c*i),1)

def test_interchannel_phase_counterexample_old_readout_cannot_separate():
    z=torch.zeros(1,2,4,32);z[:,0]=1
    changed=z.clone();changed[:,:,1]=rotate(changed[:,:,1:2],torch.tensor(.8))[:,:,0]
    assert torch.allclose(InvariantReadout()(z),InvariantReadout()(changed),atol=1e-6,rtol=0)
    assert not torch.allclose(CrossChannelPhaseReadout()(z),CrossChannelPhaseReadout()(changed),atol=1e-3,rtol=0)

def test_common_phase_and_equal_readout_dimension():
    z=torch.randn(5,2,32,64);reader=CrossChannelPhaseReadout()
    assert reader(z).shape==InvariantReadout()(z).shape==(5,320)
    assert torch.allclose(reader(z),reader(rotate(z,torch.tensor(.73))),atol=1e-6,rtol=0)

@pytest.mark.parametrize('variant',VARIANTS)
def test_own_initial_states_and_parameter_count_unchanged(variant):
    prior='energy_equivariant' if variant=='crossphase_raw' else 'energy_half'
    torch.manual_seed(47);a=prior_build(prior)
    torch.manual_seed(47);b=build(variant)
    assert sum(p.numel() for p in b.parameters())==202553
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
    assert b.contract()==crossphase_contract(variant)
    b.core.readout=InvariantReadout()
    assert b.contract()['readout_active'] is False

@pytest.mark.parametrize('variant',VARIANTS)
def test_ce_and_all_parameter_gradients_finite_on_random_weak_zero_constant(variant):
    model=build(variant)
    for x in [torch.randn(3,2,256),1e-8*torch.randn(3,2,256),torch.zeros(3,2,256),torch.ones(3,2,256)]:
        x.requires_grad_();model.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(x),torch.tensor([0,1,2]));loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

@pytest.mark.parametrize('variant',VARIANTS)
def test_whole_model_phase_and_strict_state_roundtrip(variant):
    model=build(variant).eval();x=torch.randn(4,2,256)
    state=copy.deepcopy(model.state_dict());other=build(variant).eval();other.load_state_dict(state,strict=True)
    with torch.no_grad():
        scores=model(x);assert torch.equal(scores,other(x))
        assert torch.allclose(scores,model(rotate(x,torch.tensor(.53))),atol=5e-5,rtol=0)

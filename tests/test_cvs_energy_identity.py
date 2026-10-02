import copy
import pytest
import torch
import torch.nn.functional as F
from experiments.cvs_equivariant_identity.model import RadialBlock,build as original_build,rotate_pair
from experiments.cvs_energy_identity.model import GlobalEnergyBlock,build,VARIANTS

def test_relative_channel_power_survives_global_normalization():
    b=RadialBlock(2,2,1)
    with torch.no_grad():
        b.conv.weight_real.zero_();b.conv.weight_imag.zero_()
        b.conv.weight_real[0,0,0]=1;b.conv.weight_real[1,1,0]=1
    global_block=GlobalEnergyBlock(copy.deepcopy(b))
    z=torch.zeros(1,2,2,32);z[:,0,0]=1;z[:,0,1]=2
    original=b(z).square().sum(1).mean(-1);preserved=global_block(z).square().sum(1).mean(-1)
    assert float((original[0,1]/original[0,0]).detach())==pytest.approx(1.,abs=2e-6)
    assert float((preserved[0,1]/preserved[0,0]).detach())==pytest.approx(4.,abs=2e-6)
    rotated=rotate_pair(z,z.new_tensor([1.2]))
    assert torch.allclose(global_block(rotated),rotate_pair(global_block(z),z.new_tensor([1.2])),atol=1e-6,rtol=1e-6)

def test_exact_scratch_parameters_and_initial_states_without_extra_rng():
    torch.manual_seed(61);old=original_build('equivariant_memory')
    torch.manual_seed(61);new=build('energy_equivariant')
    assert sum(p.numel() for p in old.parameters())==sum(p.numel() for p in new.parameters())==202553
    assert list(old.state_dict())==list(new.core.state_dict())
    assert all(torch.equal(v,new.core.state_dict()[k]) for k,v in old.state_dict().items())
    assert new.contract()['new_trainable_parameters']==0

@pytest.mark.parametrize('kind',['random','weak','zero','constant'])
@pytest.mark.parametrize('variant',VARIANTS)
def test_finite_ce_all_parameter_and_input_gradients(kind,variant):
    torch.set_num_threads(2);torch.manual_seed(8);m=build(variant)
    x=torch.randn(4,2,256)
    if kind=='weak':x*=1e-9
    elif kind=='zero':x.zero_()
    elif kind=='constant':x.fill_(.2)
    x.requires_grad_(True);loss=F.cross_entropy(m(x),torch.arange(4));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

@pytest.mark.parametrize('variant',VARIANTS)
def test_common_phase_property_and_strict_state_roundtrip(variant):
    torch.set_num_threads(2);torch.manual_seed(18);m=build(variant).eval();x=torch.randn(5,2,256)
    restored=build(variant).eval();restored.load_state_dict(m.state_dict(),strict=True)
    with torch.no_grad():
        reference=m(x)
        assert torch.equal(reference,restored(x))
        for theta in (.37,-1.2,2.9):assert torch.allclose(m(rotate_pair(x,x.new_full((5,),theta))),reference,atol=2e-4,rtol=2e-5)

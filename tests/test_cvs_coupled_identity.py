"""Public algebra/synthetic IQ only; no formal dataset, weight or target evidence."""
import torch
import pytest
from experiments.cvs_coupled_identity.model import VARIANTS,build,coupled_basis_from_clipped,delay,coupled_contract
from experiments.cvs_energy_identity.model import build as control
from experiments.cvs_equivariant_identity.model import behavior_basis,rotate_pair

torch.set_num_threads(2)


@pytest.mark.parametrize('variant',VARIANTS)
def test_same_initial_state_dimensions_and_parameters(variant):
    torch.manual_seed(391);base=control('energy_equivariant')
    torch.manual_seed(391);new=build(variant)
    assert sum(p.numel() for p in new.parameters())==sum(p.numel() for p in base.parameters())==202553
    assert new.state_dict().keys()==base.state_dict().keys()
    assert all(torch.equal(a,base.state_dict()[k]) for k,a in new.state_dict().items())
    assert new.core.readout.__class__==base.core.readout.__class__
    assert new.core.behavior[0].conv.cin==12
    assert not new.contract()['prototype_only'] and new.contract()['coupled_lift_active']
    assert new.contract()==coupled_contract(variant)
    new.core.behavior[0].envelope_lag=4 if variant=='coupled_lag1' else 1
    assert new.contract()!=coupled_contract(variant)


@pytest.mark.parametrize('lag',[1,4])
def test_explicit_polynomial_and_causal_envelope(lag):
    z=torch.tensor([[[1.,2.,3.,1.,2.,1.,.5,1.],[.5,1.,.2,.5,1.,.2,.5,1.]]],dtype=torch.float64)
    phi=coupled_basis_from_clipped(z,lag);power=z.square().sum(1)
    assert phi.shape==(1,2,12,8)
    for m in range(4):
        v=delay(z,m);a=delay(power,m)/4;b=delay(power,m+lag)/4
        assert torch.allclose(phi[:,:,3*m+1],v*(a+b)[:,None]/2)
        assert torch.allclose(phi[:,:,3*m+2],v*(a.square()+2*a*b+b.square())[:,None]/4)
    changed=z.clone();changed[...,5:]=17.
    assert torch.equal(phi[...,:5],coupled_basis_from_clipped(changed,lag)[...,:5])


def test_cross_memory_monomial_not_in_aligned_polynomial_lift_span():
    # Only the lift's linear span is addressed; the old full neural network
    # can express additional interactions after its nonlinear layers.
    g=torch.Generator().manual_seed(47)
    a,b=torch.rand(64,2,generator=g,dtype=torch.float64).mul(1.7).add(.1).unbind(1)
    old=torch.stack([a,b,a**3,b**3,a**5,b**5],1)
    cross=a*(a.square()+b.square())/8
    fit=old@torch.linalg.lstsq(old,cross).solution
    assert float((fit-cross).norm()/cross.norm())>.01


@pytest.mark.parametrize('variant',VARIANTS)
def test_common_phase_but_no_frequency_invariance_claim(variant):
    torch.manual_seed(51);m=build(variant).eval();x=torch.randn(3,2,256)
    phase=torch.tensor(.71);rot=rotate_pair(x,phase)
    lag=m.core.behavior[0].envelope_lag
    z=behavior_basis(x)[:,:,0];zr=behavior_basis(rot)[:,:,0]
    assert torch.allclose(coupled_basis_from_clipped(zr,lag),rotate_pair(coupled_basis_from_clipped(z,lag),phase),atol=2e-5,rtol=2e-5)
    with torch.no_grad():assert torch.allclose(m(x),m(rot),atol=1e-3,rtol=1e-4)
    assert m.contract()['whole_affine_phase_invariant'] is False


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('kind',['normal','weak','zero','constant'])
def test_ce_gradients_degenerate_inputs_and_state_roundtrip(variant,kind):
    torch.manual_seed(73);m=build(variant)
    x=torch.randn(3,2,256)
    if kind=='weak':x=x*1e-8
    if kind=='zero':x=x*0
    if kind=='constant':x=torch.ones_like(x)
    out=m(x);loss=torch.nn.functional.cross_entropy(out,torch.tensor([0,2,5]));loss.backward()
    assert out.shape==(3,6) and torch.isfinite(out).all() and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    clone=build(variant);clone.load_state_dict(m.state_dict(),strict=True)
    m.eval();clone.eval()
    with torch.no_grad():assert torch.equal(m(x),clone(x))

@pytest.mark.parametrize('variant',VARIANTS)
def test_actual_behavior_conv_input_diagnostic_does_not_update_state(variant):
    model=build(variant).eval();state={k:v.clone() for k,v in model.state_dict().items()}
    x=torch.randn(4,2,256);diagnostic=model.input_diagnostics(x)
    assert diagnostic['active'] and len(diagnostic['records'])==1
    record=diagnostic['records'][0]
    assert record['actual_envelope_lag']==coupled_contract(variant)['envelope_lag']
    assert record['packets']==4 and record['complex_terms']==12 and record['input_formula_max_abs_error']==0.
    assert record['degree3_relative_input_change_mean']>0 and record['degree5_relative_input_change_mean']>0
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert not model.core.behavior[0].conv._forward_hooks

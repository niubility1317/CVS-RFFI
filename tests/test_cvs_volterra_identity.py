"""Public synthetic IQ only: algebra, RF symmetry and matched model budget."""
import copy
import pytest
import torch
import torch.nn.functional as F
from experiments.cvs_coupled_identity.model import CoupledCVS,coupled_basis_from_clipped
from experiments.cvs_volterra_identity.model import build,VARIANTS,volterra_basis_from_clipped,volterra_contract


def rotate(z,phase):
    return torch.stack((z[:,0]*phase.cos()-z[:,1]*phase.sin(),z[:,0]*phase.sin()+z[:,1]*phase.cos()),1)


@pytest.mark.parametrize('lag',[1,4])
def test_independent_complex_formula(lag):
    torch.manual_seed(101);z=torch.randn(3,2,32,dtype=torch.float64)
    actual=volterra_basis_from_clipped(z,lag)
    w=torch.complex(z[:,0],z[:,1]);p=w.abs().square();expected=torch.zeros_like(actual)
    for n in range(32):
        for m in range(4):
            t=n-m
            if t<0:continue
            env=(p[:,t]+(p[:,t-4] if t>=4 else 0))/8
            cubic=w[:,t-lag].square()*w[:,t-2*lag].conj()/4 if t>=2*lag else torch.zeros(3,dtype=torch.complex128)
            terms=[w[:,t],.5*(w[:,t]*env+cubic),.5*(w[:,t]*env.square()+cubic*env)]
            for q,value in enumerate(terms):expected[:,:,3*m+q,n]=torch.stack((value.real,value.imag),1)
    torch.testing.assert_close(actual,expected,atol=2e-14,rtol=2e-14)


@pytest.mark.parametrize('lag',[1,4])
def test_affine_phase_covariance_of_lift_only(lag):
    torch.manual_seed(41);z=torch.randn(2,2,48,dtype=torch.float64)
    phase=.41+.19*torch.arange(48,dtype=z.dtype)
    a=volterra_basis_from_clipped(rotate(z,phase),lag)
    b=volterra_basis_from_clipped(z,lag)
    for m in range(4):
        # Affine delay-balance l+l-2l=0 matches the original z[n-m].
        expected=rotate(b[:,:,3*m:3*m+3].flatten(2,3),(.41+.19*(torch.arange(48,dtype=z.dtype)-m)).repeat(3))
        torch.testing.assert_close(a[:,:,3*m:3*m+3].flatten(2,3),expected,atol=4e-14,rtol=4e-14)


@pytest.mark.parametrize('lag',[1,4])
def test_causal_padding_and_no_future_access(lag):
    z=torch.randn(2,2,32,dtype=torch.float64);changed=z.clone();changed[...,18:]+=5
    a=volterra_basis_from_clipped(z,lag);b=volterra_basis_from_clipped(changed,lag)
    torch.testing.assert_close(a[...,:18],b[...,:18],rtol=0,atol=0)
    for m in range(4):assert torch.count_nonzero(a[:,:,3*m:3*m+3,:m])==0


@pytest.mark.parametrize('lag',[1,4])
def test_cross_phase_information_beyond_envelope_only_at_current_delay(lag):
    # Same amplitudes and current z, different history phase. Old delay0
    # envelope terms are identical; the new cubic differs. This is a local
    # lift witness, not inability of the old full nonlinear CNN to express it.
    z=torch.zeros(1,2,24,dtype=torch.float64);z[:,0]=1
    changed=z.clone();changed[:,0,16-lag]=0;changed[:,1,16-lag]=1
    old=coupled_basis_from_clipped(z,4);old_changed=coupled_basis_from_clipped(changed,4)
    torch.testing.assert_close(old[:,:,:3,16],old_changed[:,:,:3,16],atol=0,rtol=0)
    a=volterra_basis_from_clipped(z,lag);b=volterra_basis_from_clipped(changed,lag)
    assert not torch.allclose(a[:,:,1,16],b[:,:,1,16])


@pytest.mark.parametrize('variant',VARIANTS)
def test_matched_scratch_state_and_parameter_budget(variant):
    torch.manual_seed(99);control=CoupledCVS('coupled_lag4')
    torch.manual_seed(99);model=build(variant)
    assert sum(p.numel() for p in model.parameters())==202553
    assert model.state_dict().keys()==control.state_dict().keys()
    for k,v in control.state_dict().items():torch.testing.assert_close(model.state_dict()[k],v,atol=0,rtol=0)
    assert model.contract()==volterra_contract(variant)
    assert model.contract()['hardware_parameter_recovery'] is False
    assert model.contract()['whole_affine_phase_invariant'] is False


@pytest.mark.parametrize('variant',VARIANTS)
def test_ce_all_parameter_gradients_roundtrip_and_batch_independence(variant):
    torch.manual_seed(28);model=build(variant);x=torch.randn(3,2,256)
    loss=F.cross_entropy(model(x),torch.tensor([0,2,5]));loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval();state=copy.deepcopy(model.state_dict());new=build(variant);new.load_state_dict(state,strict=True);new.eval()
    with torch.no_grad():
        a=model(x);torch.testing.assert_close(new(x),a)
        torch.testing.assert_close(model(x[[2,0,1]])[[1,2,0]],a,atol=3e-5,rtol=3e-5)
        torch.testing.assert_close(model(x[:1]),a[:1],atol=4e-5,rtol=4e-5)
    assert all(torch.equal(state[k],model.state_dict()[k]) for k in state)


@pytest.mark.parametrize('variant',VARIANTS)
def test_whole_constant_phase_property_and_actual_input_diagnostics(variant):
    # The unchanged native Sinc path explicitly computes in FP32 even after
    # model.double(): adv3b02_xuc/code/model.py:251,296-310. Whole-network
    # tolerance follows the actual retained FP32 protocol; lift algebra above
    # is independently checked in FP64 without a Sinc path.
    torch.manual_seed(200);model=build(variant).eval();x=torch.randn(2,2,256)
    with torch.no_grad():
        a=model(x);b=model(rotate(x,torch.tensor(.57,dtype=x.dtype)))
        torch.testing.assert_close(a,b,atol=1e-3,rtol=1e-4)
    d=model.input_diagnostics(x)
    assert d['active'] and len(d['records'])==1
    assert d['records'][0]['input_formula_max_abs_error']==0
    assert d['records'][0]['actual_phase_lag']==volterra_contract(variant)['phase_lag']
    assert d['records'][0]['degree3_relative_input_change_mean']>0


@pytest.mark.parametrize('lag',[1,4])
def test_zero_and_short_history_are_finite(lag):
    for n in (1,3,8,256):
        z=torch.zeros(2,2,n,requires_grad=True);y=volterra_basis_from_clipped(z,lag)
        assert y.shape==(2,2,12,n) and torch.isfinite(y).all()
        y.sum().backward();assert torch.isfinite(z.grad).all()


def test_invalid_input_and_variant_rejected():
    with pytest.raises(ValueError):build('coupled_lag4')
    with pytest.raises(ValueError):volterra_basis_from_clipped(torch.zeros(2,3,256),1)
    with pytest.raises(ValueError):volterra_basis_from_clipped(torch.zeros(2,2,256),2)

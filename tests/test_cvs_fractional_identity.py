import copy,io,math
import pytest
import torch
import torch.nn.functional as F
from experiments.cvs_fractional_identity.model import build,VARIANTS
from experiments.cvs_coordinate_identity.model import rotate

@pytest.fixture(autouse=True)
def cpu_threads():torch.set_num_threads(2)

def repeated_signal(batch=2):
    n=torch.arange(256,dtype=torch.float32);k=n.remainder(20)
    amp=1+.25*torch.cos(2*torch.pi*k/20)
    phase=.002*n+.3*torch.sin(2*torch.pi*k/20)
    return torch.stack((amp*phase.cos(),amp*phase.sin()),0)[None].repeat(batch,1,1)

def test_equal_initial_output_and_only_one_extra_parameter():
    torch.manual_seed(17);a=build(VARIANTS[0]).eval()
    torch.manual_seed(17);b=build(VARIANTS[1]).eval()
    x=torch.randn(3,2,256)
    assert torch.equal(a(x),b(x))
    assert sum(p.numel() for p in a.parameters())==202553
    assert sum(p.numel() for p in b.parameters())==202554
    assert float(a.alignment_strength().detach())==float(b.alignment_strength().detach())==.5

@pytest.mark.parametrize('variant',VARIANTS)
def test_amplitudes_reconstruction_and_residual_frequency_covariance(variant):
    model=build(variant).eval();x=repeated_signal()
    y,omega,valid,_,alpha=model.coordinates(x)
    assert valid.all()
    assert torch.allclose(y.square().sum(1),x.square().sum(1),atol=8e-7,rtol=1e-6)
    assert torch.allclose(rotate(y,alpha*omega),x,atol=6e-7,rtol=1e-6)
    nu=torch.full((len(x),),2*math.pi*80000/25000000)
    z=model.coordinates(rotate(x,nu))[0]
    assert torch.allclose(z,rotate(y,(1-alpha)*nu),atol=3e-6,rtol=1e-6)
    observed=model.synchronizer.estimate(y)[0]
    assert torch.allclose(observed,(1-alpha)*omega,atol=1e-7,rtol=1e-5)
    assert model.contract()['whole_affine_phase_invariant'] is False
    assert model.contract()['hardware_parameter_estimation'] is False

@pytest.mark.parametrize('variant',VARIANTS)
def test_whole_constant_phase_and_strict_roundtrip(variant):
    torch.manual_seed(18);m=build(variant).eval();x=torch.randn(3,2,256)
    if variant==VARIANTS[1]:
        with torch.no_grad():m.alignment_logit.fill_(.7)
    theta=.731;r,i=x.unbind(1)
    z=torch.stack((r*math.cos(theta)-i*math.sin(theta),r*math.sin(theta)+i*math.cos(theta)),1)
    assert torch.allclose(m(x),m(z),atol=2e-4,rtol=1e-5)
    b=io.BytesIO();torch.save(m.state_dict(),b);b.seek(0)
    other=build(variant).eval();other.load_state_dict(torch.load(b,weights_only=False),strict=True)
    assert torch.equal(m(x),other(x))

@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('kind',['zero','weak','cancelled','random'])
def test_finite_plain_ce_and_all_parameter_gradients(variant,kind):
    torch.manual_seed(19);m=build(variant);x=torch.randn(2,2,256)
    if kind=='zero':x.zero_()
    if kind=='weak':x*=1e-9;x[0].zero_()
    if kind=='cancelled':x[:,:,80:160]=0
    x.requires_grad_();loss=F.cross_entropy(m(x),torch.tensor([0,1]));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    if kind in ('zero','weak','cancelled'):
        assert torch.equal(m.coordinates(x)[1],torch.zeros(2))
    assert all(math.isfinite(v) for v in m.diagnostics(x).values() if isinstance(v,(float,int)))

def test_alignment_strength_has_actual_nonzero_plain_ce_gradient():
    torch.manual_seed(20);m=build(VARIANTS[1]);x=repeated_signal()
    loss=F.cross_entropy(m(x),torch.tensor([0,1]));loss.backward()
    assert torch.isfinite(m.alignment_logit.grad) and float(m.alignment_logit.grad.abs())>1e-8

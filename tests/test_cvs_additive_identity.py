import io,math
import pytest
import torch
import torch.nn.functional as F
from experiments.cvs_additive_identity.model import build,VARIANTS,PARENTS,additive_contract
from experiments.cvs_coordinate_identity.model import build as parent_build,rotate

@pytest.fixture(autouse=True)
def cpu_threads():torch.set_num_threads(2)

@pytest.mark.parametrize('variant',VARIANTS)
def test_zero_init_exact_core_equivalence_and_count(variant):
    torch.manual_seed(19);m=build(variant).eval();x=torch.randn(3,2,256)
    p=parent_build(PARENTS[variant]).eval();p.load_state_dict(m.state_dict(),strict=True)
    aligned,_,_,_=m.coordinates(x)
    assert torch.equal(m.features(x),m.core.features(aligned))
    assert torch.equal(m(x),p(x))
    assert sum(a.numel() for a in m.parameters())==({'additive_equivariant':203193,'additive_gauge':164865}[variant])
    assert additive_contract(variant)['conditioner_parameters']==640
    assert 'gain' not in m.contract()

@pytest.mark.parametrize('variant',VARIANTS)
def test_coordinate_can_create_zero_component_and_bound(variant):
    m=build(variant);u=torch.zeros(2,160);u[:,0]=2
    with torch.no_grad():m.conditioner.weight.fill_(.5);m.conditioner.bias.zero_()
    d=torch.tensor([[0.,0.,0.],[-1.,.7,-.2]])
    f,delta,scale=m.inject(u,d)
    assert torch.equal(f[0],u[0])
    assert f[1,1]!=0 and u[1,1]==0
    assert (delta.abs()<=.25*scale+1e-7).all()
    assert (delta.norm(dim=1)<=.25*math.sqrt(160)*scale.squeeze()+1e-6).all()
    f.sum().backward();assert torch.isfinite(m.conditioner.weight.grad).all()

@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('kind',['zero','random','cancelled_correlation'])
def test_finite_plain_ce_gradient_and_roundtrip(variant,kind):
    torch.manual_seed(21);m=build(variant);x=torch.randn(2,2,256)
    if kind=='zero':x.zero_()
    if kind=='cancelled_correlation':x[:,:,80:160]=0
    with torch.no_grad():m.conditioner.weight.normal_(0,.1);m.conditioner.bias.normal_(0,.1)
    x.requires_grad_();loss=F.cross_entropy(m(x),torch.tensor([0,1]));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    m.eval();b=io.BytesIO();torch.save(m.state_dict(),b);b.seek(0)
    other=build(variant).eval();other.load_state_dict(torch.load(b,weights_only=False),strict=True)
    assert torch.equal(m(x.detach()),other(x.detach()))
    assert all(math.isfinite(v) for v in m.diagnostics(x.detach()).values() if isinstance(v,(float,int)))

@pytest.mark.parametrize('variant',VARIANTS)
def test_whole_constant_phase_with_active_coordinate(variant):
    torch.manual_seed(22);m=build(variant).eval();x=torch.randn(3,2,256)
    with torch.no_grad():m.conditioner.weight.normal_(0,.1);m.conditioner.bias.normal_(0,.1)
    theta=torch.full((3,),.37)
    # rotate's omega multiplies sample index; constant phase applied directly.
    c,s=theta.cos()[:,None],theta.sin()[:,None];r,i=x.unbind(1)
    z=torch.stack((r*c-i*s,r*s+i*c),1)
    assert torch.allclose(m(x),m(z),atol=2e-4,rtol=1e-5)
    assert m.contract()['whole_affine_phase_invariant'] is False
    assert m.contract()['hardware_parameter_estimation'] is False

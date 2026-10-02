"""Public algebra and scratch model checks; no dataset or checkpoint inputs."""
import copy,math
import pytest,torch
from experiments.cvs_coupled_identity.model import build as control_build,coupled_basis_from_clipped
from experiments.cvs_adaptive_volterra_identity.model import VARIANTS,build,adaptive_basis_from_clipped,adaptive_contract

@pytest.mark.parametrize('lag',[1,4])
@pytest.mark.parametrize('raw',[[0.,0.],[.2,-.3],[2.,-2.]])
def test_independent_complex_formula(lag,raw):
    torch.manual_seed(5);z=torch.randn(3,2,25,dtype=torch.float64)
    got=adaptive_basis_from_clipped(z,lag,z.new_tensor(raw));x=torch.complex(z[:,0],z[:,1]);expected=torch.zeros(3,12,25,dtype=torch.complex128)
    for n in range(25):
        for m in range(4):
            t=n-m
            if t<0:continue
            p=(x[:,t].abs().square()+(x[:,t-4].abs().square() if t>=4 else 0.))/8
            c=x[:,t-lag].square()*x[:,t-2*lag].conj()/4 if t>=2*lag else torch.zeros(3,dtype=torch.complex128)
            expected[:,3*m,t+m]=x[:,t]
            expected[:,3*m+1,t+m]=x[:,t]*p+math.tanh(raw[0])*(c-x[:,t]*p)
            expected[:,3*m+2,t+m]=x[:,t]*p.square()+math.tanh(raw[1])*(c*p-x[:,t]*p.square())
    torch.testing.assert_close(torch.complex(got[:,0],got[:,1]),expected,rtol=2e-13,atol=2e-13)

@pytest.mark.parametrize('lag',[1,4])
def test_zero_gate_exactly_retains_control_and_input_gradient(lag):
    torch.manual_seed(7);z=torch.randn(2,2,31,dtype=torch.float64,requires_grad=True)
    raw=torch.zeros(2,dtype=torch.float64,requires_grad=True)
    actual=adaptive_basis_from_clipped(z,lag,raw);expected=coupled_basis_from_clipped(z,4)
    assert torch.equal(actual,expected)
    weight=torch.randn_like(actual)
    da=torch.autograd.grad((actual*weight).sum(),z,retain_graph=True)[0]
    de=torch.autograd.grad((expected*weight).sum(),z,retain_graph=True)[0]
    torch.testing.assert_close(da,de,rtol=0,atol=0)
    gate_grad=torch.autograd.grad((actual*weight).sum(),raw)[0]
    assert torch.isfinite(gate_grad).all() and torch.all(gate_grad.abs()>1e-8)

@pytest.mark.parametrize('lag',[1,4])
def test_affine_phase_covariance_and_causality(lag):
    torch.manual_seed(6);z=torch.randn(2,2,30,dtype=torch.float64);raw=z.new_tensor([.3,-.2])
    phi=.7+.021*torch.arange(30,dtype=z.dtype);x=torch.complex(z[:,0],z[:,1]);rot=x*torch.exp(1j*phi)
    out=adaptive_basis_from_clipped(z,lag,raw);other=adaptive_basis_from_clipped(torch.stack([rot.real,rot.imag],1),lag,raw)
    for m in range(4):
        expected=torch.complex(out[:,0,3*m:3*m+3],out[:,1,3*m:3*m+3])*torch.exp(1j*(phi-.021*m))
        torch.testing.assert_close(torch.complex(other[:,0,3*m:3*m+3],other[:,1,3*m:3*m+3]),expected,rtol=2e-13,atol=2e-13)
    changed=z.clone();changed[:,:,20:]+=50
    assert torch.equal(out[...,:20],adaptive_basis_from_clipped(changed,lag,raw)[...,:20])

@pytest.mark.parametrize('v',VARIANTS)
def test_scratch_initial_state_and_logits_match_control(v):
    torch.set_num_threads(2);torch.manual_seed(43);baseline=control_build('coupled_lag4').eval()
    torch.manual_seed(43);model=build(v).eval()
    a=baseline.state_dict();b=model.state_dict();extra=set(b)-set(a)
    assert extra=={'core.behavior.0.mix_raw'} and all(torch.equal(a[k],b[k]) for k in a)
    assert torch.equal(b['core.behavior.0.mix_raw'],torch.zeros(2))
    assert sum(p.numel() for p in model.parameters())==202555
    x=torch.randn(3,2,256)
    with torch.no_grad():assert torch.equal(model(x),baseline(x))
    assert model.contract()==adaptive_contract(v)

@pytest.mark.parametrize('v',VARIANTS)
def test_ce_updates_both_gates_and_all_parameters(v):
    torch.set_num_threads(2);torch.manual_seed(123);model=build(v).train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001)
    loss=torch.nn.functional.cross_entropy(model(torch.randn(5,2,256)),torch.tensor([0,1,2,3,4]));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    g=model.core.behavior[0].mix_raw.grad;assert torch.all(g.abs()>1e-10)
    optimizer.step();assert torch.all(model.core.behavior[0].mix_raw.abs()>0)
    d=model.diagnostics(torch.randn(3,2,256))['adaptive_input']
    assert len(d['records'])==1 and d['records'][0]['input_formula_max_abs_error']==0.
    assert d['records'][0]['actual_envelope_lag']==4 and all(abs(x)<1 for x in d['coefficients'])

@pytest.mark.parametrize('v',VARIANTS)
def test_trained_gate_phase_batch_state_and_roundtrip(v):
    torch.set_num_threads(2);torch.manual_seed(2);model=build(v).eval()
    with torch.no_grad():model.core.behavior[0].mix_raw.copy_(torch.tensor([.2,-.1]))
    state=copy.deepcopy(model.state_dict());x=torch.randn(4,2,256);z=torch.complex(x[:,0],x[:,1])*torch.exp(torch.tensor(.37j))
    with torch.no_grad():
        reference=model(x);torch.testing.assert_close(model(torch.stack([z.real,z.imag],1)),reference,rtol=1e-4,atol=1e-3)
        torch.testing.assert_close(model(x.flip(0)).flip(0),reference,rtol=1e-5,atol=1e-5)
    assert all(torch.equal(state[k],model.state_dict()[k]) for k in state)
    other=build(v).eval();other.load_state_dict(state,strict=True)
    with torch.no_grad():assert torch.equal(other(x),reference)

@pytest.mark.parametrize('lag',[1,4])
def test_weak_and_short_inputs_finite(lag):
    for length in (1,3,9):
        z=torch.zeros(2,2,length,requires_grad=True);raw=torch.zeros(2,requires_grad=True)
        out=adaptive_basis_from_clipped(z,lag,raw);out.sum().backward()
        assert torch.isfinite(out).all() and torch.isfinite(z.grad).all() and torch.isfinite(raw.grad).all()

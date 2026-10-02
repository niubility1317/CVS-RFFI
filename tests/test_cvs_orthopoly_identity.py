import math
import pytest
import torch
from experiments.cvs_orthopoly_identity.model import packet_basis,basis_diagnostics,build,VARIANTS,orthopoly_contract
from experiments.cvs_coupled_identity.model import delay

@pytest.mark.parametrize('lag',[0,4])
def test_packet_inner_products_reconstruction_and_raw_retention(lag):
    torch.manual_seed(121);z=torch.randn(7,2,256,dtype=torch.float64)
    d=basis_diagnostics(z,lag)
    assert len(d['records'])==4
    for row in d['records']:
        assert row['eligible_packets']==7
        assert row['raw_order1_max_abs_error']==0
        assert row['reconstruction_max_abs_error']<2e-13
        assert row['orthogonal_normalized_gram_offdiag_max']<1e-13
        assert row['original_normalized_gram_offdiag_mean']>.5

@pytest.mark.parametrize('lag',[0,4])
def test_same_packet_only_and_no_grad_detach(lag):
    torch.manual_seed(22);z=torch.randn(4,2,256,requires_grad=True)
    alone=packet_basis(z[:1],lag);together=packet_basis(z,lag)
    assert torch.equal(alone,together[:1])
    together.square().mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum()>0

@pytest.mark.parametrize('lag',[0,4])
@pytest.mark.parametrize('omega',[0.,.08])
def test_phase_covariance_respects_slot_delay(lag,omega):
    torch.manual_seed(44);z=torch.randn(3,2,256,dtype=torch.float64)
    def rotate(z,a):
        return torch.stack((z[:,0]*a.cos()-z[:,1]*a.sin(),z[:,0]*a.sin()+z[:,1]*a.cos()),1)
    angle=.37+omega*torch.arange(256,dtype=z.dtype)
    original=packet_basis(z,lag);other=packet_basis(rotate(z,angle),lag)
    for m in range(4):
        for order in range(3):
            expected=rotate(original[:,:,3*m+order],angle-omega*m)
            assert (other[:,:,3*m+order]-expected).abs().max()<2e-12

@pytest.mark.parametrize('lag',[0,4])
@pytest.mark.parametrize('mode',['zero','constant','tiny'])
def test_degenerate_inputs_finite_and_reconstructable(lag,mode):
    z=torch.zeros(2,2,256,dtype=torch.float64);z[:,0]=0 if mode=='zero' else 1 if mode=='constant' else 1e-15
    basis=packet_basis(z,lag);d=basis_diagnostics(z,lag)
    assert torch.isfinite(basis).all()
    assert torch.equal(basis[:,:,0],z)
    assert all(r['reconstruction_max_abs_error']<1e-12 for r in d['records'])

@pytest.mark.parametrize('variant',VARIANTS)
def test_real_model_ce_all_parameters_actual_input_and_read_only(variant):
    torch.set_num_threads(2);torch.manual_seed(92);model=build(variant)
    assert sum(p.numel() for p in model.parameters())==202553
    assert model.contract()==orthopoly_contract(variant)
    x=torch.randn(4,2,256);y=torch.tensor([0,1,2,3])
    logits=model(x);assert logits.shape==(4,6) and torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits,y).backward()
    assert sum(p.numel() for p in model.parameters() if p.grad is not None)==202553
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    state={k:v.clone() for k,v in model.state_dict().items()};d=model.input_diagnostics(x)
    assert d['active'] and len(d['records'])==1 and d['records'][0]['input_formula_max_abs_error']==0
    assert len(d['records'][0]['records'])==4
    assert all(torch.equal(state[k],v) for k,v in model.state_dict().items())

def test_invalid_shape_lag_and_variant():
    with pytest.raises(ValueError):packet_basis(torch.zeros(2,3,256),0)
    with pytest.raises(ValueError):packet_basis(torch.zeros(2,2,256),1)
    with pytest.raises(ValueError):build('unregistered')

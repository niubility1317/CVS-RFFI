import torch
import pytest
from torch.nn import functional as F
from experiments.cvs_equivariant_identity.model import ComplexConv,RadialBlock,InvariantReadout,behavior_basis,rotate_pair,build

torch.set_num_threads(2)


def test_complex_conv_independent_scalar_sum():
    conv=ComplexConv(2,3,3).double();x=torch.randn(2,2,2,9,dtype=torch.float64)
    actual=conv(x);expected=torch.zeros_like(actual)
    for batch in range(2):
        for out in range(3):
            for n in range(9):
                value=0j
                for cin in range(2):
                    for k in range(3):
                        pos=n+k-1
                        if 0<=pos<9:
                            value+=complex(conv.weight_real[out,cin,k].detach(),conv.weight_imag[out,cin,k].detach())*complex(x[batch,0,cin,pos],x[batch,1,cin,pos])
                expected[batch,0,out,n]=value.real;expected[batch,1,out,n]=value.imag
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)


def test_block_equivariance_and_readout_invariance():
    block=RadialBlock(2,3,5).double();x=torch.randn(4,2,2,31,dtype=torch.float64);theta=torch.tensor([.3,-1.,2.1,-2.8],dtype=torch.float64)
    block.gate_a.data.fill_(.7);block.gate_b.data.fill_(-.4)
    a=block(x);b=block(rotate_pair(x,theta))
    torch.testing.assert_close(b,rotate_pair(a,theta),rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(InvariantReadout()(a),InvariantReadout()(b),rtol=1e-10,atol=1e-10)


def test_memory_basis_causal_formula_and_phase():
    x=torch.randn(3,2,23,dtype=torch.float64)*.2;actual=behavior_basis(x)
    for delay in range(4):
        for j,order in enumerate((1,3,5)):
            expected=torch.zeros_like(x)
            for n in range(delay,23):
                r=x[:,0,n-delay];i=x[:,1,n-delay];scale=((r*r+i*i)/4).pow((order-1)//2)
                expected[:,:,n]=x[:,:,n-delay]*scale[:,None]
            torch.testing.assert_close(actual[:,:,delay*3+j],expected,rtol=1e-12,atol=1e-12)
    theta=x.new_tensor([.4,-.8,2.])
    torch.testing.assert_close(behavior_basis(rotate_pair(x,theta)),rotate_pair(actual,theta),rtol=1e-12,atol=1e-12)
    tail=x.clone();tail[...,15:]+=1
    torch.testing.assert_close(behavior_basis(x)[...,:15],behavior_basis(tail)[...,:15])


def test_causal_complex_filter_no_future_leak():
    conv=ComplexConv(2,3,5,dilation=2,causal=True).double();x=torch.randn(2,2,2,41,dtype=torch.float64)
    tail=x.clone();tail[...,24:]+=5
    torch.testing.assert_close(conv(x)[...,:24],conv(tail)[...,:24],rtol=0,atol=0)


def test_readout_retains_relative_phase_not_only_magnitude():
    n=torch.arange(64,dtype=torch.float64);z=torch.stack([torch.cos(.13*n),torch.sin(.13*n)],0)[None,:,None]
    shifted=torch.stack([torch.cos(.27*n),torch.sin(.27*n)],0)[None,:,None]
    read=InvariantReadout()
    assert (read(z)-read(shifted)).norm()>.1


def test_whole_all_paths_phase_invariant_and_packet_separable():
    torch.manual_seed(5);model=build('equivariant_memory').eval();x=torch.randn(5,2,256)
    theta=x.new_tensor([.37,-1.2,2.9,-.8,1.7])
    with torch.no_grad():
        torch.testing.assert_close(model(x),model(rotate_pair(x,theta)),rtol=2e-4,atol=2e-4)
        torch.testing.assert_close(model(x)[:1],model(x[:1]),rtol=2e-4,atol=2e-4)
        torch.testing.assert_close(model.features(x),model.features(rotate_pair(x,theta)),rtol=2e-4,atol=2e-4)
    assert model.contract()['raw_unconstrained_identity_bypass'] is False
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())


@pytest.mark.parametrize('amplitude',[0.,1e-9,1.])
def test_full_ce_gradients_finite_all_parameters(amplitude):
    model=build('equivariant_memory').train();x=(torch.randn(4,2,256)*amplitude).requires_grad_()
    loss=F.cross_entropy(model(x),torch.tensor([0,1,2,3]));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    missing=[name for name,p in model.named_parameters() if p.grad is None]
    assert not missing,missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_fixed_tx_rx_chain_and_whole_phase():
    from experiments.cvs_equivariant_identity.physics import controlled_diagnostics
    model=build('equivariant_memory').eval();d=controlled_diagnostics(model)
    assert len(d['records'])==30 and len(d['isolated_tx_changes'])==4
    assert not d['formal_data_access'] and not d['target_access'] and not d['training_augmentation']
    assert d['phase_tolerance_pass'],d['phase_audit']
    assert max(d['confounds'].values())<1e-12

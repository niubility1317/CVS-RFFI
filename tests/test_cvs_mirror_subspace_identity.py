"""Public mathematical/operator and actual neural-use checks; no real IQ."""
import math

import pytest
import torch
from torch import nn

from experiments.cvs_mirror_subspace_identity.model import (
    VARIANTS, MirrorPairRelation, build, relation_contract, DETERMINANT_FLOOR,
)
from experiments.cvs_neural_residual_identity.model import NeuralResidualCVS

torch.set_num_threads(2)


def paired(z):return torch.stack((z.real,z.imag),1)
def complex_tensor(z):return torch.complex(z[:,0],z[:,1])


def random_z(dtype=torch.complex64):
    torch.manual_seed(1971)
    return torch.randn(5,31,2,4,dtype=dtype)


def test_mirror_pair_order_conjugation_and_excluded_self_conjugate_bins():
    module=MirrorPairRelation(False)
    real=torch.arange(64.).reshape(1,64,1).expand(1,64,7)
    imag=real+100
    actual=complex_tensor(module.pair_spectra(torch.stack((real,imag),1)))
    assert actual.shape==(1,31,2,7)
    assert torch.equal(actual[0,:,0,0].real,torch.arange(33.,64.))
    assert torch.equal(actual[0,:,1,0].real,torch.arange(31.,0.,-1.))
    assert torch.equal(actual[0,:,1,0].imag,-torch.arange(131.,100.,-1.))


@pytest.mark.parametrize('subspace',[False,True])
def test_statistics_match_complex128_direct_algebra(subspace):
    m=MirrorPairRelation(subspace).double();z=random_z(torch.complex128)
    power=z.abs().square().sum((-1,-2))
    denom=torch.maximum(power,(power.mean(-1,keepdim=True)/64).clamp_min(1e-6))
    n=z/denom.sqrt()[...,None,None]
    expected=n.mH@torch.linalg.solve(n@n.mH,n)/2 if subspace else n.mH@n
    if subspace:assert (torch.linalg.det(n@n.mH).real>DETERMINANT_FLOOR).all()
    actual=complex_tensor(m.relations(paired(z)))
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('subspace',[False,True])
def test_hermitian_psd_rank_and_bounds(subspace):
    m=MirrorPairRelation(subspace);q=complex_tensor(m.relations(paired(random_z())))
    torch.testing.assert_close(q,q.mH,atol=0,rtol=0)
    eigen=torch.linalg.eigvalsh(q)
    assert eigen.min()>-3e-6
    assert (eigen>1e-5).sum(-1).max()<=2
    assert q.diagonal(dim1=-2,dim2=-1).real.sum(-1).max()<=1+3e-6
    assert torch.linalg.matrix_norm(q).max()<=(1/math.sqrt(2) if subspace else 1)+3e-6


def test_subspace_qualified_gl2_invariance_and_energy_counterexample():
    z=random_z();matrix=torch.tensor([[1.5,.2j],[.15+.2j,.7]],dtype=torch.complex64)
    changed=matrix@z
    subspace=MirrorPairRelation(True);control=MirrorPairRelation(False)
    before,after=(subspace.components(paired(v)) for v in (z,changed))
    eligible=~(before['energy_floor_active']|after['energy_floor_active']|
                before['determinant_floor_active']|after['determinant_floor_active'])
    assert eligible.sum()>100
    q,q2=(complex_tensor(subspace.relations(paired(v))) for v in (z,changed))
    assert (q-q2).abs()[eligible].max()<2e-5
    q,q2=(complex_tensor(control.relations(paired(v))) for v in (z,changed))
    assert (q-q2).abs()[eligible].mean()>.01


@pytest.mark.parametrize('subspace',[False,True])
def test_unitary_row_mixing_invariance(subspace):
    z=random_z();unitary=torch.tensor([[1,1j],[1j,1]],dtype=torch.complex64)/math.sqrt(2)
    m=MirrorPairRelation(subspace)
    torch.testing.assert_close(m.relations(paired(z)),m.relations(paired(unitary@z)),atol=2e-6,rtol=2e-5)


def test_time_domain_iq_mixing_commutes_with_mirror_rows_and_shared_A():
    torch.manual_seed(15);x=torch.randn(8,2,256);m=MirrorPairRelation(True)
    a,b=1+.15j,.12-.07j
    z=torch.complex(x[:,0],x[:,1]);y=a*z+b*z.conj()
    matrix=torch.tensor([[a,b],[b.conjugate(),a.conjugate()]],dtype=torch.complex64)
    before=complex_tensor(m.mix_spectra(m.spectral(x)))
    after=complex_tensor(m.mix_spectra(m.spectral(paired(y))))
    torch.testing.assert_close(after,matrix@before,atol=2e-6,rtol=3e-6)
    c0,c1=m.components(paired(before)),m.components(paired(after))
    eligible=~(c0['energy_floor_active']|c1['energy_floor_active']|c0['determinant_floor_active']|c1['determinant_floor_active'])
    assert eligible.any()
    q0,q1=(complex_tensor(m.relations(paired(v))) for v in (before,after))
    assert (q0-q1).abs()[eligible].max()<3e-5


@pytest.mark.parametrize('subspace',[False,True])
@pytest.mark.parametrize('kind',['zero','rank_one','weak','large'])
def test_degenerate_inputs_have_finite_forward_and_gradients(subspace,kind):
    z=random_z()
    if kind=='zero':z.zero_()
    elif kind=='rank_one':z[...,1,:]=2j*z[...,0,:]
    elif kind=='weak':z*=1e-12
    else:z*=1e4
    v=paired(z).requires_grad_(True);m=MirrorPairRelation(subspace)
    q=m.relations(v);q.square().sum().backward()
    assert torch.isfinite(q).all() and torch.isfinite(v.grad).all()
    if kind=='zero':assert torch.count_nonzero(q)==0
    if kind=='rank_one' and subspace:assert q.abs().max()<4e-5


def test_low_energy_and_rank_floor_are_explicit_qualification_exclusions():
    m=MirrorPairRelation(True);z=random_z();z[:,0]*=1e-5;z[:,1,1]=2j*z[:,1,0]
    c=m.components(paired(z))
    assert c['energy_floor_active'][:,0].all()
    assert c['determinant_floor_active'][:,1].all()
    assert c['alpha'][:,0].max()<1e-5


@pytest.mark.parametrize('subspace',[False,True])
def test_gradcheck_away_from_floor_boundaries(subspace):
    m=MirrorPairRelation(subspace).double()
    v=paired(random_z(torch.complex128)[:1]).requires_grad_(True)
    assert torch.autograd.gradcheck(m.relations,(v,),fast_mode=True,atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('variant',VARIANTS)
def test_initial_function_parameters_and_rng_match_own_scratch(variant):
    torch.manual_seed(120);base=NeuralResidualCVS('neural_residual_shallow').eval()
    rng=torch.get_rng_state().clone()
    torch.manual_seed(120);model=build(variant).eval()
    assert torch.equal(rng,torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters())==247731
    assert sum(p.numel() for p in model.mirror_relation_parameters())==26744
    assert model.contract()==relation_contract(variant)
    actual=model.state_dict()
    for key,value in base.state_dict().items():assert torch.equal(value,actual[key]),key
    x=torch.randn(4,2,256)
    with torch.no_grad():torch.testing.assert_close(model(x),base(x),atol=0,rtol=0)


@pytest.mark.parametrize('variant',VARIANTS)
def test_original_ce_updates_every_added_tensor_and_state_roundtrip(variant):
    torch.manual_seed(70);model=build(variant)
    old={n:p.detach().clone() for n,p in model.core.mirror_relation.named_parameters()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    for _ in range(3):
        x=torch.randn(6,2,256);y=torch.arange(6)
        optimizer.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(x),y);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    for name,param in model.core.mirror_relation.named_parameters():assert not torch.equal(param,old[name]),name
    model.eval();clone=build(variant).eval();clone.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():torch.testing.assert_close(model(x),clone(x),atol=0,rtol=0)
    before={n:t.clone() for n,t in model.state_dict().items()};flags=[m.training for m in model.modules()]
    rng=torch.get_rng_state().clone();d=model.diagnostics(x)
    assert torch.equal(rng,torch.get_rng_state()) and flags==[m.training for m in model.modules()]
    assert all(torch.equal(before[n],t) for n,t in model.state_dict().items())
    record=d['mirror_relation']['records'][0]
    assert record['frequency_pairs']==31 and record['packets']==6
    assert len(record['determinant_floor_fraction_by_frequency'])==31
    assert d['mirror_relation']['active'] is True


@pytest.mark.parametrize('mutation',['window','mix_shape','floor','mode','encoder','method'])
def test_actual_contract_detects_operator_mutation(mutation):
    model=build('mirror_subspace');r=model.core.mirror_relation
    if mutation=='window':r.window[0]=.1
    elif mutation=='mix_shape':r.mix_real=nn.Parameter(torch.ones(4,6))
    elif mutation=='floor':r.determinant_floor=.1
    elif mutation=='mode':r.use_subspace=False
    elif mutation=='encoder':r.encoder[1]=nn.ReLU()
    else:r.pair_spectra=lambda x:x
    assert model.contract()['mirror_relation_active'] is False


@pytest.mark.parametrize('variant',VARIANTS)
def test_actual_forward_branch_is_called_once(variant):
    model=build(variant).eval();calls=[]
    handle=model.core.mirror_relation.register_forward_hook(lambda m,i,o:calls.append(tuple(o.shape)))
    try:
        with torch.no_grad():out=model(torch.randn(3,2,256))
    finally:handle.remove()
    assert out.shape==(3,6) and calls==[(3,160)]

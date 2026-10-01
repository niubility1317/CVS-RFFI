import pytest
import torch
import torch.nn.functional as F
from experiments.cvs_rf_operator_identity.model import CausalRFOperator, build, VARIANTS, causal_delay

torch.set_num_threads(2)

def iq(z):
    return torch.stack((z.real, z.imag), dim=1)

def complex_components(op, x):
    dr, di, vr, vi = op.components(x)
    return torch.complex(dr, di), torch.complex(vr, vi)

@pytest.mark.parametrize('variant', VARIANTS)
def test_known_amam_ampm_memory_and_image_mapping(variant):
    op = CausalRFOperator(variant, channels=1).double()
    with torch.no_grad():op.coefficients.zero_()
    z = torch.polar(torch.linspace(.05, 1.8, 80, dtype=torch.float64), torch.linspace(-1., 1., 80, dtype=torch.float64)).unsqueeze(0)
    chosen = [(1,0,0), (3,0,0), (5,2,0)] + ([(3,1,2)] if variant == 'rf_gmp' else [])
    coeff = [1+.1j, -.3+.4j, .08-.02j, .1+.2j]
    expected = torch.zeros_like(z)
    for spec, c in zip(chosen, coeff):
        j = op.basis_spec.index(spec);p,m,r = spec
        with torch.no_grad():op.coefficients[0,0,0,j]=c.real;op.coefficients[0,1,0,j]=c.imag
        expected += c * causal_delay(z,m) * causal_delay(z.abs().square(),m+r).pow((p-1)//2)/(2**(p-1))
    d, v = complex_components(op, iq(z))
    torch.testing.assert_close(d[:,0], expected, rtol=2e-13, atol=2e-13)
    assert v.abs().sum() == 0
    # A conjugate path must reproduce widely-linear IQ imaging, including phase.
    c = .07-.11j;j = op.basis_spec.index((1,0,0))
    with torch.no_grad():op.coefficients[1,0,0,j]=c.real;op.coefficients[1,1,0,j]=c.imag
    d, v = complex_components(op, iq(z))
    torch.testing.assert_close((d+v)[:,0], expected+c*z.conj(), rtol=2e-13, atol=2e-13)

@pytest.mark.parametrize('variant', VARIANTS)
def test_rotation_charges_bounded_observations_and_causality(variant):
    torch.manual_seed(17);op=CausalRFOperator(variant).double()
    # Include radial clipping, so independent I/Q clipping would fail this check.
    x=torch.randn(3,2,37,dtype=torch.float64)*3
    z=torch.complex(x[:,0],x[:,1]);angle=.71;rot=torch.polar(torch.tensor(1.,dtype=torch.float64),torch.tensor(angle,dtype=torch.float64))
    d,v=complex_components(op,x);rd,rv=complex_components(op,iq(z*rot))
    torch.testing.assert_close(rd,d*rot,rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(rv,v*rot.conj(),rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(op(x),op(iq(z*rot)),rtol=2e-12,atol=2e-12)
    out=op(x);c=op.channels
    assert (out[:,2*c:3*c].square()+out[:,3*c:].square()).max() <= 1+1e-12
    altered=x.clone();altered[...,19:]+=100
    torch.testing.assert_close(op(x)[...,:19],op(altered)[...,:19],rtol=0,atol=0)
    torch.testing.assert_close(op(x)[:1],op(x[:1]),rtol=2e-12,atol=2e-12)
    # CFO is not a promised invariance of a bank with multiple delay coefficients.
    cfo=torch.polar(torch.ones(37,dtype=torch.float64),torch.arange(37,dtype=torch.float64)*.17)
    assert (out-op(iq(z*cfo))).abs().max() > .01

@pytest.mark.parametrize('variant', VARIANTS)
def test_zero_weak_short_signal_derivatives_and_zero_padding(variant):
    op=CausalRFOperator(variant).double()
    for length in (1,2,5,31):
        for amplitude in (0.,1e-9,1.):
            x=(torch.randn(2,2,length,dtype=torch.float64)*amplitude).requires_grad_()
            out=op(x);out.sum().backward()
            assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()
    z=torch.zeros(1,2,9,dtype=torch.float64);z[0,0,4]=1
    phi=op.basis(z)
    for j,(_,m,_) in enumerate(op.basis_spec):assert phi[...,:m][:,:,j].abs().sum()==0
    x=(torch.randn(1,2,9,dtype=torch.float64)*.2).requires_grad_()
    assert torch.autograd.gradcheck(op,(x,),eps=1e-6,atol=3e-4,rtol=3e-3)

@pytest.mark.parametrize('variant', VARIANTS)
def test_real_ce_uses_rf_parameters_without_domain_or_augmentation(variant):
    torch.manual_seed(9);m=build(variant);b=m.id_backbone
    assert b.time_stability is None and b.freq_stability is None
    assert b.nmfdu_gate is None and not b.use_dac_path and not b.mixstyle_on
    x=torch.randn(4,2,256);y=torch.arange(4)%6;m.train()
    opt=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.0001)
    for _ in range(3):
        opt.zero_grad();F.cross_entropy(m(x),y).backward()
        for p in (b.pa_lift.coefficients,b.pa_b1.conv.weight,b.pa_proj[0].weight,b.t_proj.weight,b.f_proj.weight,b.cls_head.gain):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0
        for j in range(len(b.pa_lift.basis_spec)):
            assert b.pa_lift.coefficients.grad[...,j].norm()>0
        opt.step()
    m.eval()
    with torch.no_grad():
        torch.testing.assert_close(m(x)[:1],m(x[:1]),rtol=2e-4,atol=2e-4)
        for z in (torch.zeros(2,2,256),torch.ones(2,2,256)):assert torch.isfinite(m(z)).all()

def test_common_scratch_parameters_match_and_rf_terms_are_the_only_difference():
    torch.manual_seed(37);mp=build('rf_mp')
    torch.manual_seed(37);gmp=build('rf_gmp')
    for key,value in mp.state_dict().items():
        other=gmp.state_dict()[key]
        if key.endswith('pa_lift.coefficients'):other=other[...,:12]
        torch.testing.assert_close(value,other,rtol=0,atol=0)
    assert sum(p.numel() for p in gmp.parameters())-sum(p.numel() for p in mp.parameters())==512

def test_source_contract_and_performance_priority():
    from experiments.cvs_rf_operator_identity.model import operator_contract
    from experiments.cvs_rf_operator_identity.dispatch import select_source_candidate
    from experiments.cvs_rf_operator_identity.source import validate_config
    rows=[dict(variant=v,seed=s,accuracy=.98+.001*(v=='rf_gmp'),worst_rx=.95,parameters=200000 if v=='rf_gmp' else 100000,macs=20000000 if v=='rf_gmp' else 10000000,target_score=0 if v=='rf_gmp' else 1) for v in VARIANTS for s in (2026092701,2026092702,2026092703,2026092704)]
    assert select_source_candidate(rows)['selected_variant']=='rf_gmp'
    with pytest.raises(ValueError):select_source_candidate(rows[:-1])
    cfg=dict(method='cvs_rf_operator_identity',variant='rf_mp',epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005,rf_operator=operator_contract('rf_mp'))
    validate_config(cfg)
    for key,value in [('checkpoint','old.pt'),('target_truth','truth'),('augmentation',True),('domain_backbone',True),('rf_operator',operator_contract('rf_gmp'))]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**{key:value}))

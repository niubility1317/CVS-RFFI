import torch
import torch.nn.functional as F
import pytest
from experiments.cvs_observable_identity.model import build,PhysicalObservables,VARIANTS,AFFINE_COLUMNS,observable_contract

torch.set_num_threads(2)


def rotate(x,phase):
    c,s=phase.cos(),phase.sin();r,i=x[:,0],x[:,1]
    return torch.stack([r*c-i*s,r*s+i*c],dim=1)


def test_whole_model_phase_invariance_and_per_packet_independence():
    x=torch.randn(3,2,256,dtype=torch.float64)
    for v in VARIANTS:
        m=build(v).double().eval()
        with torch.no_grad():
            y=m(x);r=m(rotate(x,torch.tensor(.731,dtype=x.dtype)))
            torch.testing.assert_close(y,r,atol=1e-10,rtol=1e-10)
            torch.testing.assert_close(y,torch.cat([m(x[j:j+1]) for j in range(3)]),atol=1e-10,rtol=1e-10)


def test_affine_entire_model_invariant_but_phase_variant_keeps_offset_information():
    x=torch.randn(2,2,256,dtype=torch.float64);phase=.4+.021*torch.arange(256,dtype=x.dtype)
    m=build('observable_affine').double().eval()
    with torch.no_grad():torch.testing.assert_close(m(x),m(rotate(x,phase)),atol=1e-10,rtol=1e-10)
    p=PhysicalObservables('observable_phase')
    assert (p(x)[:,5:13]-p(rotate(x,phase))[:,5:13]).abs().max()>.01


def test_exact_handset_am_am_am_pm_response():
    n=torch.arange(256,dtype=torch.float64);a=1+.25*torch.sin(.2*n);theta=.13*n
    z=torch.stack([a*theta.cos(),a*theta.sin()],dim=0)[None]
    b=.17;p=z.square().sum(1)
    phi=PhysicalObservables('observable_phase')
    ampm=rotate(z,b*p)
    base,out=phi(z),phi(ampm)
    # For pure AM/PM |x| is unchanged, so regularized product moduli agree.
    d=1;ur,ui=base[:,5],base[:,6]
    delta=b*(p[:,d:]-p[:,:-d])
    expected=torch.stack([ur[:,d:]*delta.cos()-ui[:,d:]*delta.sin(),
                          ur[:,d:]*delta.sin()+ui[:,d:]*delta.cos()],dim=1)
    torch.testing.assert_close(out[:,5:7,d:],expected,atol=1e-12,rtol=1e-12)
    delta2=b*(p[:,2:]-2*p[:,1:-1]+p[:,:-2]);cr,ci=base[:,13],base[:,14]
    expected2=torch.stack([cr[:,2:]*delta2.cos()-ci[:,2:]*delta2.sin(),cr[:,2:]*delta2.sin()+ci[:,2:]*delta2.cos()],dim=1)
    torch.testing.assert_close(out[:,13:15,2:],expected2,atol=1e-12,rtol=1e-12)
    compressed=z/(1+.12*p[:,None])
    torch.testing.assert_close(phi(compressed)[:,0],torch.log1p(p/(1+.12*p)**2),atol=1e-12,rtol=1e-12)


def test_bounded_closure_zero_history_and_frontend_causality():
    x=torch.randn(2,2,256,dtype=torch.float64);phi=PhysicalObservables('observable_phase')
    y=phi(x)
    for pos,d in enumerate((1,2,5,20)):
        assert torch.count_nonzero(y[:,5+2*pos:7+2*pos,:d])==0
        assert torch.count_nonzero(y[:,13+2*pos:15+2*pos,:2*d])==0
        assert (y[:,5+2*pos].square()+y[:,6+2*pos].square()).max()<=1+1e-12
        assert (y[:,13+2*pos].square()+y[:,14+2*pos].square()).max()<=1+1e-12
    changed=x.clone();changed[...,170:]+=5
    torch.testing.assert_close(y[...,:170],phi(changed)[...,:170],atol=0,rtol=0)


def test_common_initialization_and_actual_contract():
    torch.manual_seed(711);a=build('observable_phase')
    torch.manual_seed(711);b=build('observable_affine')
    for key,value in a.state_dict().items():
        actual=b.state_dict()[key]
        if key in ('time_stem.0.weight','repeat_stem.0.weight','spectral_stem.0.weight'):
            assert torch.equal(value[:,AFFINE_COLUMNS],actual)
        else:assert torch.equal(value,actual),key
    assert all(m.contract()==observable_contract(m.variant) for m in (a,b))
    x=torch.randn(2,2,256)
    torch.testing.assert_close(a.observables(x)[:,AFFINE_COLUMNS],b.observables(x),atol=0,rtol=0)


@pytest.mark.parametrize('variant',VARIANTS)
def test_ce_all_branches_and_weak_zero_signals_finite(variant):
    m=build(variant);optimizer=torch.optim.AdamW(m.parameters(),lr=2e-4)
    x=torch.randn(4,2,256);y=torch.arange(4)%6
    optimizer.zero_grad();loss=F.cross_entropy(m(x),y);loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    for name in ('time_stem','repeat_stem','spectral_stem'):
        assert getattr(m,name)[0].weight.grad.abs().sum()>0
    optimizer.step()
    for level in (0.,1e-9,1.):
        optimizer.zero_grad();z=torch.full((2,2,256),level,requires_grad=True)
        loss=F.cross_entropy(m(z),torch.tensor([0,1]));loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_frontend_does_not_claim_multipath_or_rx_iq_invariance():
    x=torch.randn(2,2,256,dtype=torch.float64)
    altered=x+.2*F.pad(x[...,:-2],(2,0))
    phi=PhysicalObservables('observable_affine')
    assert (phi(x)-phi(altered)).abs().max()>.1
    image=x.clone();image[:,0]*=1.1;image[:,1]*=.9
    assert (phi(x)-phi(image)).abs().max()>.01


def test_schema_rejects_length_or_variant():
    with pytest.raises(ValueError):PhysicalObservables('observable_phase')(torch.zeros(1,2,255))
    with pytest.raises(ValueError):build('unregistered')


def test_source_config_no_target_or_inherited_checkpoint():
    from experiments.cvs_observable_identity.source import validate_config
    c=dict(method='cvs_observable_identity',variant='observable_phase',epochs=200,batch_size=128,lr=.0002,
           lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,
           extra_losses=[],selection='fixed_last_epoch',split_seed=392005,observables=observable_contract('observable_phase'))
    validate_config(c)
    for key,value in [('checkpoint','old.pt'),('target_truth','truth.json'),('augmentation',True),
                      ('domain_backbone',True),('observables',observable_contract('observable_affine'))]:
        with pytest.raises(ValueError):validate_config(dict(c,**{key:value}))


def test_frozen_synthetic_diagnostics_have_no_state_updates():
    from experiments.cvs_observable_identity.model import frozen_synthetic_diagnostics
    m=build('observable_affine');before={k:v.clone() for k,v in m.state_dict().items()}
    result=frozen_synthetic_diagnostics(m)
    assert result['phase_logit_max_abs_error']<1e-3 and result['affine_logit_max_abs_error']<1e-3
    assert result['synthetic_only'] and not result['training_augmentation'] and not result['target_access']
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())


def test_performance_source_selection_precedes_parameter_cost():
    from experiments.cvs_observable_identity.dispatch import select_source_candidate
    rows=[]
    for v in VARIANTS:
        for seed in (2026092701,2026092702,2026092703,2026092704):
            rows.append(dict(variant=v,seed=seed,accuracy=.98 if v=='observable_phase' else .978,
                             worst_rx=.95,parameters=200_000 if v=='observable_phase' else 150_000,
                             macs=15_000_000 if v=='observable_phase' else 10_000_000))
    s=select_source_candidate(rows)
    assert s['selected_variant']=='observable_phase' and not s['target_access'] and not s['target_score_used']


@pytest.mark.parametrize('dimensions',(1,2))
def test_profiler_conv1d_and_conv2d_macs(dimensions):
    from experiments.cvs_clean_design.profile import resource_profile
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=torch.nn.Conv1d(2,4,3,padding=1,bias=False) if dimensions==1 else torch.nn.Conv2d(2,4,3,padding=1,bias=False)
            self.head=torch.nn.Linear(4,6)
        def forward(self,x):
            if dimensions==2:x=x.reshape(len(x),2,16,16)
            h=self.conv(x);h=h.mean(tuple(range(2,h.ndim)))
            return self.head(h)
    p=resource_profile(Toy(),torch.device('cpu'))
    assert p['conv_linear_macs_per_sample']==4*256*2*(3 if dimensions==1 else 9)+4*6
    assert not p['benchmark_state_used_for_training']

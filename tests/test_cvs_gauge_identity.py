import pytest
import torch
from experiments.cvs_gauge_identity.model import VARIANTS,PacketPhaseGauge,build,gauge_contract
from experiments.cvs_residual_identity.model import build as residual_build
from experiments.cvs_gauge_identity.model import frozen_synthetic_diagnostics


def rotate(x,phase):
    r,i=x[:,0],x[:,1]
    return torch.stack([r*phase.cos()-i*phase.sin(),r*phase.sin()+i*phase.cos()],1)


def periodic(dtype=torch.float64):
    n=torch.arange(256,dtype=dtype);k=n.remainder(20)
    amplitude=1+.3*torch.cos(2*torch.pi*k/20)+.07*torch.sin(4*torch.pi*k/20)
    phase=.19*n+.1*torch.sin(2*torch.pi*k/20)
    return torch.stack([amplitude*phase.cos(),amplitude*phase.sin()],0)[None]


@pytest.mark.parametrize('variant',VARIANTS)
def test_global_gauge_and_information_preservation(variant):
    torch.manual_seed(8);x=torch.randn(3,2,256,dtype=torch.float64)
    gauge=PacketPhaseGauge(variant);z=gauge(x)
    assert torch.allclose(z,gauge(rotate(x,x.new_tensor(.731))),atol=1e-11,rtol=1e-11)
    original=torch.complex(x[:,0],x[:,1]);canonical=torch.complex(z[:,0],z[:,1])
    factor=canonical[:,0]/original[:,0]
    assert torch.allclose(canonical,original*factor[:,None],atol=1e-11,rtol=1e-11)
    assert torch.all(factor.abs()>0)
    assert torch.allclose(gauge(x),torch.cat([gauge(y[None]) for y in x]),atol=1e-11,rtol=1e-11)


@pytest.mark.parametrize('variant',VARIANTS)
def test_relative_am_pm_and_retained_cfo(variant):
    x=periodic();gauge=PacketPhaseGauge(variant);power=x.square().sum(1)
    base=gauge(x);distorted=gauge(rotate(x,.17*power))
    reference_power=power.max(-1).values[:,None]
    expected=rotate(base,.17*(power-reference_power))
    assert torch.allclose(distorted,expected,atol=1e-10,rtol=1e-10)
    n=torch.arange(256,dtype=x.dtype)
    assert (gauge(rotate(x,.031*n))-base).abs().max()>.1
    compressed=x/(1+.12*power[:,None]);p=gauge(compressed).square().sum(1)
    assert not torch.allclose(p,base.square().sum(1),atol=1e-4)


@pytest.mark.parametrize('variant',VARIANTS)
def test_whole_network_gauge_without_more_parameters(variant):
    torch.set_num_threads(2);torch.manual_seed(10);model=build(variant).double().eval()
    torch.manual_seed(10);control=residual_build('residual_fusion').double().eval()
    assert sum(p.numel() for p in model.parameters())==164225
    assert sum(p.numel() for p in model.gauge.parameters())==0
    assert all(torch.equal(v,control.state_dict()[k]) for k,v in model.core.state_dict().items())
    x=periodic().repeat(2,1,1);x[1]*=.8
    assert torch.allclose(model(x),model(rotate(x,x.new_tensor(.731))),atol=1e-8,rtol=1e-8)
    assert torch.allclose(model(x),torch.cat([model(y[None]) for y in x]),atol=1e-8,rtol=1e-8)
    assert model.contract()==gauge_contract(variant)
    assert torch.allclose(model(x),model.classify_features(model.features(x)),atol=1e-8,rtol=1e-8)


@pytest.mark.parametrize('variant',VARIANTS)
def test_ce_and_degenerate_gradients(variant):
    torch.set_num_threads(2);torch.manual_seed(11);model=build(variant).train()
    x=torch.randn(4,2,256,requires_grad=True)
    loss=torch.nn.functional.cross_entropy(model(x),torch.arange(4));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for scale in (0.,1e-9):
        weak=(torch.randn(2,2,256)*scale).requires_grad_(True)
        g=PacketPhaseGauge(variant)(weak);g.square().sum().backward()
        assert torch.isfinite(g).all() and torch.isfinite(weak.grad).all()


def test_coherent_fallback_preserves_nonzero_packet():
    x=torch.zeros(2,2,256,dtype=torch.float64);x[:,0,0]=2.;x[:,1,200]=1.
    g=PacketPhaseGauge('gauge_coherent');_,_,_,fallback,_=g.reference(x)
    assert fallback.all() and g(x).square().sum()>0
    assert torch.allclose(g(x),g(rotate(x,x.new_tensor(.47))),atol=1e-11)


def test_receiver_and_multipath_are_explicit_counterexamples():
    x=periodic();g=PacketPhaseGauge('gauge_coherent')
    image=x*torch.tensor([1.08,.92],dtype=x.dtype)[None,:,None]
    channel=x+.2*torch.nn.functional.pad(x[...,:-2],(2,0))
    assert not torch.allclose(g(image),g(x),atol=1e-4)
    assert not torch.allclose(g(channel),g(x),atol=1e-4)


def test_invalid_gauge_and_schema():
    with pytest.raises(ValueError):build('unknown')
    with pytest.raises(ValueError):PacketPhaseGauge('gauge_peak')(torch.zeros(2,2,128))


@pytest.mark.parametrize('variant',VARIANTS)
def test_frozen_diagnostic_does_not_update_weights(variant):
    torch.set_num_threads(2);torch.manual_seed(13);model=build(variant).eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    d=frozen_synthetic_diagnostics(model)
    assert d['phase_logit_max_abs_error']<1e-3
    assert d['target_access'] is False and d['training_augmentation'] is False
    assert d['affine_invariance_claimed'] is False
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())

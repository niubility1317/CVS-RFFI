import math
import numpy as np
import pytest
import torch
from experiments.cvs_synchronized_identity.model import RepeatedFieldSynchronizer,build,VARIANTS,synchronized_contract
from experiments.cvs_synchronized_identity.physics import frozen_synthetic_diagnostics
from experiments.cvs_reference_identity.physics import received,TX_ROWS,RX_ROWS


def rotate(x,theta=0.,hz=0.):
    phase=x.new_tensor(theta)+2*math.pi*hz/25000000*torch.arange(256,device=x.device,dtype=x.dtype)
    r,i=x.unbind(1)
    return torch.stack((r*phase.cos()-i*phase.sin(),r*phase.sin()+i*phase.cos()),1)


def public(dtype=torch.float64):
    return torch.tensor(np.stack([received(t,RX_ROWS[0]) for t in TX_ROWS]).tolist(),dtype=dtype)


def test_repeated_field_cfo_units_amplitude_preservation_and_covariance():
    sync=RepeatedFieldSynchronizer();x=public();original=x.clone()
    for hz in (-80000.,0.,80000.,500000.):
        y=rotate(x,.37,hz);omega,valid,quality=sync.estimate(y)
        assert valid.all();torch.testing.assert_close(quality,torch.ones_like(quality),atol=1e-13,rtol=1e-13)
        torch.testing.assert_close(omega*25000000/(2*math.pi),torch.full_like(omega,hz),atol=1e-7,rtol=1e-12)
        torch.testing.assert_close(sync(y),rotate(sync(x),.37),atol=1e-13,rtol=1e-13)
        torch.testing.assert_close(sync(y).square().sum(1),y.square().sum(1),atol=1e-13,rtol=1e-13)
    assert torch.equal(x,original)


def test_frequency_principal_branch_ambiguity_is_not_hidden():
    sync=RepeatedFieldSynchronizer();x=public()
    aliased=rotate(x,hz=1250000.)
    omega,valid,_=sync.estimate(aliased)
    assert valid.all();assert omega.abs().max()<1e-14
    assert (sync(aliased)-sync(x)).norm()>1
    a=rotate(x,hz=500000.);b=rotate(a,hz=300000.)
    oa,_,_=sync.estimate(a);ob,_,_=sync.estimate(b)
    torch.testing.assert_close((ob-oa)*25000000/(2*math.pi),torch.full_like(oa,-950000.),atol=1e-7,rtol=1e-12)


def test_degenerate_and_weak_windows_have_finite_gradients_and_no_frequency_correction():
    sync=RepeatedFieldSynchronizer()
    for scale in (0.,1e-10):
        x=(torch.randn(3,2,256,dtype=torch.float64)*scale).requires_grad_(True)
        omega,valid,quality=sync.estimate(x);assert not valid.any();assert omega.abs().max()==0
        if scale==0.:assert quality.abs().max()==0
        y=sync(x);torch.testing.assert_close(y,x,atol=0,rtol=0)
        y.square().sum().backward();assert torch.isfinite(x.grad).all()


def test_nonzero_energy_cancellation_reports_zero_coherence_and_safe_fallback():
    sync=RepeatedFieldSynchronizer();x=torch.zeros(1,2,256,dtype=torch.float64)
    x[0,0,80:160]=torch.tensor([1.,1.,1.,-2.],dtype=x.dtype).repeat_interleave(20)
    x.requires_grad_(True);omega,valid,quality=sync.estimate(x)
    assert not valid.any() and omega.abs().max()==0 and quality.abs().max()==0
    sync(x).square().sum().backward();assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('variant',VARIANTS)
def test_whole_identity_affine_phase_and_synthetic_rx_confounds(variant):
    torch.manual_seed(7);model=build(variant).eval()
    d=frozen_synthetic_diagnostics(model)
    assert len(d['records'])==30 and not d['target_access']
    assert all(a['estimated_cfo_principal_branch_crossing_count']==0 and a['valid_correlation_count']==30 for a in d['phase_audit'])
    assert d['phase_tolerance_pass'],d['phase_audit']
    assert d['confounds']['iq_tx_rx_waveform_max_error']<=1e-12 and d['confounds']['cubic_tx_rx_waveform_max_error']<=1e-12
    x=public(torch.float32)[:2];alone=model(x[:1]);batch=model(x)
    torch.testing.assert_close(alone,batch[:1],atol=2e-5,rtol=2e-5)
    assert model.contract()==synchronized_contract(variant)
    assert sum(p.numel() for p in model.synchronizer.parameters())==0
    assert sum(p.numel() for p in model.parameters())==(202553 if variant=='synchronized_equivariant' else 164225)


@pytest.mark.parametrize('variant',VARIANTS)
def test_ce_updates_identity_parameters_without_extra_loss_or_domain_backbone(variant):
    model=build(variant).train();x=public(torch.float32)[:4].requires_grad_(True)
    loss=torch.nn.functional.cross_entropy(model(x),torch.arange(4));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    grads=[p for p in model.parameters() if p.grad is not None]
    assert sum(p.numel() for p in grads)>150000
    assert not any('domain' in name for name,_ in model.named_modules())

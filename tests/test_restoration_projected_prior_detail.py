import pytest
import torch

from experiments.pamf_phase3_restoration.covariance_inverse import gaussian_posterior
from experiments.pamf_phase3_restoration.projected_prior_detail import (
    fixed_posterior_frame, prior_update, scratch_projected_prior_detail,
)


def test_projected_prior_equals_direct_conditioned_gaussian_mean():
    rng = torch.Generator().manual_seed(93)
    def rand(*shape):
        return torch.randn(*shape,generator=rng,dtype=torch.float64)
    a,y,mean,change = rand(2,6,6),rand(2,6),rand(6),rand(2,6)
    factor = rand(6,6)
    covariance = factor@factor.T+.3*torch.eye(6,dtype=torch.float64)
    variance = torch.tensor([.07,.5],dtype=torch.float64)
    base,ac,chol,_ = fixed_posterior_frame(a,y,mean,covariance,variance)
    correction = prior_update(a,ac,chol,change[:,:,None]).squeeze(2)
    direct = torch.stack([gaussian_posterior(a[i:i+1],y[i:i+1],mean+change[i],covariance,variance[i:i+1])[0][0]
                          for i in range(len(y))])
    torch.testing.assert_close(base+correction,direct,rtol=1e-11,atol=1e-12)


def test_source_supervision_targets_noiseless_shrinkage_only():
    a = torch.diag(torch.tensor([1.,.05],dtype=torch.float64))[None]
    y = torch.tensor([[.4,.3]],dtype=torch.float64)
    clean = torch.tensor([[.3,-.2]],dtype=torch.float64)
    mean = torch.zeros(2,dtype=torch.float64)
    covariance = torch.eye(2,dtype=torch.float64)
    variance = torch.tensor([.01],dtype=torch.float64)
    _,ac,chol,_ = fixed_posterior_frame(a,y,mean,covariance,variance)
    target = prior_update(a,ac,chol,(clean-mean)[:,:,None]).squeeze(-1)
    noiseless = gaussian_posterior(a,(a@clean[:,:,None]).squeeze(2),mean,covariance,variance)[0]
    torch.testing.assert_close(clean-target,noiseless,rtol=1e-12,atol=1e-12)
    assert (target/clean)[0,0] < .02
    assert (target/clean)[0,1] > .75


def test_actual_scratch_detail_gradient_is_isolated_and_deployed():
    model = scratch_projected_prior_detail(2026092701)
    t = torch.linspace(-1,1,256)
    received = torch.stack((.8+.1*t,.2+.2*t.square())).unsqueeze(0)
    clean = torch.stack((.4+.05*t,.1+.15*t.square())).unsqueeze(0)
    model.physical.bind_prior(torch.zeros(512),.04*torch.eye(512))
    initial,aux = model(received,return_aux=True)
    torch.testing.assert_close(initial,aux['base_iq'],rtol=0,atol=0)
    restored,debug = model.source_readout(received,clean)
    torch.testing.assert_close(restored,initial,rtol=0,atol=0)
    target = debug['source_noiseless_corrected_iq']
    assert not torch.allclose(restored,target)
    (target-clean).square().mean().backward()
    assert all(p.grad is None for p in model.physical.parameters())
    assert model.detail.output.weight.grad.abs().sum()>0
    with torch.no_grad():
        model.detail.output.weight.add_(-.05*model.detail.output.weight.grad)
        updated = model(received)
        assert not torch.equal(updated,initial)
    assert not model.initialization_report['ancestor_checkpoints']


def test_only_preregistered_arm_and_single_stage_are_selected():
    from experiments.pamf_phase3_restoration import projected_prior_detail_trial as run

    with pytest.raises(ValueError,match='fixed projected prior detail'):
        run.configuration('other')
    cfg = run.configuration(run.ARM)
    assert cfg['training_stages']==1 and len(cfg['source_draws'])==30
    assert cfg['new_G_initialization_checkpoints']==0 and cfg['F0_guidance_for_G_updates']
    assert cfg['source_actual_IQ_F0_guidance'] is False

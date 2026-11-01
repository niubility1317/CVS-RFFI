from unittest.mock import patch
import pytest
import torch
from experiments.pamf_phase3_restoration.residual_diffusion import (
    scratch_residual_diffusion, fft_iq, ifft_iq, diffusion_loss)
from experiments.pamf_phase3_restoration import residual_diffusion_trial as trial

torch.set_num_threads(2)


def test_fourier_coordinates_are_invertible_and_preserve_energy():
    x=torch.randn(3,4,256)
    torch.testing.assert_close(ifft_iq(fft_iq(x)),x,atol=5e-7,rtol=2e-5)
    torch.testing.assert_close(fft_iq(x).square().sum(),x.square().sum())


def test_gaussian_prior_heun_and_fifteen_actual_denoiser_calls():
    g=scratch_residual_diffusion(2026092701).eval();y=torch.randn(2,2,256)
    calls=[];hook=g.core.register_forward_hook(lambda *a:calls.append(1))
    before=torch.get_rng_state().clone()
    with torch.no_grad():actual=g(y)
    hook.remove();assert len(calls)==15 and torch.equal(before,torch.get_rng_state())
    schedule=g.sigma_steps.double();z=g.fixed_latent.double().expand_as(y)*schedule[0]
    for i in range(8):
        s,t=schedule[i:i+2];d=(z-.25/(s*s+.25)*z)/s
        proposal=z+(t-s)*d
        if i<7:
            d2=(proposal-.25/(t*t+.25)*proposal)/t
            z=z+(t-s)*(d+d2)/2
        else:z=proposal
    torch.testing.assert_close(actual,y+z.float(),atol=2e-5,rtol=2e-5)
    assert not torch.equal(actual,y)  # E0 is a Gaussian residual prior, not identity.


def test_sampling_uses_only_current_record_and_fixed_saved_latent():
    g=scratch_residual_diffusion(7).eval();y=torch.randn(3,2,256)
    with torch.no_grad():
        g.core.time_out.weight.normal_(0,.001);g.core.freq_out.weight.normal_(0,.001)
        actual=g(y)
        torch.testing.assert_close(actual,torch.cat([g(x[None]) for x in y]),atol=3e-5,rtol=3e-5)
        torch.testing.assert_close(g(y.flip(0)).flip(0),actual,atol=3e-5,rtol=3e-5)
    clone=scratch_residual_diffusion(8).eval();clone.load_state_dict(g.state_dict(),strict=True)
    with torch.no_grad():torch.testing.assert_close(clone(y),actual,atol=0,rtol=0)
    g.denoise(torch.randn_like(y),y,torch.ones(len(y))).square().mean().backward()
    for name in ['time_in.weight','freq_in.weight','blocks.0.attention.in_proj_weight']:
        grad=dict(g.core.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0


def test_preconditioning_and_loss_have_finite_gradients_at_all_fixed_scales():
    g=scratch_residual_diffusion(9);y=torch.randn(3,2,256);clean=torch.randn_like(y,requires_grad=True)
    sigma=torch.tensor([.002,.5,80.]);noise=torch.randn_like(y)
    restored,terms=diffusion_loss(g,y,clean,sigma,noise)
    residual=clean.detach()-y;z=residual+sigma[:,None,None]*noise
    prior=.25/(sigma.square()+.25)
    torch.testing.assert_close(restored,y+prior[:,None,None]*z)
    weight=(sigma.square()+.25)/(sigma*.5).square()
    expected=weight*(restored-clean.detach()).square().mean((1,2))
    torch.testing.assert_close(terms['diffusion_mse'],expected)
    terms['diffusion_mse'].mean().backward()
    assert clean.grad is None
    for name in ['time_out.weight','freq_out.weight']:
        grad=dict(g.core.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0


def test_fixed_legal_budget_and_strict_checkpoint_boundary(tmp_path):
    c=trial.configuration(trial.ARM)
    assert (c['epochs'],c['updates'],c['training_stages'])==(30,7410,1)
    assert c['initialization']=='scratch' and c['ancestor_checkpoints']==[]
    assert c['sampling_denoiser_calls']==15 and c['unit_power_projection'] is False
    assert all(c[k]==0 for k in ['TX_labels_read','V_reads','OE_reads','target_training_reads'])
    with pytest.raises(ValueError):trial.configuration('DURA_v1')
    trial.install(torch.ones(160));g=trial.factory(trial.ARM);p=tmp_path/'E0.pt'
    trial.trial.save(p,g,trial.ARM,'test-commit',0)
    clone=trial.trial.load(p,trial.ARM,'test-commit','cpu',epoch=0)
    assert all(torch.equal(v,clone.state_dict()[k]) for k,v in g.state_dict().items())
    with pytest.raises(ValueError):trial.trial.load(p,trial.ARM,'wrong-commit','cpu',epoch=0)
    with pytest.raises(FileExistsError):trial.trial.save(p,g,trial.ARM,'test-commit',0)


def test_queue_parent_failure_cannot_start_neural_work(tmp_path):
    parent=tmp_path/'runs'/trial.WAIT_PARENT;parent.mkdir(parents=True)
    (parent/'complete.json').write_text('{"status":"ARTIFACTS_COMPLETE"}',encoding='utf-8')
    with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':''}):trial.wait_for_parent(tmp_path)
    (parent/'terminal-train.json').write_text('{"exit_code":1}',encoding='utf-8')
    with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':''}),pytest.raises(RuntimeError):trial.wait_for_parent(tmp_path)

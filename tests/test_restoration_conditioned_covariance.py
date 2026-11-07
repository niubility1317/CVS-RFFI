import torch

from experiments.pamf_phase3_restoration.conditioned_covariance import (
    conditioned_posterior, scratch_conditioned_covariance,
)
from experiments.pamf_phase3_restoration.covariance_inverse import gaussian_posterior


def test_positive_variance_addition_matches_the_direct_gaussian_solution():
    gen = torch.Generator().manual_seed(73)
    action = torch.randn(2,6,6,generator=gen,dtype=torch.float64)
    observed = torch.randn(2,6,generator=gen,dtype=torch.float64)
    mean = torch.randn(6,generator=gen,dtype=torch.float64)
    covariance = .3*torch.eye(6,dtype=torch.float64)
    extra = torch.rand(2,6,generator=gen,dtype=torch.float64,requires_grad=True)
    variance = torch.tensor([.03,.1],dtype=torch.float64)
    output,residual = conditioned_posterior(action,observed,mean,covariance,variance,extra)
    expected = torch.cat([gaussian_posterior(action[i:i+1],observed[i:i+1],mean,
        covariance+torch.diag(extra[i]),variance[i:i+1])[0] for i in range(2)])
    torch.testing.assert_close(output,expected,rtol=1e-11,atol=1e-12)
    assert residual.max()<1e-10
    output.square().sum().backward()
    assert extra.grad is not None and torch.isfinite(extra.grad).all() and extra.grad.abs().sum()>0


def test_actual_model_variance_gradient_is_separate_and_state_is_immutable():
    model = scratch_conditioned_covariance(2026092701)
    model.physical.bind_prior(torch.zeros(512),.04*torch.eye(512))
    time = torch.linspace(-1,1,256)
    received = torch.stack((.6+.2*time,.2+.1*time.square()))[None]
    snapshot = {key:value.detach().clone() for key,value in model.state_dict().items()}
    output,aux = model(received,return_aux=True)
    assert output.shape==received.shape and torch.isfinite(output).all()
    assert (aux['added_variance']>0).all() and aux['added_variance'].shape==(1,2,256)
    (output-.5*received).square().mean().backward()
    assert all(p.grad is None for p in model.physical.parameters())
    assert model.detail.output.weight.grad.abs().sum()>0
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in snapshot.items())


def test_receiver_specific_training_moments_do_not_change_deployment_moments():
    model = scratch_conditioned_covariance(2026092701)
    model.physical.bind_prior(torch.zeros(512),.04*torch.eye(512))
    received = torch.randn(2,2,256,generator=torch.Generator().manual_seed(79))
    before = {key:value.detach().clone() for key,value in model.state_dict().items()}
    prior = (.1*torch.ones(2,512),.09*torch.eye(512)[None].expand(2,-1,-1))
    with torch.inference_mode():
        original = model(received)
        changed,_ = model.source_readout(received,prior=prior)
        repeated = model(received)
        alone = model(received[1:])
    assert not torch.equal(original,changed)
    torch.testing.assert_close(original,repeated,rtol=0,atol=0)
    torch.testing.assert_close(original[1:],alone,rtol=2e-5,atol=2e-6)
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())


def test_more_variance_can_both_remove_shrinkage_and_amplify_realized_noise():
    action = torch.ones(1,1,1,dtype=torch.float64)
    mean = torch.zeros(1,dtype=torch.float64)
    covariance = .01*torch.ones(1,1,dtype=torch.float64)
    variance = .1*torch.ones(1,dtype=torch.float64)
    extra = torch.ones(1,1,dtype=torch.float64)
    clean = torch.ones(1,1,dtype=torch.float64)
    plain = gaussian_posterior(action,clean,mean,covariance,variance)[0]
    larger,_ = conditioned_posterior(action,clean,mean,covariance,variance,extra)
    assert (larger-clean).abs().item() < (plain-clean).abs().item()
    noisy = 4*clean
    plain = gaussian_posterior(action,noisy,mean,covariance,variance)[0]
    larger,_ = conditioned_posterior(action,noisy,mean,covariance,variance,extra)
    assert (larger-clean).abs().item() > (plain-clean).abs().item()


def test_training_optimizes_the_actual_noisy_readout_and_keeps_single30():
    from experiments.pamf_phase3_restoration import conditioned_covariance_trial as trial
    from experiments.pamf_phase3_restoration.covariance_detail import detail_objective
    from experiments.pamf_phase3_restoration.objective import PairedObjective

    config = trial.configuration(trial.ARM)
    assert (config['epochs'],config['updates'],config['training_stages'])==(30,7410,1)
    assert config['source_actual_IQ_F0_guidance'] and not config['detail_initial_zero']
    model = trial.factory(trial.ARM)
    model.physical.bind_prior(torch.zeros(512),.04*torch.eye(512))
    received = torch.randn(1,2,256,generator=torch.Generator().manual_seed(83))
    clean = .7*received
    class Reader:
        def features(self,value):
            return value.flatten(1)[:,:160]
    reader = Reader()
    target = reader.features(clean)
    scale = torch.ones(160)
    objective = PairedObjective()
    priors = (torch.zeros(5,512),.08*torch.eye(512)[None].expand(5,-1,-1))
    restored,aux,loss,_ = trial.noisy_readout(model,reader,received,clean,target,scale,objective,
        source_indices=torch.tensor([0]),epoch=2,priors=priors,assignment=torch.tensor([0]))
    expected,_ = detail_objective(reader,restored,clean,target,scale,objective)
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    assert restored is aux['base_iq']
    loss.mean().backward()
    assert model.detail.output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.physical.parameters())

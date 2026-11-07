import torch

from experiments.pamf_phase3_restoration.covariance_inverse import gaussian_posterior
from experiments.pamf_phase3_restoration.mismatched_prior_training import choose_prior
from experiments.pamf_phase3_restoration.projected_prior_detail import fixed_posterior_frame, scratch_projected_prior_detail


def test_batched_source_priors_match_individual_gaussian_posteriors():
    gen = torch.Generator().manual_seed(57)
    action = torch.randn(3,6,6,generator=gen,dtype=torch.float64)
    observed = torch.randn(3,6,generator=gen,dtype=torch.float64)
    means = torch.randn(3,6,generator=gen,dtype=torch.float64)
    factor = torch.randn(3,6,6,generator=gen,dtype=torch.float64)
    cov = factor@factor.transpose(1,2)+.1*torch.eye(6,dtype=torch.float64)
    variance = torch.tensor([.01,.1,.5],dtype=torch.float64)
    actual = fixed_posterior_frame(action,observed,means,cov,variance)[0]
    expected = torch.cat([gaussian_posterior(action[i:i+1],observed[i:i+1],means[i],cov[i],variance[i:i+1])[0]
                          for i in range(3)])
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-12)


def test_every_source_record_alternates_between_global_and_its_excluded_rx():
    means = torch.arange(5*6).reshape(5,6).float()
    cov = torch.arange(5*36).reshape(5,6,6).float()
    assignment = torch.tensor([1,4,0,2,3,1])
    indices = torch.tensor([4,0,1])
    for epoch in range(1,31):
        selected = choose_prior((means,cov),assignment,indices,epoch)
        if epoch%2:
            assert selected is None
        else:
            torch.testing.assert_close(selected[0],means[[3,1,4]])
            torch.testing.assert_close(selected[1],cov[[3,1,4]])


def test_mismatch_source_path_preserves_deployment_state_and_gradient_isolation():
    model = scratch_projected_prior_detail(2026092701)
    model.physical.bind_prior(torch.zeros(512),.04*torch.eye(512))
    time = torch.linspace(-1,1,256)
    received = torch.stack((.7+.1*time,.2+.12*time.square()))[None]
    clean = .6*received
    prior = (.1*torch.ones(1,512),.08*torch.eye(512)[None])
    snapshot = {k:v.detach().clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        deployed = model(received)
    source,aux = model.source_readout(received,clean,prior=prior)
    assert not torch.equal(source,deployed)
    action = aux['action_matrix'].detach()
    noiseless = (action@clean.flatten(1).double()[:,:,None]).squeeze(-1)
    expected = gaussian_posterior(action,noiseless,prior[0][0].double(),prior[1][0].double(),aux['variance'].detach().double())[0]
    torch.testing.assert_close(aux['source_noiseless_corrected_iq'],expected.float().reshape_as(clean),rtol=1e-6,atol=1e-7)
    (aux['source_noiseless_corrected_iq']-clean).square().mean().backward()
    assert all(p.grad is None for p in model.physical.parameters())
    assert model.detail.output.weight.grad.abs().sum()>0
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in snapshot.items())
    with torch.no_grad():
        torch.testing.assert_close(model(received),deployed,rtol=0,atol=0)


def test_single_stage_configuration_retains_source_permissions_and_budget():
    from experiments.pamf_phase3_restoration import mismatched_prior_trial as trial
    cfg = trial.configuration(trial.ARM)
    assert cfg['epochs']==30 and cfg['updates']==7410 and cfg['training_stages']==1
    assert cfg['source_draws']==list(range(1,21))+list(range(1,11))
    assert cfg['training_prior_mismatch_epochs']==list(range(2,31,2))
    assert cfg['new_G_initialization_checkpoints']==0
    assert cfg['training_prior_other_RX_samples']==12600

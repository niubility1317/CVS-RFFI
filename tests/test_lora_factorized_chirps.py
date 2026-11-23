import torch

from experiments.pamf_lora_main.factorized_chirps import ChirpPrior, packet_posterior


def test_packet_posterior_matches_dense_complex_solution():
    generator = torch.Generator().manual_seed(944)
    shape = (3, 6, 12)
    residual = torch.randn(*shape, dtype=torch.complex64, generator=generator)
    noise = torch.rand(*shape, generator=generator) + .1
    common = torch.rand(12, generator=generator) + .1
    individual = torch.rand(12, generator=generator) + .1
    prior = (torch.diag_embed(individual[:, None].expand(-1,6)) + common[:, None, None]).to(residual.dtype)
    observed = prior + torch.diag_embed(noise.transpose(1,2))
    rhs = residual.transpose(1,2)[..., None]
    expected = (prior @ torch.linalg.solve(observed, rhs)).squeeze(-1).transpose(1,2)
    actual = packet_posterior(residual, noise, common, individual, coupled=True)
    torch.testing.assert_close(actual, expected)
    uncoupled = packet_posterior(residual, noise, torch.zeros_like(common), individual, coupled=False)
    collapsed = packet_posterior(residual, noise, torch.zeros_like(common), individual, coupled=True)
    torch.testing.assert_close(uncoupled, collapsed)


def test_noisy_chirp_draws_from_other_observations_only_under_joint_model():
    residual = torch.zeros((1,6,1),dtype=torch.complex64)
    residual[:,1:] = 1. + .25j
    noise = residual.real.new_full(residual.shape,.02)
    noise[:,0] = 10.
    shared, indiv = torch.tensor([2.]),torch.tensor([.01])
    coupled = packet_posterior(residual,noise,shared,indiv,coupled=True)
    single = packet_posterior(residual,noise,shared,indiv,coupled=False)
    assert single[:,0].abs().item() == 0.
    assert (coupled[:,0] - (1+.25j)).abs().item() < .02


def test_source_prior_separates_common_variation_from_private_components():
    prior=ChirpPrior()
    n=12
    generator=torch.Generator().manual_seed(945)
    packet=torch.randn(n,1,256,dtype=torch.complex64,generator=generator)
    prior.fit(packet.expand(-1,6,-1))
    torch.testing.assert_close(prior.mean,packet.mean(0)[0])
    torch.testing.assert_close(prior.shared, (packet[:,0]-prior.mean).abs().square().mean(0))
    torch.testing.assert_close(prior.individual,torch.full_like(prior.individual,1e-7))
    contrast=torch.randn(n,6,256,dtype=torch.complex64,generator=generator)
    contrast=contrast-contrast.mean(1,keepdim=True)
    prior.fit(contrast)
    torch.testing.assert_close(prior.shared,torch.zeros_like(prior.shared))
    assert prior.individual.min() > .05

import torch

from experiments.pamf_lora_main.shared_latent_mixture import (
    WITHIN_VARIANCE,
    empirical_centers,
    latent_posteriors,
    replace_common,
)


def test_empirical_prior_centers_match_gaussian_first_and_second_moments():
    generator=torch.Generator().manual_seed(96)
    common=torch.randn(150,20,generator=generator,dtype=torch.float64)
    factors=torch.randn(20,4,generator=generator,dtype=torch.float64)
    diag=torch.rand(20,generator=generator,dtype=torch.float64)+.2
    centers=empirical_centers(common,common.mean(0),diag,factors)
    eye=torch.eye(4,dtype=centers.dtype)
    torch.testing.assert_close(centers.mean(0),torch.zeros(4,dtype=centers.dtype))
    torch.testing.assert_close(centers.T @ centers/len(centers)+WITHIN_VARIANCE*eye,eye)


def test_latent_mixture_agrees_with_dense_observation_model():
    gen=torch.Generator().manual_seed(29)
    factors=torch.randn(2,9,4,generator=gen,dtype=torch.float64)
    noise=torch.rand(2,9,generator=gen,dtype=torch.float64)+.3
    observed=torch.randn(2,9,generator=gen,dtype=torch.float64)
    centers=torch.randn(17,4,generator=gen,dtype=torch.float64)*.7
    a=factors.mT @ (factors/noise[...,None])
    b=(factors.mT @ (observed/noise).unsqueeze(-1)).squeeze(-1)
    actual_gaussian,actual_mixture=latent_posteriors(a,b,centers)
    expected=[]
    for F,D,y in zip(factors,noise,observed,strict=True):
        latent=F.T @ torch.linalg.solve(torch.diag(D)+F@F.T,y)
        C=torch.diag(D)+WITHIN_VARIANCE*F@F.T
        residual=y.unsqueeze(0)-centers@F.T
        z=torch.linalg.solve(C,residual.T).T
        weights=(-.5*(residual*z).sum(-1)).softmax(0)
        mixture=weights@(centers+WITHIN_VARIANCE*z@F)
        expected.append((latent,mixture))
    expected_gaussian,expected_mix=(torch.stack(x) for x in zip(*expected,strict=True))
    torch.testing.assert_close(actual_gaussian,expected_gaussian)
    torch.testing.assert_close(actual_mixture,expected_mix)


def test_replace_common_preserves_chirp_specific_structure_and_outside_bins():
    gen=torch.Generator().manual_seed(6)
    spectrum=torch.randn(2,6,256,generator=gen,dtype=torch.complex64)
    bins=torch.tensor([13,23,65,211])
    new=torch.randn(2,len(bins),generator=gen,dtype=spectrum.dtype)
    updated=replace_common(spectrum,bins,new)
    outside=torch.ones(256,dtype=torch.bool)
    outside[bins]=False
    assert torch.equal(updated[...,outside],spectrum[...,outside])
    before=spectrum[...,bins]
    after=updated[...,bins]
    torch.testing.assert_close(after.mean(1),new)
    torch.testing.assert_close(after-after.mean(1,keepdim=True),before-before.mean(1,keepdim=True))

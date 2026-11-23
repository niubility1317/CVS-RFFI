import torch

from experiments.pamf_lora_main.coupled_echo import echo_system, integrated_echo, joint_precision
from experiments.pamf_lora_main.factorized_chirps import packet_posterior


def test_joint_echo_covariance_matches_dense_complex_posterior():
    generator = torch.Generator().manual_seed(141)
    shape = (2, 3, 5)
    residual = torch.randn(shape, generator=generator, dtype=torch.complex64)
    columns = .18 * torch.randn((*shape, 4), generator=generator, dtype=torch.complex64)
    noise = torch.rand(shape, generator=generator) + .1
    s = torch.rand(shape[-1], generator=generator) + .05
    d = torch.rand(shape[-1], generator=generator) + .05
    freqs = torch.eye(shape[-1])
    chirps = torch.eye(shape[1])
    shared = torch.kron(torch.ones_like(chirps), torch.diag(s))
    private = torch.kron(chirps, torch.diag(d))
    prior = (shared+private).to(residual.dtype)
    a = prior + torch.diag_embed(noise.flatten(1))
    flat = columns.flatten(1,2)
    augmented = a + flat@flat.mH
    rhs = residual.flatten(1)[...,None]
    expected = (prior@torch.linalg.solve(augmented,rhs)).reshape(shape)
    system = echo_system(columns,noise,s,d)
    correction,evidence = integrated_echo(residual,system)
    achieved = packet_posterior(residual-correction,noise,s,d,coupled=True)
    torch.testing.assert_close(achieved,expected)
    score = (rhs.mH@(torch.linalg.solve(a,rhs)-torch.linalg.solve(augmented,rhs))).real.flatten()
    score=score+torch.linalg.slogdet(a).logabsdet-torch.linalg.slogdet(augmented).logabsdet
    torch.testing.assert_close(evidence,score,atol=4e-6,rtol=1e-5)
    solved=joint_precision(residual[..., None],noise,s,d).flatten(1,2)
    torch.testing.assert_close(solved,torch.linalg.solve(a,rhs))


def test_zero_echo_design_preserves_original_innovation():
    residual=torch.ones(2,6,4,dtype=torch.complex64)
    noise=torch.full_like(residual.real,.04)
    spectral_shape=noise.new_full((4,),.1)
    columns=residual.new_zeros((2,6,4,8))
    system=echo_system(columns,noise,spectral_shape,spectral_shape)
    correction,evidence=integrated_echo(residual,system)
    torch.testing.assert_close(correction,torch.zeros_like(correction),atol=0,rtol=0)
    torch.testing.assert_close(evidence,torch.zeros_like(evidence),atol=0,rtol=0)

import torch

from experiments.pamf_lora_main.channel_blind_source import FitRecipe
from experiments.pamf_lora_main.fission_risk import band_masks
from experiments.pamf_lora_main.private_dose import band_energy
from experiments.pamf_lora_main.refinement_bound import bound_refinement


def test_bound_only_changes_bands_above_the_physical_budget():
    t = torch.arange(256) / 256
    mode = lambda f: torch.exp(2j * torch.pi * f * t)[None, None].expand(2, 6, -1)
    observed = .5 * mode(0)
    coarse = observed + mode(4) + mode(16) + mode(48)
    fine = observed + 2j * mode(4) + .3 * mode(16) - 3 * mode(48)
    expected = observed + 1j * mode(4) + .3 * mode(16) - mode(48)
    rms = torch.linspace(.2, 1.1, 6)[None, :, None].expand(2, -1, -1)
    base, result = bound_refinement(observed, coarse, fine, rms)
    assert torch.equal(base, coarse)
    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-5)
    masks = band_masks(result).float()
    energy = lambda y: band_energy(torch.fft.fft(y-observed, norm='ortho'), rms, masks)
    torch.testing.assert_close(energy(result), torch.minimum(energy(fine), energy(coarse)))


def test_bound_has_finite_gradient_and_does_not_train_the_coarse_budget():
    rng = torch.Generator().manual_seed(78)
    coarse = torch.randn(2, 6, 256, generator=rng, dtype=torch.complex64).requires_grad_()
    fine = torch.randn(2, 6, 256, generator=rng, dtype=torch.complex64).requires_grad_()
    zero = torch.zeros_like(coarse)
    rms = torch.ones(2, 6, 1)
    _, refined = bound_refinement(zero, coarse, fine, rms)
    refined.real.square().mean().backward()
    assert coarse.grad is None and fine.grad.isfinite().all()
    fine.grad = None
    _, empty = bound_refinement(zero, zero, fine, rms)
    empty.real.square().mean().backward()
    assert not empty.count_nonzero() and fine.grad.isfinite().all()


def test_recipe_keeps_old_stage_tuple_and_runs_configured_bound():
    original = FitRecipe('example', 1, None, None, None)
    bounded = FitRecipe('bounded', 1, None, None, None, stage_transform=bound_refinement)
    observed = torch.zeros(1, 6, 256, dtype=torch.complex64)
    edit = torch.randn_like(observed)
    edit -= edit.mean(-1, keepdim=True)
    stages = (edit, 2 * edit)
    scale = torch.ones(1, 6, 1)
    assert original.transform_stages(observed, stages, scale) is stages
    first, second = bounded.transform_stages(observed, stages, scale)
    assert torch.equal(first, edit)
    torch.testing.assert_close(second, edit)

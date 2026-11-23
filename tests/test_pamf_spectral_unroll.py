import torch

from experiments.cric_dura_comparison.g_lineage import skeleton
from experiments.pamf_unrolled_restore.spectral import SpectralUnroll, paired_loss


def test_unroll_preserves_initialized_r17_and_reaches_all_three_spectral_heads():
    torch.set_num_threads(2)
    torch.manual_seed(81)
    r17 = skeleton('r17').eval()
    value = torch.randn(2, 2, 256) * .3
    desired = value.roll(3, dims=-1)
    with torch.no_grad():
        control = r17.stage_outputs(value)[0]
    model = SpectralUnroll(r17).eval()
    stages = model.stage_outputs(value)[0]
    assert len(stages) == 3
    for new, old in zip(stages, control, strict=True):
        torch.testing.assert_close(new, old, atol=0., rtol=0.)
    paired_loss(stages, desired).mean().backward()
    for stage in model.prior:
        grad = stage.spectral.out.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    assert all(param.grad is None for param in model.frozen.parameters())


def test_paired_loss_prefers_complex_alignment_and_is_finite_for_silent_pairs():
    rng = torch.Generator().manual_seed(11)
    target = torch.randn(2, 2, 256, generator=rng)
    rotated = torch.stack((-target[:, 1], target[:, 0]), dim=1).requires_grad_()
    aligned = paired_loss((target, target, target), target)
    wrong = paired_loss((rotated, rotated, rotated), target)
    assert (aligned == 0).all() and (wrong > 0).all()
    torch.autograd.grad(wrong.sum(), rotated)
    zero = torch.zeros_like(target).requires_grad_()
    grad, = torch.autograd.grad(paired_loss((zero, zero, zero), zero.detach()).sum(), zero)
    assert torch.isfinite(grad).all() and not grad.any()

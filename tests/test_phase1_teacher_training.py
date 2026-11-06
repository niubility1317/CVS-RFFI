import torch

from experiments.phase1_dura_orthogonal.benchmark_teacher_precompute import targets
from experiments.phase1_dura_orthogonal.benchmark_teacher_training import cached_teacher_loss
from experiments.phase1_dura_orthogonal.masked_prior_constraint import masked_prior_loss


def test_cached_loss_and_gradient_match_original():
    class Teacher(torch.nn.Module):
        def forward(self, observed, mask, scale):
            return observed.square() / scale + mask[:, None].to(observed.dtype)*scale

    torch.manual_seed(72)
    clean = torch.randn(35, 2, 256)
    starts = torch.randint(1, 224, (35,))
    a = torch.randn_like(clean, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    prior = Teacher()
    target = targets(prior, clean, starts, 32)
    original = masked_prior_loss(prior, a, clean, starts)
    cached = cached_teacher_loss(prior, b, clean, starts, target)
    assert torch.equal(original, cached)
    original.backward()
    cached.backward()
    assert torch.equal(a.grad, b.grad)

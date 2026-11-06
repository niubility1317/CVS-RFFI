import torch

from experiments.phase1_dura_orthogonal.train_source import source_loss


def test_tail_guard_penalizes_largest_relative_harm_and_keeps_gradients():
    clean = torch.ones(4, 2, 256)
    anchor = clean.clone()
    output = torch.stack([clean[i] + v for i, v in enumerate((.1, .2, .3, .4))])
    output.requires_grad_()
    ordinary = source_loss(output, anchor, clean, objective='dura_guard')
    tail = source_loss(output, anchor, clean, objective='tail_excess_guard')
    torch.testing.assert_close(tail - ordinary, torch.tensor(.5 * .4**2))
    ordinary_grad, = torch.autograd.grad(ordinary, output, retain_graph=True)
    tail_grad, = torch.autograd.grad(tail, output)
    torch.testing.assert_close(tail_grad[:3], ordinary_grad[:3])
    assert torch.isfinite(tail_grad).all()
    assert tail_grad[3].abs().sum() > ordinary_grad[3].abs().sum()


def test_tail_guard_adds_no_penalty_when_all_samples_improve():
    clean = torch.ones(4, 2, 256)
    anchor = clean + .4
    output = clean + .2
    torch.testing.assert_close(source_loss(output, anchor, clean, objective='tail_excess_guard'),
                               source_loss(output, anchor, clean, objective='dura_guard'))


def test_tail_guard_has_no_extra_gradient_at_exact_anchor():
    clean = torch.ones(4, 2, 256)
    anchor = clean + .4
    output = anchor.clone().requires_grad_()
    ordinary = source_loss(output, anchor, clean, objective='dura_guard')
    tail = source_loss(output, anchor, clean, objective='tail_excess_guard')
    ordinary_grad, = torch.autograd.grad(ordinary, output, retain_graph=True)
    tail_grad, = torch.autograd.grad(tail, output)
    torch.testing.assert_close(tail_grad, ordinary_grad)

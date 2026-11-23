import torch

from experiments.pamf_unrolled_restore.joint import EDIT_RMS, JointUnroll, action, adjoint, joint_loss


def test_widely_linear_action_uses_the_real_adjoint():
    generator = torch.Generator().manual_seed(9)
    x, y, main, image = (
        torch.randn(2, 3, 64, dtype=torch.complex128, generator=generator) for _ in range(4)
    )
    left = (action(x, main, image).conj() * y).sum().real
    right = (x.conj() * adjoint(y, main, image)).sum().real
    torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)


def test_joint_unroll_gradients_and_protected_carrier():
    torch.manual_seed(42)
    model = JointUnroll()
    y = torch.randn(4, 2, 64, dtype=torch.complex64)
    noise = .15 * torch.randn_like(y)
    target = y - noise
    q = torch.rand(4, 2, 1)
    for iterative in (False, True):
        stages = model(y, q, iterative=iterative)
        torch.testing.assert_close(stages[-1][0], torch.fft.fft(y, norm='ortho'))
        loss, detail = joint_loss(stages, target, noise, y)
        assert torch.isfinite(loss) and detail > 0
        model.zero_grad()
        loss.backward()
        assert all(block.out.weight.grad.isfinite().all() and block.out.weight.grad.norm() > 0
                   for block in model.steps)
    with torch.no_grad():
        for step in model.steps:
            step.out.bias.fill_(2.)
        # Nonzero updates at frequencies other than the preserved carrier.
        for step in model.steps:
            step.out.weight.normal_(std=.03)
        expected = torch.fft.fft(y, norm='ortho')
        final = model(y, q)[-1][0]
        flat = model(y, q, iterative=False)[-1][0]
        assert not torch.allclose(final, flat)
        torch.testing.assert_close(final[..., 0], expected[..., 0], rtol=0, atol=0)
        assert ((final - expected).abs().square().mean(-1).sqrt() <= EDIT_RMS).all()

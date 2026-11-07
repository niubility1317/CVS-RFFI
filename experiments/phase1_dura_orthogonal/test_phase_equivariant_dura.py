import torch

from experiments.phase1_dura_orthogonal.phase_equivariant_dura import PhaseEquivariantDura


def rotate(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c*x[:, 0]-s*x[:, 1], s*x[:, 0]+c*x[:, 1]), dim=1)


def test_zero_head_and_clean_gate_preserve_dura_exactly():
    model = PhaseEquivariantDura(width=4)
    x, anchor, innovation = (torch.randn(4, 2, 256) for _ in range(3))
    variance = torch.tensor([0., .01, .1, 1.])
    with torch.no_grad():
        assert torch.equal(model(x, anchor, innovation, variance), anchor)
        model.head.real.normal_(0., .1)
        model.head.imag.normal_(0., .1)
        result = model(x, anchor, innovation, variance)
    assert torch.equal(result[0], anchor[0])


def test_conditional_common_phase_rotation_equivariance():
    model = PhaseEquivariantDura(width=4).eval()
    with torch.no_grad():
        model.head.real.normal_(0., .1)
        model.head.imag.normal_(0., .1)
    x, anchor, innovation = (torch.randn(3, 2, 256) for _ in range(3))
    variance = torch.full((3,), .1)
    angle = torch.tensor(.73)
    with torch.no_grad():
        expected = rotate(model(x, anchor, innovation, variance), angle)
        actual = model(rotate(x, angle), rotate(anchor, angle), rotate(innovation, angle), variance)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_both_complex_head_components_receive_gradients():
    model = PhaseEquivariantDura(width=4)
    x, anchor, innovation, clean = (torch.randn(4, 2, 256) for _ in range(4))
    result = model(x, anchor, innovation, torch.full((4,), .1))
    (result-clean).square().mean().backward()
    for param in (model.head.real, model.head.imag):
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0

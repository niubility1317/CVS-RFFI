"""Synthetic no-query checks for joint complex gain and proposal restoration."""
import torch

from experiments.phase1_joint_particle.particle import (
    JointParticleRecovery, _repeat_action, fitted_gain, physical_direction,
)


class FakeAction:
    """Exactly adjoint widely-linear action with no convolution."""
    def __init__(self, kernel, main, image):
        self.kernel, self.main, self.image = kernel, main, image

    def __call__(self, value):
        return self.main * value + self.image * value.conj()

    def adjoint(self, value):
        return self.main.conj() * value + self.image * value.conj()


def fixture(batch=2):
    generator = torch.Generator().manual_seed(12)
    received = torch.randn((batch, 2, 256), generator=generator)
    anchor = torch.randn((batch, 2, 256), generator=generator)
    innovation = torch.randn((batch, 2, 256), generator=generator)
    variance = torch.tensor([0.02 + i * .01 for i in range(batch)])
    kernel = torch.ones((batch, 1), dtype=torch.complex64)
    main = torch.full((batch, 256), 1.0 + .05j, dtype=torch.complex64)
    image = torch.full((batch, 256), .02 - .01j, dtype=torch.complex64)
    return received, anchor, innovation, variance, FakeAction(kernel, main, image)


def test_initial_model_is_exact_anchor():
    inputs = fixture()
    model = JointParticleRecovery(width=8, steps=2).eval()
    output = model(*inputs)
    torch.testing.assert_close(output, inputs[1], rtol=0, atol=0)


def test_gain_solves_ridge_and_is_per_record():
    y, x, _, _, action = fixture()
    target = torch.complex(y[:, 0], y[:, 1])
    prediction = action(torch.complex(x[:, 0], x[:, 1]))
    reference_energy = prediction.abs().square().sum(-1)
    gain = fitted_gain(action, x, y, reference_energy)
    ridge = .1 * reference_energy.clamp_min(1e-8)
    stationary = (prediction.conj() * (gain[:, None] * prediction - target)).sum(-1) + ridge * (gain - 1)
    torch.testing.assert_close(stationary, torch.zeros_like(stationary), rtol=0, atol=1e-4)
    one = FakeAction(action.kernel[:1], action.main[:1], action.image[:1])
    torch.testing.assert_close(gain[:1], fitted_gain(one, x[:1], y[:1], reference_energy[:1]))


def test_real_linear_adjoint_identity():
    y, x, _, _, action = fixture()
    a = torch.complex(x[:, 0], x[:, 1])
    b = torch.complex(y[:, 0], y[:, 1])
    lhs = (action(a).conj() * b).real.sum(-1)
    rhs = (a.conj() * action.adjoint(b)).real.sum(-1)
    torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-6)


def test_gradients_flow_through_physical_update():
    y, x, u, v, action = fixture()
    model = JointParticleRecovery(width=8, steps=2)
    with torch.no_grad():
        model.output.weight.normal_(std=.01)
    target = x * .7
    out = model(y, x, u, v, action)
    loss = (out - target).square().mean()
    loss.backward()
    assert model.output.weight.grad is not None
    assert torch.isfinite(model.output.weight.grad).all()
    assert model.output.weight.grad.abs().sum() > 0
    assert torch.isfinite(out).all()
    assert action.kernel.grad is None and action.main.grad is None and action.image.grad is None


def test_record_independence_and_fixed_latent_determinism():
    y, x, u, v, action = fixture()
    model = JointParticleRecovery(width=8, steps=2).eval()
    with torch.no_grad():
        model.output.weight.fill_(.01)
        full = model(y, x, u, v, action)
        again = model(y, x, u, v, action)
        first = FakeAction(action.kernel[:1], action.main[:1], action.image[:1])
        last = FakeAction(action.kernel[1:], action.main[1:], action.image[1:])
        separate = torch.cat((model(y[:1], x[:1], u[:1], v[:1], first),
                              model(y[1:], x[1:], u[1:], v[1:], last)))
    torch.testing.assert_close(full, again, rtol=0, atol=0)
    torch.testing.assert_close(full, separate, rtol=1e-5, atol=2e-6)


def test_particle_order_and_nonfinite_rejection():
    y, x, u, v, action = fixture()
    repeated = _repeat_action(action, 3)
    assert repeated.main.shape == (6, 256)
    torch.testing.assert_close(repeated.main[0], action.main[0])
    torch.testing.assert_close(repeated.main[3], action.main[1])
    model = JointParticleRecovery(width=8, steps=1)
    try:
        model(y, x, u, torch.tensor([1.0, float('nan')]), action)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite variance must be rejected')
    try:
        model(y, x, u, v, action, latent=torch.zeros(2, 0, 2, 256))
    except ValueError:
        pass
    else:
        raise AssertionError('Empty particle set must be rejected')


def test_gain_adjoint_gradient_matches_finite_difference():
    y, x, _, _, action = fixture(1)
    x = x.requires_grad_(True)
    energy = torch.full((1,), 500.0)
    loss = physical_direction(action, x, y, energy).square().mean()
    grad, = torch.autograd.grad(loss, x)
    index = (0, 0, 17)
    with torch.no_grad():
        plus, minus = x.detach().clone(), x.detach().clone()
        plus[index] += .002
        minus[index] -= .002
        numerical = (physical_direction(action, plus, y, energy).square().mean() -
                     physical_direction(action, minus, y, energy).square().mean()) / .004
    torch.testing.assert_close(grad[index], numerical, rtol=.02, atol=1e-4)


def test_existing_numpy_torch_boundary():
    import numpy as np
    positions = np.asarray([3, 1, 4], dtype=np.int64)
    index = torch.tensor(positions.tolist(), dtype=torch.long)
    values = torch.arange(6, dtype=torch.int64)[index].contiguous()
    output = np.from_dlpack(values).copy()
    assert output.tolist() == [3, 1, 4]


def test_single_particle_reuses_action():
    *_, action = fixture()
    assert _repeat_action(action, 1) is action


def test_polyphase_block_matches_native_forward_and_gradients():
    from torch.nn import functional as F
    from experiments.phase1_joint_particle.particle import ResidualBlock
    torch.manual_seed(29)
    block = ResidualBlock(width=8, dilation=4)
    a = torch.randn(2, 8, 256, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    actual = block(a)
    hidden = F.gelu(block.norm1(F.conv1d(b, block.first.weight, block.first.bias, padding=8, dilation=4)))
    expected = F.gelu(b + block.norm2(block.second(hidden)))
    params = tuple(block.parameters())
    actual_grad = torch.autograd.grad(actual.square().mean(), (a, *params))
    expected_grad = torch.autograd.grad(expected.square().mean(), (b, *params))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, rtol=2e-4, atol=2e-6)


def test_reused_physical_forward_matches_original_formula():
    y, x, _, _, action = fixture()
    x = x.requires_grad_(True)
    reference = torch.full((len(x),), 500.0)
    gain = fitted_gain(action, x, y, reference)
    target = torch.complex(y[:, 0], y[:, 1])
    signal = torch.complex(x[:, 0], x[:, 1])
    original = action.adjoint(gain.conj()[:, None] * (target - gain[:, None] * action(signal)))
    original = torch.stack((original.real, original.imag), 1)
    reused = physical_direction(action, x, y, reference)
    old_grad, = torch.autograd.grad(original.square().mean(), x, retain_graph=True)
    new_grad, = torch.autograd.grad(reused.square().mean(), x)
    torch.testing.assert_close(reused, original, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(new_grad, old_grad, rtol=1e-5, atol=1e-6)

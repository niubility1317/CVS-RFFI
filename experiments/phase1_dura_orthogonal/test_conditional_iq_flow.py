import torch
import pytest

from experiments.phase1_dura_orthogonal.conditional_iq_flow import ConditionalIQFlow, DirectIQFlow, flow_matching_loss, sample_flow_times, sample_epoch_flow_times


def inputs(batch=3):
    generator = torch.Generator().manual_seed(42)
    received = torch.randn(batch, 2, 256, generator=generator)
    anchor = torch.randn(batch, 2, 256, generator=generator)
    innovation = torch.randn(batch, 2, 256, generator=generator)
    clean = torch.randn(batch, 2, 256, generator=generator)
    variance = torch.full((batch,), .1)
    return received, anchor, innovation, clean, variance


def test_zero_initialized_flow_starts_at_exact_anchor():
    x, anchor, innovation, _, variance = inputs()
    model = ConditionalIQFlow(width=16, steps=4)
    assert torch.equal(model(x, anchor, innovation, variance), anchor)


def test_flow_matching_provides_finite_gradient_with_frozen_inputs():
    x, anchor, innovation, clean, variance = inputs()
    model = ConditionalIQFlow(width=16, steps=4)
    times = torch.tensor([0., .5, 1.])
    loss = flow_matching_loss(model, x, anchor, innovation, variance, clean, times)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    assert torch.isfinite(model.out.weight.grad).all() and model.out.weight.grad.abs().sum() > 0
    assert all(t.grad is None for t in (x, anchor, innovation, clean, variance))


def test_euler_integrator_follows_constant_velocity():
    x, anchor, innovation, clean, variance = inputs()
    velocity = clean-anchor
    class ConstantVelocity(ConditionalIQFlow):
        def velocity(self, state, received, reference, direction, noise, time):
            return velocity
    model = ConstantVelocity(width=8, steps=4)
    torch.testing.assert_close(model(x, anchor, innovation, variance), clean, atol=1e-6, rtol=1e-6)


def test_loss_rejects_invalid_times():
    x, anchor, innovation, clean, variance = inputs()
    model = ConditionalIQFlow(width=8, steps=4)
    with pytest.raises(ValueError, match='time'):
        flow_matching_loss(model, x, anchor, innovation, variance, clean, torch.tensor([0., 1.1, .5]))


def test_trusted_generated_times_preserve_loss_and_gradients():
    x, anchor, innovation, clean, variance = inputs()
    reference = ConditionalIQFlow(width=8, steps=4)
    fast = ConditionalIQFlow(width=8, steps=4)
    fast.load_state_dict(reference.state_dict())
    times = sample_flow_times(len(x), torch.Generator().manual_seed(71), origin_fraction=.5)
    baseline_loss = flow_matching_loss(reference, x, anchor, innovation, variance, clean, times)
    fast_loss = flow_matching_loss(fast, x, anchor, innovation, variance, clean, times, validate_time=False)
    baseline_loss.backward()
    fast_loss.backward()
    assert torch.equal(baseline_loss, fast_loss)
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(reference.parameters(), fast.parameters()))


def test_direct_flow_uses_received_iq_only_and_initially_reproduces_it():
    x, anchor, innovation, _, variance = inputs()
    model = DirectIQFlow(width=8, steps=4)
    assert torch.equal(model(x, anchor, innovation, variance), x)
    with torch.no_grad():
        model.out.bias.fill_(.1)
    actual = model(x, anchor, innovation, variance)
    assert torch.equal(actual, model(x, torch.zeros_like(anchor), torch.zeros_like(innovation), variance * 10))


def test_mixed_time_sampling_includes_actual_inference_origin_and_is_reproducible():
    first = sample_flow_times(4096, torch.Generator().manual_seed(71), origin_fraction=.5)
    second = sample_flow_times(4096, torch.Generator().manual_seed(71), origin_fraction=.5)
    assert torch.equal(first, second) and ((first >= 0) & (first < 1)).all()
    assert .45 < (first == 0).float().mean() < .55
    assert (first > 0).any()


@pytest.mark.parametrize('origin_fraction', [0., .5])
def test_epoch_sampling_matches_each_batch_and_generator_state(origin_fraction):
    per_batch = torch.Generator().manual_seed(2026110713)
    per_epoch = torch.Generator().manual_seed(2026110713)
    for _ in range(5):
        original = torch.cat([sample_flow_times(32, per_batch, origin_fraction=origin_fraction) for _ in range(3072)])
        combined = sample_epoch_flow_times(98304, per_epoch, origin_fraction=origin_fraction)
        assert torch.equal(original, combined)
        assert torch.equal(per_batch.get_state(), per_epoch.get_state())


def test_epoch_sampling_rejects_partial_batch():
    with pytest.raises(ValueError, match='batch'):
        sample_epoch_flow_times(33, torch.Generator().manual_seed(1), origin_fraction=0.)

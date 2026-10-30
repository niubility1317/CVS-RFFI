from types import SimpleNamespace
import numpy as np
import torch

from experiments.pamf_action_restore.operators import Action
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_dura_direct_train import runner as base, phase_pair


def example():
    rng = torch.Generator().manual_seed(37)
    x = torch.randn(12, 2, 32, generator=rng)
    kernel = torch.complex(torch.randn(12, 5, generator=rng), torch.randn(12, 5, generator=rng)) / 4
    main = torch.complex(torch.ones(12, 32), torch.randn(12, 32, generator=rng) / 8)
    image = torch.complex(torch.randn(12, 32, generator=rng), torch.randn(12, 32, generator=rng)) / 3
    action = Action(kernel, main, image)
    noise = torch.randn(x.shape, generator=rng) / 10
    return x, pair(action(complex_iq(x))) + noise, action, noise


def test_rotation_conjugates_full_real_linear_action_and_preserves_noise_and_power():
    x, y, action, noise = example()
    angle = torch.linspace(-3, 3, len(x))
    y2, x2, transformed = phase_pair.rotate_pair(y, x, action, angle)
    unit = torch.polar(torch.ones_like(angle), angle)[:, None]
    torch.testing.assert_close(complex_iq(y2) - transformed(complex_iq(x2)),
                               unit * complex_iq(noise), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(x2.square().sum(1), x.square().sum(1), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(transformed.adjoint(unit * complex_iq(y)),
                               unit * action.adjoint(complex_iq(y)), atol=1e-5, rtol=1e-5)
    wrong = action(complex_iq(x2))
    assert (wrong - unit * action(complex_iq(x))).abs().mean() > .1


def test_composition_and_zero_rotation_do_not_mutate_cached_action_or_iq():
    x, y, action, _ = example()
    before = [v.clone() for v in (x, y, action.kernel, action.main, action.image)]
    a = torch.linspace(-1, 2, len(x)); b = torch.linspace(2, -2, len(x))
    ya, xa, aa = phase_pair.rotate_pair(y, x, action, a)
    yab, xab, aab = phase_pair.rotate_pair(ya, xa, aa, b)
    yc, xc, ac = phase_pair.rotate_pair(y, x, action, a + b)
    torch.testing.assert_close(yab, yc, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(xab, xc, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(aab.image, ac.image, atol=2e-6, rtol=2e-6)
    yz, xz, az = phase_pair.rotate_pair(y, x, action, torch.zeros(len(x)))
    assert torch.equal(yz, y) and torch.equal(xz, x) and torch.equal(az.image, action.image)
    assert all(torch.equal(a, b) for a, b in zip(before, (x, y, action.kernel, action.main, action.image)))


def test_one_objective_call_and_independent_repeatable_phase_stream():
    x, y, action, _ = example()
    variance = torch.full((len(x),), .1); probes = [complex_iq(x)]
    calls = []
    def objective(model, received, clean, actual, noise, current_probes):
        calls.append((received, clean, actual, noise, current_probes))
        return received.square().mean((1, 2)), {}
    first = phase_pair.PhaseObjective(objective)
    second = phase_pair.PhaseObjective(objective)
    before = torch.get_rng_state().clone()
    first(None, y, x, action, variance, probes)
    second(None, y, x, action, variance, probes)
    assert len(calls) == 2 and calls[0][3] is variance and calls[0][4] is probes
    assert torch.equal(calls[0][0], calls[1][0]) and torch.equal(before, torch.get_rng_state())
    assert first.summary() == second.summary() and first.summary()['seen'] == len(x)
    assert first.summary()['calls'] == 1 and sum(first.summary()['angle_bins']) == len(x)


def test_recipe_preserves_budget_f_free_inference_and_routes_child_processes():
    keys = ('RUN', 'configuration', 'command', 'learning_rate', 'direct_objective', 'smoke', 'train')
    original = {key: getattr(base, key) for key in keys}
    try:
        phase_pair.configure(); config = base.configuration()
        assert (config['epochs'], config['updates'], config['training_stages']) == (20, 4940, 1)
        assert not config['training_F_access'] and config['ancestor_checkpoints'] == []
        assert config['G_forwards_per_training_visit'] == 1 and config['source_phase_probability'] == 1
        assert config['allowed_GPUs'] == [4, 6, 7] and config['max_total_visible_GPU_processes'] == 3
        args = SimpleNamespace(root='root', commit='a' * 40)
        assert all(base.command(args, mode, 'scratch_joint', 'target')[3] == phase_pair.MODULE
                   for mode in ('worker', 'score'))
    finally:
        for key, value in original.items():
            setattr(base, key, value)

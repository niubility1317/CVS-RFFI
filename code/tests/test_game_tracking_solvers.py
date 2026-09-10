import copy

import pytest
import torch
from torch import nn

from cvsrffi.game_tracking.jacobian_audit import local_jacobian_audit
from cvsrffi.game_tracking.parameter_roles import parameter_roles
from cvsrffi.game_tracking.response_tracking import damped_cg, implicit_head_response
from cvsrffi.game_tracking.solvers import GameSolver


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


class Game(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(1., dtype=torch.float64))
        self.adv_head = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.adv_head.weight.data.fill_(2.)

    def forward(self):
        return Reverse.apply(self.theta) * self.adv_head.weight.sum()


@pytest.mark.parametrize("mode,factor", [
    ("simultaneous", lambda h: 1+h*h),
    ("extragradient", lambda h: 1-h*h+h**4),
    ("heun", lambda h: 1+h**4/4),
])
def test_bilinear_exact_radius(mode, factor):
    model, eta = Game(), .2
    result = GameSolver(model, torch.optim.SGD(model.parameters(), lr=eta), mode).step(model)
    radius = sum(float(p.detach().square().sum()) for p in model.parameters())
    assert radius == pytest.approx(5*factor(eta), abs=1e-12)
    assert result.accepted
    assert result.origin_grad_norm == pytest.approx(5**.5)


@pytest.mark.parametrize("mode", ["extragradient", "heun", "head_lookahead"])
def test_adamw_predictor_uses_old_state_and_one_formal_step(mode):
    model = Game()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.05, weight_decay=.1)
    GameSolver(model, optimizer).step(model)
    original_model, original_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    reference = Game()
    reference.load_state_dict(original_model)
    refopt = torch.optim.AdamW(reference.parameters(), lr=.05, weight_decay=.1)
    refopt.load_state_dict(copy.deepcopy(original_optimizer))
    first = torch.autograd.grad(reference(), tuple(reference.parameters()))
    for name, p in reference.named_parameters():
        index = list(dict(reference.named_parameters())).index(name)
        p.grad = first[index].clone() if mode != "head_lookahead" or name.startswith("adv_head.") else None
    refopt.step()
    second = torch.autograd.grad(reference(), tuple(reference.parameters()))
    reference.load_state_dict(original_model)
    refopt.load_state_dict(copy.deepcopy(original_optimizer))
    for p, a, b in zip(reference.parameters(), first, second):
        p.grad = (a+b)*.5 if mode == "heun" else b.clone()
    refopt.step()
    result = GameSolver(model, optimizer, mode).step(model)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        assert optimizer.state[p]["step"].item() == 2
        torch.testing.assert_close(optimizer.state[p]["exp_avg"], refopt.state[q]["exp_avg"])
        torch.testing.assert_close(optimizer.state[p]["exp_avg_sq"], refopt.state[q]["exp_avg_sq"])
    assert result.algorithm.startswith("adamw_isolated_predictor")


def test_shared_parameters_have_one_owner_and_inactive_grad_none():
    model = Game()
    model.alias = model.adv_head
    model.unused = nn.Parameter(torch.tensor(4., dtype=torch.float64))
    roles = parameter_roles(model)
    assert len(roles) == 3
    assert len([row for row in roles if len(row.aliases) == 2]) == 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=.2)
    GameSolver(model, optimizer, "extragradient").step(model)
    assert model.unused.grad is None
    assert model.unused.item() == 4
    assert model.unused not in optimizer.state


def test_bn_and_dropout_replay_commit_entire_first_closure_only():
    torch.manual_seed(4)
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(.5), nn.Linear(3, 1)).double()
    reference = copy.deepcopy(model)
    x = torch.randn(8, 3, dtype=torch.float64)
    rng = torch.get_rng_state()
    reference(x); reference(x * 2)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    seen = []
    handle = model[1].register_forward_hook(lambda _, inp, out: seen.append(out.detach().ne(0)))
    solver = GameSolver(model, torch.optim.SGD(model.parameters(), lr=.01), "extragradient")
    solver.step(lambda: model(x).square().mean()+model(x*2).square().mean())
    handle.remove()
    assert model[0].num_batches_tracked.item() == 2
    torch.testing.assert_close(model[0].running_mean, reference[0].running_mean)
    torch.testing.assert_close(model[0].running_var, reference[0].running_var)
    assert torch.equal(seen[0], seen[2]) and torch.equal(seen[1], seen[3])
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_nonfinite_corrector_rolls_back_weights_moments_buffers_rng():
    model = Game()
    model.register_buffer("counter", torch.tensor(0.))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    GameSolver(model, optimizer).step(model)
    before, moments, rng = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), torch.get_rng_state()
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        model.counter.add_(1)
        torch.rand(3)
        return model() * (1 if calls == 1 else float("nan"))

    result = GameSolver(model, optimizer, "extragradient", nonfinite="skip").step(closure)
    assert not result.accepted and result.failure_stage == "corrector:loss"
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for key, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, moments["state"][key][name], rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())


def test_optimistic_resume_and_reset():
    model = Game()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    solver = GameSolver(model, optimizer, "optimistic")
    solver.step(model)
    restored = copy.deepcopy(model)
    restored_opt = torch.optim.SGD(restored.parameters(), lr=.1)
    restored_opt.load_state_dict(optimizer.state_dict())
    restored_solver = GameSolver(restored, restored_opt, "optimistic")
    restored_solver.load_state_dict(solver.state_dict())
    solver.step(model); restored_solver.step(restored)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    solver.reset_history("curriculum_changed")
    ordinary = copy.deepcopy(model)
    GameSolver(ordinary, torch.optim.SGD(ordinary.parameters(), lr=.1)).step(ordinary)
    solver.step(model)
    for a, b in zip(model.parameters(), ordinary.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert solver.reset_reasons[-1]["reason"] == "curriculum_changed"


def test_alternating_updates_head_once_preserves_encoder_input_gradient():
    model = Game()
    optimizer = torch.optim.SGD(model.parameters(), lr=.2)
    optimizer.zero_grad(set_to_none=True)
    model.adv_head.weight.grad = model.theta.detach().reshape(1, 1)
    optimizer.step()
    solver = GameSolver(model, optimizer, "alternating")
    result = solver.step(model, exclude_head=True)
    assert model.adv_head.weight.item() == pytest.approx(1.8)
    assert model.theta.item() == pytest.approx(1.36)
    assert result.field_evaluations == 1


def test_explicit_jacobian_bilinear_and_finite_difference():
    result = local_jacobian_audit(lambda w: torch.stack((-w[1], w[0])),
                                  torch.tensor([1., 2.], dtype=torch.float64),
                                  finite_difference_step=1e-5)
    assert result.valid and result.rotation_ratio == pytest.approx(1.)
    torch.testing.assert_close(result.jvp, -result.vjp)
    assert result.finite_difference_relative_error < 1e-9
    with pytest.raises(ValueError, match="GRL"):
        local_jacobian_audit(lambda w: w, torch.ones(2), field_kind="grl_hessian")


def test_implicit_response_matches_regularized_linear_solution():
    theta = torch.tensor([1., -2.], dtype=torch.float64, requires_grad=True)
    phi = torch.tensor([.2, .4], dtype=torch.float64, requires_grad=True)
    loss = .5 * (phi-theta).square().sum() + .3 * phi.square().sum()
    delta = torch.tensor([.1, -.1], dtype=torch.float64)
    result = implicit_head_response(loss, [phi], [theta], [delta], damping=.2, positive_definite=True, tolerance=1e-12)
    assert result.accepted
    torch.testing.assert_close(result.delta, delta / 1.8)


def test_response_non_pd_and_budget_fallback():
    rhs = torch.tensor([1., 2.], dtype=torch.float64)
    assert not damped_cg(lambda v: v, rhs, damping=.1).accepted
    negative = damped_cg(lambda v: -v, rhs, damping=.1, positive_definite=True)
    assert negative.negative_curvature and negative.delta is None
    result = damped_cg(lambda v: torch.tensor([1., 3.])*v, rhs,
                       damping=.1, max_iterations=1, positive_definite=True, tolerance=1e-14)
    assert not result.accepted and result.reason == "iteration_budget_exhausted"


def test_scaler_growth_once_and_atomic_backoff():
    model = Game()
    scaler = torch.amp.GradScaler("cpu", init_scale=8., growth_interval=2)
    solver = GameSolver(model, torch.optim.SGD(model.parameters(), lr=.1), "extragradient", scaler=scaler, nonfinite="skip")
    solver.step(model)
    assert scaler.get_scale() == 8. and scaler.state_dict()["_growth_tracker"] == 1
    solver.step(model)
    assert scaler.get_scale() == 16. and scaler.state_dict()["_growth_tracker"] == 0
    before = copy.deepcopy(model.state_dict())
    assert not solver.step(lambda: model() * float("nan")).accepted
    assert scaler.get_scale() == 8.
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_frozen_head_keeps_input_derivative():
    model = Game()
    model.adv_head.requires_grad_(False)
    optimizer = torch.optim.SGD([model.theta], lr=.1)
    GameSolver(model, optimizer).step(model)
    assert model.theta.item() == pytest.approx(1.2)
    assert model.adv_head.weight.item() == 2


def test_plain_mode_matches_native_adamw_update():
    model = Game()
    reference = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=.05, weight_decay=.2)
    refopt = torch.optim.AdamW(reference.parameters(), lr=.05, weight_decay=.2)
    for _ in range(3):
        refopt.zero_grad(set_to_none=True)
        reference().backward()
        refopt.step()
        GameSolver(model, opt).step(model)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(opt.state[p][key], refopt.state[q][key], rtol=0, atol=0)


def test_alternating_linear_map_unit_determinant_and_bounded_eigenvalues():
    def advance(theta, phi):
        model = Game()
        with torch.no_grad():
            model.theta.fill_(theta)
            model.adv_head.weight.fill_(phi)
        optimizer = torch.optim.SGD(model.parameters(), lr=.3)
        model.adv_head.weight.grad = model.theta.detach().reshape(1, 1)
        optimizer.step()
        GameSolver(model, optimizer, "alternating").step(model, exclude_head=True)
        return torch.tensor([model.theta.item(), model.adv_head.weight.item()], dtype=torch.float64)
    matrix = torch.stack((advance(1., 0.), advance(0., 1.)), dim=1)
    assert torch.linalg.det(matrix).item() == pytest.approx(1., abs=1e-12)
    torch.testing.assert_close(torch.linalg.eigvals(matrix).abs(), torch.ones(2, dtype=torch.float64))


def test_full_predictor_cannot_silently_exclude_head():
    model = Game()
    solver = GameSolver(model, torch.optim.SGD(model.parameters(), lr=.1), "extragradient")
    with pytest.raises(ValueError, match="all active head"):
        solver.step(model, exclude_head=True)

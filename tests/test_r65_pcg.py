import pytest
import torch

from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_action_restore.operators import Action, real_dot
from experiments.pamf_r46_delayed_prune.models import CompactSuccessor
from experiments.pamf_r65_four_route import pcg


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(69)


def _action(batch=2, count=9, length=6, dtype=torch.complex128):
    return Action(torch.randn(batch, length, dtype=dtype)*.3,
                  1+torch.randn(batch, count, dtype=dtype)*.2,
                  torch.randn(batch, count, dtype=dtype)*.25)


def _real(value):
    return torch.cat((value.real, value.imag), -1)


def _complex(value):
    return torch.complex(*value.chunk(2, dim=-1))


def _explicit_normal(action, variance, tau):
    """Test-only 2N system; column probes include the true reflected boundary."""
    count = action.count
    identity = torch.eye(2*count, dtype=action.main.real.dtype)
    columns = [_real(action(_complex(row[None])))[0] for row in identity]
    matrix = torch.stack(columns, -1)
    return matrix.T@matrix/variance.item()+identity/tau


def _inverse_matrix(block):
    return torch.stack((torch.stack((block.ii, block.iq), -1),
                        torch.stack((block.iq, block.qq), -1)), -2)


@pytest.mark.parametrize('count,length', [(1, 1), (7, 2), (7, 4), (8, 8)])
def test_exact_iq_blocks_match_explicit_reflected_real_normal(count, length):
    action = _action(1, count, length)
    variance = torch.tensor([[.037]], dtype=torch.float64)
    tau = .13
    normal = _explicit_normal(action, variance, tau)
    ii, iq, qq = pcg.normal_blocks(action, variance, tau)
    torch.testing.assert_close(ii[0], normal.diagonal()[:count], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(qq[0], normal.diagonal()[count:], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(iq[0], normal.diagonal(count), rtol=1e-12, atol=1e-12)
    inverse = _inverse_matrix(pcg.build_preconditioner(action, variance, tau))[0]
    blocks = torch.stack((torch.stack((ii[0], iq[0]), -1),
                          torch.stack((iq[0], qq[0]), -1)), -2)
    torch.testing.assert_close(inverse, torch.linalg.inv(blocks), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('kind', ['random', 'zero', 'rank_one_image', 'low_variance',
                                  'rank_one_low_variance'])
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_preconditioner_is_finite_spd_for_widely_linear_edge_cases(kind, dtype):
    action = _action(dtype=dtype)
    if kind == 'zero':
        action = Action(action.kernel*0, action.main*0, action.image*0)
    elif kind in ('rank_one_image', 'rank_one_low_variance'):
        action = Action(action.kernel, action.main, action.main.clone())
    variance = torch.full((2, 1), 1e-8 if 'low_variance' in kind else .1,
                          dtype=action.main.real.dtype)
    inverse = _inverse_matrix(pcg.build_preconditioner(action, variance))
    assert torch.isfinite(inverse).all()
    assert (torch.linalg.eigvalsh(inverse) > 0).all()
    if kind == 'zero':
        expected = torch.eye(2, dtype=variance.dtype)*SCALE**2
        torch.testing.assert_close(inverse, expected.expand_as(inverse))


@pytest.mark.parametrize('count,length', [(5, 3), (8, 8)])
def test_one_pcg_step_matches_explicit_real_system_and_lowers_normal_error(count, length):
    action = _action(1, count, length)
    observed, prior, initial = (torch.randn(1, count, dtype=torch.complex128) for _ in range(3))
    variance = torch.tensor([[.17]], dtype=torch.float64)
    tau = .2
    normal = _explicit_normal(action, variance, tau)
    rhs = _real(action.adjoint((observed-action(prior))/variance))[0]
    delta = _real(initial-prior)[0]
    residual = rhs-normal@delta
    ii, iq, qq = pcg.normal_blocks(action, variance, tau)
    block_matrix = torch.zeros_like(normal)
    index = torch.arange(count)
    block_matrix[index, index] = ii[0]
    block_matrix[index, index+count] = iq[0]
    block_matrix[index+count, index] = iq[0]
    block_matrix[index+count, index+count] = qq[0]
    direction = torch.linalg.solve(block_matrix, residual)
    alpha = (residual@direction)/(direction@normal@direction)
    expected_delta = delta+alpha*direction
    got = pcg.preconditioned_map(action, observed, prior, variance,
        pcg.build_preconditioner(action, variance, tau), tau, initial=initial)
    torch.testing.assert_close(_real(got-prior)[0], expected_delta, rtol=1e-12, atol=1e-12)
    assert residual@direction > 0
    solution = torch.linalg.solve(normal, rhs)
    before, after = delta-solution, expected_delta-solution
    assert after@normal@after < before@normal@before


def test_zero_residual_is_fixed_and_good_matrix_reduces_residual_energy():
    count = 7
    kernel = torch.tensor([[1.]], dtype=torch.complex128)
    action = Action(kernel, torch.linspace(.8, 1.2, count)[None].to(torch.complex128),
                    torch.ones(1, count, dtype=torch.complex128)*.07j)
    variance = torch.tensor([[.12]], dtype=torch.float64)
    prior = torch.randn(1, count, dtype=torch.complex128)
    observation = action(prior)
    preconditioner = pcg.build_preconditioner(action, variance)
    got = pcg.preconditioned_map(action, observation, prior, variance, preconditioner, initial=prior)
    torch.testing.assert_close(got, prior, rtol=0, atol=0)
    initial = torch.randn_like(prior)
    rhs = action.adjoint((observation-action(prior))/variance)
    def residual(value):
        delta = value-prior
        return rhs-action.adjoint(action(delta)/variance)-delta/SCALE**2
    before = residual(initial)
    got = pcg.preconditioned_map(action, observation, prior, variance, preconditioner, initial=initial)
    after = residual(got)
    assert (real_dot(before, before) > 0).all()
    assert (real_dot(after, after) < real_dot(before, before)).all()


def test_solver_gradients_are_finite_for_every_physical_component():
    leaves = [torch.randn(2, 5, dtype=torch.complex128, requires_grad=True),
              torch.randn(2, 9, dtype=torch.complex128, requires_grad=True),
              torch.randn(2, 9, dtype=torch.complex128, requires_grad=True)]
    action = Action(*leaves)
    observed, prior, initial = [torch.randn(2, 9, dtype=torch.complex128, requires_grad=True)
                                for _ in range(3)]
    variance = torch.full((2, 1), .07, dtype=torch.float64, requires_grad=True)
    got = pcg.preconditioned_map(action, observed, prior, variance,
        pcg.build_preconditioner(action, variance), initial=initial)
    got.abs().square().mean().backward()
    for value in [*leaves, observed, prior, initial, variance]:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


@pytest.mark.parametrize('bad', [0., -1., float('nan'), float('inf')])
def test_invalid_variance_is_rejected(bad):
    with pytest.raises(ValueError, match='variance per window'):
        pcg.build_preconditioner(_action(), torch.full((2, 1), bad))


def _alias_names(model):
    groups = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append(name)
    return sorted(tuple(names) for names in groups.values() if len(names) > 1)


def test_full_model_copies_parent_preserves_aliases_context_and_portable_state(monkeypatch):
    parent = CompactSuccessor('prune_after10').eval().requires_grad_(False)
    # Avoid only exercising the all-zero prior heads of a new shape skeleton.
    with torch.no_grad():
        for core in (parent.frozen.coarse.temporal, parent.frozen.coarse.spectral,
                     parent.frozen.estimator.core, *parent.prior):
            core.out.weight.normal_(0, .003)
    before = {key: value.clone() for key, value in parent.state_dict().items()}
    model = pcg.PreconditionedRestorer(parent).eval()
    assert sum(p.numel() for p in model.parameters()) == 273765
    assert _alias_names(model) == _alias_names(parent)
    assert all(p.requires_grad for p in model.parameters())
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert not ({id(p) for p in model.parameters()} & {id(p) for p in parent.parameters()})
    fresh = pcg.PreconditionedRestorer().eval()
    fresh.load_state_dict(model.state_dict(), strict=True)
    assert _alias_names(fresh) == _alias_names(model)
    received = torch.randn(2, 2, 256)*.1
    with torch.no_grad():
        expected_context, got_context = parent.estimate_context(received), model.estimate_context(received)
        assert torch.equal(expected_context[0], got_context[0])
        assert torch.equal(expected_context[2], got_context[2])
        for key in ('kernel', 'main', 'image'):
            assert torch.equal(getattr(expected_context[1], key), getattr(got_context[1], key))
    builds, solves = [], []
    original_build, original_solve = pcg.build_preconditioner, pcg.preconditioned_map
    def build(*args, **kwargs):
        result = original_build(*args, **kwargs)
        builds.append(result)
        return result
    def solve(*args, **kwargs):
        solves.append(args[4])
        return original_solve(*args, **kwargs)
    monkeypatch.setattr(pcg, 'build_preconditioner', build)
    monkeypatch.setattr(pcg, 'preconditioned_map', solve)
    outputs, variance, action, base = model.stage_outputs(received)
    assert len(outputs) == 3 and variance.shape == (2, 256) and base.shape == received.shape
    assert len(builds) == 1 and len(solves) == 3 and all(value is builds[0] for value in solves)
    assert all(value.shape == received.shape and torch.isfinite(value).all() for value in outputs)
    sum(value.square().mean() for value in outputs).backward()
    for component in (model.frozen.coarse.spectral, model.frozen.coarse.temporal,
                      model.frozen.estimator, *model.prior):
        gradients = [parameter.grad for parameter in component.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 0
    with torch.no_grad():
        torch.testing.assert_close(model(received), fresh(received), rtol=0, atol=0)
        model(received+.03)
    assert len(builds) == 4
    assert builds[0] is not builds[-1]
    assert all(torch.equal(value, before[key]) for key, value in parent.state_dict().items())
    assert all(parameter.grad is None for parameter in parent.parameters())
    report = model.initialization_report
    assert report['solver_changed'] and not report['exact_initial_function_equivalence']
    assert report['exact_initial_weight_equivalence'] and report['exact_initial_context_equivalence']
    assert report['new_parameters'] == report['new_calibration_IQ'] == 0


def test_wrong_parent_and_input_shapes_fail():
    parent = CompactSuccessor('prune_after5')
    with pytest.raises(ValueError, match='prune_after10'):
        pcg.PreconditionedRestorer(parent)
    with pytest.raises(ValueError, match='IQ2x256'):
        pcg.PreconditionedRestorer()(torch.zeros(1, 2, 255))
    with pytest.raises(ValueError, match='variance per window'):
        pcg.build_preconditioner(_action(), torch.ones(2, 9))


def test_full_inference_fft_count_includes_current_input_preconditioning(monkeypatch):
    model = pcg.PreconditionedRestorer().eval()
    calls = []
    for name in ('fft', 'ifft'):
        original = getattr(torch.fft, name)
        def count(*args, _original=original, **kwargs):
            calls.append(1)
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch.fft, name, count)
    with torch.no_grad():
        model(torch.randn(1, 2, 256)*.1)
    # Original context 3; original innovation/rhs/warmstart plus one H*p
    # uses 16/stage; preconditioner adds 2*(FFT taps+FFT rhs+IFFT).
    assert len(calls) == 3+3*16+6 == 57

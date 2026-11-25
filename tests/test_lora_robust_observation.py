import torch

from experiments.pamf_lora_main import dura_capacity
from experiments.pamf_lora_main.dura_capacity import ObservedAction, map_two, robust_observation_variance


def test_robust_observation_preserves_nominal_precision_and_caps_outliers():
    variance = torch.full((1, 1, 12), .25)
    residual = torch.full(variance.shape, .5, dtype=torch.complex64)
    residual[..., 4] = 2j
    residual[..., 8] = 1e3
    robust = robust_observation_variance(residual, variance)
    torch.testing.assert_close(robust, variance * torch.tensor([1., 1.75, 8.]).repeat_interleave(4))

    # With two distinct diagonal coefficients, the two CG iterations solve exactly.
    residual = residual[..., :8]
    zero = torch.zeros_like(residual)
    identity = ObservedAction(torch.ones_like(residual), zero, zero)
    nominal = map_two(identity, residual, zero, zero, variance[..., :8])
    reweighted = map_two(identity, residual, zero, zero, robust[..., :8])
    torch.testing.assert_close(reweighted[..., :4], nominal[..., :4])
    assert (reweighted[..., 4:].abs() < nominal[..., 4:].abs()).all()


@torch.no_grad()
def test_local_observation_profile_only_redistributes_solver_variance(monkeypatch):
    widths = {'test': (8, 8, 8, 8)}
    monkeypatch.setattr(dura_capacity, 'WIDTHS', widths)
    model = dura_capacity.DuraCapacity('test').eval()
    rng = torch.Generator().manual_seed(2026112593)
    observed = torch.randn((1, 6, 256), dtype=torch.complex64, generator=rng)
    level = observed.abs().square().mean(-1, keepdim=True).sqrt()
    profile = observed.real.new_tensor([.5, 2.]).repeat_interleave(128)
    used = []

    def record(action, received, current, prior, variance, residual=None):
        used.append(variance)
        return map_two(action, received, current, prior, variance, residual)

    monkeypatch.setattr(dura_capacity, 'map_two', record)
    ordinary, *_ = model.stages(observed, level, level)
    changed, *_ = model.stages(observed, level, level, solve_profile=profile)
    torch.testing.assert_close(ordinary[0], changed[0], atol=0., rtol=0.)
    torch.testing.assert_close(used[2].mean(-1), used[0].mean(-1))
    torch.testing.assert_close(used[2], used[3])
    assert not torch.equal(ordinary[-1], changed[-1])

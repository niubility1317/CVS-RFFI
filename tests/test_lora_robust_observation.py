import torch

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

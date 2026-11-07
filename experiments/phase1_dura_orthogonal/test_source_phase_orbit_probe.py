import pytest
import torch
from experiments.phase1_dura_orthogonal.source_phase_orbit_probe import rotate, evaluate_batch


def test_four_rotations_preserve_energy_and_exactly_invert():
    x = torch.randn(4, 2, 256)
    for k in range(4):
        assert torch.equal(rotate(rotate(x, k), (-k) % 4), x)
        torch.testing.assert_close(rotate(x, k).square().sum(), x.square().sum())
    with pytest.raises(ValueError):
        rotate(x, 4)


def test_identity_is_exactly_equivariant_and_preserves_signal():
    x = torch.randn(4, 2, 256)
    result = evaluate_batch(torch.nn.Identity(), x, x)
    torch.testing.assert_close(result, torch.zeros_like(result), atol=0, rtol=0)


def test_phase_orbit_averaging_removes_fixed_quadrature_bias():
    x = torch.randn(3, 2, 256)
    class Offset:
        def __call__(self, value):
            return value + .1
    result = evaluate_batch(Offset(), x, x)
    assert (result[:, 0] > 0).all() and (result[:, 2] > 0).all()
    assert (result[:, 1] < result[:, 0]).all()

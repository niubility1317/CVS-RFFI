import pytest
import torch

from experiments.phase1_dura_orthogonal import frozen_noise_response as probe


def test_noise_has_fixed_per_record_rms_and_rejects_zero_draw():
    x = torch.randn(3, 2, 256)
    random = torch.randn(3, probe.DRAWS, 2, 256)
    delta = probe.noise_deltas(x, random)
    expected = probe.EPSILON*x.square().mean((1, 2)).sqrt()
    actual = delta.square().mean((2, 3)).sqrt()
    torch.testing.assert_close(actual, expected[:, None].expand_as(actual))
    random[0, 0] = 0
    with pytest.raises(ValueError, match='nonzero'):
        probe.noise_deltas(x, random)


def test_identity_response_has_unit_noise_energy_gain():
    class Action:
        def __call__(self, value):
            return value
        def adjoint(self, value):
            return value
    class Dura:
        def estimate_context(self, x):
            return None, Action(), torch.ones(len(x))
        def stage_outputs(self, x, context):
            return [x], None, context[1], None
    class Model:
        def __call__(self, x, anchor, innovation, variance):
            return anchor
    torch.manual_seed(5)
    x = torch.randn(3, 2, 256)
    random = torch.randn(3, probe.DRAWS, 2, 256)
    result = probe.evaluate_batch(Dura(), [Model(), Model()], x, x, random)
    assert result.shape == (3, probe.DRAWS, 3, 2)
    torch.testing.assert_close(result[..., 0], torch.ones(3, probe.DRAWS, 3), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result[..., 1], torch.full((3, probe.DRAWS, 3), 1e-4), atol=1e-8, rtol=1e-4)

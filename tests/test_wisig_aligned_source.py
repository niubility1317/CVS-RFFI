import torch

from experiments.pamf_unrolled_restore.wisig_aligned_source import observe, prepare, shifted_inputs
from experiments.pamf_unrolled_restore.wisig_uncertain_prior import UncertaintyPrior, fit_loss


def test_shifted_training_arms_share_crop_and_keep_identity_rows():
    rng = torch.Generator().manual_seed(37)
    z = torch.randn((8, 256), dtype=torch.complex64, generator=rng)
    inputs = torch.stack((z, z, z), dim=1)
    shifted = shifted_inputs(inputs, rng).reshape_as(inputs)
    torch.testing.assert_close(shifted[:, 0], shifted[:, 2], rtol=0, atol=0)
    torch.testing.assert_close(shifted[-4:], inputs[-4:], rtol=0, atol=0)
    aligned, mask = prepare(shifted[:, 0].contiguous(), 'aligned', 100.)
    torch.testing.assert_close(aligned, shifted[:, 0], rtol=0, atol=0)
    assert aligned.dtype == torch.complex64 and not mask.any()


def test_aligned_training_round_trip_supports_gradient():
    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(75)
    z = torch.randn((8, 256), dtype=torch.complex64, generator=rng)
    m = UncertaintyPrior()
    value, logs = fit_loss(m, z, rng, .20, observe=lambda x, g: observe(x, g, 'aligned', 1.))
    value.backward()
    assert torch.isfinite(logs).all()
    assert m.spectral[-1].weight.grad.norm() > 0

import torch

from cvsrffi.losses import feature_norm_guard_loss


def test_feature_norm_guard_l2_uses_mean_squared_norm():
    z = torch.tensor([[3.0, 4.0], [1.0, 2.0]])

    loss, norm_mean = feature_norm_guard_loss(z, mode="l2")

    assert torch.isclose(loss, torch.tensor(15.0))
    assert abs(norm_mean - ((5.0 + (5.0 ** 0.5)) / 2.0)) < 1e-6


def test_feature_norm_guard_hinge_penalizes_only_above_target():
    z = torch.tensor([[3.0, 4.0], [1.0, 0.0]])

    loss, _ = feature_norm_guard_loss(z, mode="hinge", target=3.0)

    assert torch.isclose(loss, torch.tensor(2.0))

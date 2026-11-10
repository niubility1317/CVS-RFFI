import pytest
import torch
from torch.nn import functional as F

from experiments.pamf_physics_branch.geometry import bounded_echo, physical_branch_loss, soft_distances, tangent


def test_causal_complex_echo_is_bounded_and_does_not_wrap():
    x = torch.zeros(2, 2, 8)
    x[:, 0, 0] = 1.
    c = torch.tensor([[.06, .08], [0., -.1]])
    y = bounded_echo(x, c, torch.tensor([2, 3]))
    assert torch.allclose(y[0, :, 2], torch.tensor([.06, .08]))
    assert torch.allclose(y[1, :, 3], torch.tensor([0., -.1]))
    assert y[:, :, -1].count_nonzero() == 0
    residual = (y - x).flatten(1).norm(dim=1)
    assert bool((residual <= .100001 * x.flatten(1).norm(dim=1)).all())
    assert bool((y.flatten(1).norm(dim=1) >= .899999 * x.flatten(1).norm(dim=1)).all())
    with pytest.raises(ValueError):
        bounded_echo(x, torch.tensor([[.11, 0.], [0., 0.]]), torch.tensor([1, 1]))


def test_finite_differences_are_detached_and_soft_distance_is_bounded():
    torch.manual_seed(4)
    base = torch.randn(8, 7, requires_grad=True)
    changed = base + .05 * torch.randn_like(base)
    direction, norm = tangent(base, changed)
    assert not direction.requires_grad and bool((norm > 0).all())
    original, equal = soft_distances(base, direction, 0.)
    _, corrected = soft_distances(base, direction, .5)
    assert torch.allclose(original, equal)
    assert torch.allclose(corrected, corrected.T, atol=1e-6)
    assert bool((corrected >= 0).all()) and bool((corrected <= original + 1e-6).all())
    corrected.sum().backward()
    assert base.grad is not None and bool(torch.isfinite(base.grad).all())


def test_zero_direction_and_unaccepted_records_have_no_gradient():
    torch.manual_seed(5)
    names = ('time', 'frequency', 'pa', 'path', 'reference')
    features = {name: torch.randn(6, 5, requires_grad=True) for name in names}
    directions = {name: F.normalize(torch.randn(6, 5), dim=1) for name in names}
    labels = torch.tensor([0, 0, 1, 1, 0, 1])
    receivers = torch.tensor([1, 2, 1, 2, 1, 2])
    accepted = torch.tensor([True, True, True, True, False, False])
    value, stats = physical_branch_loss(features, directions, labels, receivers, torch.ones(6),
                                        torch.arange(6), accepted, beta=.5, single_receiver=False)
    value.backward()
    assert stats['valid_anchors'] == 4 and torch.isfinite(value)
    for feature in features.values():
        assert feature.grad[:4].norm() > 0
        assert torch.equal(feature.grad[4:], torch.zeros_like(feature.grad[4:]))
    direction, _ = tangent(torch.zeros(2, 3), torch.zeros(2, 3))
    assert torch.equal(direction, torch.zeros_like(direction))


def test_one_receiver_fallback_and_no_eligible_pairs():
    feature = torch.randn(4, 3, requires_grad=True)
    direction = torch.zeros_like(feature)
    labels = torch.tensor([0, 0, 1, 1])
    rx = torch.ones(4, dtype=torch.long)
    days = torch.ones(4, dtype=torch.long)
    ids = torch.arange(4)
    value, stats = physical_branch_loss({'time': feature}, {'time': direction}, labels, rx, days, ids,
                                        torch.ones(4, dtype=torch.bool), beta=.5, single_receiver=True)
    assert stats['valid_anchors'] == 4 and torch.isfinite(value)
    zero, stats = physical_branch_loss({'time': feature}, {'time': direction}, labels, rx, days, ids,
                                       torch.zeros(4, dtype=torch.bool), beta=.5, single_receiver=False)
    assert stats['valid_anchors'] == 0 and zero.item() == 0

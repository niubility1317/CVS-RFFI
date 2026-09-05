import pytest
import torch
from cvsrffi.selective_tangent import directional_routing_loss, chordal_sensitivity, selective_tangent_loss


def test_four_same_origin_identity_distances_with_different_origins():
    clean = torch.tensor([[1., 0., .3], [0., 1., .2]], requires_grad=True)
    medium = torch.tensor([[0., 1., .2], [1., 0., .3]], requires_grad=True)
    out = directional_routing_loss(nuisance_base_id=medium, nuisance_base_dom=medium,
        fingerprint_base_id=clean, fingerprint_base_dom=clean,
        nuisance_id=medium, nuisance_dom=medium, fingerprint_id=clean, fingerprint_dom=clean,
        nuisance_margin=.05, fingerprint_margin=.05)
    for name in ['delta_nui_id', 'delta_nui_dom', 'delta_fp_id', 'delta_fp_dom']:
        assert torch.equal(out[name], torch.zeros(2))
    out['loss'].backward()
    assert torch.isfinite(clean.grad).all() and torch.isfinite(medium.grad).all()


def test_identical_direction_conflict_control_lower_bound():
    x = torch.tensor([[1., 0.]])
    for angle in [.01, .1, .8]:
        y = torch.tensor([[1., angle]])
        out = directional_routing_loss(base_id=x, base_dom=x, nuisance_id=y,
            fingerprint_id=y, nuisance_dom=x, fingerprint_dom=x,
            nuisance_margin=.05, fingerprint_margin=.05)
        assert out['loss'] >= .1 - 1e-7


@pytest.mark.parametrize('amount', [0., 1e-8, 1e-4])
def test_finite_chordal_and_linear_budget_gradients(amount):
    x = torch.tensor([[1., .3]], requires_grad=True)
    y = x + torch.tensor([[0., amount]])
    energy = chordal_sensitivity(x, y, delta=1e-4, reference_scale=.5)
    loss = selective_tangent_loss(energy, budgets=torch.zeros(1), valid=torch.ones(1, dtype=torch.bool), penalty='linear')
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    if amount == 0:
        assert loss == 0


def test_reference_scale_and_linear_penalty_units():
    x, y = torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]])
    assert chordal_sensitivity(x, y, delta=.1, reference_scale=.2) == 8.
    assert selective_tangent_loss(torch.tensor([3.]), budgets=torch.tensor([1.]), valid=torch.tensor([True]), penalty='linear') == 2.
    assert selective_tangent_loss(torch.tensor([3.]), budgets=torch.tensor([1.]), valid=torch.tensor([True])) == 4.

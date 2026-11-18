import torch

from experiments.pamf_repl_adaptive_kl import plan
from experiments.pamf_repl_adaptive_kl.objective import sample_preference, sample_weights, symmetric_kl
from experiments.pamf_ssl_research.hard_pl_objective import masked_pair


def test_accepted_weights_preserve_mass_and_reverse_the_same_preference():
    accepted = torch.tensor([True, True, True, False])
    preference = torch.tensor([0., .75, 1., 100.], requires_grad=True)
    favorable = sample_weights(preference, accepted, 1.)
    hard = sample_weights(preference, accepted, -1.)
    torch.testing.assert_close(favorable + hard, 2 * accepted.float())
    torch.testing.assert_close(favorable.sum(), accepted.sum().float())
    assert favorable[0] < favorable[1] < favorable[2]
    assert .25 <= favorable[accepted].min() and favorable.max() <= 1.75
    assert not favorable.requires_grad
    torch.testing.assert_close(sample_weights(preference, ~torch.ones_like(accepted), 1.), torch.zeros(4))


def test_receiver_balance_uses_only_accepted_source_records():
    accepted = torch.tensor([True, True, True, False])
    rx = torch.tensor([0, 0, 1, 1])
    preference = sample_preference(None, None, None, accepted, rx, 'rx_balance')
    torch.testing.assert_close(preference, torch.tensor([.5, .5, 1., 0.]))
    weights = sample_weights(preference, accepted, .5)
    assert weights[2] > weights[0] == weights[1] > weights[3]


def test_flat_preferences_reproduce_fixed_kl_and_keep_rejected_gradient_zero():
    a = torch.tensor([[1., -1.], [-1., 1.]], requires_grad=True)
    b = torch.tensor([[-.25, .25], [1., -1.]], requires_grad=True)
    accepted = torch.tensor([True, False])
    weights = sample_weights(torch.ones(2), accepted, 1.)
    kl = symmetric_kl((a, b), weights)
    _, _, base_kl = masked_pair(a, b, torch.zeros(2, dtype=torch.long), accepted, 1.)
    torch.testing.assert_close(kl, base_kl, rtol=0, atol=0)
    for grad in torch.autograd.grad(kl, (a, b)):
        assert torch.count_nonzero(grad[1]) == 0 and torch.count_nonzero(grad[0]) > 0


def test_all_matrix_rows_differ_from_their_anchor_only_in_the_weighting_rule():
    assert len({r['row_id'] for r in plan.candidate_rows()}) == 14
    assert plan.GPUS == tuple(range(7))
    for recipe in plan.RECIPES.values():
        anchor, view = plan.ANCHORS[recipe['setting']]
        assert recipe['adaptive_kl_view'] == view
        assert {key: value for key, value in recipe.items() if not key.startswith('adaptive_')} == plan.fixed.RECIPES[anchor]

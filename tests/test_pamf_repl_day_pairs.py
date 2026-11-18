import torch

from experiments.cric_dura_comparison.f_model import contrast
from experiments.pamf_repl_day_pairs import plan
from experiments.pamf_repl_day_pairs.objective import pair_distributions


def _source_batch():
    return torch.tensor([0, 0, 0, 0, 1]), torch.tensor([0, 1, 2, 1, 2]), torch.tensor([1, 1, 3, 3, 2])


def test_distribution_changes_only_the_existing_cross_rx_positives():
    labels, rx, dates = _source_batch()
    target, base, both, valid, mix = pair_distributions(labels, rx, dates, 'joint', .5)
    torch.testing.assert_close(target[0], torch.tensor([0., 1/6, 5/12, 5/12, 0.]))
    torch.testing.assert_close(target.sum(1), valid.float())
    assert not (target.bool() & ~base.bool()).any()
    assert not target.requires_grad and not both[0, 1] and mix[0] == .5


def test_sparse_coverage_and_equal_date_balance_have_defined_fallbacks():
    labels, rx, dates = _source_batch()
    target, base, both, valid, mix = pair_distributions(labels, rx, dates, 'coverage', .75)
    assert mix[0] == .375 and mix[4] == 0 and not valid[4]
    no_dates, uniform, _, _, _ = pair_distributions(labels, rx, torch.ones_like(dates), 'joint', .75)
    torch.testing.assert_close(no_dates, uniform, atol=0, rtol=0)
    balanced, _, _, _, _ = pair_distributions(labels, rx, dates, 'date_balance', 1.)
    torch.testing.assert_close(balanced[0], torch.tensor([0., .5, .25, .25, 0.]))


def test_uniform_positive_distribution_preserves_original_objective_and_gradient():
    labels, rx, dates = _source_batch()
    features = torch.randn(len(labels), 5, generator=torch.Generator().manual_seed(91), requires_grad=True)
    base, _ = contrast({'head': features}, labels, rx)
    uniform, _, _, _, _ = pair_distributions(labels, rx, dates, 'joint', 0.)
    alt, _ = contrast({'head': features}, labels, rx, positive_distribution=uniform)
    torch.testing.assert_close(alt, base)
    grad_a, = torch.autograd.grad(base, features, retain_graph=True)
    grad_b, = torch.autograd.grad(alt, features)
    torch.testing.assert_close(grad_a, grad_b)


def test_matrix_keeps_the_existing_loss_doses():
    assert len({r['row_id'] for r in plan.candidate_rows()}) == 14
    assert plan.GPUS == tuple(range(7))
    for recipe in plan.RECIPES.values():
        anchor, _ = plan.ANCHORS[recipe['setting']]
        assert {k: v for k, v in recipe.items() if not k.startswith('day_')} == plan.fixed.RECIPES[anchor]

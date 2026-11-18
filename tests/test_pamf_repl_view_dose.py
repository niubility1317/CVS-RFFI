import json
import torch

from experiments.pamf_repl_view_dose import plan
from experiments.pamf_repl_view_dose.objective import routed_components


def test_view_changes_route_one_objective_and_preserve_nominal_dose():
    rows = plan.candidate_rows()
    assert len({row['row_id'] for row in rows}) == len(rows) == 12
    for recipe in plan.RECIPES.values():
        assert json.loads(json.dumps(recipe)) == recipe
        assert sum(recipe['kl_views']) == .8
        assert sum(recipe['ce_views']) == (1. if tuple(recipe['kl_views']) == plan.ORIGINAL_KL else 0.)


def test_view_gradients_use_exact_inherited_kl_or_redistributed_kl():
    p = torch.ones(8, requires_grad=True)
    views = tuple((p[i], p[4+i]) for i in range(4))
    parts = {'pair_kl': 2 * p[5]}
    ce, kl = routed_components(views, parts, plan.RECIPES['ten_ce_ordinary_raw'])
    assert kl is parts['pair_kl']
    expected = torch.tensor([.375, 0, 0, .375, 0, 2, 0, 0])
    torch.testing.assert_close(torch.autograd.grad(ce+kl, p)[0], expected)
    ce, kl = routed_components(views, parts, plan.RECIPES['six_kl_raw'])
    expected = torch.tensor([0., 0, 0, 0, 0, 0, 0, .9])
    torch.testing.assert_close(torch.autograd.grad(ce+kl, p)[0], expected)

import torch

from experiments.pamf_repl_style_dose import plan
from experiments.pamf_repl_style_dose.objective import DateStyleREPL


def test_preferred_donors_preserve_rx_and_pseudo_identity_with_original_fallback():
    pseudo = torch.tensor([0, 0, 0, 1, 1, 0, 0])
    accepted = torch.tensor([True, True, True, True, True, False, True])
    receivers = torch.tensor([0, 1, 2, 1, 2, 2, 0])
    days = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    original = torch.tensor([2, 0, 0, 4, 3, 5, 1])
    valid = accepted.clone()
    for rule in ('original', 'same_day', 'cross_day'):
        objective = DateStyleREPL.__new__(DateStyleREPL)
        objective.c = dict(model_seed=plan.SEED, tuning_recipe={'date_donor': rule})
        objective.donor_inputs = pseudo, accepted, receivers, days, 4440
        before = torch.random.get_rng_state()
        applied = objective.select_style_partners(original, valid)
        assert torch.equal(torch.random.get_rng_state(), before)
        if rule == 'cross_day':
            assert applied[0] == 2 and applied[1].item() in (2, 6)
        elif rule == 'same_day':
            assert applied[0] == 1 and applied[1] == 0
        else:
            assert torch.equal(applied, original)
        assert torch.equal(applied[2:], original[2:])
        assert torch.all(pseudo[valid] == pseudo[applied[valid]])
        assert torch.all(accepted[applied[valid]])
        assert torch.all(receivers[valid] != receivers[applied[valid]])
        assert objective.donor_metrics['HPL_donor_changed'] == (0 if rule == 'original' else 1)


def test_weight_and_donor_axes_preserve_the_hard_vote_anchor():
    assert len(plan.candidate_rows()) == 9
    assert {(r['weight'], r['date_donor']) for r in plan.RECIPES.values()} == {
        (weight, rule)
        for weight in (.375, .75, 1.5)
        for rule in ('original', 'same_day', 'cross_day')
    }
    for recipe in plan.RECIPES.values():
        assert {k: v for k, v in recipe.items() if k not in ('weight', 'date_donor')} == {
            k: v for k, v in plan.ANCHOR.items() if k != 'weight'
        }

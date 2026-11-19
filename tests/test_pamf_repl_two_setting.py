import torch

from experiments.pamf_repl_two_setting import plan
from experiments.pamf_repl_two_setting.objective import TenClassREPL
from experiments.pamf_repl_component_guard import plan as base


def test_full_factorial_and_held_seeds_are_preserved_in_ancestry_registration():
    rows = plan.candidate_rows()
    assert len({row['row_id'] for row in rows}) == 12
    assert {row['model_seed'] for row in rows if row['setting'] == 'source_rx_6'} == set(plan.HELD_SEEDS)
    with plan.scoped():
        assert base.candidate_rows() == rows
        recipe = base.RECIPES['ten_ce1_gap1_rx1']
        assert recipe['weight'] == .75 and recipe['kl_factor'] == 1.5
        assert recipe['raw_ce_dose'] == .25
        assert recipe['rx_balance'] and recipe['adaptive_strength'] == .5


def test_raw_ce_dose_reuses_original_full_batch_masked_term():
    ce = torch.tensor(.8, requires_grad=True)
    recipe = plan.RECIPES['ten_ce1_gap0_rx0']
    value = TenClassREPL.compute_ce(None, ce, recipe)
    torch.testing.assert_close(value, torch.tensor(.15))
    (grad,) = torch.autograd.grad(value, ce)
    torch.testing.assert_close(grad, torch.tensor(.1875))

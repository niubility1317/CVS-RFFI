import numpy as np
import torch

from experiments.pamf_repl_loss_dose import plan
from experiments.pamf_repl_loss_dose.analyze import contrasts, effect_counts
from experiments.pamf_repl_loss_dose.objective import apply_doses


def test_scaled_gradient_and_exact_passthrough():
    p = torch.tensor([2., 3.], requires_grad=True)
    parts = dict(shallow=p[1], pair_kl=2*p[1], joint=p[0]+p[1], extrap_delta=0*p[0])
    total = 4*p[0]+sum(parts.values())
    expected = {(1., 1.): (5., 4.), (.25, 1.): (2., 4.), (.5, 0.): (2., 0.), (0., 0.): (0., 0.)}
    for (ce, auxiliary), grad in expected.items():
        effective, _, _ = apply_doses(total, parts, ce, auxiliary)
        torch.testing.assert_close(torch.autograd.grad(effective, p, retain_graph=True)[0], torch.tensor(grad))
    assert apply_doses(total, parts, 1., 1.)[0] is total


def test_full_factorial_and_unique_paths():
    rows = plan.candidate_rows()
    assert len({row['row_id'] for row in rows}) == len(rows) == 16
    for setting in plan.SETTINGS:
        candidates = [r for r in plan.RECIPES.values() if r['setting'] == setting]
        assert {(r['auxiliary_dose'], r['ce_dose']) for r in candidates} == {
            (a, ce) for a in (0., 1.) for ce in (0., .25, .5, 1.)}


def test_factorial_interaction_uses_paired_correctness():
    correctness = {
        (1., .5): np.array([True, False, True]),
        (1., 1.): np.array([False, True, True]),
        (0., .5): np.array([False, True, True]),
        (0., 1.): np.array([True, False, True]),
    }
    assert effect_counts(correctness, contrasts()['interaction_ce_0.5_vs_1']) == {
        -2: 1, 0: 1, 2: 1,
    }

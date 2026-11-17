import torch
from experiments.pamf_repl_dual_optimization import plan
from experiments.pamf_repl_uaux_soft.projection import auxiliary_loss

def test_matrix_is_unique_and_has_paired_dose_controls():
    rows = plan.candidate_rows()
    assert len(rows) == len({r['row_id'] for r in rows}) == 16
    assert sum(r['setting'] == 'ten_class' for r in rows) == 8
    assert all(r['views'] == ['clean'] for r in rows)
    for name, recipe in plan.RECIPES.items():
        assert 0 <= recipe['alpha'] <= 1 and 0 <= recipe['beta'] <= 1
        assert recipe['mode'] == 'soft' and recipe['extrap_coefficient'] == 0

def test_combined_projection_matches_order_and_preserves_ce():
    p = torch.nn.Parameter(torch.zeros(2))
    labeled, ref, other, ce = p[0], p[1], -p[0]-p[1], 3*p[0]
    loss, _, _, stats = auxiliary_loss(labeled, ref+other+ce, ref, other,
                                       [('other.p', p)], alpha=.5, beta=.5)
    loss.backward()
    assert torch.allclose(p.grad, torch.tensor([3.5, .5]))
    assert stats['other_reference_remainder_dot_before'] < 0
    assert stats['other_label_aux_dot_before'] < 0

import pytest
import torch
from experiments.pamf_repl_component_guard.projection import protected_components
from experiments.pamf_repl_component_guard import plan


def test_conflicting_component_is_corrected_even_if_total_is_aligned():
    p = torch.nn.Parameter(torch.tensor([0., 0.]))
    labeled = p[0]
    selected = -p[0]
    unlabeled = selected + p[0] + p[1]
    value, details, gradients, stats = protected_components(
        labeled, unlabeled, [('encoder.other.weight', p)], {'shallow': selected}, max_u_ratio=10.)
    value.backward()
    assert torch.allclose(p.grad, torch.tensor([2., 1.]))
    assert torch.equal(p.grad, gradients[0])
    assert stats['shallow_other_conflict'] == 1
    assert stats['shallow_other_correction_norm'] == pytest.approx(1.)
    assert details['other']['raw_dot_before'] == pytest.approx(0.)
    assert details['other']['dot_before'] == pytest.approx(1.)
    assert details['other']['dot_after'] == pytest.approx(1.)


def test_tiny_labeled_gradient_does_not_create_component_correction():
    p = torch.nn.Parameter(torch.tensor([0., 0.]))
    labeled = 0. * p.sum(); selected = -p[0]; unlabeled = selected + p[1]
    _, details, gradients, stats = protected_components(
        labeled, unlabeled, [('encoder.other.weight', p)], {'shallow': selected}, max_u_ratio=1.)
    assert stats['shallow_other_conflict'] == 0
    assert stats['shallow_other_correction_norm'] == 0.
    assert details['other']['tiny_L']
    assert torch.equal(gradients[0], torch.zeros_like(p))


def test_candidate_rejects_target_or_unregistered_recipe():
    row = plan.candidate_rows()[0]
    c = {'row_id':row['row_id'], 'tuning_row':row, 'tuning_recipe':plan.RECIPES[row['recipe']],
         'output_root':f"/project/runs/{plan.RUN}/F/{row['row_id']}/source",
         'run_id':plan.RUN, 'model_seed':row['model_seed'], 'protocol_id':row['protocol_id']}
    assert plan.validate(c) == row
    with pytest.raises(ValueError, match='Target'):
        plan.validate(dict(c, target_truth='/not/permitted'))
    with pytest.raises(ValueError, match='not registered'):
        plan.validate(dict(c, tuning_recipe={'project_components':['unknown']}))

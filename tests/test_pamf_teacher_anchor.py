import pytest
import torch
from pathlib import Path

from experiments.pamf_physics_tri.teacher_anchor import distill_pair
from experiments.pamf_physics_tri import teacher_anchor_plan


def test_distillation_uses_detached_teacher_and_selected_source_u_only():
    teacher_logits = torch.tensor([[3., 0.], [3., 0.]], requires_grad=True)
    target = teacher_logits.softmax(1)
    first = torch.tensor([[0., 2.], [0., 2.]], requires_grad=True)
    second = torch.tensor([[0., 1.], [0., 1.]], requires_grad=True)
    loss = distill_pair(first, second, target, torch.tensor([True, False]))
    assert loss.item() > 0
    loss.backward()
    assert teacher_logits.grad is None
    assert first.grad[0].abs().sum() > 0 and second.grad[0].abs().sum() > 0
    assert first.grad[1].abs().sum() == 0 and second.grad[1].abs().sum() == 0


def test_invalid_teacher_mask_and_fixed_three_mode_matrix():
    with pytest.raises(ValueError):
        distill_pair(torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 3), torch.tensor([True]))
    rows = teacher_anchor_plan.candidate_rows()
    assert len(rows) == 9
    assert {r['arm'] for r in rows} == {'agree', 'teacher', 'teacher_physical'}
    assert {r['model_seed'] for r in rows} == {2026092701, 2026092702, 2026092703}


def test_source_root_is_identical_for_source_and_segment_configs():
    source = Path('/project/runs/run/F/row/source')
    for path in (source, source / 'segment30', source / 'segment50'):
        assert teacher_anchor_plan.source_root({'output_root': str(path)}) == source

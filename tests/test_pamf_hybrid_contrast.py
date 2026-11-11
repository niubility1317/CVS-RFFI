import torch

from experiments.pamf_physics_tri.hybrid_contrast import HybridContrast
from experiments.pamf_physics_tri import hybrid_contrast_plan


def objective(mode):
    obj = object.__new__(HybridContrast)
    obj.c = {'hybrid_contrast': {'mode': mode, 'temperature': .2}}
    obj.hybrid_labels = torch.tensor([0, 1]), torch.tensor([True, True])
    obj.hybrid_calls = obj.hybrid_anchors = 0
    obj.metrics = {}
    return obj


def test_symmetric_contrast_updates_both_student_views():
    obj = objective('symmetric')
    raw = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    changed = torch.tensor([[.7, .7], [0., 1.]], requires_grad=True)
    loss = obj.additional_fusion_loss([raw, raw, changed, changed], torch.tensor([True, False]))
    loss.backward()
    assert raw.grad[0].abs().sum() > 0
    assert changed.grad[0].abs().sum() > 0
    assert obj.hybrid_anchors == 2


def test_branch_contrast_detaches_raw_branch_teacher():
    obj = objective('branch')
    raw = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    changed = torch.tensor([[.7, .7], [0., 1.]], requires_grad=True)
    obj.hybrid_branches = [{'time': raw}, {'time': raw}, {'time': changed}, {'time': changed}]
    loss = obj.additional_fusion_loss([raw, raw, changed, changed], torch.tensor([True, False]))
    loss.backward()
    assert raw.grad is None and changed.grad[0].abs().sum() > 0


def test_three_contrast_structures_share_three_seeds():
    rows = hybrid_contrast_plan.candidate_rows()
    assert len(rows) == 9
    assert {r['arm'] for r in rows} == {'symmetric', 'branch', 'fusion_branch'}
    assert {r['model_seed'] for r in rows} == {2026092701, 2026092702, 2026092703}

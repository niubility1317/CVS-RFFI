import torch

from experiments.pamf_physics_tri.source_prototype import prototype_loss
from experiments.pamf_physics_tri import source_prototype_plan


def test_prototype_uses_other_days_and_detaches_labeled_bank():
    bank = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [1., 0.]], requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    days = torch.tensor([1, 1, 2, 2])
    query = torch.tensor([[0., 1.], [0., 1.]], requires_grad=True)
    pseudo = torch.tensor([0, 0])
    loss, used = prototype_loss(query, bank, labels, days, torch.tensor([1, 2]), pseudo,
                                torch.tensor([True, False]), 2)
    assert used == 1
    loss.backward()
    assert bank.grad is None
    assert query.grad[0].abs().sum() > 0
    assert query.grad[1].abs().sum() == 0
    assert loss < .01


def test_missing_other_day_class_skips_selected_query():
    query = torch.tensor([[1., 0.]], requires_grad=True)
    loss, used = prototype_loss(query, torch.eye(2), torch.tensor([0, 1]), torch.tensor([1, 2]),
                                torch.tensor([1]), torch.tensor([0]), torch.tensor([True]), 2)
    assert used == 0 and loss.item() == 0.
    loss.backward()
    assert query.grad.abs().sum() == 0


def test_registered_rows_and_segment_roots():
    rows = source_prototype_plan.candidate_rows()
    assert len(rows) == 9
    assert {row['arm'] for row in rows} == {'fusion', 'branch', 'fusion_branch'}
    from pathlib import Path
    source = Path('/project/runs/run/F/row/source')
    for suffix in ('segment30', 'segment50'):
        assert source_prototype_plan.source_root({'output_root': str(source / suffix)}) == source

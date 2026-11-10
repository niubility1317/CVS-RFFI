import pytest
import torch

from experiments.pamf_physics_tri_u import plan, objective


def test_fixed_three_seed_plan_and_strict_parent():
    assert [r['model_seed'] for r in plan.candidate_rows()] == [2026092701, 2026092702, 2026092703]
    with pytest.raises(ValueError):
        plan.configuration({}, {}, '/project', 'commit')


def test_unaccepted_u_cannot_change_physical_relation():
    torch.manual_seed(4)
    features = {name: torch.randn(6, 4, requires_grad=True) for name in ('time', 'frequency', 'pa', 'path', 'reference')}
    directions = {name: torch.nn.functional.normalize(torch.randn(6, 4), dim=1) for name in features}
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    metadata = {'rx_i': torch.tensor([1, 3, 1, 3, 1, 3]), 'day_i': torch.ones(6, dtype=torch.long),
                'branch_record_id': torch.arange(6)}
    accepted = torch.tensor([True, True, True, True, False, False])
    config = {'source_rxs': [1, 3], 'source_days': [1], 'physical_relation': {'beta': .5}}
    indices = torch.arange(6)
    a, stats = objective.selected_relation(features, directions, labels, metadata, indices, accepted, config)
    changed = labels.clone(); changed[~accepted] = torch.tensor([99, 99])
    b, other = objective.selected_relation(features, directions, changed, metadata, indices, accepted, config)
    assert torch.allclose(a, b)
    assert stats['valid_anchors'] == other['valid_anchors'] == 4
    a.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in features.values())
    assert all(torch.equal(v.grad[~accepted], torch.zeros_like(v.grad[~accepted])) for v in features.values())


def test_visible_unlabeled_label_rejected():
    fake = object.__new__(objective.PhysicalU)
    batch = (None, torch.tensor([-1, -1]), None, {'tx_label_visible': torch.tensor([False, True])})
    with pytest.raises(ValueError):
        fake.validate_unlabeled_batch(batch)

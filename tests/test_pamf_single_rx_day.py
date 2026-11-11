import pytest
import torch

from experiments.pamf_branch_training.relation import branch_relation_loss
from experiments.pamf_physics_tri.single_rx import SingleRxDay
from experiments.pamf_physics_tri import single_rx_plan


def test_cross_day_relation_excludes_unaccepted_and_same_day_pairs():
    features = {'time': torch.tensor([[1., 0.], [.9, .1], [0., 1.], [.1, .9]], requires_grad=True)}
    labels = torch.tensor([0, 0, 1, 1])
    days = torch.tensor([1, 2, 1, 2])
    ids = torch.arange(4)
    loss, counts = branch_relation_loss(features, labels, days, ids, torch.ones(4, dtype=torch.bool),
                                        single_receiver=False)
    assert counts['cross_receiver_anchors'] == 4
    loss.backward()
    assert torch.isfinite(features['time'].grad).all()
    _, no_pairs = branch_relation_loss(features, labels, torch.ones(4), ids,
                                      torch.ones(4, dtype=torch.bool), single_receiver=False)
    assert no_pairs['cross_receiver_anchors'] == 0
    _, hidden = branch_relation_loss(features, labels, days, ids, torch.tensor([True, False, True, False]),
                                    single_receiver=False)
    assert hidden['cross_receiver_anchors'] == 0


def test_labeled_and_query_groups_are_source_days():
    obj = object.__new__(SingleRxDay)
    obj.c = {'source_days': [1, 2, 3]}
    meta = {'day_i': torch.tensor([1, 3]), 'rx_i': torch.tensor([1, 1])}
    assert obj.labeled_group(meta, torch.tensor([0, 1])).tolist() == [1, 3]
    assert obj.query_group(meta, torch.tensor([1])).tolist() == [3]
    with pytest.raises(ValueError):
        obj.labeled_group({'day_i': torch.tensor([4])}, torch.tensor([0]))


def test_matrix_has_only_requested_setting_three_seeds_and_two_arms():
    rows = single_rx_plan.candidate_rows()
    assert len(rows) == 6
    assert {r['protocol_id'] for r in rows} == {'m1-natural'}
    assert {r['model_seed'] for r in rows} == {2026092701, 2026092702, 2026092703}
    assert {r['arm'] for r in rows} == {'day', 'day_physics'}

import pytest
import torch

from experiments.pamf_physics_tri.soft_channel import physical_weights
from experiments.pamf_physics_tri import soft_channel_plan


def test_continuous_strength_tracks_votes_and_disables_missing_day_anchors():
    names = ('time', 'frequency', 'pa', 'path', 'reference')
    raw_vec = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [1., 0.]])
    altered_vec = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [0., 1.]])
    raw = {name: raw_vec for name in names}
    altered = {name: altered_vec for name in names}
    labels, l_day = torch.tensor([0, 1]), torch.tensor([1, 1])
    u_day, pseudo = torch.tensor([2, 1]), torch.tensor([0, 0])
    vote, cert = physical_weights(raw, altered, labels, l_day, u_day, pseudo, 2, 'vote')
    stable, _ = physical_weights(raw, altered, labels, l_day, u_day, pseudo, 2, 'vote_stability')
    assert vote.tolist() == pytest.approx([.5, 0.])
    assert stable.tolist() == pytest.approx([0., 0.])
    assert cert.tolist() == [False, False]


def test_two_rules_and_three_seeds_in_single_receiver_setting():
    rows = soft_channel_plan.candidate_rows()
    assert len(rows) == 6
    assert {r['arm'] for r in rows} == {'vote', 'vote_stability'}
    assert {r['model_seed'] for r in rows} == {2026092701, 2026092702, 2026092703}
    assert {r['protocol_id'] for r in rows} == {'m1-natural'}

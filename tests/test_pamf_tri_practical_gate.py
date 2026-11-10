from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.pamf_physics_tri.practical_gate import PracticalGate, select_views


def test_practical_views_only_enter_accepted_physical_slots_and_gate_is_respected():
    batch = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1)
    ordinary = batch + 100
    physical = torch.tensor([True, False, True, False])
    accepted = torch.tensor([True, True, True, False])
    pseudo = torch.tensor([0, 0, 1, 1])
    metadata = {'rx_i': torch.tensor([1, 1, 3, 3])}
    student = SimpleNamespace(_cd_practical=lambda x, meta, epoch, step: x + 10)
    gate = PracticalGate.__new__(PracticalGate)
    gate.labeled = (batch, pseudo, metadata['rx_i'])
    gate.c = dict(model_seed=1, source_rxs=[1, 3], num_classes=2, practical_gate=dict(branch_gate=True))
    gate.as_calls = gate.as_offered = gate.as_used = gate.as_certified = 0
    with patch('experiments.pamf_physics_tri.practical_gate.capture_pair', return_value=({}, {})), \
         patch('experiments.pamf_physics_tri.practical_gate.branch_certificate',
               return_value=(torch.tensor([True, True]), torch.tensor([False, True]))):
        mixed = gate.mixed_view(student, batch, ordinary, physical, metadata, 1, 1, pseudo, accepted)
    torch.testing.assert_close(mixed[0], batch[0])
    torch.testing.assert_close(mixed[2], batch[2] + 10)
    torch.testing.assert_close(mixed[~physical], ordinary[~physical])
    assert (gate.as_calls, gate.as_offered, gate.as_certified, gate.as_used) == (1, 2, 1, 1)


def test_random_control_matches_certified_dose_and_preserves_global_rng():
    offered = torch.tensor([True, True, False, True, True, False])
    certified = torch.tensor([True, False, True, False, True, True])
    before = torch.random.get_rng_state()
    random = select_views(offered, certified, 'random', 123)
    branch = select_views(offered, certified, 'branch', 123)
    assert random.sum() == branch.sum() == 2
    assert (random & ~offered).sum() == 0
    assert torch.equal(random, select_views(offered, certified, 'random', 123))
    assert torch.equal(before, torch.random.get_rng_state())

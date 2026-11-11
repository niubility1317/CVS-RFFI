import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.pamf_tri_u_softdistill_m1 import objective, plan


def test_same_raw_iq_is_used_for_teacher_and_identity_student(monkeypatch):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.response_projection = nn.Linear(2, 2)
            self.sequence_pool = nn.Identity()
            self.core = SimpleNamespace(id_backbone=SimpleNamespace())

        def response(self, x):
            result = x.new_ones((len(x), 29))
            result[:, -4] = .9
            return result

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.id_backbone = nn.Module()
            self.id_backbone.encoder = Encoder()
            self.head = nn.Linear(2, 3)
            self._feature_l_steps = 1
            self.inputs = []

        def forward(self, x):
            self.inputs.append(x.detach().clone())
            return self.head(self.id_backbone.encoder.response_projection(x))

    raw = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    accepted = torch.tensor([True, True, False, True])
    net = Student()
    instance = objective.SoftDistillU.__new__(objective.SoftDistillU)
    instance.c = {'soft_recipe': {'selected': 4, 'temperature': 2.},
                  'soft_branch': {'mode': 'identity', 'frequency': False, 'weight': 1.}}
    instance._soft_batch = instance._soft_teacher = instance._raw_input = None
    instance.soft_calls = instance.soft_selected = instance.soft_accepted = 0
    instance.coherence_sum = instance.scale_sum = 0.
    instance.metrics = {}

    def parent_loss(self, student, device):
        self._raw_input = raw
        self._soft_teacher = student(raw).detach()
        self._soft_batch = (raw + .5, torch.zeros(4, dtype=torch.long), accepted)
        return torch.tensor(0.)

    monkeypatch.setattr(objective.ReceiverExtrapolation, 'loss', parent_loss)
    monkeypatch.setattr(objective, 'select_u_candidates', lambda *args: torch.arange(4))
    loss = instance.loss(net, 'cpu')
    assert torch.equal(net.inputs[0], raw)
    assert torch.equal(net.inputs[1], raw)
    assert abs(float(loss)) < 1e-6
    assert instance._raw_input is None and instance._soft_teacher is None


def test_soft_targets_detached_and_rejected_packets_have_no_gradient():
    teacher = torch.randn(3, 4, requires_grad=True)
    logits = torch.randn(3, 4, requires_grad=True)
    accepted = torch.tensor([True, False, True])
    objective.masked_loss(logits, teacher, accepted, 2.).backward()
    assert teacher.grad is None
    assert torch.count_nonzero(logits.grad[1]) == 0
    assert torch.count_nonzero(logits.grad[[0, 2]]) > 0
    with pytest.raises(ValueError):
        objective.masked_loss(logits, teacher, accepted.float(), 2.)


def test_registered_three_seed_source_only_matrix():
    path = Path(r'E:\type10-7\local_artifacts\tri_unprotected_context.json')
    if not path.exists():
        pytest.skip('Local source-handle context unavailable')
    context = json.loads(path.read_text(encoding='utf-8'))
    rows = plan.candidate_rows()
    assert len(rows) == 15
    for row in rows:
        cfg = plan.configuration(context, row, Path('/test/project'), 'test')
        plan.validate_candidate(cfg)
        assert cfg['protocol_id'] == 'm1-natural' and cfg['source_rxs'] == [1]
        bad = copy.deepcopy(cfg)
        bad['target_truth'] = '/forbidden'
        with pytest.raises(ValueError):
            plan.validate_candidate(bad)

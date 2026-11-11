from unittest.mock import patch

import pytest
import torch

from experiments.pamf_physics_tri.certified_day import CertifiedDay
from experiments.pamf_physics_tri import certified_day_plan
from experiments.pamf_physics_route import paired_report


def test_relation_mask_only_restricts_forward_pairing():
    obj = object.__new__(CertifiedDay)
    obj.relation_mask = torch.tensor([True, False, False])
    accepted = torch.tensor([True, True, False])
    with patch('experiments.pamf_physics_tri.single_rx.SingleRxDay.unlabeled_student_forward', return_value='logits') as parent:
        out = obj.unlabeled_student_forward(None, None, None, None, accepted, {})
    assert out == 'logits'
    assert torch.equal(parent.call_args.args[4], obj.relation_mask)
    assert accepted.tolist() == [True, True, False]


def test_relation_mask_cannot_add_rejected_u():
    obj = object.__new__(CertifiedDay)
    obj.relation_mask = torch.tensor([True, True])
    with pytest.raises(ValueError):
        obj.unlabeled_student_forward(None, None, None, None, torch.tensor([True, False]), {})


def test_prespecified_single_rx_two_controls_three_seeds():
    rows = certified_day_plan.candidate_rows()
    assert len(rows) == 6
    assert {r['protocol_id'] for r in rows} == {'m1-natural'}
    assert {r['arm'] for r in rows} == {'branch', 'echo'}
    assert {r['model_seed'] for r in rows} == {2026092701, 2026092702, 2026092703}


def test_paired_report_includes_matched_day_physics(monkeypatch, tmp_path):
    row, seed, pid = 'candidate', 2026092701, 'm1-natural'
    sources = {row: 'candidate-entry', f'baseline_pl-{pid}-s{seed}': 'pl-entry',
               f'tri_sigtime_ssl-{pid}-s{seed}': 'tri-entry', f'day-physics-{pid}-s{seed}': 'day-entry'}
    cfg = {'model_seed': seed, 'protocol_id': pid, 'views': list(paired_report.VIEWS)}
    def fake_read(path):
        path = str(path)
        return sources if path.endswith('sources.json') else cfg if 'score__' in path else {'inputs': True}
    monkeypatch.setattr(paired_report, 'read', fake_read)
    monkeypatch.setattr(paired_report, 'frozen_row', lambda entry, seed, runs, inputs: entry)
    _, entries = paired_report.prepare_row(tmp_path, row)
    assert entries['Day-physics'] == 'day-entry'

import gzip
import json
import zlib

import torch

from experiments.pamf_physics_tri import hybrid_tracked_plan as plan
from experiments.pamf_physics_tri.hybrid_confidence import HybridConfidence
from experiments.pamf_physics_tri.hybrid_tracked import HybridTracked
from experiments.pamf_physics_tri.hybrid_tracked_report import summarize_trace
from experiments.pamf_physics_tri.branch_guard import BranchGuard


def test_underperforming_seed_matrix_is_fixed():
    rows = plan.candidate_rows()
    assert len(rows) == len({r['row_id'] for r in rows}) == 26
    assert {r['tracked_aux_mode'] for r in rows} == {'contrast', 'positive'}
    assert all(r['auxiliary_weight'] == .1 and plan.baseline_row(r)['model_seed'] == r['model_seed'] for r in rows)
    assert {r['model_seed'] for r in rows if r['setting'] == 'receiver_split_2'} == {2026092701, 2026092702}


def test_auxiliary_view_does_not_replace_base_view(monkeypatch):
    def parent(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        self.as_physical_used += int((accepted & physical).sum())
        return ordinary + 5
    monkeypatch.setattr(HybridConfidence, 'mixed_view', parent)
    obj = HybridTracked.__new__(HybridTracked)
    obj.as_physical_used = 0
    obj.accepted_original_physical_half = 0
    obj.trace_confidence = torch.tensor([.99, .8, .96, .7])
    x = torch.zeros(4, 2); ordinary = torch.ones_like(x)
    physical = torch.tensor([True, False, True, False]); accepted = torch.tensor([True, False, True, False])
    metadata = {'base_index': torch.arange(4), 'rx_i': torch.ones(4), 'day_i': torch.ones(4)}
    mixed = obj.mixed_view(None, x, ordinary, physical, metadata, 21, 1, torch.arange(4), accepted)
    assert torch.equal(mixed[physical], x[physical]) and torch.equal(mixed[~physical], ordinary[~physical])
    assert torch.equal(obj.aux_view, ordinary + 5)
    assert obj.trace_batch['accepted'] == [True, False, True, False]
    assert obj.as_physical_used == 0
    assert obj.accepted_original_physical_half == 2


def test_trace_counts_packet_switches_and_acceptance(tmp_path):
    p = tmp_path / 'trace.jsonl.gz'
    batches = [dict(epoch=21, packet=[1, 2], rx=[1, 2], day=[1, 1], confidence=[.99, .8],
                    pseudo=[0, 1], accepted=[True, False], aux_prediction_1=[0, 1], aux_prediction_2=[0, 0]),
               dict(epoch=22, packet=[1], rx=[1], day=[1], confidence=[.97], pseudo=[1], accepted=[True],
                    aux_prediction_1=[1], aux_prediction_2=[0])]
    with gzip.open(p, 'wt', encoding='utf8') as f:
        for batch in batches:
            f.write(json.dumps(batch) + '\n')
    result = summarize_trace([p])
    assert result['exposures'] == 3 and result['accepted'] == 2
    assert result['distinct_packets'] == 2 and result['repeat_exposures'] == result['pseudo_changed'] == 1


def test_trace_flush_is_readable_during_training(tmp_path):
    obj = HybridTracked.__new__(HybridTracked)
    obj.c = {'output_root': str(tmp_path)}
    obj.trace_file = None
    obj.trace_batches = 0
    for step in range(50):
        obj.write_trace({'step': step})
    data = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress((tmp_path / 'U_trace.jsonl.gz').read_bytes())
    assert len(data.splitlines()) == 50
    obj.trace_file.close()
    with gzip.open(tmp_path / 'U_trace.jsonl.gz', 'rt') as f:
        assert len(f.readlines()) == 50


def test_auxiliary_gradient_is_logged_separately(monkeypatch):
    monkeypatch.setattr(BranchGuard, 'guide', lambda self, labeled, student: labeled)
    model = torch.nn.Linear(1, 1, bias=False)
    obj = HybridTracked.__new__(HybridTracked)
    obj.calls = 50
    obj.metrics = {}
    obj.aux_loss = (model.weight ** 2).sum()
    labeled = (-model.weight ** 2).sum()
    obj.guide(labeled, model)
    assert obj.metrics['HPL_aux_other_dot_L'] < 0
    assert obj.metrics['HPL_aux_other_norm'] > 0
    assert obj.aux_loss is None

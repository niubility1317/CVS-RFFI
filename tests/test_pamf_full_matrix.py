from collections import Counter

import torch
import pytest

from experiments.pamf_physics_tri import full_matrix_plan as plan
from experiments.pamf_physics_tri.branch_guard import BranchGuard
from experiments.pamf_physics_tri.single_rx import SingleRxDay


def test_all_remaining_settings_methods_and_seeds_are_present_once():
    rows = plan.candidate_rows()
    assert len(rows) == 60
    assert len({row['row_id'] for row in rows}) == 60
    assert len(plan.SETTINGS) == 10
    assert {'main_split', 'ten_class', 'labeled_1pct', 'labeled_5pct',
            'receiver_split_1', 'receiver_split_2', 'source_rx_2', 'source_rx_3',
            'source_rx_4', 'source_rx_6'} == set(plan.SETTINGS)
    assert set(Counter((r['setting'], r['arm']) for r in rows).values()) == {3}
    assert {r['arm'] for r in rows} == {'hybrid', 'branch_prototype'}


def test_multireceiver_keeps_actual_rx_for_native_extrapolation(monkeypatch):
    def probe(self, student, x, domain, pseudo, accepted, metadata):
        return metadata['rx_i'].clone(), list(self.c['source_rxs'])
    monkeypatch.setattr(BranchGuard, 'unlabeled_student_forward', probe)
    obj = object.__new__(SingleRxDay)
    obj.c = {'source_rxs': [1, 3], 'source_days': [1, 2, 3]}
    obj._rxe_forward_index = 4
    meta = {'rx_i': torch.tensor([1, 3]), 'day_i': torch.tensor([2, 1])}
    x = torch.ones(2, 1)
    actual, allowed = obj.unlabeled_student_forward(None, x, None, None, None, meta)
    assert actual.tolist() == [1, 3] and allowed == [1, 3]
    assert obj.c['source_rxs'] == [1, 3]


def test_single_receiver_preserves_original_day_fallback(monkeypatch):
    def probe(self, student, x, domain, pseudo, accepted, metadata):
        return metadata['rx_i'].clone(), list(self.c['source_rxs'])
    monkeypatch.setattr(BranchGuard, 'unlabeled_student_forward', probe)
    obj = object.__new__(SingleRxDay)
    obj.c = {'source_rxs': [1], 'source_days': [1, 2, 3]}
    obj._rxe_forward_index = 4
    meta = {'rx_i': torch.tensor([1, 1]), 'day_i': torch.tensor([2, 1])}
    grouped, allowed = obj.unlabeled_student_forward(None, torch.ones(2, 1), None, None, None, meta)
    assert grouped.tolist() == [2, 1] and allowed == [1, 2, 3]
    assert obj.c['source_rxs'] == [1]


def test_only_genuinely_empty_certified_batch_is_retryable(monkeypatch):
    from experiments.pamf_physics_tri import pair_runner
    from experiments.pamf_physics_tri.full_matrix import EmptyCertifiedBatch
    monkeypatch.setattr(pair_runner, 'check_single_rx', lambda _: {})
    state = {'U_fused_physics': {'selected': 0}, 'U_hybrid_contrast': {'calls': 1, 'anchors': 0}}
    with pytest.raises(EmptyCertifiedBatch):
        pair_runner.check_full_matrix(state)
    state['U_hybrid_contrast']['anchors'] = 1
    with pytest.raises(ValueError, match='inconsistent'):
        pair_runner.check_full_matrix(state)


def test_matrix_smoke_retries_only_empty_source_batch(monkeypatch, tmp_path):
    from experiments.pamf_physics_tri import full_matrix, runner
    import json
    monkeypatch.setattr(plan, 'candidate_rows', lambda: [{'row_id': 'one'}])
    seen = []
    def fake_smoke(rt, **kwargs):
        offset = kwargs['labeled_batch_offset']
        seen.append(offset)
        if offset < 2:
            raise full_matrix.EmptyCertifiedBatch()
        return [{'row': 'one', 'finite_backward': True}]
    monkeypatch.setattr(runner, 'smoke', fake_smoke)
    full_matrix.smoke({'run_root': str(tmp_path)})
    saved = json.loads((tmp_path / 'smoke.json').read_text())
    assert seen == [0, 1, 2]
    assert saved['checks'][0]['source_labeled_batch_offset'] == 2

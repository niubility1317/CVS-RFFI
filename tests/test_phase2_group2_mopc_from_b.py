import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.pamf_phase2_practical import group2_mopc_from_b as worker, group2_schedule, group1_target, group2_dadda_mopc_diagnostic


@pytest.mark.parametrize('trace', [False, True])
@pytest.mark.parametrize('mismatch', [False, True])
def test_saved_anchor_is_used_without_refitting_b(tmp_path, monkeypatch, trace, mismatch):
    cap = tmp_path/'data'/'capsules/P/M/R'; splits = cap/'splits'; splits.mkdir(parents=True)
    (cap/'manifest.json').write_text(json.dumps(dict(capsule_id='cap', protocol_schema='p2_min_v1',
                                                      phase2_data_status='VALIDATED_ONCE')))
    for n, split in ((0, 'old'), (2, 'new')):
        (splits/(split+'.json')).write_text(json.dumps(dict(k=5, new_count=n, split_id=split,
                                                           registered_classes=list(range(6+n)))))
    np.save(cap/'ids.npy', np.array([90, 101, 102, 103, 104]))
    subset = tmp_path/'subset'; subset.mkdir()
    ref = str(subset/'B-K5.pt')
    prior = dict(k=5, resources=dict(B_state_ref=ref, B_seconds=2.0, B_native={'optimizer_steps': 0}))
    (subset/'predictions.json').write_text(json.dumps(dict(capsule_id='cap', profile='protonet_cda',
                                                          model_seed=7, rows=[prior])))
    (subset/'complete.json').write_text(json.dumps(dict(status='SUBSET_PREDICTIONS_FIXED',
                                                        truth_read=False, query_fit=False)))
    job = dict(key='p-s7-c0', capsule='capsules/P/M/R', profile='protonet_cda', seed=7,
               methods=['mopc_hr'], points=[[5, 2]], output=str(tmp_path/'out'), load_B_for_mopc=str(subset),
               trace_registration=trace)
    anchor = object(); seen = []
    monkeypatch.setattr(worker, 'activate', lambda rt: None)
    monkeypatch.setattr(group2_schedule, 'closed_prediction', lambda j, n: seen.append(('subset', j['methods'], n)))
    monkeypatch.setattr(group1_target, 'fit_B', lambda *a, **k: (_ for _ in ()).throw(AssertionError('B refit')))
    monkeypatch.setattr(worker, 'support_only', lambda *a: (None, None, np.array([1, 2]), None))
    monkeypatch.setattr(worker, 'load_b_state', lambda *a: dict(anchor=anchor))
    monkeypatch.setattr(worker, 'load_task', lambda *a, **k: dict(support_y=np.array([0, 1, 6, 7]),
        support_ids=np.array([101, 999 if mismatch else 102, 103, 104]), support_x=np.zeros((4, 3), dtype=np.float32),
        query_x=np.zeros((2, 3), dtype=np.float32)))
    def register(prepared, *args, **kwargs):
        assert prepared['mopc_hr'] is anchor
        return {'mopc_hr': SimpleNamespace(resource={}, protocol_metadata={})}
    monkeypatch.setattr(worker, 'register_c', register)
    def trace_register(prepared, x, y, seed, emit):
        assert prepared['mopc_hr'] is anchor
        seen.append(('trace', seed))
        return SimpleNamespace(resource={}, protocol_metadata={})
    monkeypatch.setattr(group2_dadda_mopc_diagnostic, 'trace_registration', trace_register)
    monkeypatch.setattr(worker, 'score_fixed', lambda *a, **k: (np.array([0, 1]), np.array([0, 1])))
    monkeypatch.setattr(worker, 'split_rows', lambda *a: np.array([9, 10]))
    if mismatch:
        with pytest.raises(ValueError, match='old physical support differs'):
            worker.predict(dict(data_root=str(tmp_path/'data'), device='cpu'), job)
        assert not list((tmp_path/'out').glob('*.npz'))
        return
    worker.predict(dict(data_root=str(tmp_path/'data'), device='cpu'), job)
    result = json.loads((tmp_path/'out'/'predictions.json').read_text())
    done = json.loads((tmp_path/'out'/'complete.json').read_text())
    assert seen[0] == ('subset', ['orthogonal', 'lora'], 2)
    assert len(seen) == (2 if trace else 1)
    assert result['rows'][0]['resources']['B_state_ref'] == ref
    assert done['status'] == 'MOPC_ONLY_PREDICTIONS_FIXED' and not done['truth_read']


def test_real_capsule_support_positions_are_mapped_to_physical_ids(tmp_path, monkeypatch):
    cap = tmp_path/'data'/'capsules/P/M/R'
    for name in ('splits', 'support_labels', 'stage_rows'):
        (cap/name).mkdir(parents=True, exist_ok=True)
    old_id, new_id = 'a'*24, 'b'*24
    contract = dict(capsule_id='cap', protocol_schema='p2_min_v1', phase2_data_status='VALIDATED_ONCE')
    (cap/'manifest.json').write_text(json.dumps(dict(contract, storage='episodic-pool-complement-v1', received_count=12)))
    for n, split_id in ((0, old_id), (2, new_id)):
        support = list(range(1, 7+n))
        split = dict(contract, split_id=split_id, k=1, new_count=n, registered_classes=list(range(6+n)),
                     support_indices=support, query_count=12-len(support))
        (cap/'splits'/f'{split_id}.json').write_text(json.dumps(split))
        np.save(cap/'support_labels'/f'{split_id}.npy', np.arange(6+n, dtype=np.int64))
    np.save(cap/'received.npy', np.zeros((12,2,256), dtype=np.float32))
    np.save(cap/'ids.npy', np.arange(100,112, dtype=np.int64))
    np.save(cap/'stage_rows/N02.npy', np.arange(12, dtype=np.int32))
    subset = tmp_path/'subset'; subset.mkdir()
    ref = str(subset/'B-K1.pt')
    row = dict(k=1, resources=dict(B_state_ref=ref, B_seconds=0, B_native={}))
    (subset/'predictions.json').write_text(json.dumps(dict(capsule_id='cap', profile='protonet_cda', model_seed=7, rows=[row])))
    (subset/'complete.json').write_text(json.dumps(dict(status='SUBSET_PREDICTIONS_FIXED', truth_read=False, query_fit=False)))
    job = dict(key='p', capsule='capsules/P/M/R', profile='protonet_cda', seed=7, methods=['mopc_hr'],
               points=[[1,2]], output=str(tmp_path/'out'), load_B_for_mopc=str(subset))
    monkeypatch.setattr(worker, 'activate', lambda rt: None)
    monkeypatch.setattr(group2_schedule, 'closed_prediction', lambda *a: None)
    monkeypatch.setattr(worker, 'load_b_state', lambda *a: dict(anchor=object()))
    monkeypatch.setattr(worker, 'register_c', lambda *a, **kw: {'mopc_hr': SimpleNamespace(resource={}, protocol_metadata={})})
    monkeypatch.setattr(worker, 'score_fixed', lambda state, query, **kw: (np.zeros(len(query), np.int16), np.zeros(len(query), np.int16)))
    monkeypatch.setattr(worker, 'split_rows', lambda *a: np.array([0,9,10,11]))
    worker.predict(dict(data_root=str(tmp_path/'data'), device='cpu'), job)
    assert json.loads((tmp_path/'out/complete.json').read_text())['status'] == 'MOPC_ONLY_PREDICTIONS_FIXED'

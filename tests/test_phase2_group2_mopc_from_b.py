import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.pamf_phase2_practical import group2_mopc_from_b as worker, group2_schedule, group1_target, group2_dadda_mopc_diagnostic


@pytest.mark.parametrize('trace', [False, True])
def test_saved_anchor_is_used_without_refitting_b(tmp_path, monkeypatch, trace):
    cap = tmp_path/'data'/'capsules/P/M/R'; splits = cap/'splits'; splits.mkdir(parents=True)
    (cap/'manifest.json').write_text(json.dumps(dict(capsule_id='cap', protocol_schema='p2_min_v1',
                                                      phase2_data_status='VALIDATED_ONCE')))
    for n, split in ((0, 'old'), (2, 'new')):
        (splits/(split+'.json')).write_text(json.dumps(dict(k=5, new_count=n, split_id=split,
                                                           registered_classes=list(range(6+n)))))
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
        support_ids=np.array([1, 2, 3, 4]), support_x=np.zeros((4, 3), dtype=np.float32),
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
    worker.predict(dict(data_root=str(tmp_path/'data'), device='cpu'), job)
    result = json.loads((tmp_path/'out'/'predictions.json').read_text())
    done = json.loads((tmp_path/'out'/'complete.json').read_text())
    assert seen[0] == ('subset', ['orthogonal', 'lora'], 2)
    assert len(seen) == (2 if trace else 1)
    assert result['rows'][0]['resources']['B_state_ref'] == ref
    assert done['status'] == 'MOPC_ONLY_PREDICTIONS_FIXED' and not done['truth_read']

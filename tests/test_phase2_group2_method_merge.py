import copy
import json
import numpy as np
import pytest

from experiments.pamf_phase2_practical.group2_method_merge import combine_indices, materialize_merge


def test_combine_requires_all_cores_same_saved_b_and_split(tmp_path):
    job = dict(capsule='capsules/P/M/R', profile='protonet_cda', seed=7, points=[[5, 20]])
    path = str(tmp_path/'B-K5.pt')
    def row(method):
        return dict(k=5, new_count=20, arm='protonet_cda__'+method, split_id='split',
                    registered_classes=[0, 1], query_count=42, resources={'B_state_ref': path})
    index = dict(capsule=job['capsule'], capsule_id='cap', profile=job['profile'], model_seed=7)
    subset = dict(index, rows=[row('orthogonal'), row('lora')])
    mopc = dict(index, rows=[row('mopc_hr')])
    assert len(combine_indices(job, subset, mopc, tmp_path)['rows']) == 3
    wrong = copy.deepcopy(mopc)
    wrong['rows'][0]['resources']['B_state_ref'] = str(tmp_path/'different.pt')
    with pytest.raises(ValueError, match='same B state'):
        combine_indices(job, subset, wrong, tmp_path)
    wrong = copy.deepcopy(mopc)
    wrong['rows'][0]['split_id'] = 'different'
    with pytest.raises(ValueError, match='split or query'):
        combine_indices(job, subset, wrong, tmp_path)
    with pytest.raises(ValueError, match='incomplete or duplicated'):
        combine_indices(job, subset, dict(index, rows=[]), tmp_path)


def test_materialize_requires_same_query_indices_before_writing(tmp_path):
    subset, mopc, merged = tmp_path/'subset', tmp_path/'mopc', tmp_path/'merged'
    subset.mkdir(); mopc.mkdir()
    job = dict(key='p-s7-c0', capsule='capsules/P/M/R', profile='protonet_cda', seed=7,
               methods=['mopc_hr', 'orthogonal', 'lora'], points=[[5, 20]])
    ref = str(subset/'B-K5.pt'); (subset/'B-K5.pt').touch()
    def row(method):
        return dict(k=5, new_count=20, arm='protonet_cda__'+method, split_id='split',
                    registered_classes=[0, 1], query_count=2, prediction=method+'.npz', resources={'B_state_ref': ref})
    common = dict(capsule=job['capsule'], capsule_id='cap', profile=job['profile'], model_seed=7)
    left = dict(common, rows=[row('orthogonal'), row('lora')]); right = dict(common, rows=[row('mopc_hr')])
    for root, index, status in ((subset, left, 'SUBSET_PREDICTIONS_FIXED'),
                                (mopc, right, 'MOPC_ONLY_PREDICTIONS_FIXED')):
        (root/'predictions.json').write_text(json.dumps(index), encoding='utf-8')
        (root/'complete.json').write_text(json.dumps(dict(status=status, truth_read=False, query_fit=False,
                                                          capsule_id='cap', rows=len(index['rows']))), encoding='utf-8')
        for r in index['rows']:
            np.savez(root/r['prediction'], query_indices=np.array([4, 9]))
    np.savez(mopc/'mopc_hr.npz', query_indices=np.array([4, 8]))
    with pytest.raises(ValueError, match='query indices differ'):
        materialize_merge(job, subset, mopc, merged)
    assert not merged.exists()
    np.savez(mopc/'mopc_hr.npz', query_indices=np.array([4, 9]))
    materialize_merge(job, subset, mopc, merged)
    assert json.loads((merged/'complete.json').read_text())['rows'] == 3
    with pytest.raises(FileExistsError):
        materialize_merge(job, subset, mopc, merged)

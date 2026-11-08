import json

import pytest

from experiments.pamf_phase2_practical.group2_schedule import closed_prediction


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding='utf-8')


def test_subset_marker_cannot_pass_as_full_matrix_and_requires_saved_b(tmp_path):
    job = dict(key='p-s1-c0', profile='protonet_cda', seed=1, capsule='capsules/P/M/R',
               methods=['orthogonal', 'lora'], points=[[5, 20]], output=str(tmp_path), save_B_for_mopc=True)
    rows = [dict(k=5, new_count=20, arm='protonet_cda__'+m, prediction=m+'.npz') for m in job['methods']]
    _write(tmp_path/'predictions.json', dict(capsule=job['capsule'], capsule_id='cap', model_seed=1,
                                             profile=job['profile'], rows=rows))
    _write(tmp_path/'complete.json', dict(status='SUBSET_PREDICTIONS_FIXED', truth_read=False,
                                          query_fit=False, capsule_id='cap', rows=2))
    for method in job['methods']:(tmp_path/(method+'.npz')).touch()
    with pytest.raises(ValueError, match='Missing saved B state'):
        closed_prediction(job, 2)
    (tmp_path/'B-K5.pt').touch()
    closed_prediction(job, 2)
    with pytest.raises(ValueError, match='Prediction not frozen'):
        closed_prediction(dict(job, save_B_for_mopc=False), 2)

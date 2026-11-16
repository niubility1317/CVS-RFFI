import json
import pytest
from experiments.pamf_phase2_practical.satellite_tweak import validate_scope,verify_prediction


def test_tweak_scope_is_full_mix_and_single_scene():
    job=dict(profile='tweak_m1',capsule='capsules/P01/MIX02/1-1',seed=2026092701,
             points=[[k,n] for k in (1,5,10,20) for n in (2,5,10,20)])
    assert validate_scope(job)==[1,5,10,20]
    job['points']=[[5,20]]
    with pytest.raises(ValueError):validate_scope(job)


def test_tweak_score_rejects_batch_decision_or_old_version(tmp_path):
    job=dict(profile='tweak_m1',capsule='capsules/P01/LS01/1-1',seed=2026092701,
             points=[[5,20]],output=str(tmp_path))
    rows=[dict(stage=s,k=5,new_count=0,input_path='S0Q0',arm='tweak_m1_single_v1',
        input_condition='satellite_received',prediction=s+'.npz',query_records_per_decision=1) for s in ('A0','B')]
    for r in rows:(tmp_path/r['prediction']).touch()
    (tmp_path/'complete.json').write_text(json.dumps(dict(status='PREDICTIONS_COMPLETE',truth_read=False,
        query_fit=False,profile='tweak_m1',model_seed=job['seed'],capsule_id='id',rows=2)))
    index=dict(capsule=job['capsule'],capsule_id='id',model_seed=job['seed'],rows=rows)
    (tmp_path/'predictions.json').write_text(json.dumps(index));verify_prediction(job)
    rows[0]['query_records_per_decision']=128
    (tmp_path/'predictions.json').write_text(json.dumps(index))
    with pytest.raises(ValueError):verify_prediction(job)
    rows[0]['query_records_per_decision']=1;rows[1]['arm']='tweak_m1'
    (tmp_path/'predictions.json').write_text(json.dumps(index))
    with pytest.raises(ValueError):verify_prediction(job)

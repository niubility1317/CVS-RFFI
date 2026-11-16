import json
import pytest
from experiments.pamf_phase2_practical.satellite_issl import validate_scope,verify_prediction
from experiments.pamf_phase2_practical.satellite_native_c import predecessor_done


def test_issl_scores_each_kn_and_computes_a0_only_once():
    job=dict(profile='issl_fixed_v1',capsule='capsules/P01/MIX01/1-1',seed=2026092701,points=[[5,2]],skip_a0=False)
    assert validate_scope(job)==(5,2)
    job['points']=[[5,10]]
    with pytest.raises(ValueError):validate_scope(job)
    job['skip_a0']=True
    assert validate_scope(job)==(5,10)


def test_issl_requires_long_version_and_satellite_input(tmp_path):
    job=dict(profile='issl_fixed_v1',capsule='capsules/P01/LS01/1-1',seed=2026092701,
             points=[[5,20]],skip_a0=False,output=str(tmp_path))
    rows=[dict(stage=s,k=5,new_count=n,input_path='S0Q0',arm='issl_fixed_v1_long_v1',
               input_condition='satellite_received',prediction=s+'.npz') for s,n in (('A0',0),('C',20))]
    for row in rows:(tmp_path/row['prediction']).touch()
    (tmp_path/'complete.json').write_text(json.dumps(dict(status='PREDICTIONS_COMPLETE',truth_read=False,
        query_fit=False,profile='issl_fixed_v1',model_seed=job['seed'],capsule_id='id',rows=2)))
    idx=dict(capsule=job['capsule'],model_seed=job['seed'],capsule_id='id',rows=rows)
    (tmp_path/'predictions.json').write_text(json.dumps(idx));verify_prediction(job)
    rows[1]['arm']='issl_fixed_v1';(tmp_path/'predictions.json').write_text(json.dumps(idx))
    with pytest.raises(ValueError):verify_prediction(job)


def test_issl_waits_for_both_predecessors(tmp_path):
    paths=[tmp_path/'one',tmp_path/'two']
    for path in paths:(path/'scheduler').mkdir(parents=True)
    (paths[0]/'scheduler'/'complete.json').write_text('{}')
    config=dict(wait_runs=list(map(str,paths)))
    assert not predecessor_done(config)
    (paths[1]/'scheduler'/'complete.json').write_text('{}')
    assert predecessor_done(config)

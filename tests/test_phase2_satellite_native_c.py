import json
import pytest
from experiments.pamf_phase2_practical.satellite_native_c import predecessor_done, validate_scope, verify_prediction


def test_mopc_keeps_only_unscored_a0_while_registering_every_mix_point():
    job=dict(profile='mopc_hr',capsule='capsules/P02/MIX02/1-1',seed=2026092701,
             satellite_c_points=[[k,n] for k in (1,5,10,20) for n in (2,5,10,20)],
             satellite_a0_ks=[1,10,20],mopc_stabilization='clip1000_v1')
    points,ks=validate_scope(job)
    assert len(points)==16 and ks==[1,10,20]
    job['satellite_a0_ks']=[1,5,10,20]
    with pytest.raises(ValueError):validate_scope(job)


def test_fixed_orthogonal_needs_new_a0_and_single_scene_has_one_c():
    job=dict(profile='orthogonal',capsule='capsules/P01/LS01/1-1',seed=2026092701,
             satellite_c_points=[[5,20]],satellite_a0_ks=[5])
    assert validate_scope(job)==([(5,20)],[5])
    job['satellite_a0_ks']=[]
    with pytest.raises(ValueError):validate_scope(job)


def test_satellite_c_scoring_requires_exact_points_and_received_input(tmp_path):
    job=dict(profile='mopc_hr',capsule='capsules/P01/LS01/1-1',seed=2026092701,
             satellite_c_points=[[5,20]],satellite_a0_ks=[],mopc_stabilization='clip1000_v1',output=str(tmp_path))
    row=dict(stage='C',k=5,new_count=20,input_path='P0',arm='mopc_hr_clip1000_v1',
             input_condition='satellite_received',prediction='c.npz')
    (tmp_path/'c.npz').touch()
    (tmp_path/'complete.json').write_text(json.dumps(dict(status='PREDICTIONS_COMPLETE',truth_read=False,
        query_fit=False,profile=job['profile'],model_seed=job['seed'],capsule_id='id',rows=1)))
    index=dict(capsule=job['capsule'],capsule_id='id',model_seed=job['seed'],rows=[row])
    (tmp_path/'predictions.json').write_text(json.dumps(index))
    verify_prediction(job)
    row['input_condition']='clean_ground_pre_satellite_normalized'
    (tmp_path/'predictions.json').write_text(json.dumps(index))
    with pytest.raises(ValueError):verify_prediction(job)


def test_waits_for_previous_queue_to_drain_before_sharing_gpus(tmp_path):
    control=tmp_path/'scheduler';control.mkdir();config=dict(wait_run=str(tmp_path))
    assert not predecessor_done(config)
    (control/'state.json').write_text(json.dumps(dict(status='FAILED',active=[{'pid':1}])))
    assert not predecessor_done(config)
    (control/'state.json').write_text(json.dumps(dict(status='FAILED',active=[])))
    assert predecessor_done(config)


def test_balanced_lora_c_cannot_be_mixed_with_historical_arm(tmp_path):
    job=dict(profile='lora_triplet_knn',capsule='capsules/P01/LS01/1-1',seed=2026092701,
             satellite_c_points=[[5,20]],satellite_a0_ks=[],output=str(tmp_path))
    row=dict(stage='C',k=5,new_count=20,input_path='P0',arm='lora_triplet_knn_balanced_v1',
             input_condition='satellite_received',prediction='c.npz')
    (tmp_path/'c.npz').touch()
    (tmp_path/'complete.json').write_text(json.dumps(dict(status='PREDICTIONS_COMPLETE',truth_read=False,
        query_fit=False,profile=job['profile'],model_seed=job['seed'],capsule_id='id',rows=1)))
    index=dict(capsule=job['capsule'],capsule_id='id',model_seed=job['seed'],rows=[row])
    (tmp_path/'predictions.json').write_text(json.dumps(index));verify_prediction(job)
    row['arm']='lora_triplet_knn'
    (tmp_path/'predictions.json').write_text(json.dumps(index))
    with pytest.raises(ValueError):verify_prediction(job)

import copy
import json

import pytest

from experiments.pamf_phase2_practical.mopc_author_satellite_restart import make_plan, POINTS, PROFILES
from experiments.pamf_phase2_practical.satellite_native_c import validate_scope
from experiments.pamf_phase2_practical.satellite_native_da import RECEIVERS


def predecessor():
    jobs=[]
    for profile in PROFILES:
        for pool in ('P01','P02','P03'):
            for mix in ('MIX01','MIX02','MIX03'):
                for rx in RECEIVERS:
                    for seed in (2026092701,2026092702,2026092703):
                        common=dict(group='native' if profile=='mopc_hr' else 'hybrid',profile=profile,
                                    capsule=f'capsules/{pool}/{mix}/{rx}',seed=seed,mopc_stabilization='clip1000_v1')
                        if profile != 'mopc_hr':
                            common.update(points=POINTS,methods=['mopc_hr'],satellite_hybrid=True,save_B_for_mopc=True)
                            if profile=='protonet_cda':common['mopc_bridge']='proto_affine_v1'
                        jobs.append(common)
    return dict(entry_module='satellite_mix3',input_condition='satellite_received',jobs=jobs,
                runtimes={'hybrid':'hybrid.json','mopc_hr':'native.json'})


def test_matrix_uses_author_version_and_does_not_repeat_a0():
    plan=make_plan(predecessor(),'/tmp/new-author-root','/tmp/new-release')
    assert len(plan['jobs'])==1134
    assert plan['max_per_gpu']==6 and plan['minimum_free_MiB']==6000
    assert all(j['mopc_stabilization']=='none' for j in plan['jobs'])
    for job in plan['jobs']:
        if job['group']=='native':
            points,a0=validate_scope(job)
            assert points==[tuple(p) for p in POINTS] and a0==[]


def test_rejects_missing_or_duplicate_tasks_and_wrong_proto_bridge():
    p=predecessor();p['jobs'].pop()
    with pytest.raises(ValueError):make_plan(p,'/tmp/new-author-root','/tmp/new-release')
    p=predecessor();p['jobs'][1]=copy.deepcopy(p['jobs'][0])
    with pytest.raises(ValueError):make_plan(p,'/tmp/new-author-root','/tmp/new-release')
    p=predecessor();next(j for j in p['jobs'] if j['profile']=='protonet_cda')['mopc_bridge']='original'
    with pytest.raises(ValueError):make_plan(p,'/tmp/new-author-root','/tmp/new-release')


def test_rejects_extra_a0_or_clipping():
    job=next(j for j in make_plan(predecessor(),'/tmp/new-author-root','/tmp/new-release')['jobs'] if j['profile']=='mopc_hr')
    job['satellite_a0_ks']=[10]
    with pytest.raises(ValueError):validate_scope(job)
    job['satellite_a0_ks']=[];job['mopc_stabilization']='clip1000_v1'
    with pytest.raises(ValueError):validate_scope(job)


def test_small_author_pilot_keeps_all_six_methods_and_one_fixed_cell():
    plan=make_plan(predecessor(),'/tmp/new-author-pilot','/tmp/new-release',pilot=True)
    assert len(plan['jobs'])==6
    assert {j['profile'] for j in plan['jobs']}==set(PROFILES)
    assert {j['capsule'] for j in plan['jobs']}=={'capsules/P01/MIX01/1-1'}
    assert {j['seed'] for j in plan['jobs']}=={2026092701}
    for job in plan['jobs']:
        if job['group']=='native':
            points,a0=validate_scope(job)
            assert points==[(5,5)] and a0==[]
        else:
            assert job['points']==[[5,5]]


def test_pilot_hybrid_verifier_accepts_one_k_and_author_sampling(tmp_path):
    from experiments.pamf_phase2_practical.satellite_mix3 import verify_prediction
    j=next(j for j in make_plan(predecessor(),'/tmp/new-author-pilot','/tmp/new-release',pilot=True)['jobs']
           if j['profile']=='mrior_sda')
    j['output']=str(tmp_path)
    row=dict(k=5,new_count=5,stage='C',arm='mrior_sda__mopc_hr',input_condition='satellite_received',
             prediction='p.npz',resources={'C_protocol':{'mopc_version':'public_code_semantics_matched_IQ_extension',
             'mopc_sampling_version':'author_drop_last_v1','config':{'mopc_max_grad_norm':None}}})
    (tmp_path/'p.npz').touch();(tmp_path/'B-K5.pt').touch()
    (tmp_path/'predictions.json').write_text(json.dumps(dict(capsule=j['capsule'],rows=[row])))
    (tmp_path/'complete.json').write_text(json.dumps(dict(status='PREDICTIONS_COMPLETE',truth_read=False,
        query_fit=False,profile=j['profile'],model_seed=j['seed'],rows=1)))
    verify_prediction(j)
    del row['resources']['C_protocol']['mopc_sampling_version']
    (tmp_path/'predictions.json').write_text(json.dumps(dict(capsule=j['capsule'],rows=[row])))
    with pytest.raises(ValueError,match='sampling'):
        verify_prediction(j)

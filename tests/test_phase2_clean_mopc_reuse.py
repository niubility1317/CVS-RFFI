import json
import pytest
from experiments.pamf_phase2_practical.clean_mopc_reuse import copy_pilot_rows


def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj));return str(path)


def test_pilot_rows_reused_without_reprediction_and_bound_to_source(tmp_path):
    old=tmp_path/'pilot';old.mkdir();out=tmp_path/'out';out.mkdir()
    capsule='capsules/P01/MIX01/1-1';seed=2026092702
    old_job=dict(profile='mopc_hr',seed=seed,capsule=capsule,points=[[5,20]],output=str(old))
    rt=dict(run_root='/source',data_root='/data',clean_root='/clean',source_contract='/contract',source_cache='/cache')
    rows=[];lookup={}
    for stage,n in [('A0',0),('C',20)]:
        spec=dict(split_id='s'+str(n),k=5,new_count=n,registered_classes=list(range(6+n)))
        lookup[5,n]=spec;name=stage+'.npz';(old/name).write_bytes(stage.encode())
        rows.append(dict(stage=stage,k=5,new_count=n,split_id=spec['split_id'],registered_classes=spec['registered_classes'],
                         arm='mopc_hr',input_path='P0',input_condition='clean_ground_pre_satellite_normalized',prediction=name))
    put(old/'predictions.json',dict(capsule=capsule,capsule_id='cap',model_seed=seed,profile='mopc_hr',rows=rows))
    put(old/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id='cap',model_seed=seed,rows=2,
                                 truth_read=False,query_fit=False))
    job=dict(profile='mopc_hr',seed=seed,capsule=capsule,reuse=dict(root=str(old),
        job=put(tmp_path/'job.json',old_job),runtime=put(tmp_path/'rt.json',rt)))
    copied=copy_pilot_rows(job,rt,{'capsule_id':'cap'},lookup,out)
    assert {(r['stage'],r['k'],r['new_count']) for r in copied}=={('A0',5,0),('C',5,20)}
    assert (out/'A0.npz').read_bytes()==b'A0' and (out/'C.npz').read_bytes()==b'C'
    other=tmp_path/'other';other.mkdir()
    with pytest.raises(ValueError,match='source'):
        copy_pilot_rows(job,dict(rt,run_root='/different'),{'capsule_id':'cap'},lookup,other)
    assert not list(other.iterdir())

import json
import pytest

from experiments.pamf_phase2_practical.clean_mopc_a0_reuse import copy_a0


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return str(path)


def test_only_fixed_a0_is_reused_for_new_mopc_version(tmp_path):
    old=tmp_path/'old';old.mkdir();out=tmp_path/'out';out.mkdir()
    rt=dict(run_root='/source',data_root='/data',clean_root='/clean',source_contract='/contract',
            source_cache='/cache',phase1_release='/release')
    job=dict(profile='mopc_hr',seed=2026092701,capsule='capsules/P/M/R',points=[[1,2]],output=str(old))
    spec=dict(split_id='oldsplit',registered_classes=list(range(6)))
    row=dict(stage='A0',k=1,new_count=0,arm='mopc_hr',input_path='P0',split_id='oldsplit',
             registered_classes=list(range(6)),prediction='A0.npz',input_condition='clean_ground_pre_satellite_normalized')
    (old/'A0.npz').write_bytes(b'fixed');(old/'C.npz').write_bytes(b'old-c')
    put(old/'complete.json',dict(status='PREDICTIONS_COMPLETE',truth_read=False,query_fit=False,capsule_id='cap'))
    put(old/'predictions.json',dict(capsule=job['capsule'],profile='mopc_hr',model_seed=job['seed'],rows=[row,dict(row,stage='C',prediction='C.npz')]))
    stable=dict(job,output=str(out),mopc_stabilization='clip1000_v1',reuse_a0=dict(
        job=put(tmp_path/'oldjob.json',job),runtime=put(tmp_path/'oldrt.json',rt)))
    copied=copy_a0(stable,rt,{'capsule_id':'cap'},{(1,0):spec},out)
    assert len(copied)==1 and (out/'A0.npz').read_bytes()==b'fixed' and not (out/'C.npz').exists()
    with pytest.raises(ValueError,match='source or data'):
        copy_a0(stable,dict(rt,source_cache='/other'),{'capsule_id':'cap'},{(1,0):spec},tmp_path/'other')

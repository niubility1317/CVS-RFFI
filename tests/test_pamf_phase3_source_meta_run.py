"""Source-meta evaluation tests reuse synthetic pure-source/public fixtures."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.source_meta_run')


def identities(m,c,pm):
    return {arm:dict(run_id=m.RUN_ID,code_commit=c['code_commit'],arm=arm,updates=444,signal_parameters=335681,
        trainable_parameters=335633,identity_head_updates=444,projection_updates=444,source_only=True,target_U_used=False,target_U_reads=0,
        parent_F0_identity=pm['parent_F0_identity'],parent_signal_identity=pm['model_identities']['E0'],dura_identity=pm['dura_identity'],
        identity_head_jointly_trained=True,identity_head_replaced=False,initial_native_logits_equal_original_F0=True,
        meta_enabled=arm=='META_NATIVE',meta_alpha=.001 if arm=='META_NATIVE' else 0.,meta_beta=.5,second_order=True,
        second_order_contribution_active=arm=='META_NATIVE',test_only_nominal_completion=True) for arm in m.NEW_ARMS}


def test_meta_identity_and_exact_public_only_config(tmp_path,monkeypatch,subtests):
    m=api();monkeypatch.setattr(m.fresh,'validate_config',lambda c:c);monkeypatch.setattr(m,'_train',lambda:SimpleNamespace(validate_config=lambda c:c))
    r=dict(code_commit=m.CONTROL_CODE,output_root=str(tmp_path/'parent'),train_config=dict(output_root=str(tmp_path/'PCD'),U_public_root=str(tmp_path/'U')),
        source_control_config=dict(reference_config=dict(train_config=dict(IQ_root=str(tmp_path/'IQ')))),Q_conditions={})
    t=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=r,output_root=str(tmp_path/'train'),steps=444,lr=.0001)
    c=dict(run_id=m.RUN_ID,code_commit='a'*40,train_config=t,reference_config=r,Q_conditions={},output_root=str(tmp_path/'predict'),batch_size=128)
    assert m.validate_config(c) is c and m.NEW_ARMS==('ERM_NATIVE','META_NATIVE') and m.PRIMARY_METHOD=='META_NATIVE_predicted_ratio'
    for change in (dict(truth='PRIVATE'),dict(U_truth='PRIVATE'),dict(unknown_K=4),dict(batch_size=True),dict(output_root=r['output_root'])):
        with subtests.test(change=change),pytest.raises(ValueError):m.validate_config(dict(c,**change))
    pm=dict(parent_F0_identity={'F0':'original'},model_identities=dict(E0={'projection':'pure-r03'}),dura_identity={'G':'frozen'})
    for arm,i in identities(m,c,pm).items():assert m._identity(i,arm,c,pm) is i
    good=identities(m,c,pm)['META_NATIVE']
    for change in (dict(meta_alpha=0.),dict(meta_beta=1.),dict(second_order=False),dict(meta_enabled=False),dict(target_U_reads=576),dict(source_only=False)):
        with subtests.test(identity=change),pytest.raises(ValueError):m._identity(dict(good,**change),'META_NATIVE',c,pm)


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_fresh_world_run import fixture as parent
    from test_pamf_phase3_source_style_run import ToyJoint
    with parent(root,mp) as (fresh,r,_,b,events,pcd_id):
        m=api();r['code_commit']=m.CONTROL_CODE;r['train_config']['code_commit']=m.CONTROL_CODE;pcd_id['code_commit']=m.CONTROL_CODE
        pm=fresh.run(r,'cpu');events.clear();mp.setattr(m,'COUNTS',dict(fresh.COUNTS));mp.setattr(m,'Q_COUNT',fresh.Q_COUNT)
        t=SimpleNamespace(SCHEMA='pamf_phase3_source_meta_train_v1',STATUS='SOURCE_META_TRAIN_COMPLETE',ROW_STATUS='SOURCE_META_STUDENT_FROZEN',validate_config=lambda c:c)
        mp.setattr(m,'_train',lambda:t)
        tc=dict(run_id=m.RUN_ID,code_commit='b'*40,reference_config=deepcopy(r),output_root=str(root/'meta_train'),steps=444,lr=.0001)
        c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],train_config=tc,reference_config=deepcopy(r),Q_conditions=deepcopy(r['Q_conditions']),output_root=str(root/'meta_predict'),batch_size=128)
        ids=identities(m,c,pm);models={arm:ToyJoint(b['model'],b['reader'],j+4) for j,arm in enumerate(m.NEW_ARMS)}
        rows={a:dict(schema=t.SCHEMA,status=t.ROW_STATUS,identity=i,updates=444,source_only=True,target_U_reads=0,test_only_nominal_completion=True) for a,i in ids.items()}
        done=dict(schema=t.SCHEMA,status=t.STATUS,run_id=m.RUN_ID,code_commit=c['code_commit'],config=tc,arms=list(m.NEW_ARMS),rows=rows,identities=ids,
            total_optimizer_updates=888,new_target_UDA_updates=0,source_only=True,source_sampling_matched=True,original_models_unchanged=True,
            all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_heads_changed=True,all_student_projections_changed=True,
            target_U_reads=0,target_Q_reads=0,private_truth_reads=0,U_truth_reads=0,final_phase3_complete=False,
            full_second_order=True,all_receiver_splits_disjoint=True,virtual_updates_committed=False,meta_alpha=.001,meta_beta=.5,
            real_parent_signal_forward_rows=m.COUNTS['L']+m.COUNTS['OE'],real_student_forward_rows=480*888,
            real_student_base_forward_rows=192*888,real_student_virtual_forward_rows=288*888,
            G_forward_rows=0,real_F_forward_rows=0,test_only_nominal_completion=True)
        for arm,row in rows.items():
            folder=Path(tc['output_root'])/arm;folder.mkdir(parents=True);m._write(folder/'complete.json',row)
        m._write(Path(tc['output_root'])/'complete.json',done)
        t.load_trained_model=lambda cfg,arm,device:(models[arm],ids[arm]);mp.setattr(m,'_backend',lambda *a:b)
        role=m._iq_role;public=m._public;calibrate=m.risk.calibrate_known;smoke=m._smoke;flow=[]
        def iq(config,name,header):
            assert name in ('L','OE','V','novel') and Path(c['output_root'],'smoke.json').exists()
            if name=='V':assert Path(c['output_root'],'models/complete.json').exists()
            if name=='novel':assert Path(c['output_root'],'calibration/complete.json').exists()
            flow.append(name);return role(config,name,header)
        def query(config,condition):
            assert Path(c['output_root'],'source/META_NATIVE/novel_predictions.npz').exists();flow.append(condition)
            return public(config,condition)
        def cal(*args):
            assert Path(c['output_root'],'models/complete.json').exists() and not Path(c['output_root'],'calibration/complete.json').exists()
            old=fresh._source_rows(r['output_root'],'source_DURA','known');np.testing.assert_array_equal(args[3],old['closed'])
            assert set(args[2].tolist()) in ({4},{5});flow.append('calibrate');return calibrate(*args)
        load=np.load;forbidden=Path(r['train_config']['U_public_root']).resolve()
        def no_U(path,*args,**kwargs):
            if isinstance(path,(str,Path)):
                p=Path(path).resolve();assert not p.is_relative_to(forbidden) and p.name not in ('U_features.npz','U_restored.npz')
            return load(path,*args,**kwargs)
        mp.setattr(m,'_iq_role',iq);mp.setattr(m,'_public',query);mp.setattr(m.risk,'calibrate_known',cal)
        mp.setattr(m,'_smoke',lambda *a:(flow.append('smoke'),smoke(*a))[1]);mp.setattr(np,'load',no_U)
        yield m,c,pm,flow,done


def test_partial_meta_checkpoint_header_stops_before_source(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,flow,done):
        p=Path(c['train_config']['output_root'])/'META_NATIVE/complete.json';row=m._read(p);row['identity']['second_order']=False
        p.unlink();m._write(p,row)
        with pytest.raises(ValueError):m.run(c,'cpu')
        assert flow==[] and not Path(c['output_root']).exists()

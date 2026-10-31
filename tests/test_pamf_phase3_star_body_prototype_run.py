"""Frozen projection students, source density/calibration, and shared-DURA queries."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_body_prototype_run')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_class_density_run import fixture as inherited
    with inherited(root,mp) as (parent,pc,_,backend,public,_,_):
        m=api();pc['code_commit']=m.PARENT_CODE;pm=parent.run(pc,'cpu')
        mp.setattr(m,'COUNTS',dict(parent.COUNTS));mp.setattr(m,'Q_COUNT',parent.Q_COUNT)
        mp.setattr(m.source_eval,'COUNTS',dict(m.COUNTS));mp.setattr(m.source_eval.IQ,'COUNTS',dict(m.COUNTS))
        tc=dict(run_id=m.RUN_ID,code_commit='f'*40,reference_config=deepcopy(pc),U_public_root=str(root/'discovery/public'),U_capsule_id=m.U_CAPSULE_ID,
            U_split_id=m.U_SPLIT_ID,output_root=str(root/'body_train'),steps=444,lr=.0001)
        c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],train_config=tc,reference_config=deepcopy(pc),output_root=str(root/'prototype_predict'),batch_size=128)
        students={};identities={'E0':pm['model_identity']};events=[];inputs={name:[] for name in m.MODEL_NAMES}
        for index,arm in enumerate(m.MODEL_NAMES[1:]):
            model=deepcopy(backend['model'])
            with torch.no_grad():model.projection.network[2].bias.add_(.005*(index+1))
            model.eval().requires_grad_(False);students[arm]=model
            identities[arm]=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],arm=arm,updates=444,projection_updates=444,projection_cumulative_updates=2664,
                open_signal_updates=444,signal_encoder_updates=444,native_updates=0,encoder_frozen=False,Sinc_frozen=True,signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,
                signal_feature_cumulative_updates=13764,trainable_parameters=334673,body_trainable_parameters=231985,trainable_projection_parameters=102688,
                new_body_updates=444,new_projection_updates=444,Sinc_optimizer_steps=0,input_path='cached_P1_IQ_only',
                parent_signal_identity=pm['model_identity'],parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],
                source_only=arm=='SOURCE',target_U_used=arm=='TARGET_PCD',target_U_reads=384 if arm=='TARGET_PCD' else 0)
        Uids=np.asarray(['U'+str(i).zfill(4) for i in range(384)])
        train=SimpleNamespace(STEPS=444,U_COUNT=384,validate_config=lambda a:a)
        mp.setattr(m,'_train',lambda:train)
        mp.setattr(m,'_training_metadata',lambda *a:(identities,Uids))
        Path(tc['output_root']).mkdir()
        m._write(Path(tc['output_root'])/'complete.json',dict(real_parent_signal_forward_rows=m.COUNTS['L']+m.COUNTS['OE']+384,
            real_student_signal_forward_rows=454656,G_forward_rows=384,real_F_forward_rows=0,
            synthetic_parent_signal_forward_rows=2,synthetic_student_signal_forward_rows=2,G_synthetic_forward_rows=2,synthetic_F_forward_rows=2))
        def load(config,arm,device):
            events.append('load_'+arm)
            students[arm].register_forward_pre_hook(lambda mod,args,name=arm:inputs[name].append(args[0].detach().clone()))
            return students[arm],identities[arm]
        mp.setattr(m,'_load_model',load)
        # Only synthetic cache records are read; restore V/NV using the fixture's same G.
        training_roles={}
        for role in ('L','OE','V','novel'):
            folder=Path(pc['train_config']['IQ_root'])/role;raw=np.load(folder/'received.npy')
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'));manifest['code_commit']='a'*40
            (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            with torch.no_grad():restored=backend['dura'][1](torch.tensor(raw.tolist(),dtype=torch.float32))[0]
            restored=np.asarray(restored.tolist(),np.float32)
            if role in ('L','OE'):
                with np.load(folder/'metadata.npz',allow_pickle=False) as z:meta={k:z[k].copy() for k in z.files}
                training_roles[role]=dict(ids=np.load(folder/'ids.npy'),restored=restored,metadata=meta,
                    manifest=dict(role=role,count=len(raw),parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],source_only=True,restored_input_path='P1'))
            else:np.save(folder/'restored.npy',restored)
            if role=='V':
                old=parent._rows(pc['output_root'],'known');np.save(folder/'baseline_closed.npy',np.stack((old['closed'],old['closed']),axis=1))
        # L/OE are supplied as legal synthetic cached-P1 roles, matching real producer API.
        source=m._iq_role
        def role(config,name,identity):
            assert Path(c['output_root'],'smoke.json').exists()
            if name=='V':assert Path(c['output_root'],'models/complete.json').exists()
            if name=='novel':assert Path(c['output_root'],'calibration/complete.json').exists()
            events.append(name)
            if name in ('L','OE'):return training_roles[name]
            return source(config,name,identity)
        mp.setattr(m,'_iq_role',role)
        def opened(config,name):
            assert Path(c['output_root'],'source/source_DURA/novel_predictions.npz').exists();events.append(name);return public[name]
        mp.setattr(m,'_public',opened)
        yield m,c,pm,backend,events,students,identities,Uids,inputs


def test_strict_config_and_frozen_checkpoint_failure_precede_IQ(tmp_path,monkeypatch,subtests):
    m=api();monkeypatch.setattr(m,'validate_config',lambda c:c)
    monkeypatch.setattr(m,'_reference',lambda *a:pytest.fail('Existing output cannot read source metadata'))
    with pytest.raises(FileExistsError):m.run(dict(output_root=str(tmp_path)),'cpu')
    assert m.FIELDS=={'run_id','code_commit','train_config','reference_config','output_root','batch_size'}


def test_complete_training_metadata_and_target_U_ancestry(tmp_path,subtests):
    m=api();root=tmp_path/'train';tc=dict(output_root=str(root),U_capsule_id=m.U_CAPSULE_ID,U_split_id=m.U_SPLIT_ID)
    c=dict(code_commit='f'*40,train_config=tc);pm=dict(model_identity=dict(model='r03'),parent_F0_identity=dict(model='F0'),dura_identity=dict(model='G'))
    manifest=dict(count=384,capsule_id=m.U_CAPSULE_ID,split_id=m.U_SPLIT_ID);rows={};identities={}
    for arm in m.MODEL_NAMES[1:]:
        i=dict(run_id=m.RUN_ID,code_commit=c['code_commit'],arm=arm,updates=444,projection_updates=444,projection_cumulative_updates=2664,
            open_signal_updates=444,signal_encoder_updates=444,native_updates=0,encoder_frozen=False,Sinc_frozen=True,signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,
            signal_feature_cumulative_updates=13764,trainable_parameters=334673,body_trainable_parameters=231985,trainable_projection_parameters=102688,
            new_body_updates=444,new_projection_updates=444,Sinc_optimizer_steps=0,input_path='cached_P1_IQ_only',
            parent_signal_identity=pm['model_identity'],parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],
            source_only=arm=='SOURCE',target_U_used=arm=='TARGET_PCD',target_U_reads=384 if arm=='TARGET_PCD' else 0)
        identities[arm]=i;rows[arm]=dict(schema='pamf_phase3_body_prototype_train_v1',status='BODY_PROTOTYPE_STUDENT_FROZEN',identity=i,updates=444,
            source_only=i['source_only'],target_U_used=i['target_U_used'],target_U_reads=i['target_U_reads'],U_manifest=manifest if arm=='TARGET_PCD' else None,
            initial_state_equal_to_parent=True,student_Sinc_unchanged=True,student_body_changed=True,student_projection_changed=True,model_buffers_unchanged=True)
        (root/arm).mkdir(parents=True);m._write(root/arm/'complete.json',rows[arm])
    done=dict(schema='pamf_phase3_body_prototype_train_v1',status='BODY_PROTOTYPE_TRAINING_COMPLETE',run_id=m.RUN_ID,code_commit=c['code_commit'],
        config=tc,arms=['SOURCE','TARGET_PCD'],updates_per_arm=444,total_optimizer_updates=888,rows=rows,identities=identities,U_manifest=manifest,
        source_sampling_matched=True,SOURCE_completed_before_U_open=True,original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_projections_changed=True)
    m._write(root/'complete.json',done);ids=np.asarray(['opaque'+str(i) for i in range(384)]);np.savez(root/'U_features.npz',ids=ids)
    saved=(root/'complete.json').read_bytes();actual,loaded=m._training_metadata(c,pm)
    assert actual==dict(E0=pm['model_identity'],**identities);np.testing.assert_array_equal(loaded,ids)
    for defect in ('partial_arm','wrong_U_split','encoder_updates','duplicate_U_ids'):
        with subtests.test(defect=defect):
            (root/'complete.json').write_bytes(saved);np.savez(root/'U_features.npz',ids=ids)
            if defect=='partial_arm':
                a=deepcopy(done);a['rows'].pop('TARGET_PCD');(root/'complete.json').unlink();m._write(root/'complete.json',a)
            elif defect=='wrong_U_split':
                a=deepcopy(done);a['U_manifest']['split_id']='wrong';(root/'complete.json').unlink();m._write(root/'complete.json',a)
            elif defect=='encoder_updates':
                a=deepcopy(done);a['identities']['TARGET_PCD']['open_signal_updates']=0;(root/'complete.json').unlink();m._write(root/'complete.json',a)
            else:np.savez(root/'U_features.npz',ids=np.repeat(ids[:1],384))
            with pytest.raises(ValueError):m._training_metadata(c,pm)


def test_frozen_students_source_fit_then_calibration_then_all_DURA_predictions(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,students,identities,Uids,inputs):
        calibrate=m.risk.calibrate_known
        def source_failure(*a):
            state,report=calibrate(*a);report['constraints_passed']=False;return state,report
        monkeypatch.setattr(m.risk,'calibrate_known',source_failure)
        done=m.run(c,'cpu');out=Path(c['output_root'])
        assert events==['load_SOURCE','load_TARGET_PCD','L','OE','V','novel','clean_DURA','star_DURA']
        assert done['source_density_models_fitted']==2 and done['new_threshold_sets_fitted']==4
        assert done['all_three_DURA_conditions_complete'] and done['classification_unchanged']
        assert done['combined_real_signal_forward_rows']==done['training_costs']['real_parent_signal_forward_rows']+454656+done['real_signal_forward_rows']
        assert done['combined_G_forward_rows']==384+done['G_forward_rows'] and done['inherited_new_body_updates_total']==888
        with np.load(out/'thresholds.npz') as z:tau=z['conditional']
        np.testing.assert_array_equal(tau[:2],m._reference(c)[2])
        for role in ('known','novel'):
            a=m._source_rows(out,'source_DURA',role);old=m.parent._rows(c['reference_config']['output_root'],role)
            for key,v in old.items():np.testing.assert_array_equal(a[key][:,:2] if key in ('raw_scores','scores') else a[key][:,0] if key=='signal_features' else a[key],v[:,[1,3]] if key in ('raw_scores','scores') else v)
        for condition in m.CONDITIONS[1:]:
            a=m._target_rows(out,condition);old=m.parent._target_rows(c['reference_config']['output_root'],condition)
            np.testing.assert_array_equal(a['scores'][:,:2],old['scores'][:,[1,3]])
            assert a['signal_features'].shape==(m.Q_COUNT,3,160)
            np.testing.assert_array_equal(a['closed'],old['closed'])
        for arm in m.MODEL_NAMES[1:]:
            for x,want in zip(inputs[arm][-2:],b['dura'][0].outputs[-2:]):torch.testing.assert_close(x,want,rtol=0,atol=0)
        with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_identity_mismatch_and_partial_query_never_complete(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,students,identities,*_):
        for delta in (dict(truth='forbidden'),dict(batch_size=True),dict(output_root=c['train_config']['output_root']),dict(reference_config=dict(c['reference_config'],code_commit='b'*40))):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        identities['TARGET_PCD']['Sinc_frozen']=False
        with pytest.raises(ValueError):m.run(c,'cpu')
        assert 'L' not in events and not Path(c['output_root']).exists()
        identities['TARGET_PCD']['Sinc_frozen']=True;events.clear();opened=m._public
        monkeypatch.setattr(m,'_public',lambda config,name:(_ for _ in ()).throw(ValueError('partial star Q')) if name=='star_DURA' else opened(config,name))
        with pytest.raises(ValueError,match='partial star'):m.run(c,'cpu')
        assert Path(c['output_root'],'targets/clean_DURA/predictions.npz').exists() and not Path(c['output_root'],'complete.json').exists()

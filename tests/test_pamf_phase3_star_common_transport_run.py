"""One CCD student extends six completed body-prototype controls."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_common_transport_run')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_body_prototype_run import fixture as inherited
    with inherited(root,mp) as (old,oc,base,b,_,_,_,Uids,_):
        m=api();oc['code_commit']=m.PARENT_CODE;pm=old.run(oc,'cpu')
        mp.setattr(m,'COUNTS',dict(old.COUNTS));mp.setattr(m,'Q_COUNT',old.Q_COUNT)
        mp.setattr(m.previous_score,'COUNTS',old.COUNTS);mp.setattr(m.previous_score,'Q_COUNT',old.Q_COUNT)
        tc=dict(run_id=m.RUN_ID,code_commit='e'*40,reference_config=deepcopy(oc),U_public_root=str(root/'discovery/public'),U_capsule_id=m.U_CAPSULE_ID,
            U_split_id=m.U_SPLIT_ID,output_root=str(root/'CCD_train'),steps=444,lr=.0001)
        c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],train_config=tc,reference_config=deepcopy(oc),output_root=str(root/'CCD_predict'),batch_size=128)
        model=deepcopy(b['model'])
        with torch.no_grad():model.projection.network[2].bias.add_(.03)
        model.eval().requires_grad_(False)
        identity=dict(run_id=m.RUN_ID,code_commit=c['code_commit'],arm='TARGET_CCD',updates=444,projection_updates=444,new_projection_updates=444,
            new_body_updates=444,projection_cumulative_updates=2664,signal_feature_cumulative_updates=13764,open_signal_updates=444,signal_encoder_updates=444,
            native_updates=0,Sinc_optimizer_steps=0,encoder_frozen=False,Sinc_frozen=True,signal_parameters=334721,signal_feature_parameters=232033,
            metric_parameters=102688,trainable_parameters=334673,body_trainable_parameters=231985,trainable_projection_parameters=102688,
            input_path='cached_P1_IQ_only',source_only=False,target_U_used=True,target_U_reads=384,CCD_weight=.1,CCD_start_step=101,
            parent_signal_identity=base['model_identity'],parent_F0_identity=base['parent_F0_identity'],dura_identity=base['dura_identity'])
        mp.setattr(m,'_train',lambda:SimpleNamespace(STEPS=444,U_COUNT=384,validate_config=lambda a:a))
        costs=dict(real_parent_signal_forward_rows=12818,real_student_signal_forward_rows=340992,G_forward_rows=384,real_F_forward_rows=0,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2)
        mp.setattr(m,'_training_metadata',lambda *a:(identity,Uids,costs))
        events=[];loaded=m._iq_role;opened=m._public
        def role(config,name,header):
            assert Path(c['output_root'],'smoke.json').exists()
            if name=='V':assert Path(c['output_root'],'models/complete.json').exists()
            if name=='novel':assert Path(c['output_root'],'calibration/complete.json').exists()
            events.append(name);return loaded(config,name,header)
        def public(config,name):
            assert Path(c['output_root'],'source/source_DURA/novel_predictions.npz').exists()
            events.append(name);return opened(config,name)
        mp.setattr(m,'_iq_role',role);mp.setattr(m,'_public',public)
        mp.setattr(m,'_load_model',lambda *a:(events.append('load_CCD') or model,identity))
        mp.setattr(old,'_load_model',lambda *a:pytest.fail('Old SOURCE/PCD checkpoint must not be loaded again'))
        yield m,c,pm,b,events,identity


def test_exact_configuration_and_immutable_output(tmp_path,monkeypatch,subtests):
    m=api();assert m.FIELDS=={'run_id','code_commit','train_config','reference_config','output_root','batch_size'}
    monkeypatch.setattr(m,'validate_config',lambda c:c);monkeypatch.setattr(m,'_reference',lambda *a:pytest.fail('Existing output cannot read parents'))
    with pytest.raises(FileExistsError):m.run(dict(output_root=str(tmp_path)),'cpu')


def test_single_training_metadata_matches_exact_completed_control_and_U(tmp_path,monkeypatch,subtests):
    m=api();reader=m._training_metadata
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,identity):
        root=Path(c['train_config']['output_root']);root.mkdir()
        manifest=dict(count=384,capsule_id=m.U_CAPSULE_ID,split_id=m.U_SPLIT_ID)
        data=dict(schema='pamf_phase3_common_transport_train_v1',status='COMMON_TRANSPORT_TRAINING_COMPLETE',run_id=m.RUN_ID,code_commit=c['code_commit'],
            config=c['train_config'],arm='TARGET_CCD',updates=444,total_optimizer_updates=444,identity=identity,U_manifest=manifest,
            source_only=False,target_U_used=True,target_U_reads=384,parent_old_control_prediction=pm,
            control_training_run_id=m.previous.TRAIN_RUN_ID,control_training_code_commit=m.previous.TRAIN_CODE,
            source_sampling_matched=True,U_sampling_matched=True,original_models_unchanged=True,all_student_Sinc_unchanged=True,
            all_student_bodies_changed=True,all_student_projections_changed=True,all_student_buffers_unchanged=True,
            real_parent_signal_forward_rows=12818,real_student_signal_forward_rows=340992,G_forward_rows=384,real_F_forward_rows=0)
        m._write(root/'complete.json',data);ids=np.asarray(['U'+str(i).zfill(4) for i in range(384)]);np.savez(root/'U_features.npz',ids=ids)
        loaded,actual_ids,costs=reader(c,pm);assert loaded==identity and costs['real_student_signal_forward_rows']==340992;np.testing.assert_array_equal(ids,actual_ids)
        saved=(root/'complete.json').read_bytes()
        for defect in ('control_header','sampling','CCD_start','U_ids'):
            with subtests.test(defect=defect):
                (root/'complete.json').write_bytes(saved);np.savez(root/'U_features.npz',ids=ids)
                if defect=='U_ids':np.savez(root/'U_features.npz',ids=np.repeat(ids[:1],384))
                else:
                    a=deepcopy(data)
                    if defect=='control_header':a['parent_old_control_prediction']['code_commit']='b'*40
                    elif defect=='sampling':a['U_sampling_matched']=False
                    else:a['identity']['CCD_start_step']=102
                    (root/'complete.json').unlink();m._write(root/'complete.json',a)
                with pytest.raises(ValueError):reader(c,pm)


def test_actual_CCD_only_forward_source_fit_calibration_and_exact_six_controls(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,identity):
        cal=m.risk.calibrate_known
        def failed_source(*a):
            state,report=cal(*a);report['constraints_passed']=False;return state,report
        monkeypatch.setattr(m.risk,'calibrate_known',failed_source)
        done=m.run(c,'cpu');out=Path(c['output_root'])
        assert events==['load_CCD','L','OE','V','novel','clean_DURA','star_DURA']
        assert done['source_density_models_fitted']==1 and done['new_threshold_sets_fitted']==2 and done['control_readouts_reused_exactly']
        assert done['new_neural_updates']==0 and done['new_candidate_optimizer_updates']==444
        assert done['real_signal_forward_rows']==sum(m.COUNTS.values())+2*m.Q_COUNT
        assert done['real_F_forward_rows']==done['G_forward_rows']==2*m.Q_COUNT
        with np.load(out/'thresholds.npz') as z:tau=z['conditional']
        with np.load(Path(c['reference_config']['output_root'])/'thresholds.npz') as z:np.testing.assert_array_equal(tau[:6],z['conditional'])
        for role in ('known','novel'):
            a=m._source_rows(out,'source_DURA',role);old=m.previous._source_rows(c['reference_config']['output_root'],'source_DURA',role)
            m._exact_controls(a,old)
            assert a['signal_features'].shape==(len(a['ids']),4,160)
        for name in m.CONDITIONS[1:]:m._exact_controls(m._target_rows(out,name),m.previous._target_rows(c['reference_config']['output_root'],name))
        with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_parent_completion_identity_and_partial_query_fail_safely(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,identity):
        for delta in (dict(truth='forbidden'),dict(batch_size=True),dict(reference_config=dict(c['reference_config'],code_commit='b'*40))):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        root=Path(c['reference_config']['output_root']);p=root/'complete.json';saved=p.read_bytes();p.unlink()
        with pytest.raises(FileNotFoundError):m.run(c,'cpu')
        assert not events and not Path(c['output_root']).exists();p.write_bytes(saved)
        identity['Sinc_frozen']=False
        with pytest.raises(ValueError):m.run(c,'cpu')
        assert 'L' not in events and not Path(c['output_root']).exists();identity['Sinc_frozen']=True
        public=m._public
        monkeypatch.setattr(m,'_public',lambda config,name:(_ for _ in ()).throw(ValueError('partial star')) if name=='star_DURA' else public(config,name))
        with pytest.raises(ValueError,match='partial star'):m.run(c,'cpu')
        assert Path(c['output_root'],'targets/clean_DURA/predictions.npz').exists() and not Path(c['output_root'],'complete.json').exists()

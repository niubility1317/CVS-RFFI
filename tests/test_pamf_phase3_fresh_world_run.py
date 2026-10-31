"""Fresh receiver uses source-only controls and one independently adapted PCD."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.fresh_world_run')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_body_prototype_run import fixture as inherited
    with inherited(root,mp) as (old,oc,base,b,_,students,_,_,_):
        m=api();oc['code_commit']=m.CONTROL_CODE;pm=old.run(oc,'cpu')
        mp.setattr(m,'COUNTS',dict(old.COUNTS));mp.setattr(m,'Q_COUNT',18)
        source_i=pm['model_identities']['SOURCE'];folder=Path(oc['train_config']['output_root'])/'SOURCE';folder.mkdir()
        m._write(folder/'complete.json',dict(schema='pamf_phase3_body_prototype_train_v1',status='BODY_PROTOTYPE_STUDENT_FROZEN',identity=source_i,
            updates=444,source_only=True,target_U_used=False,target_U_reads=0,student_Sinc_unchanged=True,model_buffers_unchanged=True))
        tc=dict(run_id=m.RUN_ID,code_commit='e'*40,reference_config=deepcopy(oc),U_public_root=str(root/'fresh/discovery/public'),U_capsule_id='fresh-U-cap',
            U_split_id='fresh-U-split',U_expected_count=576,output_root=str(root/'fresh_train'),steps=444,lr=.0001)
        q={name:dict(public_root=str(root/'fresh'/role/'public'),capsule_id=name+'fresh',split_id=name+'split',expected_count=18,view_family=family)
            for name,role,family in (('clean_DURA','future_clean','clean_cross_receiver'),('star_DURA','future_star','practical_cross_receiver'))}
        c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],train_config=tc,source_control_config=deepcopy(oc),Q_conditions=q,output_root=str(root/'fresh_predict'),batch_size=128)
        model=deepcopy(b['model'])
        with torch.no_grad():model.projection.network[2].bias.add_(.03)
        model.eval().requires_grad_(False)
        identity=dict(source_i,run_id=m.RUN_ID,code_commit=c['code_commit'],arm='TARGET_PCD',source_only=False,target_U_used=True,target_U_reads=576,
            source_control_identity=source_i,source_control_source_only=True,CCD_updates=0)
        mp.setattr(m,'_train',lambda:SimpleNamespace(STEPS=444,U_COUNT=576,validate_config=lambda a:a))
        Uids=np.asarray(['fresh-U'+str(i).zfill(4) for i in range(576)])
        costs=dict(real_parent_signal_forward_rows=13010,real_student_signal_forward_rows=426240,G_forward_rows=576,real_F_forward_rows=0,synthetic_student_signal_rows=48)
        mp.setattr(m,'_training_metadata',lambda *a:(identity,Uids,costs))
        events=[];role_loader=m._iq_role
        def role(config,name,header):
            assert Path(c['output_root'],'smoke.json').exists()
            if name=='V':assert Path(c['output_root'],'models/complete.json').exists()
            if name=='novel':assert Path(c['output_root'],'calibration/complete.json').exists()
            events.append(name);return role_loader(config,name,header)
        mp.setattr(m,'_iq_role',role)
        mp.setattr(m,'_load_SOURCE',lambda *a:(events.append('SOURCE') or students['SOURCE'],source_i))
        mp.setattr(m,'_load_model',lambda *a:(events.append('fresh_PCD') or model,identity))
        mp.setattr(old,'_training_metadata',lambda *a:pytest.fail('Fresh evaluation must not open old target-U IDs'))
        ids=np.asarray(['freshQ'+str(i).zfill(3) for i in range(18)]);rng=np.random.default_rng(301)
        received=rng.normal(size=(18,2,256)).astype(np.float32)
        def public(config,name):
            assert Path(c['output_root'],'source/source_DURA/novel_predictions.npz').exists();events.append(name)
            spec=config['Q_conditions'][name];manifest=dict(schema='phase3_clean_min_v1' if name=='clean_DURA' else 'phase3_min_v1',data_status='VALIDATED_ONCE',capsule_id=spec['capsule_id'],split_id=spec['split_id'],count=18,
                registered_classes=old.parent.signal_train.native.CLASSES,processing_route='clean' if name=='clean_DURA' else 'residual')
            return manifest,ids.copy(),received.copy() if name=='clean_DURA' else received*.7+.03
        mp.setattr(m,'_public',public)
        yield m,c,pm,b,events,identity


def test_exact_roles_sourceonly_SOURCE_and_no_old_U(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,identity):
        for delta in (dict(truth='forbidden'),dict(batch_size=True),dict(Q_conditions=dict(c['Q_conditions'],star_DURA=dict(c['Q_conditions']['star_DURA'],expected_count=864)))):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        identity['target_U_reads']=384
        with pytest.raises(ValueError):m.run(c,'cpu')
        assert 'L' not in events and not Path(c['output_root']).exists()
        identity['target_U_reads']=576;events.clear()
        path=Path(c['source_control_config']['train_config']['output_root'])/'SOURCE/complete.json';saved=path.read_bytes();row=m._read(path)
        row['target_U_reads']=384;path.unlink();m._write(path,row)
        with pytest.raises(ValueError):m.run(c,'cpu')
        assert events==[] and not Path(c['output_root']).exists();path.write_bytes(saved)


def test_actual_three_future_models_dynamic_count_frozen_states_and_exact_source4(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,identity):
        oldcal=m.risk.calibrate_known
        def source_failure(*a):
            state,report=oldcal(*a);report['constraints_passed']=False;return state,report
        monkeypatch.setattr(m.risk,'calibrate_known',source_failure)
        done=m.run(c,'cpu');out=Path(c['output_root'])
        assert events==['SOURCE','fresh_PCD','L','OE','V','novel','clean_DURA','star_DURA']
        assert done['source_density_models_fitted']==1 and done['new_threshold_sets_fitted']==2 and done['source_control_readouts_reused_exactly']
        assert done['real_signal_forward_rows']==sum(m.COUNTS.values())+6*18 and done['G_forward_rows']==done['real_F_forward_rows']==36
        assert done['Q_counts']==dict(clean_DURA=18,star_DURA=18) and done['training_target_U_count']==576
        for role in ('known','novel'):
            a=m._source_rows(out,'source_DURA',role);old=m.previous._source_rows(c['source_control_config']['output_root'],'source_DURA',role);m._exact_source_controls(a,old)
        clean=m._target_rows(out,'clean_DURA');star=m._target_rows(out,'star_DURA')
        np.testing.assert_array_equal(clean['ids'],star['ids']);assert clean['signal_features'].shape==(18,3,160)
        np.testing.assert_array_equal(clean['closed'],clean['baseline_closed'])
        with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_partial_future_and_mismatched_clean_star_ids_preserve_failure(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,*_):
        public=m._public
        def bad(config,name):
            manifest,ids,iq=public(config,name)
            return manifest,ids[::-1] if name=='star_DURA' else ids,iq
        monkeypatch.setattr(m,'_public',bad)
        with pytest.raises(ValueError):m.run(c,'cpu')
        assert Path(c['output_root'],'targets/clean_DURA/predictions.npz').exists() and not Path(c['output_root'],'complete.json').exists()

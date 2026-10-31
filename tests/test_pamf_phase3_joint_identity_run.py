"""Joint heads route their own densities; original F0 remains a baseline."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn


def api():return importlib.import_module('experiments.pamf_phase3_open_world.joint_identity_run')


class ToyJoint(nn.Module):
    def __init__(self,signal,reader,offset):
        super().__init__();self.body=deepcopy(signal)
        self.body.signal.encoder.core.id_backbone.cls_head.weight=nn.Parameter(reader.core.id_backbone.cls_head.weight.detach().clone().roll(offset,0))
        self.eval().requires_grad_(False)
    def joint(self,x):
        a=self.body.features(x);head=self.body.signal.encoder.core.id_backbone.cls_head
        return dict(a,logits=30*torch.nn.functional.linear(torch.nn.functional.normalize(a['raw_fused160'],dim=1,eps=1e-4),torch.nn.functional.normalize(head.weight,dim=1,eps=1e-4)))


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_fresh_world_run import fixture as inherited
    with inherited(root,mp) as (old,oc,_,b,events,old_identity):
        m=api();oc['code_commit']=m.CONTROL_CODE;oc['train_config']['code_commit']=m.CONTROL_CODE
        old_identity['code_commit']=m.CONTROL_CODE
        pm=old.run(oc,'cpu');events.clear();mp.setattr(m,'COUNTS',dict(old.COUNTS));mp.setattr(m,'Q_COUNT',old.Q_COUNT)
        tc=dict(run_id=m.RUN_ID,code_commit='b'*40,reference_config=deepcopy(oc['train_config']),output_root=str(root/'joint_train'),steps=444,lr=.0001)
        c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],train_config=tc,reference_config=deepcopy(oc),Q_conditions=deepcopy(oc['Q_conditions']),output_root=str(root/'joint_predict'),batch_size=128)
        mp.setattr(m,'_train',lambda:SimpleNamespace(validate_config=lambda a:a))
        identities={};models={}
        for j,arm in enumerate(m.NEW_ARMS):
            models[arm]=ToyJoint(b['model'],b['reader'],j+1)
            identities[arm]=dict(run_id=m.RUN_ID,code_commit=c['code_commit'],arm=arm,updates=444,projection_updates=444,identity_head_updates=444,
                open_signal_updates=444,signal_encoder_updates=444,signal_parameters=335681,identity_head_parameters=960,trainable_parameters=335633,
                Sinc_frozen=True,Sinc_optimizer_steps=0,identity_head_jointly_trained=True,identity_head_replaced=False,native_head_structure_unchanged=True,
                source_only=arm=='SOURCE_JOINT',target_U_used=arm=='TARGET_JOINT',target_U_reads=576 if arm=='TARGET_JOINT' else 0,
                parent_signal_identity=pm['model_identities']['E0'],parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'])
        costs=dict(real_parent_signal_forward_rows=13010,real_student_signal_forward_rows=539904,G_forward_rows=576,real_F_forward_rows=0)
        Uids=np.asarray(['fresh-U'+str(i).zfill(4) for i in range(576)])
        mp.setattr(m,'_training_metadata',lambda *a:(identities,Uids,costs))
        mp.setattr(m,'_backend',lambda *a:b)
        mp.setattr(m,'_load_model',lambda c,arm,device:(models[arm],identities[arm]))
        role=old._iq_role
        def load_role(config,name,header):
            out=Path(c['output_root']);assert (out/'smoke.json').exists()
            if name=='V':assert (out/'models/complete.json').exists()
            if name=='novel':assert (out/'calibration/complete.json').exists()
            events.append('joint_'+name)
            return role(oc,name,pm)
        mp.setattr(m,'_iq_role',load_role)
        public=old._public
        def load_public(config,name):
            assert Path(c['output_root'],'source/TARGET_JOINT/novel_predictions.npz').exists()
            return public(oc,name)
        mp.setattr(m,'_public',load_public)
        yield m,c,pm,b,events,identities,models


def test_exact_config_and_identity_permissions(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,identities,_):
        for change in (dict(truth='private'),dict(batch_size=True),dict(Q_conditions=dict(c['Q_conditions'],star_DURA=dict(c['Q_conditions']['star_DURA'],capsule_id='wrong')))):
            with subtests.test(change=change),pytest.raises(ValueError):m.validate_config(dict(c,**change))
        for arm,change in (('SOURCE_JOINT',dict(target_U_reads=576)),('TARGET_JOINT',dict(identity_head_updates=0)),('TARGET_JOINT',dict(identity_head_parameters=0))):
            with subtests.test(arm=arm,change=change),pytest.raises(ValueError):m._identity(dict(identities[arm],**change),arm,c,pm)
        assert not any(e.startswith('joint_') for e in events)


def test_four_models_own_heads_two_fits_freeze_then_queries(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,_,models):
        calibrate=m.risk.calibrate_known;risks=[]
        def fail_source(*a):
            risks.append(tuple(np.asarray(v).copy() for v in a))
            state,report=calibrate(*a);report['constraints_passed']=False;return state,report
        monkeypatch.setattr(m.risk,'calibrate_known',fail_source)
        done=m.run(c,'cpu');out=Path(c['output_root'])
        assert [e for e in events if e.startswith('joint_')]==['joint_L','joint_OE','joint_V','joint_novel']
        assert done['source_density_models_fitted']==2 and done['new_threshold_sets_fitted']==4
        assert done['original_F0_unchanged'] and done['student_heads_jointly_trained'] and not done.get('classification_unchanged',False)
        assert done['real_signal_forward_rows']==2*sum(m.COUNTS.values())+4*m.Q_COUNT
        assert done['G_forward_rows']==done['real_F_forward_rows']==2*m.Q_COUNT
        tau=np.load(out/'thresholds.npz')['conditional'];assert tau.shape==(8,6)
        for index,arm in enumerate(m.NEW_ARMS):
            known=m._source_rows(out,arm,'known')
            for j in range(2):
                args=risks[2*index+j]
                np.testing.assert_array_equal(args[2],known['closed']);np.testing.assert_array_equal(args[3],known['baseline_closed'])
                np.testing.assert_array_equal(args[4],known['raw_scores'][:,j])
            assert np.any(known['closed']!=known['baseline_closed'])
        changed=0
        for condition in m.CONDITIONS:
            for arm in m.ARMS:
                a=m._source_rows(out,arm,'known') if condition=='source_DURA' else m._target_rows(out,condition,arm)
                index=m.ARMS.index(arm);np.testing.assert_array_equal(a['scores'],a['raw_scores']-tau[2*index:2*index+2,a['closed']].T)
                if arm in m.NEW_ARMS:changed+=int((a['closed']!=a['baseline_closed']).sum())
        assert changed>0
        for condition in m.CONDITIONS[1:]:
            a=m._target_rows(out,condition,'E0');b_=m._target_rows(out,condition,'TARGET_JOINT')
            np.testing.assert_array_equal(a['baseline_closed'],b_['baseline_closed'])
        with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_partial_query_never_completes(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,*_):
        public=m._public
        def broken(config,name):
            if name=='star_DURA':raise ValueError('synthetic partial')
            return public(config,name)
        monkeypatch.setattr(m,'_public',broken)
        with pytest.raises(ValueError,match='partial'):m.run(c,'cpu')
        assert Path(c['output_root'],'targets/clean_DURA/TARGET_JOINT/predictions.npz').exists()
        assert not Path(c['output_root'],'complete.json').exists()


def test_actual_train_checkpoint_loader_metadata_and_producer_forward(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_joint_identity_train import fixture as trained
    m=api()
    with trained(tmp_path,monkeypatch) as (t,tc,pm,b,events,Umanifest):
        done=t.train(tc,'cpu');monkeypatch.setattr(m,'UPDATES',2);monkeypatch.setattr(m,'U_COUNT',16)
        c=dict(code_commit=tc['code_commit'],train_config=tc,reference_config=dict(train_config=tc['reference_config']))
        original=dict(model_identities=dict(E0=pm['model_identity']),parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'])
        identities,ids,costs=m._training_metadata(c,original)
        assert len(ids)==16 and identities==done['identities']
        for arm in m.NEW_ARMS:
            model,identity=t.load_trained_model(tc,arm,'cpu');assert m._identity(identity,arm,c,original)==identities[arm]
            x=np.random.default_rng(12).normal(size=(3,2,256)).astype(np.float32)
            actual=m._joint(model,x,'cpu');expected=model.joint(torch.tensor(x.tolist(),dtype=torch.float32))
            np.testing.assert_array_equal(actual['default_logits'],np.asarray(expected['logits'].tolist(),np.float32))
            np.testing.assert_array_equal(actual['signal_features'],np.asarray(expected['z'].tolist(),np.float32))
        p=Path(tc['output_root'])/'complete.json';saved=p.read_bytes()
        for defect in ('missing_arm','SOURCE_U','head_updates'):
            with subtests.test(defect=defect):
                a=deepcopy(done)
                if defect=='missing_arm':a['rows'].pop('TARGET_JOINT')
                elif defect=='SOURCE_U':a['identities']['SOURCE_JOINT']['target_U_reads']=16
                else:a['identities']['TARGET_JOINT']['identity_head_updates']=0
                p.unlink();m._write(p,a)
                with pytest.raises(ValueError):m._training_metadata(c,original)
                p.write_bytes(saved)

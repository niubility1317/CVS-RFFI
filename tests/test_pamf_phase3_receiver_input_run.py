"""Receiver adapters alter candidate identity, never its original DURA baseline."""
import importlib
import numpy as np
import pytest
import torch
from contextlib import contextmanager
from pathlib import Path
from copy import deepcopy


def api():return importlib.import_module('experiments.pamf_phase3_open_world.receiver_input_run')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_receiver_input_train import fixture as training_fixture
    with training_fixture(root,mp) as (t,tc,pm,b,pcd,events):
        m=api();tc['code_commit']=m.TRAIN_CODE;trained=t.train(tc,'cpu');ref=deepcopy(pm['config'])
        mp.setattr(m,'COUNTS',dict(t.reg.fresh.COUNTS));mp.setattr(m,'Q_COUNT',t.reg.fresh.Q_COUNT)
        mp.setattr(m,'_backend',lambda *a:b);mp.setattr(m,'_load_model',lambda *a:(pcd,pm['model_identities']['TARGET_PCD']))
        # Remove only the inherited registry test's source/query instrumentation.
        # The actual fresh producer already wrote these legal cached rows.
        def source_rows(folder,condition,role):
            return t.reg.fresh._validate(m._npz(Path(folder)/'source'/condition/(role+'_predictions.npz')),role)
        def target_rows(folder,condition):
            return t.reg.fresh._validate(m._npz(Path(folder)/'targets'/condition/'predictions.npz'),'Q')
        mp.setattr(t.reg.fresh,'_source_rows',source_rows);mp.setattr(t.reg.fresh,'_target_rows',target_rows)
        c=dict(run_id=m.RUN_ID,code_commit='a'*40,train_config=tc,reference_config=ref,output_root=str(root/'receiver_r02/predict'),batch_size=128)
        flow=[];parent_rows=m._parent_rows;public=m._public;smoke=m._smoke
        def rows(config,key):
            assert Path(c['output_root'],'smoke.json').exists() and Path(c['output_root'],'models/complete.json').exists()
            assert Path(c['output_root'],'calibration/complete.json').exists();flow.append(key);return parent_rows(config,key)
        def query(config,condition):
            assert Path(c['output_root'],'source/TARGET_INPUT/novel_predictions.npz').exists();flow.append('IQ_'+condition);return public(config,condition)
        mp.setattr(m,'_parent_rows',rows);mp.setattr(m,'_public',query)
        mp.setattr(m,'_smoke',lambda *a:(flow.append('smoke'),smoke(*a))[1])
        mp.setattr(t.reg.fresh,'_iq_role',lambda *a:pytest.fail('Evaluation cannot open source IQ'))
        mp.setattr(t,'train',lambda *a:pytest.fail('Recovery cannot retrain adapters'))
        yield m,c,pm,b,pcd,flow,trained


def test_recovery_accepts_only_original_r01_training_run_and_code(tmp_path,monkeypatch,subtests):
    m=api();assert m.RUN_ID=='20261031-phase3-receiver-input-s2026092701-r02'
    assert m.TRAIN_RUN_ID=='20261031-phase3-receiver-input-s2026092701-r01' and m.TRAIN_CODE=='ac8bbd27277fcb1f16806c7b8b94e8c3e27e390b'
    monkeypatch.setattr(m.parent.fresh,'validate_config',lambda c:c);monkeypatch.setattr(m._train(),'validate_config',lambda c:c)
    r=dict(code_commit=m.parent.REFERENCE_CODE,output_root=str(tmp_path/'old_predict'),train_config=dict(output_root=str(tmp_path/'pcd'),U_public_root=str(tmp_path/'U')),
        source_control_config=dict(output_root=str(tmp_path/'source'),reference_config=dict(train_config=dict(IQ_root=str(tmp_path/'IQ')))),Q_conditions={})
    t=dict(run_id=m.TRAIN_RUN_ID,code_commit=m.TRAIN_CODE,reference_config=r['train_config'],reference_prediction_root=r['output_root'],output_root=str(tmp_path/'old_train'))
    c=dict(run_id=m.RUN_ID,code_commit='a'*40,train_config=t,reference_config=r,output_root=str(tmp_path/'new_predict'),batch_size=128)
    assert m.validate_config(c) is c
    for change in (dict(run_id=m.RUN_ID),dict(code_commit=c['code_commit'])):
        with subtests.test(change=next(iter(change))),pytest.raises(ValueError):m.validate_config(dict(c,train_config=dict(t,**change)))


def test_forward_two_adapters_share_one_DURA_and_route_own_F0_class():
    m=api();calls=[]
    class Reader(torch.nn.Module):
        def features(self,x):calls.append('F0');return x[:,0,0]
        def classify_features(self,x):
            out=torch.zeros((len(x),6));out[:,0]=x;out[:,1]=-x;return out
    class Signal(torch.nn.Module):
        def forward(self,x):
            calls.append('PCD');z=torch.zeros((len(x),160));z[:,0]=1;return z
    class Adapter(torch.nn.Module):
        def __init__(self,sign):super().__init__();self.sign=sign
        def forward(self,x):calls.append('adapter');return x*self.sign
    def dura(x):calls.append('G');return (x,)
    b=dict(reader=Reader(),dura=(torch.nn.Identity(),dura));adapters=dict(SOURCE_INPUT=Adapter(1),TARGET_INPUT=Adapter(-1))
    a=m._forward(b,Signal(),adapters,np.ones((3,2,256),np.float32),'cpu')
    assert calls==['G','adapter','F0','PCD','adapter','F0','PCD']
    np.testing.assert_array_equal(a['SOURCE_INPUT']['closed'],0);np.testing.assert_array_equal(a['TARGET_INPUT']['closed'],1)
    assert all(v['default_logits'].dtype==np.float32 and v['signal_features'].shape==(3,160) for v in a.values())
    with pytest.raises(ValueError):m._forward(b,Signal(),adapters,np.ones((3,256,2),np.float32),'cpu')


def test_public_rows_reject_private_labels_and_keep_baseline_distinct(subtests):
    m=api();n=3;z=np.zeros((n,160),np.float32);z[:,0]=1
    logits=np.zeros((n,6),np.float32);logits[:,1]=1;base=logits[:,::-1].copy()
    a=dict(ids=np.asarray(['q0','q1','q2']),default_logits=logits,closed=np.ones(n,np.int64),baseline_logits=base,
        baseline_closed=np.full(n,4,np.int64),signal_features=z,raw_scores=np.zeros((n,2)),scores=np.zeros((n,2)))
    assert m._validate(a,n,'Q') is a
    for change in (dict(y=np.zeros(n,np.int64)),dict(closed=np.zeros(n,np.int64)),dict(baseline_closed=np.ones(n,np.int64))):
        with subtests.test(change=next(iter(change))),pytest.raises(ValueError):m._validate(dict(a,**change),n,'Q')

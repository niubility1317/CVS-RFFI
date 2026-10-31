"""Source-only own-head models produce complete predictions before CPU truth.

Only the training/loader boundary is mocked. Its synthetic completion metadata
is nominal; the producer, density fits, calibration and prediction gate are real.
"""
from contextlib import contextmanager
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn


def api():return importlib.import_module('experiments.pamf_phase3_open_world.logit_energy_run')

def test_one_shared_DURA_and_original_head_then_two_own_joint_models():
    m=api();calls=[];seen=[]
    class Reader(nn.Module):
        def features(self,x):seen.append(x.clone());return x
        def classify_features(self,x):
            calls.append('F0');v=torch.zeros((len(x),6));v[:,0]=3.;return v
    class Joint(nn.Module):
        def __init__(self,label):super().__init__();self.label=label
        def forward(self,x):
            calls.append('joint'+str(self.label));seen.append(x.clone());z=torch.zeros((len(x),160));z[:,self.label]=1
            logits=torch.zeros((len(x),6));logits[:,self.label]=5;return z,logits
    def dura(x):calls.append('G');return (x*.7+.03,)
    models={a:Joint(j+1) for j,a in enumerate(m.NEW_ARMS)}
    original,rows=m._forward(dict(reader=Reader(),dura=(nn.Identity(),dura)),models,np.ones((3,2,256),np.float32),'cpu')
    assert calls==['G','F0','joint1','joint2']
    for x in seen:torch.testing.assert_close(x,torch.full((3,2,256),.73))
    np.testing.assert_array_equal(original.argmax(1),0)
    for j,a in enumerate(m.NEW_ARMS,1):
        np.testing.assert_array_equal(rows[a]['closed'],j);np.testing.assert_array_equal(rows[a]['baseline_closed'],0)
        assert rows[a]['signal_features'].shape==(3,160) and rows[a]['default_logits'].dtype==np.float32

class ToyEnergy(nn.Module):
    """Synthetic frozen loader output; padding only satisfies the public count."""
    def __init__(self,signal,reader,label):
        super().__init__();self.reader=deepcopy(reader)
        bias=torch.zeros(6);bias[label]=100.;self.register_buffer('head_bias',bias);self.shift=label
        self.padding=nn.Parameter(torch.zeros(232993-sum(p.numel() for p in self.parameters())))
        self.eval().requires_grad_(False)
    def forward(self,x):
        a=self.reader.features(x);z=torch.roll(a,self.shift,1)
        logits=self.reader.classify_features(a)+self.head_bias
        return z,logits

@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_fresh_world_run import fixture as inherited
    with inherited(root,mp) as (fresh,ref,_,b,events,pcd_identity):
        m=api();ref['code_commit']=m.CONTROL_CODE;ref['train_config']['code_commit']=m.CONTROL_CODE;pcd_identity['code_commit']=m.CONTROL_CODE
        pm=fresh.run(ref,'cpu');events.clear();mp.setattr(m,'COUNTS',dict(fresh.COUNTS));mp.setattr(m,'Q_COUNT',fresh.Q_COUNT)
        t=m._train();tc=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=deepcopy(ref),output_root=str(root/'energy_train'),steps=2220,lr=.0001)
        c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],train_config=tc,reference_config=deepcopy(ref),Q_conditions=deepcopy(ref['Q_conditions']),output_root=str(root/'energy_predict'),batch_size=128)
        models={arm:ToyEnergy(b['model'],b['reader'],j+4) for j,arm in enumerate(m.NEW_ARMS)}
        n=ref['source_control_config']['reference_config']['train_config']['native_config']
        handles={}
        for role in ('L','OE'):
            iq=m._iq_role(c,role,pm)
            try:handles[role]=deepcopy(iq['manifest'])
            finally:fresh.source_eval._close_role(iq)
        identities={arm:dict(run_id=m.RUN_ID,code_commit=c['code_commit'],arm=arm,checkpoint=str(Path(tc['output_root'])/'rows'/arm/'final.pt'),updates=2220,
            reader_parameters=232993,trainable_parameters=232945,default_head_parameters=960,identity_head_updates=2220,source_only=True,target_U_used=False,target_U_reads=0,
            parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],original_parent_identity=deepcopy(n),source_handles=deepcopy(handles),
            identity_head_jointly_trained=True,identity_head_replaced=False,initial_native_logits_equal_original_F0=True,test_only_nominal_completion=True) for arm in m.NEW_ARMS}
        rows={arm:dict(schema=t.SCHEMA,status=t.ROW_STATUS,identity=identities[arm],updates=2220,source_only=True,target_U_reads=0,test_only_nominal_completion=True) for arm in m.NEW_ARMS}
        done=dict(schema=t.SCHEMA,status=t.STATUS,run_id=m.RUN_ID,code_commit=c['code_commit'],config=tc,arms=list(m.NEW_ARMS),rows=rows,identities=identities,
            steps_per_arm=2220,updates_per_arm=2220,total_optimizer_updates=4440,source_sampling_matched=True,original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,
            target_U_reads=0,target_Q_reads=0,private_truth_reads=0,U_truth_reads=0,final_phase3_complete=False,test_only_nominal_completion=True,
            real_parent_signal_forward_rows=0,real_student_forward_rows=1136640,G_forward_rows=0,real_F_forward_rows=0)
        for arm,row in rows.items():
            folder=Path(tc['output_root'])/'rows'/arm;folder.mkdir(parents=True);m._write(folder/'complete.json',row)
        m._write(Path(tc['output_root'])/'complete.json',done)
        mp.setattr(m,'_backend',lambda *a:b);mp.setattr(t,'load_trained_model',lambda cfg,arm,device:(models[arm],identities[arm]))
        role=m._iq_role;public=m._public;smoke=m._smoke;calibrate=m.risk.calibrate_known;flow=[];calibrated=[]
        def iq(config,name,header):
            assert name in ('L','OE','V','novel') and Path(c['output_root'],'smoke.json').exists()
            if name=='V':assert Path(c['output_root'],'models/complete.json').exists()
            if name=='novel':assert Path(c['output_root'],'calibration/complete.json').exists()
            flow.append(name);return role(config,name,header)
        def query(config,condition):
            assert condition in m.CONDITIONS[1:] and Path(c['output_root'],'source/RXPAIR/novel_predictions.npz').exists()
            flow.append(condition);return public(config,condition)
        def cal(ids,y,closed,baseline,raw,rx,scenario):
            assert Path(c['output_root'],'models/complete.json').exists() and not Path(c['output_root'],'calibration/complete.json').exists()
            old=fresh._source_rows(ref['output_root'],'source_DURA','known');np.testing.assert_array_equal(baseline,old['closed'])
            assert set(closed.tolist()) in ({4},{5});calibrated.append((closed.copy(),raw.copy()));flow.append('calibrate')
            return calibrate(ids,y,closed,baseline,raw,rx,scenario)
        load=np.load;forbidden=Path(ref['train_config']['U_public_root']).resolve()
        def no_U(path,*args,**kwargs):
            if isinstance(path,(str,Path)):
                current=Path(path).resolve()
                assert not current.is_relative_to(forbidden) and current.name not in ('U_features.npz','U_restored.npz'), 'Evaluation cannot open U values or IDs'
            return load(path,*args,**kwargs)
        mp.setattr(m,'_iq_role',iq);mp.setattr(m,'_public',query);mp.setattr(m.risk,'calibrate_known',cal)
        mp.setattr(m,'_smoke',lambda *a:(flow.append('smoke'),smoke(*a))[1]);mp.setattr(np,'load',no_U)
        yield m,c,pm,flow,calibrated

def test_logit_energy_is_direct_own_logsumexp_and_ratio_stays_predicted(monkeypatch):
    m=api();z=np.eye(160,dtype=np.float32)[:3];c=np.asarray([0,1,2],np.int64)
    logits=np.asarray([[3,0,0,0,0,0],[0,4,0,0,0,0],[0,0,5,0,0,0]],np.float32)
    pooled=np.asarray([[-10.,7.],[-20.,8.],[-30.,9.]],np.float64)
    monkeypatch.setattr(m.fresh,'_raw',lambda *a:pooled.copy())
    for arm in m.ARMS[:2]:np.testing.assert_array_equal(m._raw({},z,c,logits,arm),pooled)
    for arm in m.NEW_ARMS:
        raw=m._raw({},z,c,logits,arm)
        np.testing.assert_array_equal(raw[:,0],np.logaddexp.reduce(logits.astype(np.float64),axis=1))
        np.testing.assert_array_equal(raw[:,1],pooled[:,1])
        shifted=m._raw({},z,c,logits+2,arm)
        np.testing.assert_allclose(shifted[:,0]-raw[:,0],2.,rtol=0,atol=1e-14)
        np.testing.assert_array_equal(shifted[:,1],raw[:,1])
    with pytest.raises(ValueError):m._raw({},z,c,logits[:,:5],'RXPAIR')
    assert m.PRIMARY_METHOD=='RXPAIR_energy' and m.PRIMARY_INDEX==6


def test_config_and_original_native_provenance_are_source_only(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,flow,calibrated):
        assert m.validate_config(c) is c
        for patch in (dict(U_truth='PRIVATE'),dict(batch_size=0),dict(Q_conditions={}),dict(code_commit='b'*40),dict(output_root=c['reference_config']['output_root'])):
            with subtests.test(config=next(iter(patch))),pytest.raises(ValueError):m.validate_config(dict(c,**patch))
        identity=m._training_metadata(c,pm)[0]['RXPAIR']
        assert m._identity(identity,'RXPAIR',c,pm) is identity
        for patch in (dict(target_U_reads=576),dict(source_only=False),dict(identity_head_updates=444),dict(reader_parameters=335681),dict(original_parent_identity={}),dict(source_handles={})):
            with subtests.test(identity=next(iter(patch))),pytest.raises(ValueError):m._identity(dict(identity,**patch),'RXPAIR',c,pm)

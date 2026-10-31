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


def api():return importlib.import_module('experiments.pamf_phase3_open_world.source_style_run')


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


def test_exact_public_config_roles_and_source_only_identity(tmp_path,monkeypatch,subtests):
    m=api();monkeypatch.setattr(m.fresh,'validate_config',lambda c:c);monkeypatch.setattr(m,'_train',lambda:SimpleNamespace(validate_config=lambda c:c))
    r=dict(code_commit=m.CONTROL_CODE,output_root=str(tmp_path/'parent'),train_config=dict(output_root=str(tmp_path/'PCD'),U_public_root=str(tmp_path/'U')),
        source_control_config=dict(reference_config=dict(train_config=dict(IQ_root=str(tmp_path/'IQ')))),Q_conditions={})
    tc=dict(run_id=m.RUN_ID,reference_config=r,code_commit='a'*40,output_root=str(tmp_path/'train'))
    c=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=r,train_config=tc,Q_conditions={},output_root=str(tmp_path/'predict'),batch_size=128)
    assert m.validate_config(c) is c
    for change in (dict(truth='private'),dict(U_truth='private'),dict(unknown_roles=[6,7]),dict(batch_size=True),dict(output_root=r['output_root'])):
        with subtests.test(config=next(iter(change))),pytest.raises(ValueError):m.validate_config(dict(c,**change))
    monkeypatch.setattr(m,'Q_COUNT',3);monkeypatch.setattr(m,'COUNTS',dict(V=3,novel=3))
    z=np.zeros((3,160),np.float32);z[:,0]=1;logits=np.eye(6,dtype=np.float32)[[1,1,1]]
    q=dict(ids=np.asarray(['q0','q1','q2']),default_logits=logits,closed=np.ones(3,np.int64),baseline_closed=np.zeros(3,np.int64),
        signal_features=z,raw_scores=np.zeros((3,2)),scores=np.zeros((3,2)))
    assert m._validate(q,'Q') is q
    for field,value in (('y',np.zeros(3,np.int64)),('true_TX',np.asarray(['private']*3)),('unknown_role',np.ones(3,bool))):
        with subtests.test(public_field=field),pytest.raises(ValueError):m._validate(dict(q,**{field:value}),'Q')
    with subtests.test(novel_labels=True),pytest.raises(ValueError):m._validate(dict(q,rx_i=np.ones(3,np.int64),day_i=np.ones(3,np.int64),y=np.zeros(3,np.int64)),'novel')
    pm=dict(parent_F0_identity={'F0':'source'},model_identities=dict(E0={'projection':'pure-r03'}),dura_identity={'G':'fixed'})
    i=dict(run_id=m.RUN_ID,code_commit=c['code_commit'],arm=m.NEW_ARMS[0],updates=444,signal_parameters=335681,trainable_parameters=335633,
        identity_head_updates=444,projection_updates=444,source_only=True,target_U_used=False,target_U_reads=0,
        parent_F0_identity=pm['parent_F0_identity'],parent_signal_identity=pm['model_identities']['E0'],dura_identity=pm['dura_identity'],
        identity_head_jointly_trained=True,identity_head_replaced=False,initial_native_logits_equal_original_F0=True,style_enabled=False)
    assert m._identity(i,m.NEW_ARMS[0],c,pm) is i
    for change in (dict(target_U_reads=576),dict(source_only=False),dict(parent_signal_identity={'projection':'adapted-PCD'})):
        with subtests.test(identity=next(iter(change))),pytest.raises(ValueError):m._identity(dict(i,**change),m.NEW_ARMS[0],c,pm)


class ToyJoint(nn.Module):
    """Synthetic frozen loader output; padding only satisfies the public count."""
    def __init__(self,signal,reader,label):
        super().__init__();self.body=deepcopy(signal);self.head=nn.Parameter(reader.core.id_backbone.cls_head.weight.detach().clone())
        bias=torch.zeros(6);bias[label]=100.;self.register_buffer('head_bias',bias);self.shift=label
        self.padding=nn.Parameter(torch.zeros(335681-sum(p.numel() for p in self.parameters())))
        self.eval().requires_grad_(False)
    def forward(self,x):
        a=self.body.features(x);z=torch.roll(a['z'],self.shift,1)
        logits=30*torch.nn.functional.linear(torch.nn.functional.normalize(a['raw_fused160'],dim=1,eps=1e-4),
            torch.nn.functional.normalize(self.head,dim=1,eps=1e-4))+self.head_bias
        return z,logits


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_fresh_world_run import fixture as inherited
    with inherited(root,mp) as (fresh,ref,_,b,events,pcd_identity):
        m=api();ref['code_commit']=m.CONTROL_CODE;ref['train_config']['code_commit']=m.CONTROL_CODE;pcd_identity['code_commit']=m.CONTROL_CODE
        pm=fresh.run(ref,'cpu');events.clear();mp.setattr(m,'COUNTS',dict(fresh.COUNTS));mp.setattr(m,'Q_COUNT',fresh.Q_COUNT)
        t=m._train();tc=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=deepcopy(ref),output_root=str(root/'style_train'),steps=444,lr=.0001)
        c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],train_config=tc,reference_config=deepcopy(ref),Q_conditions=deepcopy(ref['Q_conditions']),output_root=str(root/'style_predict'),batch_size=128)
        models={arm:ToyJoint(b['model'],b['reader'],j+4) for j,arm in enumerate(m.NEW_ARMS)}
        identities={arm:dict(run_id=m.RUN_ID,code_commit=c['code_commit'],arm=arm,updates=444,signal_parameters=335681,trainable_parameters=335633,
            identity_head_updates=444,projection_updates=444,source_only=True,target_U_used=False,target_U_reads=0,
            parent_F0_identity=pm['parent_F0_identity'],parent_signal_identity=pm['model_identities']['E0'],dura_identity=pm['dura_identity'],
            identity_head_jointly_trained=True,identity_head_replaced=False,initial_native_logits_equal_original_F0=True,style_enabled=arm=='STYLE_NATIVE',test_only_nominal_completion=True) for arm in m.NEW_ARMS}
        rows={arm:dict(schema=t.SCHEMA,status=t.ROW_STATUS,identity=identities[arm],updates=444,source_only=True,target_U_reads=0,test_only_nominal_completion=True) for arm in m.NEW_ARMS}
        done=dict(schema=t.SCHEMA,status=t.STATUS,run_id=m.RUN_ID,code_commit=c['code_commit'],config=tc,arms=list(m.NEW_ARMS),rows=rows,identities=identities,
            total_optimizer_updates=888,source_sampling_matched=True,original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,
            target_U_reads=0,target_Q_reads=0,private_truth_reads=0,U_truth_reads=0,final_phase3_complete=False,test_only_nominal_completion=True,
            real_parent_signal_forward_rows=0,real_student_signal_forward_rows=0,G_forward_rows=0,real_F_forward_rows=0)
        for arm,row in rows.items():
            folder=Path(tc['output_root'])/arm;folder.mkdir(parents=True);m._write(folder/'complete.json',row)
        m._write(Path(tc['output_root'])/'complete.json',done)
        mp.setattr(m,'_backend',lambda *a:b);mp.setattr(t,'load_trained_model',lambda cfg,arm,device:(models[arm],identities[arm]))
        role=m._iq_role;public=m._public;smoke=m._smoke;calibrate=m.risk.calibrate_known;flow=[];calibrated=[]
        def iq(config,name,header):
            assert name in ('L','OE','V','novel') and Path(c['output_root'],'smoke.json').exists()
            if name=='V':assert Path(c['output_root'],'models/complete.json').exists()
            if name=='novel':assert Path(c['output_root'],'calibration/complete.json').exists()
            flow.append(name);return role(config,name,header)
        def query(config,condition):
            assert condition in m.CONDITIONS[1:] and Path(c['output_root'],'source/STYLE_NATIVE/novel_predictions.npz').exists()
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


def test_actual_eight_readouts_gate_CPU_truth_then_seven_corruptions(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,flow,calibrated):
        done=m.run(c,'cpu');out=Path(c['output_root'])
        assert flow==['smoke','L','OE','V','calibrate','calibrate','calibrate','calibrate','novel','clean_DURA','star_DURA']
        assert len(calibrated)==4 and done['source_density_models_fitted']==2 and done['new_threshold_sets_fitted']==4
        assert done['threshold_values']==48 and done['threshold_bytes']==384 and done['U_ID_metadata_reads']==done['target_U_reads']==0
        assert done['known_count']==m.COUNTS['V'] and done['novel_count']==m.COUNTS['novel'] and not done['classification_unchanged']
        assert done['real_signal_forward_rows']==2*sum(m.COUNTS.values())+4*m.Q_COUNT
        score=importlib.import_module('experiments.pamf_phase3_open_world.source_style_score');monkeypatch.setattr(score,'COUNTS',dict(m.COUNTS));monkeypatch.setattr(score,'Q_COUNT',m.Q_COUNT)
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('CPU scoring cannot recalibrate'))
        _,_,rows=score._prediction(out)
        with np.load(out/'thresholds.npz',allow_pickle=False) as z:tau=z['conditional'].copy()
        for condition in m.CONDITIONS:
            for index,arm in enumerate(m.ARMS):
                a=rows[condition][arm]['known'] if condition=='source_DURA' else rows[condition][arm]
                np.testing.assert_array_equal(a['scores'],a['raw_scores']-tau[2*index:2*index+2,a['closed']].T)
                if arm in m.NEW_ARMS:
                    assert np.any(a['closed']!=a['baseline_closed'])
                    with np.load(out/'models'/(arm+'.npz'),allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
                    np.testing.assert_allclose(a['raw_scores'],m._raw(state,a['signal_features'],a['closed']),rtol=1e-12,atol=1e-9)
        nv=rows['source_DURA']['E0']['novel'];grid=[(tx,r,scene) for tx in score.common.UNKNOWN_TX for r in score.common.RECEIVERS for scene in score.common.SCENES]
        source=tmp_path/'source_truth.npz';paths=[source]
        np.savez(source,ids=nv['ids'],rx_i=nv['rx_i'],day_i=nv['day_i'],tx=np.asarray([v[0] for v in grid]),scenario=np.asarray([v[2] for v in grid]))
        y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64);days=np.asarray(['2021_03_15']*10+['2021_03_23']*8)
        for condition in m.CONDITIONS[1:]:
            a=rows[condition]['E0'];path=tmp_path/(condition+'_truth.npz');paths.append(path)
            np.savez(path,ids=a['ids'],y=y,tx=np.asarray(score.fresh.TX_BY_LABEL)[y],receiver=np.full(len(y),34,np.int64),day=days,
                scenario=np.asarray(['clean' if condition=='clean_DURA' else 'practical_low_urban']*len(y)))
        reads=[];st=score._source_truth;ft=score._future_truth
        monkeypatch.setattr(score,'_source_truth',lambda *a:(reads.append('source'),st(*a))[1]);monkeypatch.setattr(score,'_future_truth',lambda *a:(reads.append('future'),ft(*a))[1])
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('CPU score cannot load NN'));monkeypatch.setattr(m._train(),'load_trained_model',lambda *a:pytest.fail('CPU score cannot load checkpoint'))
        monkeypatch.setattr(m.density,'fit_known',lambda *a:pytest.fail('CPU score cannot fit known density'));monkeypatch.setattr(m.density,'fit_background',lambda *a:pytest.fail('CPU score cannot fit background'))
        result=score.score(out,*paths,tmp_path/'style_score')
        assert reads==['source','future','future'] and result['truth_last'] and result['all_three_predictions_before_truth']
        assert set(result['methods'])==set(m.METHODS) and result['methods'][m.PRIMARY_METHOD]['paths']['star_DURA']['by_day']['2021_03_23']['A_K'] is None
        main=out/'targets/star_DURA/STYLE_NATIVE/predictions.npz';old=out/'targets/star_DURA/OLD_PCD/predictions.npz';thresholds=out/'thresholds.npz'
        saved={p:p.read_bytes() for p in (main,old,thresholds)}
        for defect in ('missing_Q','private_y','old_control','new_raw','original_baseline','own_closed','threshold'):
            with subtests.test(defect=defect):
                reads.clear()
                if defect=='missing_Q':main.unlink()
                else:
                    path=old if defect=='old_control' else thresholds if defect=='threshold' else main
                    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='private_y':a['y']=np.zeros(m.Q_COUNT,np.int64)
                    elif defect=='old_control':a['scores'][0,1]+=.1
                    elif defect=='new_raw':a['raw_scores'][0,1]+=.1
                    elif defect=='original_baseline':a['baseline_closed'][0]=(a['baseline_closed'][0]+1)%6
                    elif defect=='own_closed':a['closed'][0]=(a['closed'][0]+1)%6
                    else:a['conditional'][7,0]+=.1
                    np.savez(path,**a)
                with pytest.raises((ValueError,FileNotFoundError)):score.score(out,*paths,tmp_path/defect)
                assert reads==[]
                for p,raw in saved.items():p.write_bytes(raw)
        with pytest.raises(FileExistsError):m.run(c,'cpu')

"""Native students own their identity; completed artifacts precede CPU truth.

The integrated fixture mocks only the already-tested training/loader boundary;
its completion counters are nominal synthetic metadata, not real training.
"""
from contextlib import contextmanager
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.native_identity_run')


def test_shared_DURA_original_F0_PCD_and_two_own_native_default_heads(monkeypatch):
    m=api();calls=[];p1_seen=[]
    class Reader(torch.nn.Module):
        def __init__(self,name,label):super().__init__();self.name=name;self.label=label
        def features(self,x):p1_seen.append(x.clone());return x
        def classify_features(self,x):
            calls.append(self.name);v=torch.zeros((len(x),6),dtype=torch.float32);v[:,self.label]=3.;return v
    class PCD(torch.nn.Module):
        def forward(self,x):
            calls.append('PCD');p1_seen.append(x.clone());z=torch.zeros((len(x),160));z[:,0]=1.;return z
    class Native(torch.nn.Module):
        def __init__(self,label):super().__init__();self.reader=Reader('native'+str(label),label)
    def native_logits(model,x):return model.reader.classify_features(model.reader.features(x))
    def dura(x):calls.append('G');return (x*.7+.03,)
    monkeypatch.setattr(m,'_train',lambda:SimpleNamespace(native_logits=native_logits))
    values=np.ones((3,2,256),np.float32);b=dict(reader=Reader('F0',0),dura=(torch.nn.Identity(),dura))
    original,z,own=m._forward(b,PCD(),dict(SOURCE_ID=Native(1),TARGET_ID=Native(2)),values,'cpu')
    assert calls==['G','F0','PCD','native1','native2']
    for x in p1_seen:torch.testing.assert_close(x,torch.full((3,2,256),.73))
    np.testing.assert_array_equal(original.argmax(1),0);np.testing.assert_array_equal(own['SOURCE_ID'].argmax(1),1);np.testing.assert_array_equal(own['TARGET_ID'].argmax(1),2)
    assert z.shape==(3,160) and z.dtype==np.float32 and all(x.dtype==np.float32 for x in own.values())


def test_exact_public_config_and_prediction_roles_reject_private_fields(tmp_path,monkeypatch,subtests):
    m=api();monkeypatch.setattr(m.base.parent.fresh,'validate_config',lambda c:c);monkeypatch.setattr(m,'_train',lambda:SimpleNamespace(validate_config=lambda c:c))
    r=dict(output_root=str(tmp_path/'parent'),train_config=dict(output_root=str(tmp_path/'PCD'),U_public_root=str(tmp_path/'U')),
        source_control_config=dict(reference_config=dict(train_config=dict(IQ_root=str(tmp_path/'IQ')))),Q_conditions={})
    tc=dict(reference_config=r,code_commit='a'*40,output_root=str(tmp_path/'train'))
    c=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=r,train_config=tc,output_root=str(tmp_path/'predict'),batch_size=128)
    assert m.validate_config(c) is c
    for change in (dict(truth='private'),dict(U_truth='private'),dict(unknown_roster=[6,7]),dict(batch_size=True),dict(output_root=r['output_root'])):
        with subtests.test(config=next(iter(change))),pytest.raises(ValueError):m.validate_config(dict(c,**change))
    n=3;logits=np.eye(6,dtype=np.float32)[[1,1,1]];baseline=np.eye(6,dtype=np.float32)[[0,0,0]];z=np.zeros((n,160),np.float32);z[:,0]=1
    q=dict(ids=np.asarray(['q0','q1','q2']),default_logits=logits,closed=np.ones(n,np.int64),baseline_logits=baseline,baseline_closed=np.zeros(n,np.int64),signal_features=z,
        raw_scores=np.zeros((n,2)),scores=np.zeros((n,2)))
    assert m._validate(q,n,'Q') is q
    for field,value in (('y',np.zeros(n,np.int64)),('true_TX',np.asarray(['secret']*n)),('unknown_role',np.ones(n,bool))):
        with subtests.test(public_field=field),pytest.raises(ValueError):m._validate(dict(q,**{field:value}),n,'Q')
    with subtests.test(novel_labels=True),pytest.raises(ValueError):m._validate(dict(q,rx_i=np.ones(n,np.int64),day_i=np.ones(n,np.int64),y=np.zeros(n,np.int64)),n,'novel')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_fresh_world_run import fixture as original
    with original(root,mp) as (fresh,ref,_,b,events,pcd_identity):
        m=api();ref['code_commit']=m.base.parent.REFERENCE_CODE;ref['train_config']['code_commit']=ref['code_commit'];pcd_identity['code_commit']=ref['code_commit']
        pm=fresh.run(ref,'cpu');pcd,_=fresh._load_model(ref,'cpu');train=m._train();events.clear()
        mp.setattr(m,'COUNTS',dict(V=fresh.COUNTS['V'],novel=fresh.COUNTS['novel']));mp.setattr(m,'Q_COUNT',fresh.Q_COUNT)
        tc=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=deepcopy(ref),output_root=str(root/'native/train'),steps=444,lr=.0001)
        c=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=deepcopy(ref),train_config=tc,output_root=str(root/'native/predict'),batch_size=128)
        class NativeReader(torch.nn.Module):
            def __init__(self,reader,label):
                super().__init__();self.original=deepcopy(reader);bias=torch.zeros(6,dtype=torch.float32);bias[label]=100.;self.register_buffer('head_bias',bias)
            def features(self,x):return self.original.features(x)
            def classify_features(self,x):return self.original.classify_features(x)+self.head_bias
        class Native(torch.nn.Module):
            def __init__(self,reader,label):super().__init__();self.reader=NativeReader(reader,label)
        models={arm:Native(b['reader'],i+4).eval().requires_grad_(False) for i,arm in enumerate(m.NEW_ARMS)}
        identities={arm:dict(run_id=m.RUN_ID,code_commit=c['code_commit'],arm=arm,checkpoint=str(Path(tc['output_root'])/arm/'final.pt'),updates=444,
            parent_F0_identity=pm['parent_F0_identity'],parent_PCD_identity=pcd_identity,dura_identity=pm['dura_identity'],identity_parameters=232993,
            trainable_parameters=232945,identity_head_updates=444,source_only=arm=='SOURCE_ID',target_U_used=arm=='TARGET_ID',target_U_reads=576 if arm=='TARGET_ID' else 0,
            system_PCD_has_target_U_ancestor=True,test_only_nominal_completion=True) for arm in m.NEW_ARMS}
        rows={arm:dict(schema=train.SCHEMA,status=train.ROW_STATUS,identity=identities[arm],updates=444) for arm in m.NEW_ARMS}
        done=dict(schema=train.SCHEMA,status=train.STATUS,run_id=m.RUN_ID,code_commit=c['code_commit'],config=tc,rows=rows,identities=identities,total_optimizer_updates=888,
            source_sampling_matched=True,SOURCE_completed_before_U_open=True,target_U_reads=576,target_Q_reads=0,private_truth_reads=0,U_truth_reads=0,final_phase3_complete=False,
            original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,test_only_nominal_completion=True)
        for arm,row in rows.items():
            folder=Path(tc['output_root'])/arm;folder.mkdir(parents=True);m._write(folder/'complete.json',row)
        m._write(Path(tc['output_root'])/'complete.json',done)
        mp.setattr(train,'load_trained_model',lambda cfg,arm,device:(models[arm],identities[arm]))
        mp.setattr(train,'native_logits',lambda model,x:model.reader.classify_features(model.reader.features(x)))
        mp.setattr(m.base,'_backend',lambda *a:b);mp.setattr(m.base,'_load_model',lambda *a:(pcd,pcd_identity))
        role=fresh._iq_role;parent_rows=m._parent_rows;public=m._public;smoke=m._smoke;calibrate=m.risk.calibrate_known;flow=[];calibrated=[]
        def iq(config,name,header):
            assert name in ('V','novel') and Path(c['output_root'],'smoke.json').exists() and Path(c['output_root'],'models/complete.json').exists()
            if name=='novel':assert Path(c['output_root'],'calibration/complete.json').exists()
            flow.append('IQ_'+name);return role(config,name,header)
        def parent_row(config,key):
            assert Path(c['output_root'],'models/complete.json').exists()
            if key!='source_V':assert Path(c['output_root'],'calibration/complete.json').exists()
            flow.append(key);return parent_rows(config,key)
        def query(config,condition):
            assert Path(c['output_root'],'source/TARGET_ID/novel_predictions.npz').exists();flow.append('public_'+condition);return public(config,condition)
        def cal(ids,y,closed,baseline,raw,rx,scene):
            assert not Path(c['output_root'],'calibration/complete.json').exists()
            expected=m._old(parent_rows(c,'source_V'));np.testing.assert_array_equal(baseline,expected['baseline_closed'])
            assert set(closed.tolist()) in ({4},{5});calibrated.append(1);flow.append('calibrate')
            return calibrate(ids,y,closed,baseline,raw,rx,scene)
        mp.setattr(fresh,'_iq_role',iq);mp.setattr(m,'_parent_rows',parent_row);mp.setattr(m,'_public',query);mp.setattr(m.risk,'calibrate_known',cal)
        mp.setattr(m,'_smoke',lambda *args:(flow.append('smoke'),smoke(*args))[1])
        yield m,c,pm,flow,calibrated


def test_actual_producer_gate_then_CPU_three_truths_own_routes_and_corruption(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,flow,calibrated):
        done=m.run(c,'cpu');out=Path(c['output_root'])
        assert len(calibrated)==4 and done['new_density_fits']==0 and done['new_threshold_sets_fitted']==4
        assert done['threshold_values']==36 and done['known_count']==m.COUNTS['V'] and done['novel_count']==m.COUNTS['novel']
        assert flow==['smoke','source_V','IQ_V','calibrate','calibrate','calibrate','calibrate','source_NV','IQ_novel','Q_clean_DURA','public_clean_DURA','Q_star_DURA','public_star_DURA']
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('CPU scorer cannot recalibrate'))
        _,state,tau,cal,rows=m._prediction_gate(out)
        for key,pair in rows.items():
            old=pair['OLD_PCD']
            for j,arm in enumerate(m.ARMS):
                a=pair[arm];np.testing.assert_array_equal(a['baseline_logits'],old['default_logits']);np.testing.assert_array_equal(a['baseline_closed'],old['closed'])
                if j:assert np.any(a['closed']!=old['closed'])
                raw=m.base.parent.fresh._raw(state,a['signal_features'],a['closed'])
                np.testing.assert_allclose(a['raw_scores'],raw,atol=2e-6,rtol=0);np.testing.assert_allclose(a['scores'],raw-tau[2*j:2*j+2,a['closed']].T,atol=2e-6,rtol=0)
        score=importlib.import_module('experiments.pamf_phase3_open_world.native_identity_score')
        paths=[];nv=rows['source_NV']['OLD_PCD'];grid=[(tx,r,scene) for tx in score.common.UNKNOWN_TX for r in score.common.RECEIVERS for scene in score.common.SCENES]
        source=tmp_path/'source_truth.npz';paths.append(source);np.savez(source,ids=nv['ids'],rx_i=nv['rx_i'],day_i=nv['day_i'],tx=np.asarray([v[0] for v in grid]),scenario=np.asarray([v[2] for v in grid]))
        y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64);days=np.asarray(['2021_03_15']*10+['2021_03_23']*8)
        for condition in m.CONDITIONS[1:]:
            a=rows['Q_'+condition]['OLD_PCD'];path=tmp_path/(condition+'_truth.npz');paths.append(path)
            np.savez(path,ids=a['ids'],y=y,tx=np.asarray(score.fresh.TX_BY_LABEL)[y],receiver=np.full(len(y),34,np.int64),day=days,
                scenario=np.asarray(['clean' if condition=='clean_DURA' else 'practical_low_urban']*len(y)))
        reads=[];st=score._source_truth;ft=score._future_truth
        monkeypatch.setattr(score,'_source_truth',lambda *a:(reads.append('source'),st(*a))[1]);monkeypatch.setattr(score,'_future_truth',lambda *a:(reads.append('future'),ft(*a))[1])
        monkeypatch.setattr(m.base,'_backend',lambda *a:pytest.fail('No score NN load'));monkeypatch.setattr(m._train(),'load_trained_model',lambda *a:pytest.fail('No score checkpoint load'))
        result=score.score(out,*paths,tmp_path/'native/score')
        assert reads==['source','future','future'] and result['truth_last'] and result['all_three_predictions_before_truth']
        assert set(result['methods'])==set(m.METHODS) and result['methods'][m.PRIMARY_METHOD]['paths']['star_DURA']['by_day']['2021_03_23']['A_K'] is None
        main=out/'targets/star_DURA/TARGET_ID/predictions.npz';oldpath=out/'targets/star_DURA/OLD_PCD/predictions.npz';taupath=out/'thresholds.npz'
        saved={p:p.read_bytes() for p in (main,oldpath,taupath)}
        for defect in ('missing_Q','private_y','old_control','new_raw','original_baseline','own_closed','threshold'):
            with subtests.test(defect=defect):
                reads.clear()
                if defect=='missing_Q':main.unlink()
                else:
                    path=oldpath if defect=='old_control' else taupath if defect=='threshold' else main;a=m._npz(path)
                    if defect=='private_y':a['y']=np.zeros(m.Q_COUNT,np.int64)
                    elif defect=='old_control':a['scores'][0,1]+=.1
                    elif defect=='new_raw':a['raw_scores'][0,1]+=.1
                    elif defect=='original_baseline':a['baseline_closed'][0]=(a['baseline_closed'][0]+1)%6
                    elif defect=='own_closed':a['closed'][0]=(a['closed'][0]+1)%6
                    else:a['conditional'][5,0]+=.1
                    np.savez(path,**a)
                with pytest.raises((ValueError,FileNotFoundError)):score.score(out,*paths,tmp_path/defect)
                assert reads==[]
                for path,raw in saved.items():path.write_bytes(raw)
        with pytest.raises(FileExistsError):m.run(c,'cpu')

"""Paired student evaluation and all-artifact-before-truth scoring handoff."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.momentum_prototype_run')


def test_one_same_record_restoration_and_original_identity_feed_both_students():
    m=api();seen=[]
    class Reader(torch.nn.Module):
        def features(self,x):seen.append('F0');return x.mean((1,2))[:,None].repeat(1,160)
        def classify_features(self,z):return z[:,:6]
    class Student(torch.nn.Module):
        def __init__(self,name):super().__init__();self.name=name
        def forward(self,x):seen.append(self.name);return torch.nn.functional.normalize(torch.arange(1,161,dtype=x.dtype)[None].repeat(len(x),1),dim=1)
    def restore(x):seen.append('G');return(x*.7,)
    values=np.ones((3,2,256),np.float32);students={n:Student(n)for n in m.ARMS}
    a=m._forward(dict(reader=Reader(),dura=(None,restore)),students,values,'cpu')
    assert seen==['G','F0',*m.ARMS]and a['signal_features'].shape==(3,2,160)
    np.testing.assert_array_equal(a['closed'],a['baseline_closed'])


def test_private_role_override_rejected_before_model_or_source_access():
    m=api()
    with pytest.raises(ValueError):m.validate_config(dict(run_id=m.RUN_ID,code_commit='a'*40,train_config={},source_control_config={},Q_conditions={},output_root='new',batch_size=128,U_truth='forbidden'))


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_fresh_world_run import fixture as inherited
    with inherited(root,mp)as(old,ref,_,b,events,old_identity):
        m=api();ref['code_commit']=m.control.REFERENCE_CODE;ref['train_config']['code_commit']=m.control.REFERENCE_CODE;old_identity['code_commit']=m.control.REFERENCE_CODE
        pm=old.run(ref,'cpu');mp.setattr(m,'COUNTS',dict(old.COUNTS));mp.setattr(m,'Q_COUNT',old.Q_COUNT)
        tc=dict(run_id=m.RUN_ID,code_commit='c'*40,reference_config=deepcopy(ref),output_root=str(root/'momentum_train'),steps=444,lr=.0001)
        c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],train_config=tc,source_control_config=deepcopy(ref),Q_conditions=deepcopy(ref['Q_conditions']),output_root=str(root/'momentum_predict'),batch_size=128)
        train=SimpleNamespace(STATUS='MOMENTUM_PROTOTYPE_TRAIN_COMPLETE',validate_config=lambda c:c)
        models={};identities={}
        for j,arm in enumerate(m.ARMS):
            model=deepcopy(b['model'])
            with torch.no_grad():model.projection.network[2].bias.add_(.015+.01*j)
            models[arm]=model.eval().requires_grad_(False)
            identities[arm]=dict(run_id=m.RUN_ID,code_commit=c['code_commit'],arm=arm,updates=444,signal_parameters=334721,Sinc_frozen=True,source_only=False,target_U_used=True,target_U_reads=576,
                parent_signal_identity=pm['model_identities']['E0'],parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'])
        train.load_trained_model=lambda config,arm,device:(models[arm],identities[arm]);mp.setattr(m,'_train',lambda:train);mp.setattr(m,'_backend',lambda *a:b)
        Path(tc['output_root']).mkdir();m._write(Path(tc['output_root'])/'complete.json',dict(status=train.STATUS,run_id=m.RUN_ID,code_commit=c['code_commit'],config=tc,arms=list(m.ARMS),updates_per_arm=444,total_optimizer_updates=888,
            target_U_reads=576,source_control_source_only=True,source_control_frozen_before_training=True,source_sampling_matched=True,original_models_unchanged=True,all_student_buffers_unchanged=True,
            all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_projections_changed=True,target_Q_reads=0,private_truth_reads=0,U_truth_reads=0,final_phase3_complete=False,
            source_control_identity=pm['model_identities']['SOURCE'],identities=identities,test_only_simulated_training=True))
        folder=Path(ref['train_config']['output_root']);folder.mkdir(exist_ok=True)
        np.savez(folder/'U_features.npz',ids=np.asarray(['toy-U'+str(j)for j in range(576)]))
        role=m._iq_role;public=m._public;calls=[]
        def iq_role(config,name,header):
            out=Path(c['output_root']);assert(out/'smoke.json').exists()
            if name=='V':assert(out/'models/complete.json').exists()
            if name=='novel':assert(out/'calibration/complete.json').exists()
            calls.append(name);return role(config,name,header)
        def query(config,name):
            assert Path(c['output_root'],'source/source_DURA/novel_predictions.npz').exists();calls.append(name);return public(config,name)
        mp.setattr(m,'_iq_role',iq_role);mp.setattr(m,'_public',query)
        yield m,c,pm,calls


def test_actual_two_students_eight_readouts_then_only_three_cpu_truths(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_class_density_score import truth_files
    s=importlib.import_module('experiments.pamf_phase3_open_world.momentum_prototype_score')
    with fixture(tmp_path,monkeypatch)as(m,c,pm,calls):
        done=m.run(c,'cpu');root=Path(c['output_root'])
        assert calls==['L','OE','V','novel','clean_DURA','star_DURA']
        assert done['source_density_models_fitted']==2 and done['new_threshold_sets_fitted']==4
        assert done['real_signal_forward_rows']==2*sum(m.COUNTS.values())+4*m.Q_COUNT
        assert done['G_forward_rows']==done['real_F_forward_rows']==2*m.Q_COUNT
        assert done['threshold_bytes']==384 and done['source_control_readouts_reused_exactly']
        with pytest.raises(FileExistsError):m.run(c,'cpu')
        proxy=type('Rows',(),dict(_rows=staticmethod(lambda root,role:m._source_rows(root,'source_DURA',role)),_target_rows=staticmethod(m._target_rows)))
        sp=type('Scoring',(),dict(common=s.common,CONDITIONS=()))
        paths=[truth_files(tmp_path,sp,proxy,root)[0]];y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64)
        for name in m.CONDITIONS[1:]:
            a=m._target_rows(root,name);path=tmp_path/(name+'_truth.npz');paths.append(path)
            np.savez(path,ids=a['ids'],y=y,tx=np.asarray(s.fresh.TX_BY_LABEL)[y],receiver=np.full(len(y),34,np.int64),day=np.asarray(['2021_03_15']*10+['2021_03_23']*8),scenario=np.asarray(['clean'if name=='clean_DURA'else'practical_low_urban']*len(y)))
        reads=[];source=s.fresh._source_truth;target=s.fresh._future_truth
        monkeypatch.setattr(s.fresh,'_source_truth',lambda *a:(reads.append('source'),source(*a))[1])
        monkeypatch.setattr(s.fresh,'_future_truth',lambda path,*a:(reads.append(Path(path).name),target(path,*a))[1])
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('Scorer must not load networks'))
        monkeypatch.setattr(m._train(),'load_trained_model',lambda *a:pytest.fail('Scorer must not load checkpoints'))
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('Scorer must not recalibrate'))
        monkeypatch.setattr(m.density,'fit_known',lambda *a:pytest.fail('Scorer must not fit density'))
        result=s.score(root,*paths,tmp_path/'score')
        assert reads==['source','clean_DURA_truth.npz','star_DURA_truth.npz']and result['truth_last']
        assert type(result['adaptation_success'])is bool and not result['eligible_primary']and result['full_goal_status']=='ACTIVE'
        late=result['methods'][s.PRIMARY]['paths']['star_DURA']['by_day']['2021_03_23']
        assert late['A_K']is None and late['R_U']is not None
        path=root/'targets/star_DURA/predictions.npz';saved=path.read_bytes()
        for defect in('missing','private','old_control','new_raw','baseline'):
            with subtests.test(defect=defect):
                path.write_bytes(saved);reads.clear()
                if defect=='missing':path.unlink()
                else:
                    a=m._npz(path)
                    if defect=='private':a['y']=y
                    elif defect=='old_control':a['scores'][0,0]=np.nextafter(a['scores'][0,0],np.inf)
                    elif defect=='new_raw':a['raw_scores'][0,7]+=.1
                    else:a['baseline_closed'][0]=(a['closed'][0]+1)%6
                    np.savez(path,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(root,*paths,tmp_path/('bad_'+defect))
                assert reads==[]

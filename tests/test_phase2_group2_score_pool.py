import json
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pytest

from experiments.pamf_phase2_practical.group2_c_backbone_score_pool import score_pool
from experiments.pamf_phase2_practical.group2_c_backbone_schedule import _score


def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj),encoding='utf-8')


def fixture(tmp_path):
    data=tmp_path/'data';capsule='capsules/P01/LS01/1-1';cap=data/capsule
    split='a'*24
    put(cap/'manifest.json',{'capsule_id':'cap','physical_pool':'P01','episode_id':'LS01','receiver':'1-1'})
    put(cap/'splits'/(split+'.json'),{'split_id':split,'new_count':2,'k':1,
        'registered_classes':list(range(8)),'support_indices':list(range(0,16,2)),'query_count':8})
    (cap/'stage_rows').mkdir(parents=True,exist_ok=True)
    np.save(cap/'stage_rows/N02.npy',np.arange(16,dtype=np.int64))
    truth=data/'score_only/P01/LS01/1-1.npz';truth.parent.mkdir(parents=True,exist_ok=True)
    np.savez(truth,labels=np.arange(16)//2,days=np.arange(16)%2,scene_indices=np.arange(16)%3)
    jobs=[]
    for i in range(3):
        pred=tmp_path/f'prediction-{i}';pred.mkdir()
        np.savez(pred/'p.npz',query_indices=np.arange(1,16,2),predicted_indices=np.arange(8),
                 old_restricted_indices=np.array([0,1,2,3,4,5,0,0]))
        row={'split_id':split,'stage':'C','input_path':'P0','arm':f'B{i}',
             'prediction':'p.npz','k':1,'new_count':2,'registered_classes':list(range(8))}
        put(pred/'predictions.json',{'capsule':capsule,'capsule_id':'cap','model_seed':7,'rows':[row]})
        put(pred/'complete.json',{'status':'PREDICTIONS_COMPLETE','truth_read':False,'rows':1,'capsule_id':'cap'})
        jobs.append({'key':f'job{i}','capsule':capsule,'output':str(pred)})
    return data,jobs


def test_persistent_pool_matches_independent_one_process_scoring(tmp_path):
    data,jobs=fixture(tmp_path)
    old={'run_root':str(tmp_path/'old'),'data_root':str(data),'score_processes':2}
    new={'run_root':str(tmp_path/'new'),'data_root':str(data),'score_processes':2}
    for rt in (old,new):put(Path(rt['run_root'])/'predictions_complete.json',{'status':'ALL_PREDICTIONS_FIXED'})
    for job in jobs:_score(old,job)
    score_pool(new,jobs)
    for job in jobs:
        a=json.loads((Path(old['run_root'])/'scores'/job['key']/'metrics.json').read_text(encoding='utf-8'))
        b=json.loads((Path(new['run_root'])/'scores'/job['key']/'metrics.json').read_text(encoding='utf-8'))
        assert a==b and b['truth_last'] and not b['target_feedback']
    state=json.loads((Path(new['run_root'])/'score_state.json').read_text(encoding='utf-8'))
    assert state['status']=='COMPLETE' and state['completed']==len(jobs)


def test_pool_will_not_score_before_all_predictions_are_fixed(tmp_path):
    data,jobs=fixture(tmp_path);root=tmp_path/'pending'
    put(root/'predictions_complete.json',{'status':'NOT_COMPLETE'})
    with pytest.raises(ValueError,match='Truth barrier'):
        score_pool({'run_root':str(root),'data_root':str(data),'score_processes':2},jobs)
    assert not (root/'scores').exists()


def test_pool_stops_admission_after_score_failure(tmp_path):
    data,jobs=fixture(tmp_path);root=tmp_path/'failed'
    put(root/'predictions_complete.json',{'status':'ALL_PREDICTIONS_FIXED'})
    jobs[0]['output']=str(tmp_path/'missing')
    with pytest.raises(ValueError,match='scor'):
        score_pool({'run_root':str(root),'data_root':str(data),'score_processes':1},jobs)
    state=json.loads((root/'score_state.json').read_text(encoding='utf-8'))
    assert state['status']=='FAILED' and state['completed']==0 and state['pending']==2
    assert not (root/'scores/job1').exists()


def test_pool_records_submit_failure_and_stops_admission(tmp_path,monkeypatch):
    from experiments.pamf_phase2_practical import group2_c_backbone_score_pool as module
    root=tmp_path/'broken';put(root/'predictions_complete.json',{'status':'ALL_PREDICTIONS_FIXED'})
    class FakeExecutor:
        def __init__(self,*args,**kwargs): self.count=0
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def submit(self,fn,job):
            self.count+=1
            if self.count>1:raise BrokenProcessPool('worker exited')
            future=Future();future.set_result({'pid':9,'key':job['key']});return future
    monkeypatch.setattr(module,'ProcessPoolExecutor',FakeExecutor)
    jobs=[{'key':f'job{i}'} for i in range(3)]
    with pytest.raises(ValueError,match='scoring failure'):
        score_pool({'run_root':str(root),'score_processes':1},jobs)
    state=json.loads((root/'score_state.json').read_text(encoding='utf-8'))
    assert state['status']=='FAILED' and state['completed']==1 and state['pending']==2
    assert state['failed'][0]['index']==1 and 'BrokenProcessPool' in state['failed'][0]['error']


def test_pool_records_first_submit_failure_with_no_active_futures(tmp_path,monkeypatch):
    from experiments.pamf_phase2_practical import group2_c_backbone_score_pool as module
    root=tmp_path/'broken';put(root/'predictions_complete.json',{'status':'ALL_PREDICTIONS_FIXED'})
    class FakeExecutor:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def submit(self,fn,job):raise BrokenProcessPool('worker exited')
    monkeypatch.setattr(module,'ProcessPoolExecutor',FakeExecutor)
    with pytest.raises(ValueError,match='scoring failure'):
        score_pool({'run_root':str(root),'score_processes':1},[{'key':'job0'}])
    state=json.loads((root/'score_state.json').read_text(encoding='utf-8'))
    assert state['status']=='FAILED' and state['completed']==0 and state['pending']==1
    assert state['active']==[] and state['failed'][0]['index']==0


def test_pool_records_executor_creation_failure(tmp_path,monkeypatch):
    from experiments.pamf_phase2_practical import group2_c_backbone_score_pool as module
    root=tmp_path/'broken';put(root/'predictions_complete.json',{'status':'ALL_PREDICTIONS_FIXED'})
    def fail(*args,**kwargs):raise OSError('cannot create process pool')
    monkeypatch.setattr(module,'ProcessPoolExecutor',fail)
    with pytest.raises(ValueError,match='scoring failure'):
        score_pool({'run_root':str(root),'score_processes':1},[{'key':'job0'}])
    state=json.loads((root/'score_state.json').read_text(encoding='utf-8'))
    assert state['status']=='FAILED' and state['pending']==1
    assert state['failed'][0]['phase']=='executor'


def test_pool_reuses_pilot_natural_and_common_scores(tmp_path):
    data=tmp_path/'data'; capsule='capsules/P01/MIX01/1-1'; cap=data/capsule
    put(cap/'manifest.json',{'capsule_id':'cap','physical_pool':'P01','episode_id':'MIX01','receiver':'1-1'})
    (cap/'stage_rows').mkdir(parents=True,exist_ok=True)
    np.save(cap/'stage_rows/N00.npy',np.arange(12,dtype=np.int64))
    np.save(cap/'stage_rows/N02.npy',np.arange(16,dtype=np.int64))
    ids={}
    for k,count in ((1,1),(5,2),(10,3),(20,4)):
        for n in (0,2):
            sid=f'{k:02d}{n:02d}'+'a'*20;ids[k,n]=sid
            put(cap/'splits'/(sid+'.json'),{'split_id':sid,'k':k,'new_count':n,
                'registered_classes':list(range(6+n)),'support_indices':list(range(count)),
                'query_count':(12 if n==0 else 16)-count})
    truth=data/'score_only/P01/MIX01/1-1.npz';truth.parent.mkdir(parents=True,exist_ok=True)
    np.savez(truth,labels=np.arange(16)//2,days=np.arange(16)%2,scene_indices=np.arange(16)%3)
    def prediction(folder,k,name):
        folder.mkdir(parents=True,exist_ok=True)
        start={1:1,5:2}[k];query=np.arange(start,16,dtype=np.int64)
        np.savez(folder/name,query_indices=query,predicted_indices=query//2,
                 old_restricted_indices=np.minimum(query//2,5))
        return {'split_id':ids[k,2],'stage':'C','input_path':'P0','arm':'B',
            'prediction':name,'k':k,'new_count':2,'registered_classes':list(range(8)),
            'query_count':len(query),'resources':{'B_fit_shared_across_N':True}}
    pilot=tmp_path/'pilotpred'; oldrow=prediction(pilot,5,'old.npz')
    def index(folder,rows):
        put(folder/'predictions.json',{'capsule':capsule,'capsule_id':'cap','model_seed':7,'rows':rows})
        put(folder/'complete.json',{'status':'PREDICTIONS_COMPLETE','truth_read':False,'rows':len(rows),'capsule_id':'cap'})
    index(pilot,[oldrow]);pilotroot=tmp_path/'pilotrun'
    rt={'data_root':str(data),'K':[1,5,10,20],'N':[2],'score_processes':1}
    put(pilotroot/'predictions_complete.json',{'status':'ALL_PREDICTIONS_FIXED'})
    _score(dict(rt,run_root=str(pilotroot)),{'key':'pilot','capsule':capsule,'output':str(pilot)})
    combined=tmp_path/'combined';newrow=prediction(combined,1,'new.npz')
    (combined/'old.npz').write_bytes((pilot/'old.npz').read_bytes())
    reused=dict(oldrow,resources={'B_fit_shared_across_N':None},reuse={
        'prediction_source':str(pilot),'score_source':str(pilotroot/'scores/pilot'),
        'B_state_across_N':'not_verified_against_new_B'})
    index(combined,[newrow,reused])
    poolroot=tmp_path/'pooled';put(poolroot/'predictions_complete.json',{'status':'ALL_PREDICTIONS_FIXED'})
    job={'key':'combined','capsule':capsule,'output':str(combined),'points':[(1,2),(5,2)],
         'reuse_prediction':str(pilot),'reuse_score':str(pilotroot/'scores/pilot')}
    score_pool(dict(rt,run_root=str(poolroot)),[job])
    score=poolroot/'scores/combined'
    for name in ('metrics.json','common_query_metrics.json'):
        result=json.loads((score/name).read_text(encoding='utf-8'))
        pilot_result=json.loads((pilotroot/'scores/pilot'/name).read_text(encoding='utf-8'))
        assert result['rows'][1]['metrics']==pilot_result['rows'][0]['metrics']
        assert result['rows'][1]['resources']['B_fit_shared_across_N'] is None

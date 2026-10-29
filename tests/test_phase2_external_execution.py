import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_phase2_practical import external_execution as execute
from experiments.pamf_phase2_practical.external_dispatch import make_jobs,external_ready,priority_key


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def test_source_runtime_has_no_target_or_truth_handles():
    rt=dict(source_cache='/Ls',source_contract='/source.json',data_root='/target',target_truth='/truth',
            new_classes=['hidden'],shared_dura_root='/targetG',capsules=['future'],source_x='/secret')
    assert execute.source_runtime(rt)==dict(source_cache='/Ls',source_contract='/source.json')


def test_core_score_without_seed_remains_schedulable(tmp_path):
    old=tmp_path/'core';old.mkdir()
    put(old/'jobs.json',[dict(id='score-predict',kind='score',output='/oldscore',success='SCORED_COMPLETE',dependencies=[],gpu=False)])
    put(tmp_path/'old.json',dict(run_root=str(old),runtime_file='/oldruntime',release_root='/oldrelease'))
    rt=dict(run_root='/new',source_plan=[],profile_matrix={},capsules=[],model_seeds=[1],continue_core=True,shared_core_runtime=str(tmp_path/'old.json'))
    j=make_jobs(rt)[0]
    assert 'seed' not in j and sorted([j],key=lambda x:priority_key(x,1))==[j]


def test_all_profiles_are_scheduled_with_shared_source_and_pair_score_gate(tmp_path):
    rt=dict(run_root=str(tmp_path),source_plan=[dict(id='shared',profile='a',action='reuse',paths={'1':'/model'})],model_seeds=[1],
            profile_matrix={p:dict(active=True,source_id='shared') for p in ['a','b']},capsules=['capsules/P01/MIX01/r'])
    matrix=make_jobs(rt);by={j['id']:j for j in matrix}
    assert len([j for j in matrix if j['kind']=='source'])==1
    assert len([j for j in matrix if j['kind']=='predict'])==4
    assert len([j for j in matrix if j['kind']=='score'])==2
    for j in matrix:
        if j['kind']=='predict':assert 'source-shared-s1' in j['dependencies']
        if j['kind']=='score':assert len(j['dependencies'])==2 and all(by[d]['input_path'] in ('P0','P1') for d in j['dependencies'])
    p1=next(j for j in matrix if j.get('input_path')=='P1');rt['shared_dura_root']=str(tmp_path/'G')
    assert not external_ready(rt,p1)
    put(Path(rt['shared_dura_root'])/p1['capsule']/'complete.json',dict(status='RESTORED_COMPLETE'))
    assert external_ready(rt,p1)
    rt['continue_core']=True
    assert not external_ready(rt,p1,set())
    assert external_ready(rt,p1,{p1['core_producer_id']})


class ToyNetwork(nn.Module):
    def __init__(self,count=6):super().__init__();self.anchor=nn.Parameter(torch.zeros(1));self.count=count
    def forward(self,x):
        value=torch.floor(x[:,0,0]);return -(value[:,None]-torch.arange(self.count,device=x.device)[None,:]).square()+self.anchor*0


def state(model):
    return dict(model=model,classes=tuple(range(model.count)),score_fn=lambda current,x:current(x),resources={})


def test_stage_reader_native_forks_and_paired_truth_last_end_to_end(tmp_path,monkeypatch):
    data=tmp_path/'data';relative='capsules/P01/MIX01/r';cap=data/relative
    labels=np.repeat(np.arange(26),4);raw=np.zeros((len(labels),2,256),np.float32)
    raw[:,0,:]=np.repeat(labels[:,None],256,axis=1)+np.tile(np.arange(4),26)[:,None]/100
    cap.mkdir(parents=True);np.save(cap/'received.npy',raw);np.save(cap/'ids.npy',np.array([f'id{i}' for i in range(len(labels))]))
    put(cap/'manifest.json',dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id='c',storage='episodic-pool-complement-v1',received_count=len(raw),physical_pool='P01',episode_id='MIX01',receiver='r'))
    for d in ('stage_rows','support_labels','splits'):(cap/d).mkdir()
    for n in (0,2):
        sid=str(n)*24;support=np.arange(0,(6+n)*4,4);reg=np.arange((6+n)*4,dtype=np.int32)
        np.save(cap/'stage_rows'/f'N{n:02d}.npy',reg);np.save(cap/'support_labels'/(sid+'.npy'),np.arange(6+n,dtype=np.int16))
        put(cap/'splits'/(sid+'.json'),dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id='c',split_id=sid,new_count=n,k=1,registered_classes=[str(i) for i in range(6+n)],support_indices=support.tolist(),query_count=3*(6+n)))
    g=tmp_path/'G'/relative;g.mkdir(parents=True);np.save(g/'received.npy',raw+.001);put(g/'complete.json',dict(status='RESTORED_COMPLETE',count=len(raw)))
    truth=data/'score_only/P01/MIX01';truth.mkdir(parents=True);np.savez(truth/'r.npz',labels=labels,days=np.zeros(len(raw),int),scene_indices=np.zeros(len(raw),int))
    trace=[]
    class Bridge:
        @staticmethod
        def load_source(*args):return state(ToyNetwork())
        @staticmethod
        def fit_B(rt,bundle,sx,sy,seed,logger=None):
            assert len(sx)==6 and sy.tolist()==list(range(6));trace.append(('B',sx[:,0,0].tolist()))
            return state(copy.deepcopy(bundle['model']))
        @staticmethod
        def native_C(rt,bundle,B,sx,sy,new_ids,seed,logger=None):
            assert len(sx)==2 and sy.tolist()==[6,7] and new_ids==[6,7];trace.append(('C',sx[:,0,0].tolist()))
            B['model'].count=8;B['classes']=tuple(range(8));return B
    monkeypatch.setattr(execute,'bridge',lambda p:Bridge);monkeypatch.setattr(execute,'activate',lambda rt:None)
    rt=dict(data_root=str(data),shared_dura_root=str(tmp_path/'G'),capsules=[relative],model_seeds=[1],shots=[1],new_counts=[0,2],
        profile_matrix={'toy':dict(native_C=True,chains=[dict(id='J02',kind='native',nature='EXPLICIT_REGISTRATION_EXTENSION')],source_nature='SOURCE_ONLY',B_nature='NATIVE')})
    parent=tmp_path/'pred';base=dict(profile='toy',seed=1,capsule=relative)
    execute.predict(rt,dict(base,input_path='P0',output=str(parent/'P0')))
    with pytest.raises(FileNotFoundError):execute.score_pair(rt,dict(prediction=str(parent),output=str(tmp_path/'score')))
    assert not (parent/'complete.json').exists()
    execute.predict(rt,dict(base,input_path='P1',output=str(parent/'P1')))
    execute.score_pair(rt,dict(prediction=str(parent),output=str(tmp_path/'score')))
    metrics=json.loads((tmp_path/'score/metrics.json').read_text())
    assert len(metrics['rows'])==6 and all(x['metrics']['accuracy']==1 for x in metrics['rows'])
    assert [x[0] for x in trace]==['B','C','B','C']
    assert max(trace[0][1])<6 and min(trace[1][1])>=6 and max(trace[1][1])<8
    assert np.allclose(np.asarray(trace[2][1])-trace[0][1],.001,atol=1e-6)

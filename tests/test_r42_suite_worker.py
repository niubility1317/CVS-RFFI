import json
from pathlib import Path
import types
import numpy as np
import pytest
import torch
from experiments.pamf_phase2_practical import r42_suite_worker as worker


def test_desired_rows_full_and_diagnostic_are_unique():
    full=worker.desired_rows([[k,n] for k in (1,5,10,20) for n in (2,5,10,20)])
    diagnostic=worker.desired_rows([[5,20],[5,20]])
    assert len(full)==208 and len(diagnostic)==22
    assert len({worker.row_key(r) for r in full})==208
    assert sum(r['stage']=='A0' for r in full)==8
    assert sum(r['arm']=='R42' for r in full)==120
    assert sum(r['stage']=='B' and r['arm']=='LocalRidge' for r in full)==16
    with pytest.raises(ValueError):worker.desired_rows([[2,20]])


def setup_case(tmp_path):
    cap=tmp_path/'data/capsule';(cap/'splits').mkdir(parents=True);(cap/'stage_rows').mkdir()
    names=[f'c{i:02d}' for i in range(26)]
    bank=np.zeros((26*6,2,256),np.float32);bank[:,0,0]=np.repeat(np.arange(26),6)
    np.save(cap/'received.npy',bank);np.save(cap/'stage_rows/N20.npy',np.arange(len(bank)))
    worker.write(cap/'manifest.json',dict(receiver='rx',capsule_id='cap1'))
    for n,count in ((0,6),(20,26)):
        support=np.concatenate([np.arange(i*6,i*6+5) for i in range(count)])
        worker.write(cap/'splits'/f'{n}.json',dict(k=5,new_count=n,split_id=f'split{n}',registered_classes=names[:count],support_indices=support.tolist(),query_count=count))
    run=tmp_path/'run';run.mkdir();worker.write(run/'source_selection.json',dict(K1_rule=dict(fusion_policy='rf_distance_margin_half_v1')))
    source=tmp_path/'source.json';worker.write(source,dict(classes=names[:6][::-1]))
    rt=dict(data_root=str(tmp_path/'data'),run_root=str(run),source_contract=str(source),model_seeds=[1],device='cpu',decision_batch=7,batch_size=13)
    job=dict(key='case',seed=1,capsule='capsule',points=[[5,20]],reuse_file=None)
    return rt,job,cap,bank


def fake_runtime(monkeypatch,cap,bank):
    calls=dict(fits=[],extract=[],columns=[],G_rows=0)
    reader=torch.nn.Linear(1,1).eval();gmodel=torch.nn.Linear(1,1).eval()
    calls.update(reader=reader,gmodel=gmodel)
    class Engine:
        pipeline=types.SimpleNamespace(model=gmodel)
        def __call__(self,x):
            calls['G_rows']+=len(x);out=x.clone();out[:,0,1]=1;return (out,)
    monkeypatch.setattr(worker,'initialize',lambda rt:(reader,Engine()))
    def task(cap,split_id,stage):
        spec=worker.read(Path(cap)/'splits'/f'{split_id.removeprefix("split")}.json');count=len(spec['registered_classes'])
        return dict(support_x=bank[spec['support_indices']],support_y=np.repeat(np.arange(count),5),support_ids=list(map(str,spec['support_indices'])))
    monkeypatch.setattr(worker,'load_task',task)
    def query(cap,spec):return np.arange(5,len(spec['registered_classes'])*6,6)
    monkeypatch.setattr(worker,'split_rows',query)
    def extract(kind,reader,raw,device,engine=None,batch=128):
        assert engine is None
        raw=np.asarray(raw);code=raw[:,0,0].astype(int);restored=raw[:,0,1].astype(bool)
        calls['extract'].append((kind,len(raw),batch,restored.tolist()))
        features=dict(z_id=code[:,None],code=code,restored=restored)
        logits=np.zeros((len(raw),6));logits[np.arange(len(raw)),5-code%6]=1
        return features,logits
    monkeypatch.setattr(worker,'extract_orbit_ensemble',lambda *args,**kwargs:extract('R42',*args,**kwargs))
    monkeypatch.setattr(worker,'extract_baseline',lambda *args,**kwargs:extract('LocalRidge',*args,**kwargs))
    class Bridge:
        def transform(self,features,*,restored,old_columns):
            assert np.all(features['restored']==restored)
            calls['columns'].append(old_columns);return features
    monkeypatch.setattr(worker.AdaptiveResidualBridge,'load',lambda path:Bridge())
    class State:
        def __init__(self,arm,features,labels,classes):
            calls['fits'].append((arm,len(labels),features['restored'].tolist()))
            self.arm=arm;self.count=int(np.max(labels))+1;self.classes=classes
            self.alpha=np.zeros((len(labels),self.count));self.selection=dict(query_fit=False);self.config=('fixed',);self.ridge=.1;self.tau=1.
        def evaluate(self,features):
            c=features['code'];g=features['restored'];scores=np.zeros((len(c),self.count));rows=np.arange(len(c))
            scores[rows,c]=np.where(g,.9,.4);scores[rows,(c+1)%self.count]=np.where(g,.1,.6)
            return scores,np.where(g,.1,.5)
        def score(self,**features):
            c=features['code'];out=np.zeros((len(c),self.count));out[np.arange(len(c)),c]=1;return out
        def audit_dict(self):return dict(optimizer_steps=0)
    monkeypatch.setattr(worker,'fit_r42',lambda f,y,ids,fixed:State('R42',f,y,None))
    def fit_base(support_labels,support_ids,classes,old_classes,arm,**features):
        assert arm=='local_ridge';return State('LocalRidge',features,support_labels,classes)
    monkeypatch.setattr(worker,'fit_branch_local_ridge',fit_base)
    return calls


def make_reuse(tmp_path,rt,job,cap,rows=None,seed=1,wrong_split=False):
    old=tmp_path/'old';old.mkdir();specs={(s['k'],s['new_count']):s for s in (worker.read(p) for p in (cap/'splits').glob('*.json'))}
    rows=worker.desired_rows(job['points']) if rows is None else rows
    indexed=[]
    for i,row in enumerate(rows):
        spec=specs[row['k'],row['new_count']];name=f'{i}.npz';query=np.arange(5,len(spec['registered_classes'])*6,6)
        np.savez_compressed(old/name,query_indices=query,predicted_indices=np.arange(len(query),dtype=np.int16),old_restricted_indices=np.arange(len(query),dtype=np.int16)%6)
        indexed.append(dict(row,split_id='wrong' if wrong_split else spec['split_id'],registered_classes=spec['registered_classes'],prediction=name))
    worker.write(old/'predictions.json',dict(capsule='capsule',capsule_id='cap1',model_seed=seed,rows=indexed))
    worker.write(old/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id='cap1',truth_read=False,query_fit=False,rows=len(indexed)))
    entries=[dict(key=list(worker.row_key(row)),prediction_dir=str(old),row=row,metrics_file=None) for row in indexed]
    path=tmp_path/'reuse.json';worker.write(path,entries);job['reuse_file']=str(path)
    return old


@pytest.mark.parametrize('failure',['seed','split'])
def test_reuse_rejects_wrong_seed_or_split_before_model_access(tmp_path,monkeypatch,failure):
    rt,job,cap,bank=setup_case(tmp_path)
    make_reuse(tmp_path,rt,job,cap,seed=2 if failure=='seed' else 1,wrong_split=failure=='split')
    monkeypatch.setattr(worker,'initialize',lambda rt:pytest.fail('model must not load for bad reuse'))
    with pytest.raises(ValueError,match='seed mismatch|split/class'):worker.run(rt,job)
    assert not (Path(rt['run_root'])/'predictions/rx/complete.json').exists()


def test_all_completed_rows_reuse_without_fit_or_model_load(tmp_path,monkeypatch):
    rt,job,cap,bank=setup_case(tmp_path);make_reuse(tmp_path,rt,job,cap)
    monkeypatch.setattr(worker,'initialize',lambda rt:pytest.fail('completed rows must not initialize models'))
    worker.run(rt,job);root=Path(rt['run_root'])/'predictions/rx';complete=worker.read(root/'complete.json')
    assert complete['rows']==22 and complete['new_prediction_rows']==0 and complete['reused_prediction_rows']==22
    assert all(r['original_arm']==r['arm'] for r in worker.read(root/'predictions.json')['rows'])


def test_three_query_paths_strict_support_and_original_A0(tmp_path,monkeypatch):
    rt,job,cap,bank=setup_case(tmp_path);calls=fake_runtime(monkeypatch,cap,bank)
    worker.run(rt,job);root=Path(rt['run_root'])/'predictions/rx';payload=worker.read(root/'predictions.json');complete=worker.read(root/'complete.json')
    assert len(payload['rows'])==22 and len(calls['fits'])==8
    assert all(flags==[flags[0]]*count for _,count,flags in calls['fits'])
    assert sum(flags[0] for _,_,flags in calls['fits'])==4
    assert all(columns==[5,4,3,2,1,0] for columns in calls['columns'])
    for row in payload['rows']:
        with np.load(root/row['prediction']) as z:
            truth=np.arange(len(row['registered_classes']))
            expected=(truth+1)%len(truth) if row['arm']=='R42' and row['input_path'].endswith('Q0') else truth
            assert np.array_equal(z['predicted_indices'],expected)
    assert complete['truth_read'] is False and complete['raw_score_cache_final_rows']==0
    resources=list(root.glob('*-resources.json'));assert len(resources)==10
    assert all(worker.read(p)['native_optimizer_steps']==0 and worker.read(p)['CPU_RSS_sampled_peak']>0 for p in resources)
    assert all(batch==5 for _,n,batch,_ in calls['extract'] if n!=len(bank))


def test_partial_reuse_keeps_existing_rows_and_only_fits_missing_state(tmp_path,monkeypatch):
    rt,job,cap,bank=setup_case(tmp_path);desired=worker.desired_rows(job['points'])
    # Only the R42 C/S1 fusion row is missing: recompute its raw/G scores,
    # while preserving both already-frozen endpoint prediction files.
    missing=('R42','C',5,20,'S1Q2')
    make_reuse(tmp_path,rt,job,cap,[r for r in desired if worker.row_key(r)!=missing]);calls=fake_runtime(monkeypatch,cap,bank)
    worker.run(rt,job);root=Path(rt['run_root'])/'predictions/rx';done=worker.read(root/'complete.json')
    assert done['new_prediction_rows']==1 and done['reused_prediction_rows']==21
    assert len(calls['fits'])==1 and calls['fits'][0][:2]==('R42',130) and all(calls['fits'][0][2])
    payload=worker.read(root/'predictions.json');new=[r for r in payload['rows'] if 'reused_from' not in r]
    assert [worker.row_key(r) for r in new]==[missing]
    reused_raw=next(r for r in payload['rows'] if worker.row_key(r)==('R42','C',5,20,'S1Q0'))
    with np.load(root/reused_raw['prediction']) as z:assert np.array_equal(z['predicted_indices'],np.arange(26))


@pytest.mark.parametrize('model_name',['reader','gmodel'])
def test_frozen_model_change_is_rejected(tmp_path,monkeypatch,model_name):
    rt,job,cap,bank=setup_case(tmp_path);calls=fake_runtime(monkeypatch,cap,bank);original=worker.fit_r42
    def changing(*args,**kwargs):
        with torch.no_grad():calls[model_name].weight.add_(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(worker,'fit_r42',changing)
    with pytest.raises(ValueError,match='Frozen suite [FG] changed'):worker.run(rt,job)
    assert not (Path(rt['run_root'])/'predictions/rx/complete.json').exists()

"""No training: original T1 loader, original R17 IQ, separate CPU truth-last."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from experiments.pamf_restoration.runner import array_tensor, tensor_array, code_commit
from experiments.pamf_restoration.data import IQSource, write_json
from experiments.pamf_restoration.frozen import classification_statistics
from experiments.pamf_restoration.dispatcher import capacity
from experiments.pamf_unrolled_restore.runtime import load_checkpoint as load_r17

ROOT=Path(__file__).resolve().parents[2]
RUN='20261020-evaluation-T1E30-R17-s2026092701-r01'
T1_RUN='20261019-phase1-FcrossRX-AsTemperature-e30-s2026092701-m2-r02'
T1_COMMIT='409fbf49571fb39cfd02701992824d7d71a6c712'
T1_RELEASE='cvs_FcrossRX_AsTemperature_E30_20261019_r02'
T1_ROW='T1-s2026092701'
R17_RUN='20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17'
R17_COMMIT='cd97744884b5e460926e48c8b7efb7d78c6da0f4'
R17_ROW='unroll_unshared'
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
COUNT=168000
GPUS=(6,7)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def locations(base):
    t1=base/'runs'/T1_RUN/T1_ROW
    r17=base/'runs'/R17_RUN/R17_ROW
    return t1,r17,r17/'target_confirmation/native_map_prior/prediction'

def validate_source_metadata(initial,done,frozen,segment):
    c=done['config'];epochs,updates=(20,4440) if segment==20 else (10,2220)
    if (done.get('status')!='SOURCE_TRAINED' or done.get('epoch')!=epochs or done.get('optimizer_steps')!=updates
            or done.get('target_access') is not False or done.get('target_evaluated') is not False or done.get('head_fit_updates')!=0
            or c.get('run_id')!=T1_RUN or c.get('row_id')!=T1_ROW or c.get('segment')!=segment
            or c.get('cumulative_epochs')!=segment or c.get('checkpoint_selection')!='final_only'
            or c.get('separate_head_training') is not False or c.get('target_access') is not False):
        raise ValueError('Exact source-only T1 E20/ownE10 lineage required')
    temp=c.get('temperature',{})
    if any(temp.get(k)!=v for k,v in dict(T=1.,As_KL=0.,ordinary_KL=1.,raw_KL=1.,deployment_temperature=1.).items()):
        raise ValueError('T1 As-only KL-off with original deployment head required')
    if (frozen.get('status')!='SOURCE_FROZEN' or frozen.get('selection')!='final_only'
            or frozen.get('config')!=c or frozen.get('target_access') is not False
            or frozen.get('checkpoint')!=done.get('checkpoint') or initial.get('target_contact') is not False
            or initial.get('source_roles')!='EXACT_MATCH' or initial.get('seed')!=2026092701):
        raise ValueError('Frozen fixed-final source provenance required')
    sources=initial.get('checkpoint_sources')
    if segment==20:
        if initial.get('scratch_only') is not True or sources!=[] or initial.get('ancestors')!=[]:
            raise ValueError('T1 must originate from scratch')
    elif len(sources or [])!=1 or not sources[0].endswith('/'+T1_ROW+'/stage20/final_ssdg.pth') or sources!=c.get('checkpoint_sources'):
        raise ValueError('T1 own E20 parent required')
    return c

def provenance(args):
    t1,r17,pred=locations(args.base)
    expected=read(args.source_cache.parent/'source/source_contract.json')
    for segment,folder in ((20,t1/'stage20'),(30,t1/'source')):
        actual=read(folder/'source_contract.json')
        if any(actual.get(k)!=expected.get(k) for k in ('role_ids','source_rxs','source_days')):
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        validate_source_metadata(*[read(folder/n) for n in ('initialization.json','completion.json','source_frozen.json')],segment)
    if (args.base/'releases'/T1_RELEASE/'release_commit.txt').read_text().strip()!=T1_COMMIT:
        raise ValueError('Original T1 code identity differs')
    if read(args.base/'runs'/R17_RUN/'launch.json')['commit']!=R17_COMMIT:
        raise ValueError('Original R17 code identity differs')
    trained=read(r17/'training_complete.json')
    required=dict(status='SOURCE_TRAINED',candidate=R17_ROW,epochs=20,updates=4940,TX_labels_read=0,
                  F_outputs_read=0,F_weights_read=0,target_reads=0,frozen_context_updates=0,L_exposures=126000,U_exposures=1134000)
    if any(trained.get(k)!=v for k,v in required.items()):raise ValueError('Original TX/F/target-free R17 training required')
    return dict(status='VERIFIED',T1_run=T1_RUN,T1_commit=T1_COMMIT,T1_row=T1_ROW,T1_epochs=30,
                R17_run=R17_RUN,R17_commit=R17_COMMIT,R17_row=R17_ROW,physical_roles='EXACT_MATCH',
                new_training_updates=0,head_fit_updates=0,source_target_training_access=False)

def original_t1(args,device):
    # Use the historical loader/build directly; no reimplementation or export.
    historical=args.base/'releases'/T1_RELEASE
    runpy.run_path(str(historical/'experiments/cvs_as_temperature84/runner.py'),run_name='t1_original_bootstrap')['bootstrap']()
    from experiments.cvs_as_temperature84 import design as old
    from experiments.cvs_as_temperature84.model import own_model
    config=old.config(next(r for r in old.rows() if r['arm']=='T1'))
    return own_model(config,device).requires_grad_(False).eval()

@torch.no_grad()
def smoke(args):
    proof=provenance(args)
    t1,r17,_=locations(args.base)
    restorer,payload=load_r17(r17/'native_map_prior/final.pt','cpu')
    restorer.requires_grad_(False).eval()
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu')
    restored=restorer(x)
    model=original_t1(args,'cpu')
    raw=model.forward_identity_only(x)['tx_logits'];after=model.forward_identity_only(restored)['tx_logits']
    if restored.shape!=(2,2,256) or raw.shape!=after.shape or raw.shape!=(2,6) or not all(torch.isfinite(v).all() for v in (restored,raw,after)):
        raise ValueError('Real T1/R17 source-only no-query smoke failed')
    if payload['epoch']!=20 or payload['updates']!=4940 or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen original checkpoints required')
    write_json(args.output/'smoke.json',dict(status='PASS',source_L_samples=2,target_reads=0,TX_labels_read=0,
               new_training_updates=0,head_fit_updates=0,original_T1_loader=True,original_R17_loader=True,provenance=proof))

def validate_prediction_metadata(value,*,original):
    if original:
        expected=dict(status='PREDICTIONS_COMPLETE',row_id=T1_ROW,count=COUNT,views=list(VIEWS),truth_read=False,
                      query_fit=False,all_parameters_frozen=True,head_fit_updates=0)
    else:
        expected=dict(status='PREDICTIONS_COMPLETE',kind='native_map_prior',role='target',count=COUNT,views=list(VIEWS),
                      readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('All seven complete frozen original predictions required')

def observations(args):
    t1,r17,pred=locations(args.base)
    validate_prediction_metadata(read(pred/'complete.json'),original=False)
    validate_prediction_metadata(read(t1/'prediction/complete.json'),original=True)
    ids=np.load(pred/'physical_ids.npy',allow_pickle=False)
    if ids.shape!=(COUNT,) or len(set(ids.tolist()))!=COUNT:raise ValueError('Original opaque physical IDs required')
    baseline={}
    for view in VIEWS:
        iq=np.load(pred/(view+'.restored.iq.npy'),mmap_mode='r',allow_pickle=False)
        if iq.shape!=(COUNT,2,256) or iq.dtype!=np.float32 or not np.isfinite(iq).all():
            raise ValueError('Incomplete original R17 IQ')
        with np.load(t1/'prediction'/(view+'_scores.npz'),allow_pickle=False) as data:
            logits=data['logits']
            if not np.array_equal(data['ids'],ids) or logits.shape!=(COUNT,6) or not np.isfinite(logits).all():
                raise ValueError('Original same-T1 baseline IDs/scores differ')
            baseline[view]=logits.copy()
    return ids,baseline

@torch.no_grad()
def predict(args):
    if read(args.output/'smoke.json')['status']!='PASS':raise ValueError('Actual source smoke must pass first')
    provenance(args);ids,_=observations(args)
    model=original_t1(args,'cuda:0')
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    _,_,original=locations(args.base);out=args.output/'prediction';out.mkdir()
    times={};started=time.perf_counter()
    for view in VIEWS:
        iq=np.load(original/(view+'.restored.iq.npy'),mmap_mode='r',allow_pickle=False)
        values=np.lib.format.open_memmap(out/(view+'.logits.npy'),mode='w+',dtype=np.float32,shape=(COUNT,6))
        start=time.perf_counter()
        for begin in range(0,COUNT,256):
            x=array_tensor(iq[begin:begin+256],'cuda:0')
            logits=model.forward_identity_only(x)['tx_logits']
            if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise ValueError('Finite native T1 default-head logits required')
            values[begin:begin+len(x)]=tensor_array(logits)
        values.flush();torch.cuda.synchronize();times[view]=time.perf_counter()-start
        print(json.dumps(dict(event='VIEW_COMPLETE',view=view,count=COUNT,seconds=times[view])),flush=True)
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Frozen T1 state changed')
    np.save(out/'physical_ids.npy',ids,allow_pickle=False)
    write_json(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',run=RUN,count=COUNT,views=list(VIEWS),
               T1_run=T1_RUN,T1_row=T1_ROW,T1_commit=T1_COMMIT,R17_run=R17_RUN,R17_row=R17_ROW,
               original_default_head=True,state_updates=0,head_fit_updates=0,TX_truth_reads=0,
               reused_R17_IQ=True,reused_T1_raw_predictions=True,per_view_seconds=times,seconds=time.perf_counter()-started))

def completed_predictions(args):
    out=args.output/'prediction';done=read(out/'complete.json')
    expected=dict(status='PREDICTIONS_COMPLETE',run=RUN,count=COUNT,views=list(VIEWS),T1_run=T1_RUN,
                  T1_row=T1_ROW,T1_commit=T1_COMMIT,R17_run=R17_RUN,R17_row=R17_ROW,
                  original_default_head=True,state_updates=0,head_fit_updates=0,TX_truth_reads=0)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Truth closed: incomplete or wrong reader predictions')
    ids,baseline=observations(args)
    if not np.array_equal(np.load(out/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Truth closed: IDs differ')
    restored={}
    for view in VIEWS:
        a=np.load(out/(view+'.logits.npy'),mmap_mode='r',allow_pickle=False)
        if a.shape!=(COUNT,6) or not np.isfinite(a).all():raise ValueError('Truth closed: incomplete/nonfinite logits')
        restored[view]=a
    return ids,baseline,restored

def score(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU scorer required')
    ids,before,after=completed_predictions(args)
    # First truth open; all seven new and reused original predictions are fixed.
    truth=read(args.base/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')
    y=np.asarray([truth[str(pid)]['label'] for pid in ids],np.int64)
    rx=np.asarray([str(truth[str(pid)]['receiver']) for pid in ids])
    day=np.asarray([str(truth[str(pid)].get('day','unavailable')) for pid in ids])
    records=[]
    for view in VIEWS:
        b=before[view].argmax(1);r=after[view].argmax(1)
        groups=[('overall','all',np.ones(COUNT,bool))]
        for dim,values in (('TX',y),('RX',rx),('day',day)):
            groups.extend((dim,str(v),values==v) for v in np.unique(values))
        for dim,value,mask in groups:
            records.append(dict(view=view,dimension=dim,value=value,**classification_statistics(y[mask],r[mask],b[mask])))
    overall={r['view']:r for r in records if r['dimension']=='overall'}
    rawclean=overall['clean']['baseline_accuracy'];limit=rawclean-5
    goal=dict(lowurban_pass=overall[VIEWS[-1]]['accuracy']>=70,other_five={v:overall[v]['accuracy']>=limit for v in VIEWS[1:-1]},
              clean_reference=rawclean,other_threshold=limit,clean_drop_pp=rawclean-overall['clean']['accuracy'])
    goal['all_targets_met']=goal['lowurban_pass'] and all(goal['other_five'].values())
    clean=before['clean'].argmax(1);mask=clean==y;low=after[VIEWS[-1]].argmax(1)
    identity=dict(clean_correct_count=int(mask.sum()),clean_wrong_count=int((~mask).sum()),
        lowurban_accuracy_clean_correct=float((low[mask]==y[mask]).mean()*100),
        lowurban_accuracy_clean_wrong=float((low[~mask]==y[~mask]).mean()*100),
        raw_all_six_correct=int(np.all(np.stack([before[v].argmax(1)==y for v in VIEWS[1:]]),axis=0).sum()),
        restored_all_six_correct=int(np.all(np.stack([after[v].argmax(1)==y for v in VIEWS[1:]]),axis=0).sum()),
        per_detail_identity_proved=False)
    write_json(args.output/'summary.json',dict(status='ANALYZED',run=RUN,count=COUNT,model_seed=2026092701,
        T1_run=T1_RUN,T1_commit=T1_COMMIT,T1_row=T1_ROW,R17_run=R17_RUN,R17_commit=R17_COMMIT,R17_row=R17_ROW,
        views=list(VIEWS),records=records,goal=goal,identity=identity,new_training_updates=0,head_fit_updates=0,
        original_default_head=True,all_predictions_before_truth=True,benchmark='Previously exposed simulated practical benchmark',
        target_feedback=False,goal_complete=False))
    write_json(args.output/'complete.json',dict(status='ANALYZED',views=list(VIEWS),count=COUNT,goal_complete=False))

def command(args,mode):
    return [sys.executable,'-u','-m','experiments.pamf_t1_r17_eval.runner',mode,'--base',str(args.base),
            '--source-cache',str(args.source_cache),'--output',str(args.output),'--expected-commit',args.expected_commit]

def run(args):
    import fcntl
    if args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New immutable root and CPU controller required')
    args.output.mkdir();write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,owner='side/T1-R17',time=time.time()))
    try:
        smoke(args)
        while True:
            with (args.base/'runs/receiver_residual_capacity4.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);state=capacity()
                cards=[g for g in GPUS if state[g]['free_mb']>=12000 and len(state[g]['pids'])<2]
                if cards:
                    gpu=min(cards,key=lambda g:(len(state[g]['pids']),-state[g]['free_mb'],GPUS.index(g)))
                    with (args.output/'prediction.log').open('x',encoding='utf-8') as log:
                        p=subprocess.Popen(command(args,'predict'),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    write_json(args.output/'worker.json',dict(pid=p.pid,GPU=gpu,cwd=str(ROOT),argv=command(args,'predict'),log=str(args.output/'prediction.log')))
                    deadline=time.monotonic()+45
                    while p.poll() is None and time.monotonic()<deadline and p.pid not in capacity()[gpu]['pids']:time.sleep(.5)
                    if p.poll() is None and p.pid not in capacity()[gpu]['pids']:raise RuntimeError('CUDA not visible in45s;preserve process,no retry')
                    break
            time.sleep(5)
        if p.wait()!=0:raise RuntimeError('Prediction failed;preserve outputs,no automatic retry')
        subprocess.run(command(args,'score'),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
    except BaseException as e:
        write_json(args.output/'failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','predict','score'))
    for name in ('base','source-cache','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-commit',required=True);args=p.parse_args()
    for name in ('base','source_cache','output'):setattr(args,name,getattr(args,name).resolve())
    if args.output.name!=RUN or code_commit()!=args.expected_commit:raise ValueError('Pinned code/new run required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {'run':run,'predict':predict,'score':score}[args.mode](args)

if __name__=='__main__':main()

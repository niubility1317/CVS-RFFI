"""Automatic seven-view frozen target confirmation after the source rule passes."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .models import load_checkpoint
from experiments.pamf_feature_refine.target import readers,SEEDS,READOUTS
from experiments.pamf_feature_refine.practical_target import reference_handles,waveform_gap
from experiments.pamf_restoration.data import SCENES,IQSource,write_json
from experiments.pamf_restoration.runner import array_tensor,tensor_array,code_commit
from experiments.pamf_restoration.dispatcher import capacity
from experiments.pamf_restoration.frozen import classification_statistics

ROOT=Path(__file__).resolve().parents[2];COUNT=168000;VIEWS=('clean',*SCENES)


def source_check(args):
    source=args.source_run;launch=json.loads((source/'launch.json').read_text())
    done=json.loads((source/'source_complete.json').read_text());decision=json.loads((source/'source_decision.json').read_text())
    training=json.loads((source/d.ARM/'training_complete.json').read_text())
    if (source.name!=d.RUN or launch['commit']!=args.expected_commit or done.get('status')!='ANALYZED' or
        not done.get('source_only') or done.get('target_reads')!=0 or done.get('arms')!=list(d.ARMS) or
        not decision.get('qualified') or decision.get('target_reads')!=0 or
        training.get('epochs')!=20 or training.get('updates')!=4940 or
        any(training.get(k)!=0 for k in ('F_optimizer_updates','head_fit_updates','coarse_updates','TX_labels_read','target_reads'))):
        raise ValueError('Completed source-qualified fixed posterior checkpoint required')


@torch.no_grad()
def smoke(args):
    model,item=load_checkpoint(args.source_run/d.ARM/'final.pt',d.ARM,'cpu')
    if (item['epoch'],item['updates'])!=(20,4940):raise ValueError('Fixed final source checkpoint required')
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu');value=model(x)
    for reader in readers(args,'cpu').values():
        scores=reader(value)
        if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Real source original-head smoke failed')
    write_json(args.output/'smoke.json',dict(status='PASS',real_source_samples=2,actual_new_checkpoint=True,
        F_seeds=list(SEEDS),original_default_heads=True,target_reads=0,TX_truth_reads=0,updates=0))


def acquire(args):
    import fcntl
    while True:
        with (args.source_run.parent/'receiver_residual_capacity4.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=capacity()[d.GPUS[d.ARM]]
            if state['free_mb']>=12000 and len(state['pids'])<2:
                model,_=load_checkpoint(args.source_run/d.ARM/'final.pt',d.ARM,'cuda:0')
                frozen=readers(args,'cuda:0');torch.cuda.synchronize();return model,frozen
        time.sleep(5)


@torch.no_grad()
def predict(args,model,frozen):
    ids,paths=reference_handles(args.formal_residual)
    paths['clean']=(args.formal_residual/'reference/clean.npy',args.formal_residual/'residual_deterministic/prediction/clean.npy')
    saved=args.output/'prediction';saved.mkdir();elapsed={}
    for view in VIEWS:
        observed,parent=[np.load(p,mmap_mode='r',allow_pickle=False) for p in paths[view]]
        if any(v.shape!=(COUNT,2,256) or v.dtype!=np.float32 for v in (observed,parent)):raise ValueError('Fixed target IQ layout required')
        iq=np.lib.format.open_memmap(saved/(view+'_refined_iq.npy'),mode='w+',dtype=np.float32,shape=(COUNT,2,256))
        decisions={name:np.empty((COUNT,3),np.uint8) for name in READOUTS};started=time.perf_counter()
        for offset in range(0,COUNT,256):
            y=array_tensor(observed[offset:offset+256],'cuda:0');p=array_tensor(parent[offset:offset+256],'cuda:0');value=model(y);end=offset+len(y)
            if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite frozen target restoration')
            iq[offset:end]=tensor_array(value)
            for column,seed in enumerate(SEEDS):
                for name,z in zip(READOUTS,(y,p,value)):
                    scores=frozen[seed](z)
                    if scores.shape!=(len(y),6) or not torch.isfinite(scores).all():raise ValueError('Invalid original defaulthead decisions')
                    decisions[name][offset:end,column]=np.asarray(scores.argmax(1).tolist(),np.uint8)
            if end%16384==0 or end==COUNT:print(json.dumps(dict(event='TARGET_PROGRESS',view=view,count=end,total=COUNT,seconds=time.perf_counter()-started)),flush=True)
        iq.flush();del iq
        with (saved/(view+'.npz')).open('xb') as stream:np.savez_compressed(stream,ids=ids,seeds=np.asarray(SEEDS),**decisions)
        elapsed[view]=time.perf_counter()-started
    write_json(saved/'complete.json',dict(status='PREDICTIONS_COMPLETE',views=list(VIEWS),count=COUNT,F_seeds=list(SEEDS),
        readouts=list(READOUTS),TX_truth_reads=0,paired_clean_in_practical_inference=False,state_updates=0,per_view_seconds=elapsed))


def completed_predictions(output,ids):
    saved=output/'prediction';done=json.loads((saved/'complete.json').read_text())
    expected=dict(status='PREDICTIONS_COMPLETE',views=list(VIEWS),count=len(ids),F_seeds=list(SEEDS),readouts=list(READOUTS),
        TX_truth_reads=0,paired_clean_in_practical_inference=False,state_updates=0)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('All seven views/3F/IQ required before truth')
    result={}
    for view in VIEWS:
        with np.load(saved/(view+'.npz'),allow_pickle=False) as p:
            if set(p.files)!={'ids','seeds',*READOUTS} or not np.array_equal(p['ids'],ids) or p['seeds'].tolist()!=list(SEEDS):raise ValueError('Same physical IDs/F seeds required')
            result[view]={}
            for name in READOUTS:
                value=p[name]
                if value.shape!=(len(ids),3) or value.dtype.kind not in 'iu' or (value<0).any() or (value>5).any():raise ValueError('Complete legal defaulthead predictions required')
                result[view][name]=value.copy()
        value=np.load(saved/(view+'_refined_iq.npy'),mmap_mode='r',allow_pickle=False)
        if value.shape!=(len(ids),2,256) or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('Complete finite restored IQ required')
    return result


def score(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU scorer required')
    ids,paths=reference_handles(args.formal_residual);packets=completed_predictions(args.output,ids)
    # Paired clean and target truth below are scoring references only, after all seven predictions.
    clean=np.load(args.formal_residual/'reference/clean.npy',mmap_mode='r',allow_pickle=False)
    truth=json.loads(args.target_truth.read_text());labels=np.asarray([truth[str(i)]['label'] for i in ids],np.int64)
    rx=np.asarray([str(truth[str(i)]['receiver']) for i in ids]);day=np.asarray([str(truth[str(i)].get('day','unavailable')) for i in ids])
    groups=[('overall','all',np.ones(len(ids),bool))]
    for dim,v in (('TX',labels),('RX',rx),('day',day)):groups.extend((dim,str(k),v==k) for k in np.unique(v))
    paths['clean']=(args.formal_residual/'reference/clean.npy',args.formal_residual/'residual_deterministic/prediction/clean.npy')
    records=[];waveform=[];means={}
    clean_accuracy=[float((packets['clean']['identity'][:,i]==labels).mean()*100) for i in range(3)]
    for view in VIEWS:
        values=dict(identity=np.load(paths[view][0],mmap_mode='r',allow_pickle=False),parent_R03=np.load(paths[view][1],mmap_mode='r',allow_pickle=False),
            refined=np.load(args.output/'prediction'/(view+'_refined_iq.npy'),mmap_mode='r',allow_pickle=False))
        waveform.append(dict(view=view,readouts={k:waveform_gap(v,clean) for k,v in values.items()}))
        for column,seed in enumerate(SEEDS):
            for name in READOUTS:
                for baseline in ('identity','parent_R03'):
                    for dimension,v,mask in groups:
                        records.append(dict(view=view,F_seed=seed,readout=name,baseline=baseline,dimension=dimension,value=v,
                            **classification_statistics(labels[mask],packets[view][name][mask,column],packets[view][baseline][mask,column])))
        means[view]={}
        for name in READOUTS:
            rows=[r for r in records if r['view']==view and r['readout']==name and r['baseline']=='identity' and r['dimension']=='overall']
            accuracy=[r['accuracy'] for r in rows]
            means[view][name]=dict(accuracy_mean=float(np.mean(accuracy)),accuracy_std=float(np.std(accuracy,ddof=1)),
                macro_f1_mean=float(np.mean([r['macro_f1'] for r in rows])),gap_to_clean_pp=float(np.mean(clean_accuracy)-np.mean(accuracy)),
                repaired_sum=sum(r['repaired'] for r in rows),damaged_sum=sum(r['damaged'] for r in rows),net_repaired_sum=sum(r['net_repaired'] for r in rows))
    write_json(args.output/'summary.json',dict(status='ANALYZED',views=list(VIEWS),count=COUNT,F_seeds=list(SEEDS),restorer_seed=d.MODEL_SEED,
        truth_last=True,training_updates=0,head_fit_updates=0,records=records,means=means,waveform=waveform,
        goal_lowurban70=means['practical_low_urban']['refined']['accuracy_mean']>=70.,
        goal_other5_near_clean2pp=all(means[v]['refined']['gap_to_clean_pp']<=2 for v in SCENES if v!='practical_low_urban'),
        other5_gap_is_agent_interpretation=True,no_target_feedback=True,goal_complete=False))


def run(args):
    if args.output.exists():raise FileExistsError('Immutable target stage output required')
    source_check(args);args.output.mkdir();smoke(args)
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,GPU=0))
    model,frozen=acquire(args);predict(args,model,frozen);del model,frozen;torch.cuda.empty_cache()
    subprocess.run([sys.executable,'-u','-m','experiments.pamf_posterior_refine.target','score','--output',str(args.output),
        '--formal-residual',str(args.formal_residual),'--target-truth',str(args.target_truth)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
    write_json(args.output/'complete.json',dict(status='ANALYZED',count=COUNT,views=list(VIEWS),truth_last=True,training_updates=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','score'))
    for key in ('output','formal-residual','target-truth'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('source-run','source-cache','reader-root'):p.add_argument('--'+key,type=Path)
    p.add_argument('--expected-commit');args=p.parse_args()
    for k,v in vars(args).items():
        if isinstance(v,Path):setattr(args,k,v.resolve())
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='score':score(args);return
    if any(getattr(args,k) is None for k in ('source_run','source_cache','reader_root','expected_commit')):p.error('Fixed source checkpoint/reader/commit required')
    if code_commit()!=args.expected_commit or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Pinned commit andGPU0 required')
    run(args)


if __name__=='__main__':main()

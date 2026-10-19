"""Frozen two-arm full clean target comparison through all three original F50s."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT/'code') not in sys.path:sys.path.append(str(ROOT/'code'))
import numpy as np
import torch
from . import design as d
from .models import load_checkpoint
from experiments.pamf_consensus_target.runner import reference_handles,COUNT
from experiments.pamf_fcrossrx_readout.frozen import load_reader,REFERENCE_RUN
from experiments.pamf_change_preserve.model import relative_change
from experiments.pamf_restoration.runner import array_tensor,tensor_array,code_commit
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.dispatcher import capacity
from experiments.pamf_restoration.frozen import classification_statistics

RUN='20261019-phase1-restoration-feature-target-clean-m3-r01'
SOURCE_COMMIT='d896e0552f1f17e7b421fb7d94d45af9256899b4'
SEEDS=(2026092701,2026092702,2026092703)
READOUTS=('identity','parent_R03','refined')


def configuration():
    return dict(run_id=RUN,source_run=d.RUN,source_commit=SOURCE_COMMIT,arms=list(d.ARMS),views=['clean'],count=COUNT,
        F_run=REFERENCE_RUN,F_seeds=list(SEEDS),restorer_seed=2026092701,restorer_checkpoint='Both fixed finalE20',
        original_default_heads=True,training_updates=0,head_fit_updates=0,adaptation_updates=0,
        decisions='Independent perpacket defaulthead argmax;no ensemblevote/truebatchcounts/globalassignment',
        primary='PerFseed accuracy,pairedrepair/damage/net;mean3F accuracychange vsidentity andmatchedcontrol',
        source_eligibility='Preregis lowurban+.5pp/clean-.1pp/NMSE<=1.05control passed before target access',
        no_target_feedback=True,GPUs=d.GPUS,CPU_threads=2,CPU_scorers=1,
        scope='Full168000clean;one restorationseed with3frozenFreadouts,not3restorertrainingseeds;exposedbenchmark',
        stop_rule='Invalid source/parents/code,collision,nonfinite,incompleteprediction:abort,preserve,noautoretry')


def source_check(source):
    launch=json.loads((source/'launch.json').read_text());done=json.loads((source/'complete.json').read_text())
    decision=json.loads((source/'source_decision.json').read_text())
    if (source.name!=d.RUN or launch.get('commit')!=SOURCE_COMMIT or done.get('status')!='ANALYZED' or
        done.get('arms')!=list(d.ARMS) or done.get('target_reads')!=0 or not done.get('source_only') or
        not decision.get('qualified') or decision.get('target_reads')!=0):
        raise ValueError('Completed source-qualified fixed comparison required')
    for arm in d.ARMS:
        row=json.loads((source/arm/'training_complete.json').read_text())
        if (row.get('epochs')!=20 or row.get('updates')!=4940 or
            any(row.get(k)!=0 for k in ('F_optimizer_updates','head_fit_updates','coarse_updates','TX_labels_read','target_reads'))):
            raise ValueError('Matched source budget/permissions differ')


def readers(args,device):
    return {seed:load_reader(args.reader_root/f'CROSS_RX-s{seed}'/'source',args.source_cache,device,seed=seed) for seed in SEEDS}


@torch.no_grad()
def smoke(args):
    models=readers(args,'cpu');x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu')
    for arm in d.ARMS:
        model,item=load_checkpoint(args.source_run/arm/'final.pt',arm,'cpu')
        if (item['epoch'],item['updates'])!=(20,4940):raise ValueError('Fixed final source model required')
        output=model(x)
        if not torch.isfinite(output).all():raise FloatingPointError('Real source model smoke failed')
        for seed,reader in models.items():
            scores=reader(output)
            if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Real original-defaulthead smoke failed')
    write_json(args.output/'smoke.json',dict(status='PASS',real_source_samples=2,arms=list(d.ARMS),F_seeds=list(SEEDS),
        actual_new_checkpoints=True,original_default_heads=True,TX_labels_read=0,target_reads=0,updates=0))


def acquire(args):
    import fcntl
    while True:
        with (args.output.parent/'receiver_residual_capacity4.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=capacity()[d.GPUS[args.arm]]
            if state['free_mb']>=12000 and len(state['pids'])<2:
                model,item=load_checkpoint(args.source_run/args.arm/'final.pt',args.arm,'cuda:0')
                if (item['epoch'],item['updates'])!=(20,4940):raise ValueError('Fixed final source checkpoint required')
                frozen=readers(args,'cuda:0');torch.cuda.synchronize();return model,frozen
        time.sleep(5)


@torch.no_grad()
def predict(args,model,frozen):
    ids,received,parent=reference_handles(args.formal_residual)
    row=args.output/args.arm;saved=row/'prediction';saved.mkdir()
    iq=np.lib.format.open_memmap(saved/'clean_refined_iq.npy',mode='w+',dtype=np.float32,shape=(COUNT,2,256))
    decisions={name:np.empty((COUNT,3),np.uint8) for name in READOUTS}
    parent_change=np.empty(COUNT,np.float32);refined_change=np.empty(COUNT,np.float32);started=time.perf_counter()
    for offset in range(0,COUNT,64):
        y=array_tensor(received[offset:offset+64],'cuda:0');p=array_tensor(parent[offset:offset+64],'cuda:0');value=model(y)
        if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite frozen refinement output')
        end=offset+len(y);iq[offset:end]=tensor_array(value)
        for column,seed in enumerate(SEEDS):
            for name,x in zip(READOUTS,(y,p,value)):
                logits=frozen[seed](x)
                if logits.shape!=(len(y),6) or not torch.isfinite(logits).all():raise ValueError('Invalid original defaulthead scores')
                decisions[name][offset:end,column]=np.asarray(logits.argmax(1).tolist(),np.uint8)
        parent_change[offset:end]=tensor_array(relative_change(y,p));refined_change[offset:end]=tensor_array(relative_change(y,value))
        if end%8192==0 or end==COUNT:
            print(json.dumps(dict(event='TARGET_CLEAN_PROGRESS',arm=args.arm,count=end,total=COUNT,seconds=time.perf_counter()-started)),flush=True)
    iq.flush()
    with (saved/'clean.npz').open('xb') as stream:
        np.savez_compressed(stream,ids=ids,seeds=np.asarray(SEEDS,np.int64),**decisions,parent_change=parent_change,refined_change=refined_change)
    write_json(saved/'complete.json',dict(status='PREDICTIONS_COMPLETE',views=['clean'],count=COUNT,F_seeds=list(SEEDS),
        readouts=list(READOUTS),TX_truth_reads=0,state_updates=0,seconds=time.perf_counter()-started))


def completed_predictions(row,expected_count=COUNT):
    saved=row/'prediction';done=json.loads((saved/'complete.json').read_text())
    required=dict(status='PREDICTIONS_COMPLETE',views=['clean'],count=expected_count,F_seeds=list(SEEDS),
        readouts=list(READOUTS),TX_truth_reads=0,state_updates=0)
    if any(done.get(k)!=v for k,v in required.items()):raise ValueError('All three F predictions required before truth access')
    with np.load(saved/'clean.npz',allow_pickle=False) as packet:
        if set(packet.files)!={'ids','seeds',*READOUTS,'parent_change','refined_change'}:raise ValueError('Prediction fields differ')
        values={k:packet[k].copy() for k in packet.files}
    if values['ids'].shape!=(expected_count,) or values['seeds'].tolist()!=list(SEEDS):raise ValueError('Declared target coverage differs')
    for name in READOUTS:
        v=values[name]
        if v.shape!=(expected_count,3) or v.dtype.kind not in 'iu' or (v<0).any() or (v>5).any():raise ValueError('Invalid perFseed predictions')
    for name in ('parent_change','refined_change'):
        if values[name].shape!=(expected_count,) or not np.isfinite(values[name]).all() or (values[name]<0).any():raise ValueError('Invalid IQ change evidence')
    iq=np.load(saved/'clean_refined_iq.npy',mmap_mode='r',allow_pickle=False)
    if iq.shape!=(expected_count,2,256) or not np.isfinite(iq).all():raise ValueError('Incomplete restored target IQ')
    return values


def score(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU scorer required')
    row=args.output/args.arm;packet=completed_predictions(row)
    # All registered views, classifiers and restored IQ are fixed before this opening.
    truth=json.loads(args.target_truth.read_text());ids=packet['ids']
    labels=np.asarray([truth[str(i)]['label'] for i in ids],np.int64)
    rx=np.asarray([str(truth[str(i)]['receiver']) for i in ids]);day=np.asarray([str(truth[str(i)].get('day','unavailable')) for i in ids])
    groups=[('overall','all',np.ones(COUNT,bool))]
    for dim,values in (('TX',labels),('RX',rx),('day',day)):
        groups.extend((dim,str(v),values==v) for v in np.unique(values))
    records=[]
    for column,seed in enumerate(SEEDS):
        for name in READOUTS:
            for baseline in ('identity','parent_R03'):
                for dimension,value,mask in groups:
                    records.append(dict(F_seed=seed,readout=name,baseline=baseline,dimension=dimension,value=value,
                        **classification_statistics(labels[mask],packet[name][mask,column],packet[baseline][mask,column])))
    overall=[r for r in records if r['dimension']=='overall' and r['baseline']=='identity']
    means={}
    for name in READOUTS:
        rows=[r for r in overall if r['readout']==name]
        means[name]=dict(n_F_seeds=3,accuracy_mean=float(np.mean([r['accuracy'] for r in rows])),
            accuracy_std=float(np.std([r['accuracy'] for r in rows],ddof=1)),macro_f1_mean=float(np.mean([r['macro_f1'] for r in rows])),
            repaired_sum=sum(r['repaired'] for r in rows),damaged_sum=sum(r['damaged'] for r in rows),net_repaired_sum=sum(r['net_repaired'] for r in rows))
    write_json(row/'summary.json',dict(status='ANALYZED',arm=args.arm,count=COUNT,views=['clean'],F_seeds=list(SEEDS),
        restorer_model_seed=2026092701,truth_last=True,training_updates=0,head_fit_updates=0,records=records,means=means,
        NMSE=dict(identity=0.,parent_R03=float(packet['parent_change'].mean(dtype=np.float64)),refined=float(packet['refined_change'].mean(dtype=np.float64))),
        no_target_feedback=True,limitation=configuration()['scope']))


def worker(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(d.GPUS[args.arm]):raise ValueError('Registered physical GPU required')
    row=args.output/args.arm;row.mkdir()
    try:
        model,frozen=acquire(args)
        write_json(row/'gpu_ready.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,physical_GPU=d.GPUS[args.arm]))
        predict(args,model,frozen);del model,frozen;torch.cuda.empty_cache()
        import fcntl
        with (args.output/'cpu_score.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            subprocess.run([sys.executable,'-X','utf8','-m','experiments.pamf_feature_refine.target','score',
                '--output',str(args.output),'--arm',args.arm,'--target-truth',str(args.target_truth)],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        write_json(row/'complete.json',dict(status='ANALYZED',count=COUNT,F_seeds=list(SEEDS),views=['clean'],training_updates=0,truth_last=True))
    except BaseException as error:
        write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise


def run(args):
    if args.output.exists():raise FileExistsError('Immutable target output required')
    source_check(args.source_run);args.output.mkdir(parents=True)
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,
        owner='codex/root/feature-target-clean'))
    write_json(args.output/'resolved_config.json',configuration());smoke(args)
    logs=args.output/'logs';logs.mkdir();processes={};launches=[]
    for arm in d.ARMS:
        command=[sys.executable,'-u','-m','experiments.pamf_feature_refine.target','worker','--arm',arm]
        for key in ('output','source_run','formal_residual','reader_root','source_cache','target_truth','expected_commit'):
            command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
        with (logs/(arm+'.log')).open('xb') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d.GPUS[arm])),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[arm]=process;launches.append(dict(arm=arm,pid=process.pid,GPU=d.GPUS[arm],argv=command))
    write_json(args.output/'workers.json',dict(workers=launches));codes={arm:p.wait() for arm,p in processes.items()}
    if any(v!=0 for v in codes.values()):
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',exit_codes=codes,healthy_rows_preserved=True,automatic_retry=False));return
    write_json(args.output/'complete.json',dict(status='ANALYZED',arms=list(d.ARMS),count=COUNT,views=['clean'],
        F_seeds=list(SEEDS),restorer_model_seed=2026092701,training_updates=0,truth_last=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','worker','score'))
    for name in ('output','target-truth'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('source-run','formal-residual','source-cache','reader-root'):p.add_argument('--'+name,type=Path)
    p.add_argument('--arm',choices=d.ARMS);p.add_argument('--expected-commit');args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='score':
        if args.arm is None:p.error('Declared arm required')
        score(args);return
    if any(getattr(args,k) is None for k in ('source_run','formal_residual','source_cache','reader_root','expected_commit')):p.error('Frozen inputs/parents/commit required')
    if code_commit()!=args.expected_commit:raise ValueError('Pinned Git code/config commit required')
    if args.mode=='run':run(args)
    else:
        if args.arm is None:p.error('Declared arm required')
        worker(args)


if __name__=='__main__':main()

"""Frozen satellite target evaluation: paired-clean IQ gap and original-F identity."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .target import source_check,readers,smoke,acquire,SEEDS,READOUTS
from .target import completed_predictions as clean_predictions
from experiments.pamf_consensus_target.runner import PARENT,PARENT_COMMIT,COUNT
from experiments.pamf_restoration.data import SCENES,write_json
from experiments.pamf_restoration.runner import array_tensor,tensor_array,code_commit
from experiments.pamf_restoration.frozen import classification_statistics

ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-restoration-feature-target-practical-m3-r01'
CLEAN_RUN='20261019-phase1-restoration-feature-target-clean-m3-r01'
BATCH=256


def configuration():
    return dict(run_id=RUN,source_run=d.RUN,arms=list(d.ARMS),views=list(SCENES),count=COUNT,
        F_seeds=list(SEEDS),restorer_seed=d.MODEL_SEED,readouts=list(READOUTS),physical_batch=BATCH,
        checkpoint_selection='Both previously source-qualified fixed finalE20;no new selection',
        training_updates=0,adaptation_updates=0,head_fit_updates=0,original_default_heads=True,
        axis1='Unaligned full/prefix232/tail24 IQ NMSE and log FFT power gap to same physical clean;scoring only',
        axis2='PerFseed trueTX accuracy/MacroF1/repair/damage/net vs same received IQ and parentR03;clean-oracle gap',
        clean_reuse=CLEAN_RUN,clean_inference_access=False,target_truth_inference_access=False,
        source_rule='Existing source comparison passed before target;no retuning from clean target scores',
        target_lowurban_accuracy_min=70.,other_scene_clean_gap_pp=2.,other_scene_gap_is_agent_interpretation=True,
        GPUS=d.GPUS,CPU_threads=2,CPU_scorers=1,all_owner_cap_per_GPU=2,min_free_MB=12000,
        claim='Six fixed simulated practical views,168000 physicals,1restorer seed/3Freadouts;exposed benchmark,not inorbit',
        stop_rule='Input/commit/checkpoint mismatch,collision,nonfinite,incompleteprediction:abort,preserve,noautoretry')


def reference_handles(root,expected_count=COUNT):
    root=Path(root)
    launch=json.loads((root/'launch.json').read_text());done=json.loads((root/'complete.json').read_text())
    pred=json.loads((root/'residual_deterministic/prediction/complete.json').read_text())
    manifest=json.loads((root/'reference/manifest.json').read_text())
    if (root.name!=PARENT or launch.get('commit')!=PARENT_COMMIT or done.get('status')!='ANALYZED' or
        done.get('count')!=expected_count or done.get('training_updates')!=0 or done.get('adaptation_updates')!=0 or
        not done.get('truth_last') or pred.get('status')!='PREDICTIONS_COMPLETE' or
        pred.get('candidate')!='residual_deterministic' or pred.get('count')!=expected_count or
        pred.get('views')!=['clean',*SCENES] or pred.get('TX_truth_reads')!=0 or pred.get('state_updates')!=0 or
        manifest.get('status')!='FORMAL_REFERENCE_READY' or manifest.get('count')!=expected_count or
        manifest.get('views')!=['clean',*SCENES] or not manifest.get('original_IQ_bytes_reused') or
        manifest.get('TX_truth_reads')!=0):raise ValueError('Unchanged completed formal six-view handles required')
    ids=np.load(root/'reference/physical_ids.npy',allow_pickle=False)
    if ids.shape!=(expected_count,):raise ValueError('Declared target physical coverage differs')
    # Only current received and fixed parent IQ enter inference; paired clean is not opened here.
    paths={view:(root/'reference'/(view+'.npy'),root/'residual_deterministic/prediction'/(view+'.npy')) for view in SCENES}
    for pair in paths.values():
        for path in pair:
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            if value.shape!=(expected_count,2,256) or value.dtype!=np.float32:raise ValueError('Original target IQ layout differs')
    return ids,paths


@torch.no_grad()
def predict(args,model,frozen):
    ids,paths=reference_handles(args.formal_residual)
    saved=args.output/args.arm/'prediction';saved.mkdir();elapsed={}
    for view in SCENES:
        observed,parent=[np.load(path,mmap_mode='r',allow_pickle=False) for path in paths[view]]
        iq=np.lib.format.open_memmap(saved/(view+'_refined_iq.npy'),mode='w+',dtype=np.float32,shape=(COUNT,2,256))
        decisions={name:np.empty((COUNT,3),np.uint8) for name in READOUTS};started=time.perf_counter()
        for offset in range(0,COUNT,BATCH):
            y=array_tensor(observed[offset:offset+BATCH],'cuda:0');p=array_tensor(parent[offset:offset+BATCH],'cuda:0')
            value=model(y);end=offset+len(y)
            if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite frozen restoration')
            iq[offset:end]=tensor_array(value)
            for column,seed in enumerate(SEEDS):
                for name,x in zip(READOUTS,(y,p,value)):
                    logits=frozen[seed](x)
                    if logits.shape!=(len(y),6) or not torch.isfinite(logits).all():raise ValueError('Invalid defaulthead scores')
                    decisions[name][offset:end,column]=np.asarray(logits.argmax(1).tolist(),np.uint8)
            if end%16384==0 or end==COUNT:
                print(json.dumps(dict(event='PRACTICAL_PROGRESS',arm=args.arm,view=view,count=end,total=COUNT,
                    seconds=time.perf_counter()-started)),flush=True)
        iq.flush();del iq
        with (saved/(view+'.npz')).open('xb') as stream:
            np.savez_compressed(stream,ids=ids,seeds=np.asarray(SEEDS,np.int64),**decisions)
        elapsed[view]=time.perf_counter()-started
    write_json(saved/'complete.json',dict(status='PREDICTIONS_COMPLETE',views=list(SCENES),count=COUNT,
        F_seeds=list(SEEDS),readouts=list(READOUTS),TX_truth_reads=0,paired_clean_reads=0,state_updates=0,per_view_seconds=elapsed))


def completed_predictions(row,ids):
    saved=row/'prediction';done=json.loads((saved/'complete.json').read_text())
    required=dict(status='PREDICTIONS_COMPLETE',views=list(SCENES),count=len(ids),F_seeds=list(SEEDS),
        readouts=list(READOUTS),TX_truth_reads=0,paired_clean_reads=0,state_updates=0)
    if any(done.get(k)!=v for k,v in required.items()):raise ValueError('All six views and three F predictions required before clean/truth')
    packets={}
    for view in SCENES:
        with np.load(saved/(view+'.npz'),allow_pickle=False) as packet:
            if set(packet.files)!={'ids','seeds',*READOUTS} or not np.array_equal(packet['ids'],ids) or packet['seeds'].tolist()!=list(SEEDS):
                raise ValueError('Same physical IDs and original F seeds required')
            packets[view]={}
            for name in READOUTS:
                value=packet[name]
                if value.shape!=(len(ids),3) or value.dtype.kind not in 'iu' or (value<0).any() or (value>5).any():
                    raise ValueError('Complete legal defaulthead predictions required')
                packets[view][name]=value.copy()
        iq=np.load(saved/(view+'_refined_iq.npy'),mmap_mode='r',allow_pickle=False)
        if iq.shape!=(len(ids),2,256) or iq.dtype!=np.float32 or not np.isfinite(iq).all():raise ValueError('Complete finite restored IQ required')
    return packets


def waveform_gap(estimate,clean,batch=2048):
    if estimate.shape!=clean.shape or estimate.ndim!=3 or estimate.shape[1:]!=(2,256):raise ValueError('Matched physical IQ required')
    values={k:[] for k in ('full_nmse','prefix232_nmse','tail24_nmse','log_power_mse')}
    for offset in range(0,len(clean),batch):
        x=np.asarray(clean[offset:offset+batch],np.float64);e=np.asarray(estimate[offset:offset+batch],np.float64)
        if not np.isfinite(x).all() or not np.isfinite(e).all():raise ValueError('Finite paired waveform required')
        for name,part in (('full_nmse',slice(None)),('prefix232_nmse',slice(0,232)),('tail24_nmse',slice(232,None))):
            error=np.square(e[:,:,part]-x[:,:,part]).mean((1,2))/np.square(x[:,:,part]).mean((1,2)).clip(1e-4)
            values[name].append(error)
        z=e[:,0]+1j*e[:,1];target=x[:,0]+1j*x[:,1]
        gap=np.square(np.log1p(np.abs(np.fft.fft(z,norm='ortho'))**2)-np.log1p(np.abs(np.fft.fft(target,norm='ortho'))**2)).mean(1)
        values['log_power_mse'].append(gap)
    return {k:dict(mean=float((a:=np.concatenate(v)).mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9))) for k,v in values.items()}


def score(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU scorer required')
    ids,paths=reference_handles(args.formal_residual);row=args.output/args.arm
    packets=completed_predictions(row,ids)
    if args.clean_run.name!=CLEAN_RUN:raise ValueError('Previously fixed full clean comparison required')
    clean_packet=clean_predictions(args.clean_run/args.arm)
    if not np.array_equal(clean_packet['ids'],ids):raise ValueError('Clean reference physical IDs differ')
    # All six restored IQ and 3F decisions have been fixed before paired clean or truth is opened.
    clean=np.load(args.formal_residual/'reference/clean.npy',mmap_mode='r',allow_pickle=False)
    if clean.shape!=(len(ids),2,256) or clean.dtype!=np.float32:raise ValueError('Original scoring-only clean required')
    truth=json.loads(args.target_truth.read_text());labels=np.asarray([truth[str(i)]['label'] for i in ids],np.int64)
    rx=np.asarray([str(truth[str(i)]['receiver']) for i in ids]);day=np.asarray([str(truth[str(i)].get('day','unavailable')) for i in ids])
    groups=[('overall','all',np.ones(len(ids),bool))]
    for dim,v in (('TX',labels),('RX',rx),('day',day)):groups.extend((dim,str(k),v==k) for k in np.unique(v))
    records=[];waveform=[];means={}
    clean_accuracy=[float((clean_packet['identity'][:,c]==labels).mean()*100) for c in range(3)]
    for view in SCENES:
        values=dict(identity=np.load(paths[view][0],mmap_mode='r',allow_pickle=False),
            parent_R03=np.load(paths[view][1],mmap_mode='r',allow_pickle=False),
            refined=np.load(row/'prediction'/(view+'_refined_iq.npy'),mmap_mode='r',allow_pickle=False))
        waveform.append(dict(view=view,readouts={name:waveform_gap(v,clean) for name,v in values.items()}))
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
                macro_f1_mean=float(np.mean([r['macro_f1'] for r in rows])),clean_accuracy_mean=float(np.mean(clean_accuracy)),
                gap_to_clean_pp=float(np.mean(clean_accuracy)-np.mean(accuracy)),repaired_sum=sum(r['repaired'] for r in rows),
                damaged_sum=sum(r['damaged'] for r in rows),net_repaired_sum=sum(r['net_repaired'] for r in rows))
    write_json(row/'summary.json',dict(status='ANALYZED',arm=args.arm,count=len(ids),views=list(SCENES),F_seeds=list(SEEDS),
        restorer_model_seed=d.MODEL_SEED,truth_last=True,paired_clean_scoring_only=True,training_updates=0,head_fit_updates=0,
        records=records,means=means,waveform=waveform,clean_reference_accuracy_perF=clean_accuracy,
        no_target_feedback=True,limitation=configuration()['claim']))


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
            command=[sys.executable,'-u','-m','experiments.pamf_feature_refine.practical_target','score','--arm',args.arm]
            for key in ('output','formal_residual','clean_run','target_truth'):command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        write_json(row/'complete.json',dict(status='ANALYZED',views=list(SCENES),count=COUNT,training_updates=0,truth_last=True))
    except BaseException as error:
        write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise


def run(args):
    if args.output.exists():raise FileExistsError('Immutable new target output required')
    source_check(args.source_run)
    if args.output.name!=RUN or args.clean_run.name!=CLEAN_RUN:raise ValueError('Registered target inputs/outputs required')
    args.output.mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,
        owner='codex/root/feature-target-practical'))
    write_json(args.output/'resolved_config.json',configuration());smoke(args)
    logs=args.output/'logs';logs.mkdir();processes={};launches=[]
    for arm in d.ARMS:
        command=[sys.executable,'-u','-m','experiments.pamf_feature_refine.practical_target','worker','--arm',arm]
        for key in ('output','source_run','formal_residual','clean_run','reader_root','source_cache','target_truth','expected_commit'):
            command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
        with (logs/(arm+'.log')).open('xb') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d.GPUS[arm])),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[arm]=process;launches.append(dict(arm=arm,pid=process.pid,GPU=d.GPUS[arm],argv=command))
    write_json(args.output/'workers.json',dict(workers=launches));codes={arm:p.wait() for arm,p in processes.items()}
    if any(v!=0 for v in codes.values()):
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',exit_codes=codes,healthy_rows_preserved=True,automatic_retry=False));return
    write_json(args.output/'complete.json',dict(status='ANALYZED',arms=list(d.ARMS),views=list(SCENES),count=COUNT,
        F_seeds=list(SEEDS),restorer_model_seed=d.MODEL_SEED,training_updates=0,truth_last=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','worker','score'))
    for key in ('output','formal-residual','clean-run','target-truth'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('source-run','source-cache','reader-root'):p.add_argument('--'+key,type=Path)
    p.add_argument('--arm',choices=d.ARMS);p.add_argument('--expected-commit');args=p.parse_args()
    for k,v in vars(args).items():
        if isinstance(v,Path):setattr(args,k,v.resolve())
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='score':score(args);return
    if any(getattr(args,k) is None for k in ('source_run','source_cache','reader_root','expected_commit')):p.error('Fixed source/reader/commit required')
    if code_commit()!=args.expected_commit:raise ValueError('Pinned code/config commit required')
    if args.mode=='run':run(args)
    else:worker(args)


if __name__=='__main__':main()

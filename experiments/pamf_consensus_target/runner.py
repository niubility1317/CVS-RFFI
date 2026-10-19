import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from experiments.pamf_consensus_preserve.model import select_iq
from experiments.pamf_quality_preserve.runner import load_checkpoint
from experiments.pamf_change_preserve.runner import load_restorer
from experiments.pamf_change_preserve.model import relative_change
from experiments.pamf_fcrossrx_readout.frozen import load_reader,REFERENCE_RUN
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.runner import ROOT,array_tensor,tensor_array,code_commit
from experiments.pamf_restoration.dispatcher import capacity
from experiments.pamf_restoration.frozen import classification_statistics

RUN='20261019-phase1-restoration-consensus-target-clean-s2026092701-r01'
PARENT='20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05'
PARENT_COMMIT='9522a4edb63f30293542de7326ce1e53aa816ed5'
COUNT=168000
MODES=('identity','R03','consensus')


def configuration():
    return dict(run_id=RUN,views=['clean'],count=COUNT,readouts=list(MODES),F_run=REFERENCE_RUN,F_seed=2026092701,
        default_head=True,training_updates=0,head_fit_updates=0,quality='Fixed finalE20/4940',
        policy='Restore iff frozen qualitylogit>=0 AND relativechange>.02;no new fitting',
        primary='Paired target clean TX accuracy change vs identity,repairs minus damages',
        secondary='MacroF1,perTX/RX/day,cleanNMSE,preservation count',
        source_status='Consensus source clean decisions unchanged,but2IQchanged,NMSE0rulefailed;negativeconfirmation retained',
        target_feedback_to_training_selection=False,physical_GPU=1,CPU_threads=2,
        scope='Full168000 clean,one fixed F seed;exposed engineering benchmark,not independentblind/inorbit',
        stop_rule='Invalid handles/parents/code,collision,nonfinite,incompleteprediction:abort,preserve,noautoretry')


def reference_handles(root,expected_count=COUNT):
    root=Path(root)
    launch=json.loads((root/'launch.json').read_text());done=json.loads((root/'complete.json').read_text())
    prediction=json.loads((root/'residual_deterministic/prediction/complete.json').read_text())
    manifest=json.loads((root/'reference/manifest.json').read_text())
    if (root.name!=PARENT or launch['commit']!=PARENT_COMMIT or done['status']!='ANALYZED' or
        done['count']!=expected_count or done['training_updates']!=0 or done['adaptation_updates']!=0 or not done['truth_last'] or
        prediction['status']!='PREDICTIONS_COMPLETE' or prediction['candidate']!='residual_deterministic' or
        prediction['count']!=expected_count or prediction['TX_truth_reads']!=0 or prediction['state_updates']!=0 or
        manifest['status']!='FORMAL_REFERENCE_READY' or manifest['count']!=expected_count or
        not manifest['original_IQ_bytes_reused'] or manifest['TX_truth_reads']!=0):
        raise ValueError('Completed unchanged formal R03/clean handles required')
    ids=np.load(root/'reference/physical_ids.npy',allow_pickle=False)
    if ids.shape!=(expected_count,):raise ValueError('Declared target physical coverage differs')
    arrays=[np.load(root/path,mmap_mode='r',allow_pickle=False) for path in
            ('reference/clean.npy','residual_deterministic/prediction/clean.npy')]
    if any(a.shape!=(expected_count,2,256) or a.dtype!=np.float32 for a in arrays):
        raise ValueError('Declared clean target IQ layout differs')
    return ids,*arrays


def smoke(args):
    gate,item=load_checkpoint(args.quality_checkpoint,'cpu')
    if (item['epoch'],item['updates'])!=(20,4940):raise ValueError('Fixed quality final required')
    parent=load_restorer(args.restorer_checkpoint);reader=load_reader(args.reader_folder,args.source_cache,'cpu')
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu')
    with torch.no_grad():
        restored=parent(x);selected,_=select_iq(x,restored,gate(x))
        if any(not torch.isfinite(reader(v)).all() for v in (x,restored,selected)):
            raise ValueError('Real parent/defaulthead no-query smoke failed')
    write_json(args.output/'smoke.json',dict(status='PASS',real_source_samples=2,actual_frozen_parents=True,
        original_defaulthead=True,TX_labels_read=0,target_reads=0,updates=0))


@torch.no_grad()
def predict(args):
    ids,received,restored=reference_handles(args.formal_residual)
    gate,_=load_checkpoint(args.quality_checkpoint,'cuda:0');gate.requires_grad_(False)
    reader=load_reader(args.reader_folder,args.source_cache,'cuda:0')
    saved={k:np.empty(COUNT,np.uint8) for k in MODES}
    keep_all=np.empty(COUNT,bool);q_all=np.empty(COUNT,np.float32);logit_all=np.empty(COUNT,np.float32)
    started=time.perf_counter()
    for offset in range(0,COUNT,64):
        y=array_tensor(received[offset:offset+64],'cuda:0');r=array_tensor(restored[offset:offset+64],'cuda:0')
        logits=gate(y);output,keep=select_iq(y,r,logits);end=offset+len(y)
        for name,value in zip(MODES,(y,r,output)):
            scores=reader(value)
            if scores.shape!=(len(y),6) or not torch.isfinite(scores).all():raise ValueError('Invalid original head prediction')
            saved[name][offset:end]=np.asarray(scores.argmax(1).tolist(),np.uint8)
        keep_all[offset:end]=np.asarray(keep.tolist(),bool);q_all[offset:end]=tensor_array(relative_change(y,r))
        logit_all[offset:end]=tensor_array(logits)
        if end%8192==0 or end==COUNT:print(json.dumps(dict(event='CLEAN_TARGET_PROGRESS',count=end,total=COUNT,seconds=time.perf_counter()-started)),flush=True)
    path=args.output/'prediction';path.mkdir()
    with (path/'clean.npz').open('xb') as f:
        np.savez_compressed(f,ids=ids,**saved,keep=keep_all,change=q_all,damage_logit=logit_all)
    write_json(path/'complete.json',dict(status='PREDICTIONS_COMPLETE',views=['clean'],count=COUNT,
        readouts=list(MODES),TX_truth_reads=0,state_updates=0,seconds=time.perf_counter()-started))


def completed_predictions(output,expected_count=COUNT):
    done=json.loads((output/'prediction/complete.json').read_text())
    if (done.get('status')!='PREDICTIONS_COMPLETE' or done.get('views')!=['clean'] or done.get('count')!=expected_count or
        done.get('readouts')!=list(MODES) or done.get('TX_truth_reads')!=0 or done.get('state_updates')!=0):
        raise ValueError('All declared clean predictions required before truth access')
    with np.load(output/'prediction/clean.npz',allow_pickle=False) as packet:
        if set(packet.files)!={'ids',*MODES,'keep','change','damage_logit'}:raise ValueError('Prediction fields differ')
        values={k:packet[k].copy() for k in packet.files}
    if any(a.shape!=(expected_count,) for a in values.values()):raise ValueError('Incomplete target prediction coverage')
    for name in MODES:
        if values[name].dtype.kind not in 'iu' or (values[name]>5).any() or (values[name]<0).any():
            raise ValueError('Invalid defaulthead class decisions')
    if (values['keep'].dtype.kind!='b' or not np.isfinite(values['change']).all() or (values['change']<0).any() or
        not np.isfinite(values['damage_logit']).all()):raise ValueError('Invalid fixed packet decisions')
    return values


def score(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU scorer required')
    packet=completed_predictions(args.output)
    # This is the first and only TX-truth opening in this run.
    truth=json.loads(args.target_truth.read_text());ids=packet['ids']
    labels=np.asarray([truth[str(i)]['label'] for i in ids],np.int64)
    rx=np.asarray([str(truth[str(i)]['receiver']) for i in ids]);day=np.asarray([str(truth[str(i)].get('day','unavailable')) for i in ids])
    records=[]
    for name in MODES:
        for baseline in ('identity','R03'):
            groups=[('overall','all',np.ones(COUNT,bool))]
            for dim,values in (('TX',labels),('RX',rx),('day',day)):
                groups.extend((dim,str(v),values==v) for v in np.unique(values))
            for dimension,value,mask in groups:
                records.append(dict(readout=name,baseline=baseline,dimension=dimension,value=value,
                    **classification_statistics(labels[mask],packet[name][mask],packet[baseline][mask])))
    q=packet['change'];keep=packet['keep']
    write_json(args.output/'summary.json',dict(status='ANALYZED',count=COUNT,views=['clean'],F_seed=2026092701,
        F_run=REFERENCE_RUN,original_default_head=True,training_updates=0,truth_last=True,records=records,
        preserved=int(keep.sum()),changed=COUNT-int(keep.sum()),consensus_decisions_changed=int((packet['consensus']!=packet['identity']).sum()),
        NMSE=dict(identity=0.,R03=float(q.mean(dtype=np.float64)),consensus=float(np.where(keep,0.,q).mean(dtype=np.float64))),
        target_feedback=False,goal_complete=False,limitation=configuration()['scope']))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','score'))
    for name in ('output','target-truth'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('formal-residual','source-cache','reader-folder','restorer-checkpoint','quality-checkpoint'):p.add_argument('--'+name,type=Path)
    p.add_argument('--expected-commit');args=p.parse_args();torch.set_num_threads(2)
    if args.mode=='score':score(args);return
    if any(getattr(args,k) is None for k in ('formal_residual','source_cache','reader_folder','restorer_checkpoint','quality_checkpoint','expected_commit')):p.error('Frozen inputs/parents/commit required')
    if args.output.exists():raise FileExistsError('Immutable new target output required')
    if code_commit()!=args.expected_commit or os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Registered commit/GPU1 required')
    args.output.mkdir(parents=True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,physical_GPU=1))
    write_json(args.output/'resolved_config.json',configuration())
    try:
        smoke(args)
        import fcntl
        while True:
            with (args.output.parent/'receiver_residual_capacity4.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);state=capacity()[1]
                if state['free_mb']>=12000 and len(state['pids'])<2:
                    torch.empty(1,device='cuda:0');torch.cuda.synchronize();break
            time.sleep(5)
        write_json(args.output/'gpu_ready.json',dict(pid=os.getpid(),physical_GPU=1))
        predict(args)
        subprocess.run([sys.executable,'-X','utf8','-m','experiments.pamf_consensus_target.runner','score',
            '--output',str(args.output),'--target-truth',str(args.target_truth)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        write_json(args.output/'complete.json',dict(status='ANALYZED',count=COUNT,views=['clean'],truth_last=True,training_updates=0))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':main()

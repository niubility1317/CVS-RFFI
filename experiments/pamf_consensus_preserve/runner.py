"""One frozen source-only two-signal candidate; all views before CPU truth."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
import torch
from .model import select_iq,CHANGE_THRESHOLD
from experiments.pamf_quality_preserve.runner import load_checkpoint,predict
from experiments.pamf_change_preserve.runner import load_restorer,score
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.runner import ROOT,array_tensor,code_commit
from experiments.pamf_restoration.dispatcher import capacity
from experiments.pamf_fcrossrx_readout.source_diagnostic import VIEWS

RUN='20261019-phase1-restoration-consensus-preserve-source-s2026092701-r01'


def configuration():
    return dict(schema='frozen_consensus_preserve_v1',run_id=RUN,quality_logit_threshold=0.,
        change_threshold=CHANGE_THRESHOLD,policy='Restore only if qualitylogit>=0 AND R03relativechange>.02',
        parameter_origin='QualityGate fixed finalE20;change.02 inherited prior L/U calibration,no new fit',
        training_updates=0,head_fit_updates=0,TX_training_access=False,F_training_access=False,
        source_only=True,target_reads=0,physical_GPU=1,CPU_threads=2,
        source_rule='Allclean IQ unchanged and decisions equal identity;eachpractical accuracy/F1/NMSE no worse than R03',
        stop_rule='Invalid source/parent/code,collision,nonfinite or incompleteprediction:abort,preserve,noautoretry')


def smoke(args):
    gate,item=load_checkpoint(args.quality_checkpoint,'cpu')
    if item['epoch']!=20 or item['updates']!=4940:raise ValueError('Frozen finalE20 quality gate required')
    parent=load_restorer(args.restorer_checkpoint)
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu')
    with torch.no_grad():
        restored=parent(x);logits=gate(x);output,keep=select_iq(x,restored,logits)
        split=torch.cat([select_iq(x[i:i+1],restored[i:i+1],logits[i:i+1])[0] for i in range(2)])
    if not torch.equal(output,split):raise ValueError('Independent packet smoke failed')
    write_json(args.output/'smoke.json',dict(status='PASS',real_source_samples=2,actual_R03_and_quality_weights=True,
        TX_labels_read=0,source_V_reads=0,F_training_access=False,target_reads=0,optimizer_updates=0))


def run(args):
    for name in ('source_cache','parent_cache','reader_folder','restorer_checkpoint','quality_checkpoint','output'):
        setattr(args,name,getattr(args,name).resolve())
    if args.output.exists():raise FileExistsError('Immutable new output required')
    if code_commit()!=args.expected_commit or os.environ.get('CUDA_VISIBLE_DEVICES')!='1':
        raise ValueError('Declared commit and physicalGPU1 required')
    args.output.mkdir(parents=True);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,owner='codex/root/consensus-preserve'))
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
        predict(args,gate_checkpoint=args.quality_checkpoint,selector=select_iq)
        command=[sys.executable,'-X','utf8','-m','experiments.pamf_consensus_preserve.runner','score',
            '--parent-cache',str(args.parent_cache),'--source-cache',str(args.source_cache),'--output',str(args.output)]
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_only=True,target_reads=0,
            training_updates=0,views=list(VIEWS),policy=configuration()['policy']))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),
            traceback=traceback.format_exc(),automatic_retry=False,outputs_preserved=True));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('run','score'))
    for name in ('source-cache','parent-cache','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('reader-folder','restorer-checkpoint','quality-checkpoint'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--expected-commit');args=parser.parse_args()
    if args.mode=='score':score(args.parent_cache/'validation',args.source_cache,args.output)
    else:
        if any(getattr(args,k) is None for k in ('reader_folder','restorer_checkpoint','quality_checkpoint','expected_commit')):
            parser.error('Actual frozen parents and Git commit required')
        run(args)


if __name__=='__main__':main()

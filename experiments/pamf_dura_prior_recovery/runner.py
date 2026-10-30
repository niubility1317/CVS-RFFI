import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from experiments.pamf_dura_prior_refine.runner import (
    paths,read_json,write_json,source_training,load_model,capacity,F_RUN)
from experiments.pamf_dura_prior_refine.model import ARMS
from experiments.pamf_dura_diagnostics.runner import tensor

RUN='20261030-phase1-DURA-prior-anchor-s2026092701-m2-r02'
SOURCE='20261030-phase1-DURA-prior-anchor-s2026092701-m2-r01'
SOURCE_COMMIT='44bb16ed08b29e7e4ee61c21e1e1287d18791c75'


def setup(args):
    p=paths(args.root)
    args.checkpoint_root=args.root/'runs'/SOURCE
    args.checkpoint_commit=SOURCE_COMMIT
    p['output']=args.root/'runs'/RUN
    return p


def source_ready(args,arm):
    row=args.checkpoint_root/arm
    if not (row/'training_complete.json').exists():return False
    done=read_json(row/'training_complete.json')
    expected=dict(status='SOURCE_TRAINED',arm=arm,epoch=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        training_TX_labels_read=0,training_F_access=False,V_reads=0,target_reads=0,teacher_updates=0,head_fit_updates=0,parameters=266951)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Only original complete frozen source rows allowed')
    if not (row/'final.pt').exists():raise ValueError('Source completion without checkpoint')
    return True


def smoke(args,p):
    if not source_ready(args,'continuation'):raise ValueError('Original frozen continuation checkpoint required')
    clean,roles=source_training(p);take=np.flatnonzero(roles=='L_s')[:2]
    received=np.load(p['cache']/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    model=load_model(args.checkpoint_root/'continuation/final.pt','cpu',arm='continuation',commit=SOURCE_COMMIT)
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    with torch.no_grad():
        output=model(tensor(received[take],'cpu'));logits=reader(output)
        if output.shape!=(2,2,256) or logits.shape!=(2,6) or not torch.isfinite(logits).all():
            raise ValueError('Actual frozen source checkpoint/defaulthead smoke failed')
    write_json(p['output']/'smoke.json',dict(status='PASS',real_Ls=2,source_checkpoint=str(args.checkpoint_root/'continuation/final.pt'),
        source_commit=SOURCE_COMMIT,new_optimizer_updates=0,TX_labels_read=0,target_reads=0,F_updates=0))
    print(json.dumps(dict(event='FROZEN_SOURCE_NO_QUERY_SMOKE_PASS')),flush=True)


def command(args,mode,arm=None,role=None):
    argv=[sys.executable,'-u','-m','experiments.pamf_dura_prior_recovery.runner',mode,
          '--root',str(args.root),'--commit',args.commit]
    if arm is not None:argv+=['--arm',arm]
    if role is not None:argv+=['--role',role]
    return argv


def worker(args,p):
    row=p['output']/args.arm;row.mkdir()
    write_json(row/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=os.environ['CUDA_VISIBLE_DEVICES']))
    write_json(row/'checkpoint_reference.json',dict(source_run=SOURCE,source_commit=SOURCE_COMMIT,
        checkpoint=str(args.checkpoint_root/args.arm/'final.pt'),new_training_updates=0,source_epoch=20,source_updates=4940))
    try:
        if not source_ready(args,args.arm):raise ValueError('Frozen source before evaluation required')
        from experiments.pamf_dura_prior_refine.evaluation import profile,predict
        profile(args,p)
        import fcntl
        for role in ('source','target'):
            predict(args,p,role)
            with (p['output']/'score.lock').open('a') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX)
                subprocess.run(command(args,'score',args.arm,role),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        write_json(row/'complete.json',dict(status='ANALYZED',source_run=SOURCE,new_training_updates=0,all_predictions_before_truth=True,target_feedback=False))
    except BaseException as error:
        write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False))
        raise


def dispatch(args,p):
    import fcntl
    if p['output'].exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New immutable root and CPU controller required')
    p['output'].mkdir();(p['output']/'logs').mkdir()
    write_json(p['output']/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,owner='codex/root/dura-prior-evaluation-recovery'))
    write_json(p['output']/'plan.json',dict(run=RUN,arms=list(ARMS),source_run=SOURCE,source_commit=SOURCE_COMMIT,
        new_training_updates=0,fix='Construct models under no_grad; engine still inference_mode/graph/reuse',
        no_target_feedback=True,all_versions_seven_views=True))
    smoke(args,p)
    pending=list(ARMS);active={};failed=[];last=0.
    while pending or active:
        for arm,(process,gpu) in list(active.items()):
            if process.poll() is None:continue
            if process.returncode or not (p['output']/arm/'complete.json').exists():failed.append(dict(arm=arm,exit_code=process.returncode))
            del active[arm]
        with (args.root/'runs/receiver_residual_capacity4.lock').open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            state=capacity();used={gpu for _,gpu in active.values()}
            free=[gpu for gpu,row in state.items() if gpu in range(8) and gpu not in used and row['free_mb']>=12000 and len(row['pids'])<2]
            free.sort(key=lambda gpu:-state[gpu]['free_mb'])
            commands=subprocess.check_output(['ps','-eo','args='],text=True).splitlines()
            goal_workers=sum(any(name in line for name in ('pamf_dura_prior_refine.runner worker','pamf_dura_prior_recovery.runner worker','pamf_dura_direct_train.runner worker')) for line in commands)
            for arm in pending[:]:
                if not free or len(active)>=2 or goal_workers>=2:break
                if not source_ready(args,arm):continue
                gpu=free.pop(0)
                with (p['output']/'logs'/(arm+'.log')).open('xb') as log:
                    process=subprocess.Popen(command(args,'worker',arm),env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                        cwd=Path.cwd(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[arm]=(process,gpu);pending.remove(arm);goal_workers+=1
                write_json(p['output']/('submitted-'+arm+'.json'),dict(pid=process.pid,GPU=gpu,command=command(args,'worker',arm)))
        if time.monotonic()-last>30:
            print(json.dumps(dict(event='RECOVERY_QUEUE',pending=pending,active={a:dict(pid=v[0].pid,GPU=v[1]) for a,v in active.items()},failed=failed)),flush=True)
            last=time.monotonic()
        if pending or active:time.sleep(5)
    write_json(p['output']/'complete.json',dict(status='FAILED_TECHNICAL' if failed else 'ANALYZED',failed=failed,new_training_updates=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description='Zero-training recovery of source-frozen DURA evaluation')
    parser.add_argument('mode',choices=('run','worker','score'));parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--commit',required=True);parser.add_argument('--arm',choices=ARMS);parser.add_argument('--role',choices=('source','target'))
    args=parser.parse_args();args.root=args.root.resolve()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned recovery release required')
    if args.mode!='run' and args.arm is None:parser.error('Fixed arm required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=setup(args)
    if args.mode=='run':dispatch(args,p)
    elif args.mode=='worker':worker(args,p)
    else:
        if args.role is None:parser.error('Fixed score role required')
        from experiments.pamf_dura_prior_refine.evaluation import score
        score(args,p,args.role)


if __name__=='__main__':main()

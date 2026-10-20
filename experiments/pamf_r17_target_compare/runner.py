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

from . import design as d
from .evaluation import predict,score,validate_source
from experiments.pamf_r17_task.runtime import load_checkpoint,validate_teacher_root
from experiments.pamf_r17_task.training import acquire
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.runner import read_json,array_tensor
from experiments.pamf_restoration.data import write_json

ROOT=Path(__file__).resolve().parents[2]
PATH_ARGS=('source_run','teacher_run','parent_action','source_cache','reader_folder','formal_reference','formal_f50','target_truth','output')


def code_commit():
    path=ROOT/'release_commit.txt'
    return path.read_text().strip() if path.exists() else subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def smoke(args):
    validate_source(args.source_run);validate_teacher_root(args.teacher_run)
    cache=args.parent_action/'cache/train';roles=np.load(cache/'roles.npy',allow_pickle=False)
    selected=np.flatnonzero(roles=='L_s')[:2]
    if len(selected)!=2:raise ValueError('Two legal source training IQ required')
    iq=array_tensor(np.load(cache/'clean.npy',mmap_mode='r',allow_pickle=False)[selected],'cpu')
    reader=load_reader(args.reader_folder,args.source_cache,'cpu').eval().requires_grad_(False)
    records={}
    with torch.no_grad():
        for arm in d.ARMS:
            model,payload=load_checkpoint(args.source_run/arm/d.KIND/'final.pt','cpu');model.eval().requires_grad_(False)
            if (payload['arm'],payload['epoch'],payload['updates'])!=(arm,20,4940):raise ValueError('Frozen final source model required')
            output=model(iq);features=reader.features(output);logits=reader.classify_features(features)
            if output.shape!=(2,2,256) or logits.shape!=(2,6) or not all(torch.isfinite(v).all() for v in (output,features,logits)):
                raise ValueError('Real compact checkpoint/original F finite no-query smoke failed')
            if sum(p.numel() for p in model.parameters())!=d.PARAMETERS[arm]:raise ValueError('Actual full compact parameter count differs')
            records[arm]=dict(parameters=d.PARAMETERS[arm],source_epoch=20,source_updates=4940,F_seed=2026092701,output_finite=True)
    write_json(args.output/'smoke.json',dict(status='PASS',real_Ls_IQ=2,arms=records,TX_labels_read=0,V_reads=0,target_reads=0,
        optimizer_updates=0,state_updates=0,head_fit_updates=0))


def command(args,mode,arm):
    result=[sys.executable,'-u','-m','experiments.pamf_r17_target_compare.runner',mode,'--arm',arm]
    for name in PATH_ARGS:result.extend(['--'+name.replace('_','-'),str(getattr(args,name))])
    return result+['--expected-commit',args.expected_commit]


def worker(args):
    import fcntl
    row=args.output/args.arm;row.mkdir()
    token=acquire(args,False);predict(args);del token
    torch.cuda.empty_cache()
    with (args.output/'score.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run(command(args,'score',args.arm),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)


def write_status(path,value):
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value)+'\n',encoding='utf-8');temporary.replace(path)


def run(args):
    if args.output.name!=d.RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Immutable new target comparison root and CPU controller required')
    validate_source(args.source_run);args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(commit=args.expected_commit,pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,
        source_run=d.SOURCE_RUN,source_commit=d.SOURCE_COMMIT,owner='codex/root/R17-fixed-target-comparison'))
    write_json(args.output/'comparison_plan.json',d.configuration())
    try:
        smoke(args);active={};workers=[];failed=[]
        print(json.dumps(dict(event='COMPARISON_SMOKE_PASS',rows=list(d.ARMS))),flush=True)
        for arm in d.ARMS:
            argv=command(args,'worker',arm);log=args.output/'logs'/(arm+'.log')
            with log.open('xb') as stream:
                process=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d.GPUS[arm])),
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            active[arm]=process;workers.append(dict(arm=arm,role='target',pid=process.pid,GPU=d.GPUS[arm],cwd=str(ROOT),argv=argv,log=str(log)))
        write_json(args.output/'workers.json',dict(workers=workers))
        while active:
            for arm,process in list(active.items()):
                if process.poll() is None:continue
                if process.returncode or not (args.output/arm/'complete.json').exists():failed.append(dict(arm=arm,exit_code=process.returncode))
                del active[arm]
            write_status(args.output/'stages.json',dict(stage='target',active={a:p.pid for a,p in active.items()},failed=failed))
            if active:time.sleep(2)
        if failed:raise RuntimeError('Technical evaluation failure;healthy rows preserved/no automaticretry:'+str(failed))
        write_json(args.output/'complete.json',dict(status='ANALYZED',target_rows=list(d.ARMS),new_training_updates=0,target_feedback=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False));raise


def main():
    parser=argparse.ArgumentParser(description='Frozen two-row compact R17 target comparison')
    parser.add_argument('mode',choices=('run','worker','score'));parser.add_argument('--arm',choices=d.ARMS)
    for name in PATH_ARGS:parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    for name in PATH_ARGS:setattr(args,name,getattr(args,name).resolve())
    if args.output.name!=d.RUN or code_commit()!=args.expected_commit:raise ValueError('Pinned comparison code and root required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':run(args)
    elif args.arm is None:parser.error('Registered row required')
    elif args.mode=='worker':worker(args)
    else:score(args)


if __name__=='__main__':main()

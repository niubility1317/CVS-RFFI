import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import torch
from . import design as d
from .runtime import ROOT,load_model,source_inputs
from .models import counterfactuals,precision
from experiments.pamf_unrolled_action_fidelity.models import trajectory as isotropic
from .evaluation import predict,score
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.runner import array_tensor,code_commit
from experiments.pamf_restoration.dispatcher import capacity

PATH_ARGS=('source_cache','parent_run','parent_action','reference_run','covariance_run','reader_folder','output')


def command(args,mode):
    result=[sys.executable,'-u','-m','experiments.pamf_noise_covariance.runner',mode]
    for key in (*PATH_ARGS,'expected_commit'):result+=['--'+key.replace('_','-'),str(getattr(args,key))]
    return result


def smoke(args):
    source_inputs(args);model,parent=load_model(args,'cpu')
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu')
    main=torch.ones((2,256),dtype=torch.complex64);known=Action(torch.ones((2,1),dtype=torch.complex64),main,torch.zeros_like(main))
    variance=torch.full((2,),1e-8);values=counterfactuals(model,x,known,variance)
    base,estimated,_=model.frozen.estimate(x)
    with torch.no_grad():
        torch.testing.assert_close(values['estimated_covariance'],isotropic(model,x,base,estimated,variance)[0],atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(values['known_covariance'],isotropic(model,x,base,known,variance)[0],atol=2e-6,rtol=2e-5)
        probe=torch.complex(x[:,0],x[:,1]);assert torch.isfinite(precision(probe,torch.full((2,),.01),torch.full_like(probe,.1+.05j))).all()
    reader=load_reader(args.reader_folder,args.source_cache,'cpu').requires_grad_(False).eval()
    with torch.no_grad():
        if any(not torch.isfinite(reader(value)).all() for value in values.values()):raise ValueError('OriginalF source checkpoint smoke failed')
    write_json(args.output/'smoke.json',dict(status='PASS',real_Ls_samples=2,fixed_full_parent=parent,
        readouts=list(d.ROWS),optimizer_updates=0,TX_truth_reads=0,target_reads=0,head_fit_updates=0,oracle_smoke_only=True))


def run(args):
    import fcntl
    if args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New immutable output and CPUcontroller required')
    args.output.mkdir();write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,owner='codex/root/IQ-noise-covariance',started=time.time()))
    write_json(args.output/'resolved_config.json',d.configuration())
    try:
        smoke(args);process=None
        while process is None:
            with (args.output.parent/d.configuration()['capacity_lock']).open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);state=capacity()
                cards=[gpu for gpu in d.GPUS if len(state[gpu]['pids'])<2 and state[gpu]['free_mb']>=12000]
                cards.sort(key=lambda g:(len(state[g]['pids']),-state[g]['free_mb'],list(d.GPUS).index(g)))
                if cards:
                    gpu=cards[0];cmd=command(args,'predict');log=args.output/'predict.log'
                    with log.open('x',encoding='utf-8') as stream:
                        process=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    write_json(args.output/'worker.json',dict(pid=process.pid,GPU=gpu,cwd=str(ROOT),argv=cmd,log=str(log),started=time.time()))
                    deadline=time.monotonic()+45
                    while process.poll() is None and time.monotonic()<deadline:
                        if process.pid in capacity()[gpu]['pids']:break
                        time.sleep(.5)
                    if process.poll() is None and process.pid not in capacity()[gpu]['pids']:
                        raise RuntimeError('CUDA PID not visible within45s;hold,preserve process,noautomaticretry')
            if process is None:time.sleep(2)
        if process.wait()!=0:raise RuntimeError('Source prediction failed;preserve outputs,noautomaticretry')
        subprocess.run(command(args,'score'),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_only=True,target_reads=0,epochs=0,
            optimizer_updates=0,source_V_state_updates=0,all_predictions_before_truth=True,oracle_not_deployable=True,
            target_promotion=False,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False));raise


def main():
    parser=argparse.ArgumentParser(description='Frozen source IQ covariance counterfactual;no target/training mode')
    parser.add_argument('mode',choices=('run','predict','score'))
    for key in PATH_ARGS:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    for key in PATH_ARGS:setattr(args,key,getattr(args,key).resolve())
    if args.output.name!=d.RUN or code_commit()!=args.expected_commit:raise ValueError('Fixed source diagnostic run/code required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':run(args)
    elif args.mode=='predict':predict(args)
    else:score(args)


if __name__=='__main__':main()

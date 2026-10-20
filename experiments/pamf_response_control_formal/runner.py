import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import torch
from . import design as d
from .runtime import ROOT,load_control,parent_selection,reference,audit_target
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.runner import array_tensor,code_commit,read_json
from experiments.pamf_restoration.dispatcher import capacity,progress
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_uncertainty_restore.evaluation import predict as original_predict,score as original_score
from experiments.pamf_native_formal.runner import assess_goal

PATH_ARGS=('parent_run','unrolled_run','reference_run','source_cache','reader_folder','formal_reference','formal_f50','target_truth','output')


def command(args,mode):
    cmd=[sys.executable,'-u','-m','experiments.pamf_response_control_formal.runner',mode]
    for key in (*PATH_ARGS,'expected_commit'):cmd+=['--'+key.replace('_','-'),str(getattr(args,key))]
    return cmd


@torch.no_grad()
def smoke(args):
    model,payload=load_control(args.output/d.KIND/'final.pt','cpu')
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu')
    reader=load_reader(args.reader_folder,args.source_cache,'cpu').requires_grad_(False).eval()
    output=model(x);logits=reader(output)
    if output.shape!=x.shape or logits.shape!=(2,6) or not torch.isfinite(output).all() or not torch.isfinite(logits).all():
        raise ValueError('Actual R32-control/F50 no-query checkpoint smoke failed')
    write_json(args.output/'smoke.json',dict(status='PASS',candidate=d.CANDIDATE,epoch=payload['epoch'],updates=payload['updates'],
        actual_source_samples=2,TX_truth_reads=0,target_reads=0,new_training_updates=0,head_fit_updates=0,original_default_head=True))


def score(args):
    audit_target(args)
    original_score(args,'target',reference_loader=reference)
    write_json(args.output/'goal_assessment.json',assess_goal(read_json(args.output/'target_confirmation'/d.KIND/'summary.json')))
    write_json(args.output/'target_complete.json',dict(status='ANALYZED',candidate=d.CANDIDATE,new_training_updates=0,
        original_default_head=True,all_predictions_before_truth=True,parent_decisions_modified=False,goal_complete=False))


def queue(args):
    import fcntl
    while True:
        with (args.output.parent/'receiver_residual_capacity4.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=capacity()
            cards=[g for g in d.GPUS if state[g]['free_mb']>=12000 and len(state[g]['pids'])<2]
            cards.sort(key=lambda g:(len(state[g]['pids']),-state[g]['free_mb'],list(d.GPUS).index(g)))
            if cards:
                gpu=cards[0];cmd=command(args,'worker');log=args.output/'logs/target.log'
                with log.open('x',encoding='utf-8') as stream:
                    process=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),start_new_session=True)
                record=dict(pid=process.pid,GPU=gpu,cwd=str(ROOT),argv=cmd,log=str(log),started=time.time())
                write_json(args.output/'worker.json',record);deadline=time.monotonic()+45
                while process.poll() is None and time.monotonic()<deadline:
                    if process.pid in capacity()[gpu]['pids']:break
                    time.sleep(.5)
                if process.poll() is None and process.pid not in capacity()[gpu]['pids']:
                    raise RuntimeError('CUDA PID not visible within45s;preserve process;no retry')
                break
        progress(args.output/'queue_state.json',dict(status='WAITING_CAPACITY',time=time.time()));time.sleep(5)
    code=process.wait()
    if code or not (args.output/'target_complete.json').is_file():raise RuntimeError('Technical prediction/scoring failure;no automatic retry')


def run(args):
    if args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New immutable root and CPU controller required')
    # No new root or processes while the original parent is still running.
    decision=parent_selection(args)
    args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,
        owner='codex/root/r32-positive-control-formal',started=time.time()))
    write_json(args.output/'resolved_config.json',d.configuration(args.expected_commit))
    write_json(args.output/'source_decision.json',decision)
    try:
        if decision['qualified']:
            row=args.output/d.KIND;row.mkdir()
            with (args.parent_run/d.CANDIDATE/d.KIND/'final.pt').open('rb') as source,(row/'final.pt').open('xb') as target:
                shutil.copyfileobj(source,target)
            smoke(args);queue(args)
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_qualified=decision['qualified'],
            target_rows=[d.CANDIDATE] if decision['qualified'] else [],source_only=not decision['qualified'],
            new_training_updates=0,parent_decisions_modified=False,target_feedback=False,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),
            outputs_preserved=True,automatic_retry=False));raise


def main():
    parser=argparse.ArgumentParser(description='Source-positive-only frozen R32 paired U-Net control formal evaluation')
    parser.add_argument('mode',choices=('run','worker','score'))
    for key in PATH_ARGS:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args();args.kind=d.KIND
    for key in PATH_ARGS:setattr(args,key,getattr(args,key).resolve())
    if args.output.name!=d.RUN or code_commit()!=args.expected_commit:raise ValueError('Pinned new run/code required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':run(args)
    elif args.mode=='score':score(args)
    else:
        original_predict(args,'target',checkpoint_loader=load_control,reference_loader=reference)
        subprocess.run(command(args,'score'),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)


if __name__=='__main__':main()

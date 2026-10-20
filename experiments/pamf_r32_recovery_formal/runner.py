import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import torch
from . import design as d
from .runtime import ROOT,parent_ready,load_final,reference,audit_target
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.runner import array_tensor,code_commit,read_json
from experiments.pamf_restoration.dispatcher import capacity,progress
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_uncertainty_restore.evaluation import predict as original_predict,score as original_score
from experiments.pamf_native_formal.runner import assess_goal

PATH_ARGS=('parent_run','parent_action','source_cache','reader_folder','formal_reference','formal_f50','target_truth','output')


def row_args(args,candidate):
    return SimpleNamespace(**{**vars(args),'candidate':candidate,'kind':d.KIND,'output':args.output/candidate})


def command(args,mode,candidate):
    cmd=[sys.executable,'-u','-m','experiments.pamf_r32_recovery_formal.runner',mode,'--candidate',candidate]
    for key in (*PATH_ARGS,'expected_commit'):cmd+=['--'+key.replace('_','-'),str(getattr(args,key))]
    return cmd


@torch.no_grad()
def smoke(args):
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu')
    reader=load_reader(args.reader_folder,args.source_cache,'cpu').requires_grad_(False).eval()
    records={}
    for candidate in d.ROWS:
        row=args.output/candidate;model,payload=load_final(row/d.KIND/'final.pt','cpu')
        output=model(x);logits=reader(output)
        if output.shape!=x.shape or logits.shape!=(2,6) or not torch.isfinite(output).all() or not torch.isfinite(logits).all():
            raise ValueError('Actual R32/F50 no-query checkpoint smoke failed')
        records[candidate]=dict(epoch=payload['epoch'],updates=payload['updates'],status='PASS')
    write_json(args.output/'smoke.json',dict(status='PASS',rows=records,actual_source_samples=2,
        TX_truth_reads=0,target_reads=0,new_training_updates=0,head_fit_updates=0,original_default_head=True))


def score(args):
    audit_target(args);original_score(args,'target',reference_loader=reference)
    write_json(args.output/'goal_assessment.json',assess_goal(read_json(args.output/'target_confirmation'/d.KIND/'summary.json')))
    write_json(args.output/'target_complete.json',dict(status='ANALYZED',candidate=args.candidate,role='target',
        all_predictions_before_truth=True,training_TX_F_access=False,new_training_updates=0,reused=False,goal_complete=False))


def queue(args,names):
    import fcntl
    pending=list(names);active={};launches=[];failed=[];completed=[]
    while pending or active:
        for name,(process,record) in list(active.items()):
            if process.poll() is None:continue
            if process.returncode or not (args.output/name/'target_complete.json').is_file():
                failed.append(dict(row=name,exit_code=process.returncode))
            else:completed.append(name)
            del active[name]
        if pending and not failed and len(active)<2:
            with (args.output.parent/'receiver_residual_capacity4.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);state=capacity();reserved={r['GPU'] for _,r in active.values()}
                cards=[g for g in d.GPUS if g not in reserved and state[g]['free_mb']>=12000 and len(state[g]['pids'])<2]
                cards.sort(key=lambda g:(len(state[g]['pids']),-state[g]['free_mb'],list(d.GPUS).index(g)))
                if cards:
                    gpu=cards[0];name=pending.pop(0);cmd=command(args,'worker',name);log=args.output/'logs'/('target-'+name+'.log')
                    with log.open('x',encoding='utf-8') as stream:
                        process=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),start_new_session=True)
                    record=dict(row=name,pid=process.pid,GPU=gpu,cwd=str(ROOT),argv=cmd,log=str(log),started=time.time())
                    launches.append(record);active[name]=(process,record);progress(args.output/'workers.json',dict(rows=launches))
                    deadline=time.monotonic()+45
                    while process.poll() is None and time.monotonic()<deadline:
                        if process.pid in capacity()[gpu]['pids']:break
                        time.sleep(.5)
                    if process.poll() is None and process.pid not in capacity()[gpu]['pids']:
                        raise RuntimeError('CUDA PID not visible within45s;preserve processes;no automatic retry')
        progress(args.output/'queue_state.json',dict(active=[r for _,r in active.values()],pending=pending,completed=completed,failed=failed,time=time.time()))
        if failed and not active:break
        if pending or active:time.sleep(2)
    if failed:raise RuntimeError('Technical failure;healthy peers preserved,pending held,no automatic retry:'+str(failed))


def prepare_sources(args):
    from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference
    from experiments.pamf_uncertainty_restore.evaluation import prediction_arrays
    from experiments.pamf_response_restore import response_evaluation
    ids,_,_=source_reference(args.parent_action/'cache/validation',args.source_cache)
    for candidate in d.ROWS:
        parent=args.parent_run/candidate;row=args.output/candidate
        shutil.copytree(parent/d.KIND/'prediction',row/d.KIND/'prediction')
        shutil.copytree(parent/'source_response',row/'source_response')
        (row/'reference_controls').symlink_to(parent/'reference_controls',target_is_directory=True)
    # All registered standard and response cells validate before any new CPU scorer opens labels.
    for candidate in d.ROWS:
        row=args.output/candidate
        prediction_arrays(row/d.KIND/'prediction',ids,d.KIND,'source')
        response_evaluation.load_predictions(row/'source_response',ids,candidate)
    write_json(args.output/'source_predictions_reused.json',dict(status='ALL_SOURCE_PREDICTIONS_VALIDATED',
        origin_run=d.PARENT,rows=list(d.ROWS),source_count=1800,standard_cells=14,response_cells=24,
        TX_truth_reads=0,source_model_forwards=0,new_training_updates=0))


def source_score(args):
    from experiments.pamf_response_restore.evaluation import score as recovered_score
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU source scorer required')
    saved=read_json(args.output/'source_predictions_reused.json')
    if saved.get('status')!='ALL_SOURCE_PREDICTIONS_VALIDATED' or saved.get('rows')!=list(d.ROWS):
        raise ValueError('All original standard and response source predictions must validate first')
    args.row=args.candidate;recovered_score(args,'source')
    write_json(args.output/args.candidate/'complete.json',dict(status='ANALYZED',candidate=args.candidate,role='source',
        all_predictions_before_truth=True,training_TX_F_access=False,recovered_from=d.PARENT,new_training_updates=0))


def run(args):
    if args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New immutable root and CPU controller required')
    matrix=parent_ready(args)
    args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,
        owner='codex/root/r32-recovery-all-versions-formal',started=time.time()))
    write_json(args.output/'resolved_config.json',d.configuration(args.expected_commit))
    write_json(args.output/'formal_matrix.json',matrix)
    try:
        for candidate in d.ROWS:
            row=args.output/candidate;row.mkdir();(row/d.KIND).mkdir()
            with (args.parent_run/candidate/d.KIND/'final.pt').open('rb') as source,(row/d.KIND/'final.pt').open('xb') as target:
                shutil.copyfileobj(source,target)
        smoke(args);prepare_sources(args)
        for candidate in d.ROWS:
            subprocess.run(command(args,'score_source',candidate),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        write_json(args.output/'source_recovery_complete.json',dict(status='ANALYZED',rows=list(d.ROWS),
            origin_run=d.PARENT,source_model_forwards=0,new_training_updates=0,source_performance_gate=False))
        queue(args,d.ROWS)
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_rows=list(d.ROWS),target_rows=list(d.ROWS),reused_rows=[],new_prediction_rows=list(d.ROWS),
            source_performance_gate=False,new_training_updates=0,parent_decisions_modified=False,target_feedback=False,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),
            outputs_preserved=True,automatic_retry=False));raise


def main():
    parser=argparse.ArgumentParser(description='Formal seven-view testing of both frozen R32 versions regardless of source performance')
    parser.add_argument('mode',choices=('run','worker','score','score_source'))
    parser.add_argument('--candidate',choices=d.ROWS)
    for key in PATH_ARGS:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    for key in PATH_ARGS:setattr(args,key,getattr(args,key).resolve())
    if args.output.name!=d.RUN or code_commit()!=args.expected_commit:raise ValueError('Pinned immutable formal run/code required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':run(args);return
    if args.candidate is None:raise ValueError('Registered candidate required')
    if args.mode=='score_source':source_score(args);return
    effective=row_args(args,args.candidate)
    if args.mode=='score':score(effective)
    else:
        original_predict(effective,'target',checkpoint_loader=load_final,reference_loader=reference)
        subprocess.run(command(args,'score',args.candidate),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)


if __name__=='__main__':main()

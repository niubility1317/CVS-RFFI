"""Source-positive later U-Net audit; historical source qualification remains unchanged."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch
from . import design as d
from .runtime import load,reference,audit_contract,validate_source_matrix
from experiments.pamf_restoration.runner import read_json,array_tensor,tensor_array,code_commit
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.dispatcher import capacity,progress
from experiments.pamf_fcrossrx_readout.frozen import load_reader,REFERENCE_RUN
from experiments.pamf_uncertainty_restore.evaluation import VIEWS,score as original_score,target_baseline
from experiments.pamf_native_formal.runner import assess_goal

ROOT=Path(__file__).resolve().parents[2]
PATH_ARGS=('runs_root','source_cache','reader_folder','formal_reference','formal_f50','target_truth','output')


def configuration(commit):
    return dict(schema='formal_backfill_v1',run=d.RUN,commit=commit,authorization=d.AUTHORIZATION,rows=d.ROWS,
        new_training_updates=0,head_fit_updates=0,source_decisions_modified=False,source_qualification_required=False,source_positive_policy=d.SOURCE_POLICY,
        target_feedback=False,count=168000,views=list(VIEWS),F_run=REFERENCE_RUN,F_seed=2026092701,
        original_default_head=True,batch_size=128,output='One IQ;all PAMF branches derived by original F',
        sampling='Original fixed per-ID noise for R01/02/03;original primary sequential generator for R20/21;no seed selection',
        GPUS=list(d.GPUS),max_own_workers=2,all_owner_cap=2,min_free_MB=12000,
        capacity_lock='receiver_residual_capacity4.lock',truth_last=True,
        stop_rule='On technical failure hold pending, preserve healthy workers and all artifacts; no automatic retry',
        benchmark='Previously exposed formal test;fixed audit,not a new blind test')


def command(args,mode,row):
    result=[sys.executable,'-u','-m','experiments.pamf_positive_unet_restore.runner',mode,'--row',row]
    for key in (*PATH_ARGS,'expected_commit'):result+=['--'+key.replace('_','-'),str(getattr(args,key))]
    return result


@torch.no_grad()
def smoke(args):
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu');ids=np.asarray(['source-smoke-0','source-smoke-1'])
    reader=load_reader(args.reader_folder,args.source_cache,'cpu').requires_grad_(False).eval();rows=[]
    for spec in d.ROWS:
        model,payload=load(args,spec['row_id'],'cpu');out=model(x,ids);logits=reader(out)
        if out.shape!=x.shape or logits.shape!=(2,6) or not torch.isfinite(out).all() or not torch.isfinite(logits).all():
            raise ValueError('Actual original restorer/F50 no-query checkpoint smoke failed')
        if any(p.requires_grad for p in model.parameters()):raise ValueError('Formal models must remain frozen')
        rows.append(dict(row=spec['row_id'],epoch=payload['epoch'],updates=payload['updates'],finite=True))
        del model,payload
    write_json(args.output/'smoke.json',dict(status='PASS',source_samples=2,rows=rows,target_reads=0,TX_labels_read=0,
        optimizer_updates=0,head_fit_updates=0,original_strict_loaders=True))


def scoring_args(args):
    return SimpleNamespace(**dict(vars(args),output=args.output/args.row,kind=d.KIND))


def scoring_reference(adapted,role):
    original=SimpleNamespace(**dict(vars(adapted),output=adapted.output.parent))
    return reference(original,role)


@torch.no_grad()
def predict(args):
    source,ids,_,_=reference(args,'target');spec=d.BY_ID[args.row]
    if 'reuse' in spec:raise ValueError('Comparable completed predictions must be reused')
    model,payload=load(args,args.row,'cuda:0')
    reader=load_reader(args.reader_folder,args.source_cache,'cuda:0').requires_grad_(False).eval()
    row=args.output/args.row/'target_confirmation'/d.KIND;row.mkdir(parents=True)
    folder=row/'prediction';folder.mkdir();timings={};started_all=time.perf_counter()
    for view in VIEWS:
        received=np.load(source/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if received.shape!=(len(ids),2,256):raise ValueError('Original formal IQ shape differs')
        arrays={key:np.lib.format.open_memmap(folder/(view+'.restored.'+key+'.npy'),mode='w+',dtype=np.float32,shape=shape)
            for key,shape in (('iq',received.shape),('features',(len(ids),160)),('logits',(len(ids),6)))}
        start=time.perf_counter()
        for begin in range(0,len(ids),128):
            end=min(begin+128,len(ids));r=array_tensor(received[begin:end],'cuda:0');out=model(r,ids[begin:end])
            features=reader.features(out);logits=reader.classify_features(features)
            if out.shape!=r.shape or features.shape!=(len(r),160) or logits.shape!=(len(r),6):raise ValueError('Current-IQ/default-head output shapes differ')
            for key,value in (('iq',out),('features',features),('logits',logits)):
                if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite frozen formal prediction')
                arrays[key][begin:end]=tensor_array(value)
        for value in arrays.values():value.flush()
        torch.cuda.synchronize();timings[view]=time.perf_counter()-start
        print(json.dumps(dict(event='FORMAL_BACKFILL_VIEW_COMPLETE',row=args.row,view=view,count=len(ids))),flush=True)
    np.save(folder/'physical_ids.npy',ids,allow_pickle=False)
    write_json(folder/'complete.json',dict(status='PREDICTIONS_COMPLETE',kind=d.KIND,role='target',count=len(ids),views=list(VIEWS),
        readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=REFERENCE_RUN,F_seed=2026092701,
        original_default_head=True,per_view_seconds=timings,seconds=time.perf_counter()-started_all,
        authorization=d.AUTHORIZATION,checkpoint=spec,source_decision_modified=False))


def score(args):
    audit_contract(args)
    original_score(scoring_args(args),'target',reference_loader=scoring_reference)
    row=args.output/args.row
    summary=read_json(row/'target_confirmation'/d.KIND/'summary.json')
    write_json(row/'goal_assessment.json',assess_goal(summary,expected_kind=d.KIND))
    write_json(row/'complete.json',dict(status='ANALYZED',role='target',row=args.row,authorization=d.AUTHORIZATION,
        new_training_updates=0,source_decision_modified=False,all_predictions_before_truth=True))


def reuse(args,spec):
    args=SimpleNamespace(**dict(vars(args),row=spec['row_id']))
    _,ids,_,_=reference(args,'target');meta=spec['reuse'];root=args.runs_root/meta['run'];prediction=root/meta['prediction']
    done=read_json(prediction/'complete.json')
    expected=dict(status='PREDICTIONS_COMPLETE',count=168000,views=list(VIEWS),TX_truth_reads=0,state_updates=0,F_run=REFERENCE_RUN,F_seed=2026092701)
    if any(done.get(k)!=v for k,v in expected.items()) or meta['readout'] not in done.get('readouts',[]):raise ValueError('Existing formal result is not comparable')
    if not np.array_equal(np.load(prediction/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Reuse formal IDs differ')
    for view in VIEWS:
        for key,width in (('logits',6),('features',160)):
            a=np.load(prediction/(view+'.'+meta['readout']+'.'+key+'.npy'),mmap_mode='r',allow_pickle=False)
            if a.shape!=(len(ids),width) or not np.isfinite(a).all():raise ValueError('Incomplete comparable formal result')
    summary_path=(root/'summary.json') if meta['run'].endswith('-r09') else prediction.parent/'summary.json'
    summary=read_json(summary_path)
    if (summary.get('status')!='ANALYZED' or summary.get('count')!=168000 or summary.get('original_default_head') is not True
            or summary.get('all_predictions_before_truth') is not True or summary.get('F_seed')!=2026092701):
        raise ValueError('Completed original-default-head truth-last result required')
    if {r['view'] for r in summary['records'] if r['readout']==meta['readout'] and r['dimension']=='overall'}!=set(VIEWS):
        raise ValueError('All seven scored views required for reuse')
    row=args.output/spec['row_id'];row.mkdir()
    write_json(row/'reuse.json',dict(status='REUSED_COMPLETE',source_summary=str(summary_path),prediction=str(prediction),readout=meta['readout'],
        all_seven_views_verified=True,original_default_head=True,new_prediction_calls=0,original_summary=summary))
    write_json(row/'complete.json',dict(status='ANALYZED',row=spec['row_id'],reused=True,new_training_updates=0,new_prediction_calls=0))


def queue(args):
    import fcntl
    pending=list(d.NEW_ROWS);active={};launches=[];completed=[];failed=[]
    while pending or active:
        for name,(process,record) in list(active.items()):
            if process.poll() is None:continue
            if process.returncode or not (args.output/name/'complete.json').is_file():failed.append(dict(row=name,exit_code=process.returncode))
            else:completed.append(name)
            del active[name]
        if pending and not failed and len(active)<2:
            with (args.output.parent/'receiver_residual_capacity4.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);state=capacity();reserved={record['GPU'] for _,record in active.values()}
                cards=[g for g in d.GPUS if g not in reserved and state[g]['free_mb']>=12000 and len(state[g]['pids'])<2]
                cards.sort(key=lambda g:(len(state[g]['pids']),-state[g]['free_mb'],list(d.GPUS).index(g)))
                if cards:
                    gpu=cards[0];name=pending.pop(0);cmd=command(args,'worker',name);log=args.output/'logs'/f'{name}.log'
                    with log.open('x',encoding='utf-8') as stream:
                        process=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,
                            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    record=dict(row=name,pid=process.pid,GPU=gpu,cwd=str(ROOT),argv=cmd,log=str(log),started=time.time())
                    launches.append(record);active[name]=(process,record);progress(args.output/'workers.json',dict(rows=launches))
                    deadline=time.monotonic()+45
                    while process.poll() is None and time.monotonic()<deadline:
                        if process.pid in capacity()[gpu]['pids']:break
                        time.sleep(.5)
                    if process.poll() is None and process.pid not in capacity()[gpu]['pids']:
                        raise RuntimeError('CUDA PID not visible within45s;hold launches,preserve process')
        progress(args.output/'queue_state.json',dict(active=[r for _,r in active.values()],pending=pending,completed=completed,failed=failed,time=time.time()))
        if failed and not active:break
        if pending or active:time.sleep(2)
    if failed:raise RuntimeError('Failed technical row;healthy preserved,pending held,noautomaticretry:'+str(failed))


def run(args):
    if args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New immutable root and CPU controller required')
    args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,owner='codex/root/formal-backfill',started=time.time()))
    write_json(args.output/'resolved_config.json',configuration(args.expected_commit))
    try:
        selection=validate_source_matrix(args)
        write_json(args.output/'source_positive_selection.json',selection)
        smoke(args)
        for spec in d.ROWS:
            if 'reuse' in spec:reuse(args,spec)
        queue(args)
        write_json(args.output/'complete.json',dict(status='ANALYZED',rows=[r['row_id'] for r in d.ROWS],new_rows=list(d.NEW_ROWS),
            new_training_updates=0,source_decisions_modified=False,target_feedback=False,authorization=d.AUTHORIZATION))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False));raise


def main():
    p=argparse.ArgumentParser(description='Source-positive later U-Net formal tests;original source decisions preserved')
    p.add_argument('mode',choices=('run','worker','score'));p.add_argument('--row',choices=tuple(d.BY_ID),default=d.NEW_ROWS[0])
    for key in PATH_ARGS:p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    p.add_argument('--expected-commit',required=True);args=p.parse_args()
    for key in PATH_ARGS:setattr(args,key,getattr(args,key).resolve())
    if args.output.name!=d.RUN or code_commit()!=args.expected_commit:raise ValueError('Pinned audit run/code required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':run(args)
    elif args.mode=='score':score(args)
    else:
        predict(args)
        subprocess.run(command(args,'score',args.row),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)


if __name__=='__main__':main()

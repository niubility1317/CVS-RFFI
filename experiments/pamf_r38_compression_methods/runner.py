import argparse
import gc
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
from .runtime import ROOT,initialize,save_checkpoint,load_checkpoint,validate_parent_root
from .training import train
from .models import objective
from .evaluation import predict,score
from experiments.pamf_iq_joint_compact.runtime import load_checkpoint as load_parent
from experiments.pamf_r17_task.runtime import load_original
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.runner import read_json,array_tensor,code_commit
from experiments.pamf_restoration.data import IQSource,write_json

PATH_ARGS=('teacher_run','compact_parent','parent_action','source_cache','reader_folder','formal_reference','formal_f50','target_truth','output')


def command(args,mode,arm):
    argv=[sys.executable,'-u','-m','experiments.pamf_r38_compression_methods.runner',mode,'--arm',arm]
    for key in PATH_ARGS:argv.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
    return argv+['--expected-commit',args.expected_commit]


def smoke(args):
    validate_parent_root(args.compact_parent)
    cache=args.parent_action/'cache';roles=np.load(cache/'train/roles.npy',allow_pickle=False);take=np.flatnonzero(roles=='L_s')[:2]
    if len(take)!=2:raise ValueError('Actual two L_s paired examples required')
    x=array_tensor(np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)[take],'cpu')
    received=array_tensor(np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)[take],'cpu')
    reader=load_reader(args.reader_folder,args.source_cache,'cpu',seed=d.F_SEED).eval().requires_grad_(False)
    before={k:v.detach().clone() for k,v in reader.state_dict().items()};records={}
    for arm in d.ARMS:
        model,teacher,parent=initialize(arm,args.compact_parent,'cpu');teacher_before={k:v.detach().clone() for k,v in teacher.state_dict().items()}
        with torch.no_grad():desired=teacher.stage_outputs(received)[0]
        outputs=model.stage_outputs(received)[0];loss,_=objective(outputs,desired,x,1);loss.mean().backward()
        groups={'coarse':model.frozen.coarse,'action':model.frozen.estimator,**{'prior'+str(i):v for i,v in enumerate(model.prior)}}
        norms={name:sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None) for name,module in groups.items()}
        if any(not np.isfinite(v) or v<=0 for v in norms.values()):raise ValueError('Every successor component must receive finite nonzero gradients')
        if any(p.grad is not None for p in teacher.parameters()) or any(not torch.equal(v,teacher_before[k]) for k,v in teacher.state_dict().items()):raise ValueError('R38 parent must stay unchanged')
        model.eval();path=args.output/('smoke-'+arm+'.pt');save_checkpoint(path,model,None,0,0,parent);fresh,_=load_checkpoint(path,'cpu')
        with torch.no_grad():
            if not torch.equal(model(received),fresh(received)) or not torch.isfinite(reader(fresh(received))).all():raise ValueError('Standalone strict successor reload/default head failed')
        records[arm]=dict(parameters=d.PARAMETERS[arm],component_gradient_norms=norms,checkpoint_output_exact=True,R38_parent_unchanged=True,
            training_F_access=False,F_knowledge_in_loss=False)
    if any(p.grad is not None for p in reader.parameters()) or any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()):raise ValueError('Original F/default head changed')
    write_json(args.output/'smoke.json',dict(status='PASS',real_Ls_IQ=2,arms=records,TX_labels_read=0,V_reads=0,target_reads=0,
        teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,optimizer_updates=0))


def predecessor_ready(args,arm):
    gpu=d.GPUS[arm];previous=args.output.parent/d.PREDECESSOR_BY_GPU[gpu];saved=previous/'workers.json'
    if not saved.exists():return False
    entries=[entry for entry in read_json(saved)['workers'] if entry['GPU']==gpu]
    if len(entries)!=1:raise ValueError('Exactly one registered preceding GPU worker required')
    entry=entries[0];row=previous/entry['arm']
    if not any((row/name).exists() for name in ('complete.json','failure.json')):return False
    path=Path('/proc')/str(entry['pid'])
    try:
        state=(path/'stat').read_text().rsplit(')',1)[1].split()[0];argv=(path/'cmdline').read_bytes().decode().split(chr(0))
        if state!='Z' and str(previous) in argv and d.PREDECESSOR_MODULE[previous.name] in argv:return False
    except FileNotFoundError:pass
    return True


def score_subprocess(args,role):
    import fcntl
    with (args.output.parent/d.SCORING_RUN/'score.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run(command(args,'score_'+role,args.arm),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)


@torch.no_grad()
def profile(args):
    values=array_tensor(np.load(args.parent_action/'cache/validation/practical_low_urban.npy',mmap_mode='r',allow_pickle=False)[:16],'cuda:0');results={}
    for name in ('original_R17','parent_R38_794529','compact'):
        if name=='original_R17':model,payload=load_original(args.teacher_run/'unroll_unshared'/d.KIND/'final.pt','cpu');budget=(20,4940)
        elif name=='parent_R38_794529':model,payload=load_parent(args.compact_parent/d.COMPACT_PARENT_ROW/d.KIND/'final.pt','cpu');budget=(20,4940)
        else:model,payload=load_checkpoint(args.output/args.arm/d.KIND/'final.pt','cpu');budget=(20,4940)
        if (payload['epoch'],payload['updates'])!=budget:raise ValueError('Fixed source final profile checkpoints required')
        del payload;gc.collect();model.eval().requires_grad_(False).to('cuda:0')
        for _ in range(2):model(values[:1])
        torch.cuda.synchronize();resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
        for value in values:
            torch.cuda.synchronize();started=time.perf_counter();model(value[None]);torch.cuda.synchronize();times.append((time.perf_counter()-started)*1000)
        peak=torch.cuda.max_memory_allocated();results[name]=dict(samples=16,warmup=2,parameters=sum(p.numel() for p in model.parameters()),median_ms=float(np.median(times)),
            p90_ms=float(np.quantile(times,.9)),resident_before_forward_bytes=resident,peak_allocated_bytes=peak,incremental_peak_bytes=peak-resident)
        del model;gc.collect();torch.cuda.empty_cache()
    write_json(args.output/args.arm/'matched_batch1_profile.json',dict(status='PROFILE_COMPLETE',physical_GPU=d.GPUS[args.arm],includes_full_context_and_CG=True,
        checkpoint_payload_on_GPU=False,optimizer_on_GPU=False,source_V_IQ=16,TX_labels_read=0,state_updates=0,results=results))


def worker(args):
    try:
        train(args);torch.cuda.empty_cache();predict(args,'source');profile(args);score_subprocess(args,'source')
        write_json(args.output/args.arm/'source_complete.json',dict(status='ANALYZED',arm=args.arm,role='source',diagnostic_only=True,
            source_selection=False,all_predictions_before_truth=True,target_feedback=False))
        torch.cuda.empty_cache();predict(args,'target');score_subprocess(args,'target')
        write_json(args.output/args.arm/'complete.json',dict(status='ANALYZED',arm=args.arm,role='target',all_predictions_before_truth=True,target_feedback=False))
    except BaseException as error:
        row=args.output/args.arm;row.mkdir(exist_ok=True)
        write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False));raise


def write_status(path,value):
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)


def run(args):
    if args.output.name!=d.RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Immutable new root and CPU controller required')
    validate_parent_root(args.compact_parent);args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,owner='codex/root/R38-compression-methods',
        compact_parent_run=d.COMPACT_PARENT_RUN,compact_parent_commit=d.COMPACT_PARENT_COMMIT))
    write_json(args.output/'resolved_config.json',d.configuration());write_json(args.output/'comparison_plan.json',d.configuration())
    try:
        smoke(args);print(json.dumps(dict(event='COMPRESSION_METHOD_SMOKE_PASS',arms=list(d.ARMS))),flush=True)
        waiting=list(d.ARMS);active={};launches=[];failed=[];last_wait=0.
        write_status(args.output/'workers.json',dict(workers=launches))
        while waiting or active:
            for arm in waiting[:]:
                if not predecessor_ready(args,arm):continue
                argv=command(args,'worker',arm);log=args.output/'logs'/('worker-'+arm+'.log')
                with log.open('xb') as stream:
                    process=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d.GPUS[arm])),stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                active[arm]=process;waiting.remove(arm);launches.append(dict(arm=arm,pid=process.pid,GPU=d.GPUS[arm],cwd=str(ROOT),argv=argv,log=str(log)))
                write_status(args.output/'workers.json',dict(workers=launches))
            for arm,process in list(active.items()):
                if process.poll() is None:continue
                if process.returncode or not (args.output/arm/'complete.json').exists():failed.append(dict(arm=arm,exit_code=process.returncode))
                del active[arm]
            write_status(args.output/'stages.json',dict(stage='wait_actual_predecessor_then_fixed_source_target',waiting=waiting,active={a:p.pid for a,p in active.items()},failed=failed))
            if waiting and time.monotonic()-last_wait>=30:
                print(json.dumps(dict(event='WAIT_ACTUAL_PRECEDING_GPU_HANDLES',waiting=waiting,source_score_reads=0,target_score_reads=0)),flush=True);last_wait=time.monotonic()
            if waiting or active:time.sleep(2)
        if failed:raise RuntimeError('Technical failures preserved after healthy rows completed:'+str(failed))
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_rows=list(d.ARMS),target_rows=list(d.ARMS),no_source_score_selection=True,target_feedback=False,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False));raise


def main():
    parser=argparse.ArgumentParser(description='User-designated 794529-parameter R38 further compression')
    parser.add_argument('mode',choices=('run','worker','score_source','score_target'));parser.add_argument('--arm',choices=d.ARMS)
    for key in PATH_ARGS:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    for key in PATH_ARGS:setattr(args,key,getattr(args,key).resolve())
    if args.output.name!=d.RUN or code_commit()!=args.expected_commit:raise ValueError('Pinned successor code/root required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':run(args)
    elif args.arm is None:parser.error('Fixed successor arm required')
    elif args.mode.startswith('score_'):score(args,args.mode[6:])
    else:worker(args)


if __name__=='__main__':main()

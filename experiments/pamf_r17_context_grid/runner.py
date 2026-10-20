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
from .runtime import ROOT,initialize,save_checkpoint,load_checkpoint,load_original,validate_teacher_root,validate_aliases
from .objective import clean_objective
from .training import train
from .evaluation import predict,score
from tools.r17_task_fidelity import frozen_task_fidelity
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.runner import array_tensor,read_json,code_commit
from experiments.pamf_restoration.data import IQSource,write_json

PATH_ARGS=('teacher_run','parent_action','source_cache','reader_folder','formal_reference','formal_f50','target_truth','output')


def command(args,mode,arm):
    result=[sys.executable,'-u','-m','experiments.pamf_r17_context_grid.runner',mode,'--arm',arm]
    for key in PATH_ARGS:result.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
    result.extend(['--expected-commit',args.expected_commit]);return result


def smoke(args):
    validate_teacher_root(args.teacher_run)
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu');r=x+torch.linspace(-.01,.01,256)[None,None]
    reader=load_reader(args.reader_folder,args.source_cache,'cpu').requires_grad_(False).eval();records={}
    reader_before={key:value.detach().clone() for key,value in reader.state_dict().items()}
    for arm in d.ARMS:
        model,teacher,parent=initialize(arm,args.teacher_run,'cpu');model.train()
        teacher_before={key:value.detach().clone() for key,value in teacher.state_dict().items()}
        output=model.stage_outputs(r)[0];combined,_=clean_objective(output,x,reader)
        probe=output[-1].detach().requires_grad_(True)
        probe_loss,_=frozen_task_fidelity(probe,x.detach()+torch.linspace(-.01,.01,256)[None,None],reader)
        probe_gradient=torch.autograd.grad(probe_loss.mean(),probe)[0]
        if not torch.isfinite(probe_gradient).all() or probe_gradient.abs().sum()<=0:
            raise ValueError('Original F/default head must allow finite nonzero IQ input gradients')
        combined.mean().backward()
        if any(p.grad is not None for p in reader.parameters()):raise ValueError('F head must remain frozen')
        norms=[sum(float(p.grad.norm()) for p in core.parameters() if p.grad is not None) for core in model.prior]
        if any(not np.isfinite(v) or v<=0 for v in norms) or any(p.grad is not None for p in teacher.parameters()):
            raise ValueError('Every compact prior must receive finite nonzero gradients;original teacher stays frozen')
        context_norms={name:sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None) for name,module in (('coarse',model.frozen.coarse),('action',model.frozen.estimator))}
        if any(not np.isfinite(v) or v<=0 for v in context_norms.values()):raise ValueError('Joint coarse/action input paths must receive finite nonzero gradients')
        if any(not torch.equal(v,teacher_before[k]) for k,v in teacher.state_dict().items()):raise ValueError('Initialization parent must remain unchanged')
        model.eval();path=args.output/('smoke-'+arm+'.pt');save_checkpoint(path,model,None,0,0,parent)
        fresh,_=load_checkpoint(path,'cpu')
        with torch.no_grad():
            expected=model(r);actual=fresh(r)
            if not torch.equal(expected,actual) or not torch.isfinite(reader(actual)).all():
                raise ValueError('Actual compact checkpoint and original default-head smoke failed')
        records[arm]=dict(parameters=sum(p.numel() for p in model.parameters()),all_prior_gradient_norms=norms,joint_context_gradient_norms=context_norms,
                          checkpoint_output_exact=True,original_R17_initialization_only=True,R17_training_forward_examples=0,original_F50_output_finite=True,original_F50_input_gradient_finite_nonzero=True)
    if any(not torch.equal(value,reader_before[key]) for key,value in reader.state_dict().items()):
        raise ValueError('Original F/default head parameters and buffers must remain unchanged')
    write_json(args.output/'smoke.json',dict(status='PASS',real_Ls_IQ=2,arms=records,TX_labels_read=0,V_reads=0,target_reads=0,
                                           teacher_parameter_updates=0,F_training=False,F_knowledge_in_loss=True,F_parameter_updates=0,head_fit_updates=0,optimizer_updates=0))


def wait_preceding_GPU(args):
    previous=args.output.parent/d.GPU_PRECEDING_RUN
    while True:
        live=[];saved=previous/'workers.json'
        entries=read_json(saved)['workers'] if saved.exists() else []
        for entry in entries:
            path=Path('/proc')/str(entry['pid'])
            try:
                state=(path/'stat').read_text().rsplit(')',1)[1].split()[0]
                argv=(path/'cmdline').read_bytes().decode().split(chr(0))
                if state!='Z' and str(previous) in argv and 'experiments.pamf_r17_joint_optimize.runner' in argv:live.append(entry['pid'])
            except FileNotFoundError:pass
        if any((previous/name).exists() for name in ('complete.json','failure.json')) and not live:
            write_json(args.output/'predecessor_released.json',dict(status='ACTUAL_OWNED_GPU_HANDLES_EXITED',run=d.GPU_PRECEDING_RUN,score_reads=0));return
        print(json.dumps(dict(event='WAIT_R09_OWNED_HANDLES',live_PIDs=live,score_reads=0)),flush=True);time.sleep(30)


def score_subprocess(args,role):
    import fcntl
    with (args.output.parent/d.SCORING_RUN/'score.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run(command(args,'score_'+role,args.arm),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)


@torch.no_grad()
def profile(args):
    values=array_tensor(np.load(args.parent_action/'cache/validation/practical_low_urban.npy',mmap_mode='r',allow_pickle=False)[:16],'cuda:0')
    results={}
    for name in ('original_R17','compact'):
        if name=='original_R17':model,payload=load_original(args.teacher_run/'unroll_unshared'/d.KIND/'final.pt','cpu')
        else:model,payload=load_checkpoint(args.output/args.arm/d.KIND/'final.pt','cpu')
        if (payload['epoch'],payload['updates'])!=(20,4940):raise ValueError('Fixed final profile checkpoint required')
        del payload;gc.collect()
        model.eval().requires_grad_(False).to('cuda:0')
        if name=='compact':validate_aliases(model)
        for _ in range(2):model(values[:1])
        torch.cuda.synchronize();resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
        for value in values:
            torch.cuda.synchronize();start=time.perf_counter();model(value[None]);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
        peak=torch.cuda.max_memory_allocated()
        results[name]=dict(samples=16,warmup=2,median_ms=float(np.median(times)),p90_ms=float(np.quantile(times,.9)),
            resident_before_forward_bytes=resident,peak_allocated_bytes=peak,incremental_peak_bytes=peak-resident,
            parameters=sum(p.numel() for p in model.parameters()))
        del model;gc.collect();torch.cuda.empty_cache()
    write_json(args.output/args.arm/'matched_batch1_profile.json',dict(status='PROFILE_COMPLETE',physical_GPU=d.GPUS[args.arm],
        includes_full_frozen_context_and_CG=True,checkpoint_payload_on_GPU=False,optimizer_on_GPU=False,
        source_V_IQ=16,TX_labels_read=0,state_updates=0,results=results))


def worker(args):
    try:
        train(args);torch.cuda.empty_cache();predict(args,'source');profile(args);score_subprocess(args,'source')
        write_json(args.output/args.arm/'source_complete.json',dict(status='ANALYZED',arm=args.arm,role='source',
            diagnostic_only=True,source_selection=False,all_predictions_before_truth=True,target_feedback=False))
        # All four rows were fixed before source or target access. No score gates.
        torch.cuda.empty_cache();predict(args,'target');score_subprocess(args,'target')
        write_json(args.output/args.arm/'complete.json',dict(status='ANALYZED',arm=args.arm,role='target',
            all_predictions_before_truth=True,target_feedback=False))
    except BaseException as error:
        row=args.output/args.arm;row.mkdir(exist_ok=True)
        write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),
            outputs_preserved=True,automatic_retry=False));raise


def write_status(path,value):
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)


def run(args):
    if args.output.name!=d.RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Immutable new root and CPU controller required')
    validate_teacher_root(args.teacher_run);args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,
        owner='codex/root/R17-context-grid',parent_run=d.PARENT_RUN,parent_commit=d.PARENT_COMMIT))
    write_json(args.output/'resolved_config.json',d.configuration());write_json(args.output/'comparison_plan.json',d.configuration())
    try:
        smoke(args)
        print(json.dumps(dict(event='COMPACT_SMOKE_PASS',arms=list(d.ARMS),TX_labels_read=0,V_reads=0,target_reads=0)),flush=True)
        wait_preceding_GPU(args)
        active={};launches=[];failed=[]
        for arm in d.ARMS:
            argv=command(args,'worker',arm);log=args.output/'logs'/('worker-'+arm+'.log')
            with log.open('xb') as stream:
                process=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d.GPUS[arm])),
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            record=dict(arm=arm,pid=process.pid,GPU=d.GPUS[arm],cwd=str(ROOT),argv=argv,log=str(log))
            active[arm]=process;launches.append(record)
        write_json(args.output/'workers.json',dict(workers=launches))
        while active:
            for arm,process in list(active.items()):
                if process.poll() is None:continue
                if process.returncode or not (args.output/arm/'complete.json').exists():failed.append(dict(arm=arm,exit_code=process.returncode))
                del active[arm]
            write_status(args.output/'stages.json',dict(stage='fixed_source_then_target',active={a:p.pid for a,p in active.items()},failed=failed))
            if active:time.sleep(2)
        if failed:raise RuntimeError('Technical failures preserved;healthy rows completed;no retry:'+str(failed))
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_rows=list(d.ARMS),target_rows=list(d.ARMS),
            no_source_score_selection=True,target_feedback=False,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False));raise


def main():
    parser=argparse.ArgumentParser(description='Original R17 shared low-rank fixed source/target matrix')
    parser.add_argument('mode',choices=('run','worker','score_source','score_target'));parser.add_argument('--arm',choices=d.ARMS)
    for name in PATH_ARGS:parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    for name in PATH_ARGS:setattr(args,name,getattr(args,name).resolve())
    if args.output.name!=d.RUN or code_commit()!=args.expected_commit:raise ValueError('Pinned compact code/root required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':run(args)
    elif args.arm is None:parser.error('Registered compact arm required')
    elif args.mode.startswith('score_'):score(args,args.mode[6:])
    else:worker(args)


if __name__=='__main__':main()

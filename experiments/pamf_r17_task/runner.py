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
from .calibration import smoke_info
from .runtime import ROOT,initialize,save_checkpoint,load_checkpoint,load_original,validate_teacher_root
from .models import stages,distillation_loss
from tools.r17_task_fidelity import frozen_task_fidelity
from .training import train,acquire
from .evaluation import predict,score,select_source
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.runner import array_tensor,read_json,code_commit
from experiments.pamf_restoration.data import IQSource,write_json

PATH_ARGS=('teacher_run','parent_action','source_cache','reader_folder','formal_reference','formal_f50','target_truth','output')


def command(args,mode,arm):
    result=[sys.executable,'-u','-m','experiments.pamf_r17_task.runner',mode,'--arm',arm]
    for key in PATH_ARGS:result.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
    result.extend(['--expected-commit',args.expected_commit])
    return result


def smoke(args):
    validate_teacher_root(args.teacher_run)
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu');r=x+torch.linspace(-.01,.01,256)[None,None]
    reader=load_reader(args.reader_folder,args.source_cache,'cpu').requires_grad_(False).eval();records={}
    reader_before={key:value.detach().clone() for key,value in reader.state_dict().items()}
    for arm in d.ARMS:
        model,teacher,parent=initialize(arm,args.teacher_run,'cpu',calibration_batches=[r],calibration_info=smoke_info());model.train()
        with torch.no_grad():
            context=model.frozen.estimate(r);desired=stages(teacher.prior,r,context)[0]
            if not all(torch.equal(a,b) for a,b in zip(desired,teacher.stage_outputs(r)[0])):
                raise ValueError('Reused teacher stage function must exactly reproduce original R17')
        output=model.stage_outputs(r,context)[0];wave,_=distillation_loss(output,desired,x)
        task,_=frozen_task_fidelity(output[-1],desired[-1],reader)
        probe=output[-1].detach().requires_grad_(True)
        probe_loss,_=frozen_task_fidelity(probe,desired[-1].detach()+torch.linspace(-.01,.01,256)[None,None],reader)
        probe_gradient=torch.autograd.grad(probe_loss.mean(),probe)[0]
        if not torch.isfinite(probe_gradient).all() or probe_gradient.abs().sum()<=0:
            raise ValueError('Original F/default head must allow finite nonzero IQ input gradients')
        (wave+task).mean().backward()
        if any(p.grad is not None for p in reader.parameters()):raise ValueError('F head must remain frozen')
        norms=[sum(float(p.grad.norm()) for p in core.parameters() if p.grad is not None) for core in model.prior]
        if any(not np.isfinite(v) or v<=0 for v in norms) or any(p.grad is not None for p in model.frozen.parameters()) or any(p.grad is not None for p in teacher.parameters()):
            raise ValueError('Every compact prior must receive finite nonzero gradients;teacher/context frozen')
        model.eval();path=args.output/('smoke-'+arm+'.pt');save_checkpoint(path,model,None,0,0,parent)
        fresh,_=load_checkpoint(path,'cpu')
        with torch.no_grad():
            expected=model(r);actual=fresh(r)
            if not torch.equal(expected,actual) or not torch.isfinite(reader(actual)).all():
                raise ValueError('Actual compact checkpoint and original default-head smoke failed')
        records[arm]=dict(parameters=sum(p.numel() for p in model.parameters()),all_prior_gradient_norms=norms,
                          checkpoint_output_exact=True,teacher_stage_function_exact=True,original_F50_output_finite=True,original_F50_input_gradient_finite_nonzero=True)
    if any(not torch.equal(value,reader_before[key]) for key,value in reader.state_dict().items()):
        raise ValueError('Original F/default head parameters and buffers must remain unchanged')
    write_json(args.output/'smoke.json',dict(status='PASS',real_Ls_IQ=2,arms=records,TX_labels_read=0,V_reads=0,target_reads=0,
                                           teacher_parameter_updates=0,F_training=False,F_knowledge_in_loss=True,F_parameter_updates=0,head_fit_updates=0,optimizer_updates=0))


def previous_GPU_rows_live(previous,proc_root=Path('/proc')):
    saved=previous/'workers.json';workers=read_json(saved)['workers'] if saved.exists() else [];live=[]
    for worker in workers:
        entries=[worker];target=previous/worker['arm']/'target_launch.json'
        if target.exists():entries.append(read_json(target))
        for entry in entries:
            path=proc_root/str(entry['pid'])
            try:
                state=(path/'stat').read_text().rsplit(')',1)[1].split()[0]
                argv=(path/'cmdline').read_bytes().decode().replace(chr(0),' ')
                if state!='Z' and str(previous) in argv and 'experiments.pamf_r17_calibrated.runner' in argv:live.append(entry['pid'])
            except FileNotFoundError:pass
    return live


def wait_GPU(args):
    previous=args.output.parent/d.GPU_PRECEDING_RUN;last=-30.
    while True:
        live=previous_GPU_rows_live(previous)
        if any((previous/name).exists() for name in ('complete.json','failure.json')) and not live:
            write_json(args.output/'GPU_predecessor_released.json',dict(status='PREVIOUS_OWNED_GPU_HANDLES_EXITED',
                        run=d.GPU_PRECEDING_RUN,live_owned_PIDs=[],scores_read=0));return
        if time.monotonic()-last>=30:
            print(json.dumps(dict(event='WAIT_PREVIOUS_GPU',run=d.GPU_PRECEDING_RUN,live_owned_PIDs=live,scores_read=0)),flush=True);last=time.monotonic()
        time.sleep(5)


def score_subprocess(args,role):
    import fcntl
    previous=args.output.parent/d.CPU_PRECEDING_RUN;last=-30.
    while not any((previous/name).exists() for name in ('complete.json','failure.json')):
        if time.monotonic()-last>=30:
            print(json.dumps(dict(event='WAIT_PREVIOUS_CPU',run=d.CPU_PRECEDING_RUN,scores_read=0)),flush=True);last=time.monotonic()
        time.sleep(5)
    with (args.output/'score.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run(command(args,'score_'+role,args.arm),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)


@torch.no_grad()
def profile(args):
    values=array_tensor(np.load(args.parent_action/'cache/validation/practical_low_urban.npy',mmap_mode='r',allow_pickle=False)[:16],'cuda:0')
    results={}
    for name in ('original_R17','compact'):
        if name=='original_R17':model,_=load_original(args.teacher_run/'unroll_unshared'/d.KIND/'final.pt','cuda:0')
        else:model,_=load_checkpoint(args.output/args.arm/d.KIND/'final.pt','cuda:0')
        model.eval().requires_grad_(False)
        for _ in range(2):model(values[:1])
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
        for value in values:
            torch.cuda.synchronize();start=time.perf_counter();model(value[None]);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
        results[name]=dict(samples=16,warmup=2,median_ms=float(np.median(times)),p90_ms=float(np.quantile(times,.9)),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters()))
        del model;torch.cuda.empty_cache()
    write_json(args.output/args.arm/'matched_batch1_profile.json',dict(status='PROFILE_COMPLETE',physical_GPU=d.GPUS[args.arm],
                 includes_full_frozen_context_and_CG=True,source_V_IQ=16,TX_labels_read=0,state_updates=0,results=results))


def worker(args,role):
    if role=='source':train(args)
    elif role=='target':acquire(args,training=False)
    else:raise ValueError('Registered worker role required')
    torch.cuda.empty_cache();predict(args,role)
    if role=='source':profile(args)
    score_subprocess(args,role)
    write_json(args.output/args.arm/('complete.json' if role=='source' else 'target_complete.json'),
               dict(status='ANALYZED',arm=args.arm,role=role,all_predictions_before_truth=True,target_feedback=False))


def launch(args,arm,role):
    argv=command(args,'worker_'+role,arm);log=args.output/'logs'/(role+'-'+arm+'.log')
    with log.open('xb') as stream:
        process=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d.GPUS[arm])),
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(arm=arm,role=role,pid=process.pid,GPU=d.GPUS[arm],cwd=str(ROOT),argv=argv,log=str(log))
    return process,record


def write_status(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def run(args):
    if args.output.name!=d.RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Immutable new compact root and CPU controller required')
    validate_teacher_root(args.teacher_run);args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,
               owner='codex/root/R17-task',parent_run=d.PARENT_RUN,parent_commit=d.PARENT_COMMIT))
    write_json(args.output/'resolved_config.json',d.configuration())
    try:
        smoke(args)
        print(json.dumps(dict(event='COMPACT_SMOKE_PASS',arms=list(d.ARMS),TX_labels_read=0,V_reads=0,target_reads=0)),flush=True)
        wait_GPU(args);active={};launches=[];failed=[]
        for arm in d.ARMS:
            process,record=launch(args,arm,'source');active[arm]=process;launches.append(record)
        write_json(args.output/'workers.json',dict(workers=launches))
        while active:
            for arm,process in list(active.items()):
                if process.poll() is None:continue
                if process.returncode or not (args.output/arm/'complete.json').exists():failed.append(dict(arm=arm,exit_code=process.returncode))
                del active[arm]
            write_status(args.output/'stages.json',dict(stage='source',active={a:p.pid for a,p in active.items()},failed=failed,target_reads=0))
            if active:time.sleep(2)
        if failed:raise RuntimeError('Technical source failures;all outputs preserved;no target dispatch:'+str(failed))
        decision=select_source({a:read_json(args.output/a/d.KIND/'summary.json') for a in d.ARMS},
                               read_json(args.teacher_run/'unroll_unshared'/d.KIND/'summary.json'))
        write_json(args.output/'source_decision.json',decision)
        if decision['selected']:
            arm=decision['selected'];process,record=launch(args,arm,'target');launches.append(record)
            write_status(args.output/'workers.json',dict(workers=launches))
            write_status(args.output/'stages.json',dict(stage='target',active={arm:process.pid},source_decision_frozen=True))
            code=process.wait()
            if code or not (args.output/arm/'target_complete.json').exists():raise RuntimeError('Technical target failure,outputs preserved,no retry')
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_qualified=decision['qualified'],
                    source_rows=list(d.ARMS),target_rows=decision['target_rows'],source_only=not decision['qualified'],
                    target_feedback=False,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False));raise


def main():
    parser=argparse.ArgumentParser(description='Original R17 frozen-default-head task fidelity')
    parser.add_argument('mode',choices=('run','worker_source','worker_target','score_source','score_target'))
    parser.add_argument('--arm',choices=d.ARMS)
    for name in PATH_ARGS:parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    for name in PATH_ARGS:setattr(args,name,getattr(args,name).resolve())
    if args.output.name!=d.RUN or code_commit()!=args.expected_commit:raise ValueError('Pinned compact code/root required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':run(args)
    elif args.arm is None:parser.error('Registered compact arm required')
    elif args.mode.startswith('score_'):score(args,args.mode[6:])
    else:worker(args,args.mode[7:])


if __name__=='__main__':main()

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
from .engine import InferenceEngine
from .runtime import ROOT,load_model,checkpoint,validate_source
from .evaluation import predict,score
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.runner import read_json,code_commit
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.dispatcher import capacity
from experiments.pamf_r46_multipoint_fast.tensor_io import array_tensor

PATH_ARGS=('parent_run','teacher_run','parent_action','source_cache','reader_folder','formal_reference','formal_f50','target_truth','output')


def command(args,mode,arm):
    value=[sys.executable,'-u','-m','experiments.pamf_stage2_inference.runner',mode,'--arm',arm]
    for key in PATH_ARGS:value+=['--'+key.replace('_','-'),str(getattr(args,key))]
    return value+['--expected-commit',args.expected_commit]


def acquire(args):
    import fcntl
    last=0.
    while True:
        with (args.output.parent/'.pamf-reception-capacity4.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=capacity()[d.GPUS[args.arm]]
            if state['free_mb']>=18000 and len(state['pids'])<1:
                reservation=torch.zeros(1,device='cuda:0');torch.cuda.synchronize()
                return reservation
        if time.monotonic()-last>=30:
            print(json.dumps(dict(event='WAIT_GPU_CAPACITY',arm=args.arm,GPU=d.GPUS[args.arm],existing_workers=len(state['pids']))),flush=True)
            last=time.monotonic()
        time.sleep(5)


def smoke(args,model,reader,pipeline):
    roles=np.load(args.parent_action/'cache/train/roles.npy',allow_pickle=False)
    take=np.flatnonzero(roles=='L_s')[:2]
    if len(take)!=2:raise ValueError('Two actual Ls source examples required')
    values=array_tensor(np.load(args.parent_action/'cache/epochs/01/received.npy',mmap_mode='r',allow_pickle=False)[take],'cuda:0')
    snapshots=[{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for m in (model,reader)]
    direct=InferenceEngine(model,reader,graph=False,native=True)
    records={};memory=[]
    for batch in (1,8,32,64,128):
        x=values.repeat((batch+1)//2,1,1)[:batch]
        other=values.flip(0).repeat((batch+1)//2,1,1)[:batch]
        native=direct(x);first=pipeline(x);held=tuple(v.clone() for v in first)
        second=pipeline(other);desired=direct(other)
        for a,b in zip(first,held):
            if not torch.equal(a,b):raise ValueError('Replay overwrote returned output')
        for index,(a,b) in enumerate((*zip(native,first),*zip(desired,second))):
            atol=1e-5 if index%4<2 else 1e-3
            torch.testing.assert_close(a,b,rtol=2e-4,atol=atol)
        records[str(batch)]=dict(max_abs=[float((a-b).abs().max()) for a,b in zip(native,first)],
            bitwise=[torch.equal(a,b) for a,b in zip(native,first)],input_replaced=True,outputs_owned=True)
        del native,first,held,second,desired,x,other
        gc.collect();torch.cuda.synchronize()
        memory.append(dict(batch=batch,allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved()))
    for module,snapshot in zip((model,reader),snapshots):
        if any(p.grad is not None or p.requires_grad for p in module.parameters()):raise ValueError('Frozen inference only')
        if any(not torch.equal(v.cpu(),snapshot[k]) for k,v in module.state_dict().items()):raise ValueError('Original weights changed')
    write_json(args.output/args.arm/'smoke.json',dict(status='PASS',real_Ls_IQ=2,batches=records,TX_labels_read=0,target_reads=0,
        optimizer_updates=0,F_parameter_updates=0,head_fit_updates=0,original_weights_unchanged=True,native_fallback=False))
    write_json(args.output/args.arm/'standalone_graph_memory.json',dict(scope='One original G+F and cumulative graph pools B1/8/32/64/128; no control model or optimizer',
        per_batch_cumulative=memory,setup=pipeline.setup()))
    print(json.dumps(dict(event='REAL_CHECKPOINT_LS_SMOKE_PASS',arm=args.arm)),flush=True)


def profile(args,model,reader,current_pipeline):
    values=array_tensor(np.load(args.parent_action/'cache/validation/practical_low_urban.npy',mmap_mode='r',allow_pickle=False)[:128],'cuda:0')
    results={};setup={}
    for mode in ('G_only','G_plus_F'):
        selected=reader if mode=='G_plus_F' else None
        calls={name:current_pipeline if name==args.arm and selected is not None else
            InferenceEngine(model,selected,reuse=name==d.ARMS[0],graph=name!='native',native=name=='native')
            for name in ('native',*d.ARMS)}
        for batch,warm,iterations in ((1,16,64),(8,4,16),(32,4,16),(128,4,8)):
            for fn in calls.values():fn(values[:batch])
            samples={name:[] for name in calls};medians={name:[] for name in calls};memory={name:[] for name in calls}
            for repeat in range(3):
                for name in (list(calls) if repeat%2==0 else list(reversed(calls))):
                    fn=calls[name]
                    for _ in range(warm):fn(values[:batch])
                    gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                    resident=torch.cuda.memory_allocated();times=[]
                    for index in range(iterations):
                        offset=(index*batch)%(129-batch);x=values[offset:offset+batch]
                        torch.cuda.synchronize();started=time.perf_counter();y=fn(x);torch.cuda.synchronize()
                        times.append((time.perf_counter()-started)*1000);del y
                    samples[name].extend(times);medians[name].append(float(np.median(times)))
                    memory[name].append(dict(resident_bytes=resident,peak_allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.memory_reserved()))
            for name in calls:
                times=np.asarray(samples[name]);results[mode+'_B'+str(batch)+'_'+name]=dict(batch=batch,execution=name,rounds=3,warmup_per_round=warm,
                    iterations_per_round=iterations,source_V_unique_IQ=128,median_ms=float(np.median(times)),p90_ms=float(np.quantile(times,.9)),
                    p95_ms=float(np.quantile(times,.95)),throughput_samples_per_second=float(batch*1000/times.mean()),round_median_ms=medians[name],memory_rounds=memory[name])
        setup[mode]={name:fn.setup() for name,fn in calls.items()}
        del calls;gc.collect();torch.cuda.empty_cache()
    write_json(args.output/args.arm/'matched_inference_profile.json',dict(status='PROFILE_COMPLETE',arm=args.arm,physical_GPU=d.GPUS[args.arm],
        parameters=d.PARAMETERS,ConvLinear_MAC_per_IQ=25068072,FFT_calls=d.configuration()['FFT_calls'],results=results,setup=setup,
        TX_labels_read=0,target_reads=0,state_updates=0,training_F_access=False,
        timing_scope='InferenceEngine shared by actual formal prediction and timing; synchronized FP32 preloaded-device; input copy, owned outputs and one device-validity check included; setup separate',
        memory_scope='One G+F plus multiple execution graph pools; independent deployment pool figures in standalone_graph_memory.json',
        MAC_scope='Conv1d+Linear only; excludes FFT/CG/norm/activation/F/memory'))


def score_subprocess(args,role):
    import fcntl
    with (args.output.parent/d.SCORING_RUN/'score.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run(command(args,'score_'+role,args.arm),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)


def worker(args):
    row=args.output/args.arm;row.mkdir()
    try:
        reservation=acquire(args)
        (row/'reference_controls').symlink_to(args.teacher_run/'unroll_unshared/reference_controls',target_is_directory=True)
        write_json(row/'gpu_ready.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=d.GPUS[args.arm]))
        model,payload=load_model(args,'cuda:0')
        reader=load_reader(args.reader_folder,args.source_cache,'cuda:0',seed=d.F_SEED).eval().requires_grad_(False)
        pipeline=InferenceEngine(model,reader,reuse=True)
        smoke(args,model,reader,pipeline)
        write_json(row/'runtime_ready.json',dict(status='FROZEN_RUNTIME_READY',arm=args.arm,parameters=d.PARAMETERS,new_training_updates=0,F_parameter_updates=0,head_fit_updates=0))
        write_json(row/'checkpoint_source.json',dict(path=str(checkpoint(args)),source_arm=payload['arm'],epoch=payload['epoch'],updates=payload['updates'],
            checkpoint_bytes=checkpoint(args).stat().st_size,new_training_updates=0,training_F_access=False))
        del payload
        predict(args,'source',pipeline);profile(args,model,reader,pipeline);score_subprocess(args,'source')
        write_json(row/'source_complete.json',dict(status='ANALYZED',arm=args.arm,source_selection=False,target_feedback=False))
        predict(args,'target',pipeline);score_subprocess(args,'target')
        write_json(row/'complete.json',dict(status='ANALYZED',arm=args.arm,all_predictions_before_truth=True,target_feedback=False))
        del reservation
    except BaseException as error:
        write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False))
        raise


def write_status(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)


def run(args):
    if args.output.name!=d.RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Immutable new root and CPU controller required')
    validate_source(args);args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,owner='codex/root/stage2-inference'))
    write_json(args.output/'comparison_plan.json',d.configuration());write_json(args.output/'resolved_config.json',d.configuration())
    active={};workers=[];failed=[]
    try:
        for arm in d.ARMS:
            argv=command(args,'worker',arm);log=args.output/'logs'/('worker-'+arm+'.log')
            with log.open('xb') as stream:
                p=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d.GPUS[arm])),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            active[arm]=p;workers.append(dict(arm=arm,pid=p.pid,GPU=d.GPUS[arm],cwd=str(ROOT),argv=argv,log=str(log)))
        write_status(args.output/'workers.json',dict(workers=workers))
        while active:
            for arm,p in list(active.items()):
                if p.poll() is None:continue
                if p.returncode or not (args.output/arm/'complete.json').exists():failed.append(dict(arm=arm,exit_code=p.returncode))
                del active[arm]
            write_status(args.output/'stages.json',dict(active={a:p.pid for a,p in active.items()},failed=failed))
            if active:time.sleep(2)
        if failed:raise RuntimeError('Technical failures preserved after healthy rows completed: '+str(failed))
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_rows=list(d.ARMS),target_rows=list(d.ARMS),target_feedback=False,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False))
        raise


def main():
    parser=argparse.ArgumentParser(description='Frozen two-stage distillation with graph/synchronization/residual-reuse optimization')
    parser.add_argument('mode',choices=('run','worker','score_source','score_target'));parser.add_argument('--arm',choices=d.ARMS)
    for key in PATH_ARGS:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    for key in PATH_ARGS:setattr(args,key,getattr(args,key).resolve())
    if args.output.name!=d.RUN or code_commit()!=args.expected_commit:raise ValueError('Pinned code and immutable run required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':run(args)
    elif args.arm is None:parser.error('Registered row required')
    elif args.mode.startswith('score_'):score(args,args.mode[6:])
    else:worker(args)


if __name__=='__main__':main()

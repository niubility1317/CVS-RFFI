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
from .runtime import ROOT,initialize,save_checkpoint,load_checkpoint,validate_parent
from .training import train,objective
from .evaluation import predict,score
from experiments.pamf_r46_delayed_prune.runtime import load_checkpoint as load_parent
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.runner import read_json,code_commit
from experiments.pamf_r46_multipoint_fast.tensor_io import array_tensor
from experiments.pamf_restoration.data import write_json

PATH_ARGS=('parent_run','teacher_run','parent_action','source_cache','reader_folder','formal_reference','formal_f50','target_truth','output')


def command(args,mode,arm):
    argv=[sys.executable,'-u','-m','experiments.pamf_r65_four_route.runner',mode,'--arm',arm]
    for key in PATH_ARGS:argv.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
    return argv+['--expected-commit',args.expected_commit]


def smoke(args):
    cache=args.parent_action/'cache';roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    take=np.flatnonzero(roles=='L_s')[:2]
    if len(take)!=2:raise ValueError('Two actual legal source L_s examples required')
    x=array_tensor(np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)[take],'cpu')
    received=array_tensor(np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)[take],'cpu')
    reader=load_reader(args.reader_folder,args.source_cache,'cpu',seed=d.F_SEED).eval().requires_grad_(False)
    reader_before={k:v.clone() for k,v in reader.state_dict().items()};records={}
    for arm in d.ARMS:
        torch.manual_seed(d.MODEL_SEED)
        model,teacher=initialize(arm,args.parent_run,'cpu')
        before={k:v.clone() for k,v in teacher.state_dict().items()}
        with torch.no_grad():desired=teacher.stage_outputs(received)[0]
        outputs=model.stage_outputs(received)[0]
        loss,_=objective(outputs,desired,x,1,arm);loss.mean().backward()
        grads={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
        if not grads or any(not np.isfinite(v) for v in grads.values()) or not any(v>0 for v in grads.values()):
            raise ValueError('Finite nonzero source student gradients required')
        if any(p.grad is not None for p in teacher.parameters()) or any(not torch.equal(v,before[k]) for k,v in teacher.state_dict().items()):
            raise ValueError('Frozen source teacher changed')
        model.eval();path=args.output/('smoke-'+arm+'.pt')
        save_checkpoint(path,model,0,0,args.expected_commit);fresh,payload=load_checkpoint(path,'cpu')
        with torch.no_grad():
            a=model(received);b=fresh(received)
            if not torch.equal(a,b) or not torch.isfinite(reader(b)).all():raise ValueError('Portable strict reload/original F50 smoke failed')
        records[arm]=dict(parameters=sum(p.numel() for p in model.parameters()),stages=len(outputs),
            checkpoint_output_exact=True,loss=float(loss.detach().mean()),nonzero_gradient_tensors=sum(v>0 for v in grads.values()),
            initialization=model.initialization_report)
    if any(not torch.equal(v,reader_before[k]) for k,v in reader.state_dict().items()):raise ValueError('Original F/default head changed')
    write_json(args.output/'smoke.json',dict(status='PASS',real_Ls_IQ=2,arms=records,TX_labels_read=0,V_reads=0,target_reads=0,
        teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,optimizer_updates=0))


def score_subprocess(args,role):
    import fcntl
    with (args.output.parent/d.SCORING_RUN/'score.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        subprocess.run(command(args,'score_'+role,args.arm),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)


@torch.no_grad()
def profile(args):
    values=array_tensor(np.load(args.parent_action/'cache/validation/practical_low_urban.npy',mmap_mode='r',allow_pickle=False)[:128],'cuda:0')
    reader=load_reader(args.reader_folder,args.source_cache,'cuda:0',seed=d.F_SEED).eval().requires_grad_(False)
    parent,_=load_parent(args.parent_run/d.PARENT_ROW/d.KIND/'final.pt','cpu')
    student,_=load_checkpoint(args.output/args.arm/d.KIND/'final.pt','cpu')
    models={name:model.eval().requires_grad_(False).to('cuda:0') for name,model in dict(R65=parent,student=student).items()}
    results={name:dict(parameters=sum(p.numel() for p in model.parameters()),cases={}) for name,model in models.items()}
    for mode in ('G_only','G_plus_F'):
        for batch,warmup,iterations in ((1,16,64),(8,4,16),(32,4,16),(128,4,16)):
            samples={k:[] for k in models};rounds={k:[] for k in models};peaks={k:[] for k in models};resident={k:[] for k in models}
            for repeat in range(3):
                names=list(models) if repeat%2==0 else list(reversed(models))
                for name in names:
                    model=models[name]
                    def forward(x):
                        y=model(x)
                        return reader(y) if mode=='G_plus_F' else y
                    for _ in range(warmup):forward(values[:batch])
                    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
                    base_memory=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
                    for index in range(iterations):
                        start=(index*batch)%(len(values)-batch+1);x=values[start:start+batch]
                        torch.cuda.synchronize();started=time.perf_counter();y=forward(x);torch.cuda.synchronize()
                        times.append((time.perf_counter()-started)*1000)
                        if not torch.isfinite(y).all():raise FloatingPointError('Nonfinite matched latency output')
                        del y
                    samples[name].extend(times);rounds[name].append(float(np.median(times)))
                    peaks[name].append(torch.cuda.max_memory_allocated());resident[name].append(base_memory)
            for name in models:
                times=np.asarray(samples[name]);key=mode+'_B'+str(batch)
                results[name]['cases'][key]=dict(batch=batch,warmup_per_round=warmup,rounds=3,iterations_per_round=iterations,
                    median_ms=float(np.median(times)),p90_ms=float(np.quantile(times,.9)),p95_ms=float(np.quantile(times,.95)),
                    round_median_ms=rounds[name],throughput_samples_per_second=float(batch*1000/times.mean()),
                    resident_before_forward_bytes=max(resident[name]),peak_allocated_bytes=max(peaks[name]),
                    incremental_peak_bytes=max(p-r for p,r in zip(peaks[name],resident[name])))
    from experiments.pamf_iq_joint_compact.models import convolution_linear_macs
    for name,model in models.items():results[name]['ConvLinear_MAC_per_IQ']=convolution_linear_macs(model,values[:1])
    write_json(args.output/args.arm/'matched_inference_profile.json',dict(status='PROFILE_COMPLETE',physical_GPU=d.GPUS[args.arm],results=results,
        source_V_IQ=128,TX_labels_read=0,state_updates=0,checkpoint_payload_on_GPU=False,optimizer_on_GPU=False,
        memory_scope='TwoGandoneFsharedresident;incrementalpeakperforward;notstandaloneprocessVRAM',
        MAC_scope='Conv1d+Linear only;excludes FFT/CG/norm/activation/preconditioner/F',
        timing_scope='Samecard alternatingforwardreverseorder;FP32 no_grad;preloadedGPUinputs;excludesloader/transfer/queue/artifactIO',
        whole_pipeline='GplusF includes original defaulthead FFT/PA/time/stats;both realprediction andtiming use same model path'))


def worker(args):
    try:
        train(args);gc.collect();torch.cuda.empty_cache()
        predict(args,'source');gc.collect();torch.cuda.empty_cache()
        profile(args);gc.collect();torch.cuda.empty_cache()
        score_subprocess(args,'source')
        write_json(args.output/args.arm/'source_complete.json',dict(status='ANALYZED',arm=args.arm,diagnostic_only=True,source_selection=False,target_feedback=False))
        predict(args,'target');gc.collect();torch.cuda.empty_cache()
        score_subprocess(args,'target')
        write_json(args.output/args.arm/'complete.json',dict(status='ANALYZED',arm=args.arm,all_seven_predictions_before_truth=True,target_feedback=False))
    except BaseException as error:
        write_json(args.output/(args.arm+'-failure.json'),dict(status='FAILED_TECHNICAL',arm=args.arm,error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True))
        raise


def write_status(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)


def run(args):
    if args.output.name!=d.RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable run root and CPU controller required')
    validate_parent(args.parent_run);args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,
        owner='codex/root/R65-four-route',physical_GPUs=[0,1]))
    write_json(args.output/'comparison_plan.json',d.configuration());write_json(args.output/'resolved_config.json',d.configuration())
    try:
        smoke(args);print(json.dumps(dict(event='R65_FOUR_ROUTE_SMOKE_PASS',arms=list(d.ARMS))),flush=True)
        waiting=list(d.ARMS);active={};launches=[];failed=[]
        while waiting or active:
            used={d.GPUS[arm] for arm in active}
            for arm in waiting[:]:
                if d.GPUS[arm] in used:continue
                argv=command(args,'worker',arm);log=args.output/'logs'/('worker-'+arm+'.log')
                with log.open('xb') as stream:
                    process=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d.GPUS[arm])),
                        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                active[arm]=process;waiting.remove(arm);used.add(d.GPUS[arm])
                launches.append(dict(arm=arm,pid=process.pid,GPU=d.GPUS[arm],cwd=str(ROOT),argv=argv,log=str(log)))
                write_status(args.output/'workers.json',dict(workers=launches))
            for arm,process in list(active.items()):
                if process.poll() is None:continue
                if process.returncode or not (args.output/arm/'complete.json').exists():failed.append(dict(arm=arm,exit_code=process.returncode))
                del active[arm]
            write_status(args.output/'stages.json',dict(waiting=waiting,active={a:p.pid for a,p in active.items()},failed=failed))
            if waiting or active:time.sleep(3)
        if failed:raise RuntimeError('Technical failures preserved after healthy routes completed:'+str(failed))
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_rows=list(d.ARMS),target_rows=list(d.ARMS),no_source_selection=True,target_feedback=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False))
        raise


def main():
    parser=argparse.ArgumentParser(description='Four fixed source-only R65 architecture and latency comparisons')
    parser.add_argument('mode',choices=('run','worker','score_source','score_target'));parser.add_argument('--arm',choices=d.ARMS)
    for key in PATH_ARGS:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    for key in PATH_ARGS:setattr(args,key,getattr(args,key).resolve())
    if args.output.name!=d.RUN or code_commit()!=args.expected_commit:raise ValueError('Pinned code and output run identity required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':run(args)
    elif args.arm is None:parser.error('Registered route required')
    elif args.mode.startswith('score_'):score(args,args.mode[6:])
    else:worker(args)


if __name__=='__main__':main()
